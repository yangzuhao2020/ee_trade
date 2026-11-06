"""MATD3 / GRPO 算法选择、公共进度显示及多次独立训练管理。"""

import shutil
import sys
from pathlib import Path
from time import monotonic
from typing import Any
import numpy as np
from .checkpoints import (
    _default_checkpoint_root,
    _write_run_manifest,
)
from ..config import load_market_settings
from ..errors import InputValidationError
from .learning_metrics import _EpisodeMetrics, _write_metrics, _write_run_summary
from ..models import MarketSettings, SimulationResult
from ..reporting import write_results
from ..simulation import learning_period_settings, market_openings

_DEFAULT_INDEPENDENT_RUNS = 1


def _training_progress(total: int, run: int):
    """终端中按轮刷新进度；预计时间只按本次实际完成的轮数计算。"""
    if not sys.stderr.isatty():
        yield from range(1, total + 1)
        return
    started = monotonic()
    current = 0
    width = 0
    try:
        while True:
            elapsed = monotonic() - started
            eta = f"{elapsed * (total - current) / current:.0f}s" if current else "--"
            filled = 20 * current // total
            message = (
                f"Run {run} [{'#' * filled}{'-' * (20 - filled)}] "
                f"{current}/{total} | elapsed {elapsed:.0f}s | ETA {eta}"
            )
            print("\r" + message.ljust(width), end="", file=sys.stderr, flush=True)
            width = len(message)
            if current == total:
                break
            current += 1
            yield current
    finally:
        print(file=sys.stderr, flush=True)


def _independent_run_seed(base_seed: int | None, run: int) -> int:
    """按运行序号偏移场景随机种子；场景种子为 null 时随机生成一个会被记录的种子。"""
    if base_seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    return base_seed + run - 1


def _steps_per_episode(settings: MarketSettings, period: str) -> int:
    """返回指定数据时段一个回合包含的交付时段数。"""
    period_settings = learning_period_settings(settings, period)
    return sum(len(opening.products) for opening in market_openings(period_settings))


def _copy_if_distinct(source: Path, destination: Path) -> None:
    """复制文件；源与目标为同一文件时跳过。"""
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _train_independent_run(
    input_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    scenario: str = "base",
    *,
    run: int,
    seed: int,
    training_episodes: int | None = None,
    training_steps: int | None = None,
) -> tuple[SimulationResult, list[_EpisodeMetrics], Path, float]:
    """读取配置，调用对应算法完成一次独立训练。"""

    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = settings.learning_config
    steps_per_episode = _steps_per_episode(settings, "train")
    if training_episodes is not None and (
        training_steps is not None or config.training_steps or config.algorithm == "grpo"
    ):
        raise InputValidationError("GRPO 使用 training_steps；training_episodes 不能与步数预算同时指定。")
    if config.algorithm == "grpo":
        from .grpo_training import train_single_run
        return train_single_run(
            input_path, output_path, checkpoint_root, scenario, config=config,
            run=run, seed=seed, steps_per_episode=steps_per_episode,
            training_steps=training_steps,
        )
    from .matd3_training import train_single_run
    return train_single_run(
        input_path, output_path, checkpoint_root, scenario, config=config,
        run=run, seed=seed, steps_per_episode=steps_per_episode,
        training_episodes=training_episodes, training_steps=training_steps,
    )


def train_learning_scenario(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    training_episodes: int | None = None,
    independent_runs: int | None = None,
    training_steps: int | None = None,
) -> SimulationResult:
    """从头训练多次独立运行，按验证总奖励选模型并写出运行清单。
    返回入选运行在验证时段上的结果；测试时段由独立测试命令运行。
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = settings.learning_config
    run_count = _DEFAULT_INDEPENDENT_RUNS if independent_runs is None else independent_runs
    if run_count <= 0:
        raise InputValidationError("Independent learning runs must be positive.")

    checkpoint_root = _default_checkpoint_root(config, output_path)
    use_run_dirs = run_count > 1
    all_metrics: list[_EpisodeMetrics] = []
    run_records: list[dict[str, Any]] = []
    selected_reward = float("-inf")
    for run in range(1, run_count + 1):
        run_output = (
            output_path if not use_run_dirs else output_path / "runs" / f"run_{run:02d}"
        )
        run_checkpoints = (
            checkpoint_root if not use_run_dirs else checkpoint_root / f"run_{run:02d}"
        )
        seed = _independent_run_seed(settings.seed, run)
        result, metrics, best_path, best_reward = _train_independent_run(
            input_path,
            run_output,
            run_checkpoints,
            scenario,
            run=run,
            seed=seed,
            training_episodes=training_episodes,
            training_steps=training_steps,
        )
        # 验证总奖励打平时保留更早的运行，只保留当前最佳的完整结果。
        if run == 1 or best_reward > selected_reward:
            selected_run = run
            selected_reward = best_reward
            root_result = result
        del result
        
        if use_run_dirs:
            _write_metrics(run_output / "learning_metrics.csv", metrics)
        all_metrics.extend(metrics)
        run_records.append({
            "run": run,
            "seed": seed,
            "best_checkpoint": best_path.relative_to(checkpoint_root).as_posix(),
            "validation_total_reward": best_reward,
        })

    selected_entry = run_records[selected_run - 1]
    _copy_if_distinct(
        checkpoint_root / selected_entry["best_checkpoint"],
        checkpoint_root / "best.pt",
    )
    _write_run_manifest(checkpoint_root, config, run_records, selected_run)
    write_results(output_path, root_result)

    _write_metrics(output_path / "learning_metrics.csv", all_metrics)
    _write_run_summary(output_path / "learning_run_summary.csv", all_metrics)
    return root_result
