"""MATD3 / GRPO 从头训练、验证选模及多次独立运行管理。"""

import shutil
import sys
from pathlib import Path
from time import monotonic
from typing import Any
import numpy as np
from .agent import (
    MATD3Agent,
    _default_checkpoint_root,
    _write_run_manifest,
)
from ..config import load_market_settings
from ..errors import InputValidationError
from .learning_metrics import _EpisodeMetrics, _metrics, _write_metrics, _write_run_summary, _write_records
from ..models import MarketSettings, SimulationResult
from ..reporting import write_results
from ..simulation import LearningEpisodeRunner, learning_period_settings, market_openings
from .test import _evaluate_actor_and_baseline
from .data_coverage import validate_learning_data_coverage

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


def _train_single_run(
    input_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    scenario: str = "base",
    *,
    run: int,
    seed: int,
    training_episodes: int | None = None,
    training_steps: int | None = None,
) -> tuple[SimulationResult, list[_EpisodeMetrics], Path, float, int | None]:
    """完成一次统计上独立的 MATD3 训练：只用训练时段更新，只用验证时段选模型。"""

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
    episode_count = (
        config.training_episodes if training_episodes is None else training_episodes
    )
    if episode_count <= 0:
        raise InputValidationError("Training episodes must be positive.")
    budget = config.training_steps if training_steps is None else training_steps
    if budget:
        if budget <= 0 or budget % steps_per_episode:
            raise InputValidationError(f"MATD3 training_steps 必须是 {steps_per_episode} 的正整数倍。")
        episode_count = budget // steps_per_episode
    elif training_steps is not None:
        raise InputValidationError("training_steps 必须为正整数。")
    regular_episode_count = max(0, episode_count - config.initial_experience_episodes)
    initial_transition_count = (
        min(episode_count, config.initial_experience_episodes) * steps_per_episode
    )
    regular_transition_count = regular_episode_count * steps_per_episode
    warmup_transitions = max(0, config.batch_size - initial_transition_count - 1)
    eligible_regular_steps = max(0, regular_transition_count - warmup_transitions)
    agent = MATD3Agent(
        config,
        total_regular_steps=max(1, eligible_regular_steps),
        seed=seed,
        run=run,
    )
    best_path = checkpoint_root / "best.pt"
    best_path.unlink(missing_ok=True)

    # 只有训练时段的经验进入经验池；验证时段只运行确定性 Actor。
    train_runner = LearningEpisodeRunner(input_path, scenario=scenario, period="train")
    validation_runner = LearningEpisodeRunner(
        input_path, scenario=scenario, period="validation"
    )
    metrics: list[_EpisodeMetrics] = []
    best_total_reward = float("-inf")
    progress = []
    started = monotonic()
    validation_seconds = 0.0
    for episode in _training_progress(episode_count, run):
        initial_experience = episode <= config.initial_experience_episodes
        training_result = train_runner.run_episode(
            lambda state, initial=initial_experience: agent.exploration_action(
                state, initial_experience=initial
            ),
            transition_consumer=lambda transition, enabled=not initial_experience: (
                agent.observe(transition, train=enabled)
            ),
        )
        metrics.append(
            _metrics(
                training_result,
                run=run,
                phase="initial_experience" if initial_experience else "train",
                episode=episode,
                gamma=config.gamma,
            )
        )
        is_validation_due = episode % config.validation_interval == 0
        is_last_episode = episode == episode_count
        should_validate = is_validation_due or is_last_episode
        if should_validate:
            validation_started = monotonic()
            validation_result = validation_runner.run_episode(agent.deterministic_action)
            validation_seconds += monotonic() - validation_started
            validation_metrics = _metrics(
                validation_result,
                run=run,
                phase="validation",
                episode=episode,
                gamma=config.gamma,
            )
            metrics.append(validation_metrics)
            progress.append({
                "episode": episode, "interaction_steps": episode * steps_per_episode,
                "elapsed_seconds": monotonic() - started, "validation_seconds": validation_seconds,
                "total_reward": validation_metrics.total_reward,
            })
            # 按验证时段总奖励选模型；打平时保留更早的检查点。
            if (
                not initial_experience
                and validation_metrics.total_reward > best_total_reward
            ):
                best_total_reward = validation_metrics.total_reward
                agent.save_checkpoint(
                    best_path,
                    episode=episode,
                    validation_total_reward=best_total_reward,
                )

    if best_total_reward == float("-inf") or not best_path.is_file():
        raise InputValidationError(
            "Training did not produce a best checkpoint; increase training_episodes "
            "or reduce episodes_collecting_initial_experience."
        )

    seed_used = agent.seed
    best_payload = agent.load_checkpoint(best_path)
    validation_result, comparison = _evaluate_actor_and_baseline(
        agent=agent,
        input_path=input_path,
        output_path=output_path / "validation",
        scenario=scenario,
        period="validation",
        episode=int(best_payload["episode"]),
        gamma=config.gamma,
        run=run,
    )
    metrics.extend(comparison)
    _write_metrics(output_path / "learning_metrics.csv", metrics)
    _write_records(output_path / "learning_progress.csv", progress)
    return (
        validation_result,
        metrics,
        best_path,
        best_total_reward,
        seed_used,
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

    validate_learning_data_coverage(input_path, settings)

    checkpoint_root = _default_checkpoint_root(config, output_path)
    use_run_dirs = run_count > 1
    all_metrics: list[_EpisodeMetrics] = []
    run_records: list[dict[str, Any]] = []
    results: dict[int, SimulationResult] = {}
    for run in range(1, run_count + 1):
        run_output = (
            output_path if not use_run_dirs else output_path / "runs" / f"run_{run:02d}"
        )
        run_checkpoints = (
            checkpoint_root if not use_run_dirs else checkpoint_root / f"run_{run:02d}"
        )
        result, metrics, best_path, best_reward, seed = _train_single_run(
            input_path,
            run_output,
            run_checkpoints,
            scenario,
            run=run,
            seed=_independent_run_seed(settings.seed, run),
            training_episodes=training_episodes,
            training_steps=training_steps,
        )
        results[run] = result
        all_metrics.extend(metrics)
        run_records.append({
            "run": run,
            "seed": seed,
            "best_checkpoint": best_path.relative_to(checkpoint_root).as_posix(),
            "validation_total_reward": best_reward,
        })

    # 验证总奖励打平时保留更早的运行。
    selected_entry = max(
        run_records, key=lambda entry: float(entry["validation_total_reward"])
    )
    selected_run = int(selected_entry["run"])
    _copy_if_distinct(
        checkpoint_root / selected_entry["best_checkpoint"],
        checkpoint_root / "best.pt",
    )
    _write_run_manifest(checkpoint_root, config, run_records, selected_run)
    root_result = results[selected_run]
    write_results(output_path, root_result)

    _write_metrics(output_path / "learning_metrics.csv", all_metrics)
    _write_run_summary(output_path / "learning_run_summary.csv", all_metrics)
    return root_result
