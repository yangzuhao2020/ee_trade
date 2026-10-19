"""第五版 MATD3 训练、恢复训练、验证选模及多运行清单编排。"""

import shutil
from pathlib import Path
from typing import Any
import numpy as np
from .agent import (
    MATD3Agent,
    _RUN_MANIFEST,
    _checkpoint_from_setting,
    _default_checkpoint_root,
    _learning_config,
    _periods_payload,
    _read_checkpoint_payload,
    _read_run_manifest,
    _write_run_manifest,
)
from ..config import load_market_settings
from ..errors import InputValidationError
from .learning_metrics import _EpisodeMetrics, _metrics, _write_metrics, _write_run_summary
from ..models import LearningConfig, MarketSettings, SimulationResult
from ..reporting import write_results
from ..simulation import LearningEpisodeRunner, learning_period_settings, market_openings
from .test import _evaluate_actor_and_baseline

_DEFAULT_INDEPENDENT_RUNS = 3


def _restore_best_checkpoint(
    *,
    restored_path: Path,
    destination: Path,
    expected_reward: float,
    expected_seed: int | None,
    expected_max_bid_price: float,
    expected_periods: dict[str, list[str]],
) -> None:
    """复制与之配套且真实匹配的最佳检查点，不改变 Agent。"""
    candidates = [destination]
    companion = (
        restored_path
        if restored_path.name == "best.pt"
        else restored_path.parent / "best.pt"
    )
    if companion not in candidates:
        candidates.append(companion)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = _read_checkpoint_payload(candidate, "cpu")
        saved_reward = payload.get("validation_total_reward")
        try:
            matches = (
                saved_reward is not None and float(saved_reward) == expected_reward
            )
            matches = (
                matches
                and float(payload["max_bid_price"]) == expected_max_bid_price
                and payload.get("periods") == expected_periods
                and (expected_seed is None or payload.get("seed") == expected_seed)
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            continue
        if candidate.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
        return
    raise InputValidationError(
        "Resume checkpoint records a historical best reward but no matching "
        f"best.pt is available beside {restored_path}."
    )


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


def _resume_run_number(
    config: LearningConfig,
    checkpoint_root: Path,
    checkpoint_path: str | Path | None,
) -> int | None:
    """读取继续训练的检查点所记录的运行编号，作为恢复运行的身份。
    检查点缺失 run 字段时按 1 处理；continue_learning 的文件不存在时
    返回 None，由 load_checkpoint 给出一致的不存在报错。
    """
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
    else:
        resolved = _checkpoint_from_setting(
            config.trained_policies_load_path, checkpoint_root, "latest.pt"
        )
        if not resolved.is_file():
            return None
    payload = _read_checkpoint_payload(resolved, "cpu")
    run = payload.get("run")
    if isinstance(run, int) and not isinstance(run, bool):
        return run
    return 1


def _train_single_run(
    input_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    scenario: str = "base",
    *,
    run: int,
    seed: int,
    checkpoint_path: str | Path | None = None,
    training_episodes: int | None = None,
) -> tuple[SimulationResult, list[_EpisodeMetrics], Path, Path, float, int | None]:
    """完成一次统计上独立的 MATD3 训练：只用训练时段更新，只用验证时段选模型。"""

    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    episode_count = (
        config.training_episodes if training_episodes is None else training_episodes
    )
    if episode_count <= 0:
        raise InputValidationError("Training episodes must be positive.")
    steps_per_episode = _steps_per_episode(settings, "train")
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
    latest_path = checkpoint_root / "latest.pt"
    best_path = checkpoint_root / "best.pt"

    restored: dict[str, Any] | None = None
    restored_path: Path | None = None
    requested_checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else None
    )
    if requested_checkpoint is not None:
        restored_path = requested_checkpoint
        restored = agent.load_checkpoint(requested_checkpoint, load_optimizers=True)
    elif config.continue_learning:
        load_path = _checkpoint_from_setting(
            config.trained_policies_load_path,
            checkpoint_root,
            "latest.pt",
        )
        restored_path = load_path
        restored = agent.load_checkpoint(load_path, load_optimizers=True)

    # 只有训练时段的经验进入经验池；验证时段只运行确定性 Actor。
    train_runner = LearningEpisodeRunner(input_path, scenario=scenario, period="train")
    validation_runner = LearningEpisodeRunner(
        input_path, scenario=scenario, period="validation"
    )
    metrics: list[_EpisodeMetrics] = []
    best_total_reward = float("-inf")
    first_episode = 1
    if restored is not None:
        first_episode = int(restored["episode"]) + 1
    if restored is not None and restored.get("validation_total_reward") is not None:
        best_total_reward = float(restored["validation_total_reward"])
        assert restored_path is not None
        _restore_best_checkpoint(
            restored_path=restored_path,
            destination=best_path,
            expected_reward=best_total_reward,
            expected_seed=(
                int(restored["seed"]) if restored.get("seed") is not None else None
            ),
            expected_max_bid_price=config.max_bid_price,
            expected_periods=_periods_payload(config),
        )
    if best_total_reward == float("-inf"):
        # No historical best is being continued: drop a stale best.pt left by
        # an unrelated earlier training so it cannot pass as this run's best.
        best_path.unlink(missing_ok=True)

    for episode in range(first_episode, episode_count + 1):
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
            validation_result = validation_runner.run_episode(agent.deterministic_action)
            validation_metrics = _metrics(
                validation_result,
                run=run,
                phase="validation",
                episode=episode,
                gamma=config.gamma,
            )
            metrics.append(validation_metrics)
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
        agent.save_checkpoint(
            latest_path,
            episode=episode,
            validation_total_reward=(
                None if best_total_reward == float("-inf") else best_total_reward
            ),
        )

    if best_total_reward == float("-inf") or not best_path.is_file():
        raise InputValidationError(
            "Training did not produce a best checkpoint; increase training_episodes "
            "or reduce episodes_collecting_initial_experience."
        )

    seed_used = agent.seed
    best_payload = agent.load_checkpoint(best_path, load_optimizers=False)
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
    return (
        validation_result,
        metrics,
        latest_path,
        best_path,
        best_total_reward,
        seed_used,
    )


def train_learning_scenario(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    checkpoint_path: str | Path | None = None,
    training_episodes: int | None = None,
    independent_runs: int | None = None,
) -> SimulationResult:
    """训练多次独立的 MATD3 运行，按验证时段总奖励选模型；不运行测试时段。
    恢复单个检查点时沿用其运行编号，写回对应的 run_XX 目录并把记录合并进
    现有运行清单；单运行根布局的条目会先迁移到各自的 run_XX 目录，其余
    运行的模型文件不受影响。新实验应使用独立输出目录。
    返回入选运行（或恢复运行）在验证时段上的结果。
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    if independent_runs is None:
        run_count = (
            1
            if checkpoint_path is not None or config.continue_learning
            else _DEFAULT_INDEPENDENT_RUNS
        )
    else:
        run_count = independent_runs
    if run_count != 1 and run_count < _DEFAULT_INDEPENDENT_RUNS:
        raise InputValidationError(
            "Independent learning runs must be 1 for resume/debug or at least 3 "
            "for Version 5 reporting."
        )
    if run_count > 1 and (checkpoint_path is not None or config.continue_learning):
        raise InputValidationError(
            "Resume training operates on one checkpoint; use independent_runs=1."
        )

    checkpoint_root = _default_checkpoint_root(config, output_path)
    manifest_path = checkpoint_root / _RUN_MANIFEST
    resuming = run_count == 1 and (
        checkpoint_path is not None or config.continue_learning
    )
    resume_run = (
        _resume_run_number(config, checkpoint_root, checkpoint_path)
        if resuming
        else None
    )
    # 恢复运行沿用检查点里的运行编号，并把记录合并进现有运行清单。
    existing_runs = (
        _read_run_manifest(manifest_path, config)["runs"]
        if resuming and manifest_path.is_file()
        else []
    )
    use_run_dirs = run_count > 1 or (
        resuming
        and (
            (resume_run or 1) > 1
            or any(entry["best_checkpoint"] != "best.pt" for entry in existing_runs)
        )
    )
    all_metrics: list[_EpisodeMetrics] = []
    run_records: list[dict[str, Any]] = [dict(entry) for entry in existing_runs]
    record_index = {
        int(entry["run"]): index for index, entry in enumerate(run_records)
    }
    results: dict[int, SimulationResult] = {}
    produced_runs: list[int] = []

    # 单运行布局的清单条目引用根目录文件；切换到多运行布局时，先把这些文件
    # 复制到各自的 run_XX 目录并改写条目路径，避免随后复制到根目录的
    # best.pt/latest.pt 覆盖掉原模型、让条目指向错误的权重。
    root_entries = [
        entry for entry in run_records if "/" not in entry["best_checkpoint"]
    ]
    if use_run_dirs and root_entries:
        for entry in root_entries:
            run_dir = checkpoint_root / f"run_{int(entry['run']):02d}"
            source = checkpoint_root / entry["best_checkpoint"]
            if source.is_file():
                _copy_if_distinct(source, run_dir / "best.pt")
            entry["best_checkpoint"] = f"run_{int(entry['run']):02d}/best.pt"
        if len(root_entries) == 1:
            latest = checkpoint_root / "latest.pt"
            if latest.is_file():
                _copy_if_distinct(
                    latest,
                    checkpoint_root
                    / f"run_{int(root_entries[0]['run']):02d}"
                    / "latest.pt",
                )

    run_numbers = [resume_run] if resume_run is not None else range(1, run_count + 1)
    for run in run_numbers:
        run_output = (
            output_path if not use_run_dirs else output_path / "runs" / f"run_{run:02d}"
        )
        run_checkpoints = (
            checkpoint_root if not use_run_dirs else checkpoint_root / f"run_{run:02d}"
        )
        result, metrics, latest_path, best_path, best_reward, seed = _train_single_run(
            input_path,
            run_output,
            run_checkpoints,
            scenario,
            run=run,
            seed=_independent_run_seed(settings.seed, run),
            checkpoint_path=checkpoint_path,
            training_episodes=training_episodes,
        )
        results[run] = result
        produced_runs.append(run)
        all_metrics.extend(metrics)
        record = {
            "run": run,
            "seed": seed,
            "best_checkpoint": best_path.relative_to(checkpoint_root).as_posix(),
            "validation_total_reward": best_reward,
        }
        if run in record_index:
            run_records[record_index[run]] = record
        else:
            run_records.append(record)
        if use_run_dirs:
            _copy_if_distinct(latest_path, checkpoint_root / "latest.pt")

    assert produced_runs
    # 入选运行按全部清单条目的验证总奖励确定，打平时保留更早编号。
    run_records.sort(key=lambda entry: entry["run"])
    selected_entry = max(
        run_records, key=lambda entry: float(entry["validation_total_reward"])
    )
    selected_run = int(selected_entry["run"])
    _copy_if_distinct(
        checkpoint_root / selected_entry["best_checkpoint"],
        checkpoint_root / "best.pt",
    )
    _write_run_manifest(checkpoint_root, config, run_records, selected_run)
    # 输出根目录保存主结果供绘图和浏览：入选运行在本次产出中取其结果，
    # 否则（恢复运行）写本次运行自身的验证段结果。
    root_result = results.get(selected_run, results[produced_runs[-1]])
    write_results(output_path, root_result)

    _write_metrics(output_path / "learning_metrics.csv", all_metrics)
    _write_run_summary(output_path / "learning_run_summary.csv", all_metrics)
    return root_result
