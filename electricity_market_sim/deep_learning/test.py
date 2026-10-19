"""第五版学习策略的确定性评估、独立测试及 Actor 与边际成本基线对比。"""

from pathlib import Path
from .agent import (
    MATD3Agent,
    _RUN_MANIFEST,
    _checkpoint_from_setting,
    _default_checkpoint_root,
    _learning_config,
    _read_run_manifest,
)
from ..config import load_market_settings
from ..errors import InputValidationError
from .learning_metrics import (
    _EpisodeMetrics,
    _comparison_phases,
    _metrics,
    _write_metrics,
    _write_run_summary,
)
from ..models import LearningConfig, SimulationResult
from ..reporting import write_results
from ..simulation import LearningEpisodeRunner

_EVALUATION_PERIODS = ("train", "validation")


def _default_evaluation_checkpoint(config: LearningConfig, output_path: Path) -> Path:
    if config.trained_policies_load_path is not None:
        return _checkpoint_from_setting(
            config.trained_policies_load_path,
            output_path / "checkpoints",
            "best.pt",
        )
    if config.trained_policies_save_path is not None:
        return Path(config.trained_policies_save_path) / "best.pt"
    return output_path / "checkpoints" / "best.pt"


def _marginal_cost_action(state: tuple[float, ...]) -> tuple[float, float]:
    """第五版的两个报价段都按观测到的边际成本报价。"""

    normalized_marginal_cost = state[-1]
    return normalized_marginal_cost, normalized_marginal_cost


def _evaluate_actor_and_baseline(
    *,
    agent: MATD3Agent,
    input_path: Path,
    output_path: Path,
    scenario: str,
    period: str,
    episode: int,
    gamma: float,
    run: int,
) -> tuple[SimulationResult, list[_EpisodeMetrics]]:
    """在指定时段的全新市场上分别运行确定性 Actor 和边际成本基准策略。
    Actor 结果写入 output_path，基线结果写入 output_path/baseline；不更新任何参数。
    """
    actor_result = LearningEpisodeRunner(
        input_path, scenario=scenario, period=period
    ).run_episode(agent.deterministic_action)
    baseline_result = LearningEpisodeRunner(
        input_path, scenario=scenario, period=period
    ).run_episode(_marginal_cost_action, enforce_action_bounds=False)

    write_results(output_path, actor_result)
    write_results(output_path / "baseline", baseline_result)
    actor_phase, baseline_phase = _comparison_phases(period)
    comparison = [
        _metrics(actor_result, run=run, phase=actor_phase, episode=episode, gamma=gamma),
        _metrics(
            baseline_result, run=run, phase=baseline_phase, episode=episode, gamma=gamma
        ),
    ]
    return actor_result, comparison


def evaluate_learning_scenario(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    checkpoint_path: str | Path | None = None,
    period: str = "validation",
) -> SimulationResult:
    """在训练或验证时段上对比单个检查点与边际成本基线；测试时段请使用 test 命令。"""

    if period not in _EVALUATION_PERIODS:
        raise InputValidationError(
            "evaluate 只能评估 train 或 validation 时段；测试时段只能通过 "
            "--learning-mode test 运行。"
        )
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    agent = MATD3Agent(config, total_regular_steps=1)
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else _default_evaluation_checkpoint(config, output_path)
    )
    payload = agent.load_checkpoint(checkpoint, load_optimizers=False)
    saved_run = payload.get("run")
    run = (
        saved_run
        if isinstance(saved_run, int) and not isinstance(saved_run, bool)
        else 0
    )
    result, comparison = _evaluate_actor_and_baseline(
        agent=agent,
        input_path=input_path,
        output_path=output_path / period,
        scenario=scenario,
        period=period,
        episode=int(payload["episode"]),
        gamma=config.gamma,
        run=run,
    )
    _write_metrics(output_path / "learning_evaluation_metrics.csv", comparison)
    write_results(output_path, result)
    return result


def run_learning_test(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    manifest_path: str | Path | None = None,
) -> SimulationResult:
    """在测试时段上评估每个独立运行的最佳 Actor 及边际成本基线。
    不加探索噪声，不更新网络、经验池或检查点，测试成绩也不参与任何选择。
    返回训练时按验证结果选中的运行在测试时段上的结果。
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    if manifest_path is None:
        manifest_file = _default_checkpoint_root(config, output_path) / _RUN_MANIFEST
    else:
        manifest_file = Path(manifest_path)
        if manifest_file.is_dir():
            manifest_file = manifest_file / _RUN_MANIFEST
    manifest = _read_run_manifest(manifest_file, config)
    entries = manifest["runs"]
    metrics: list[_EpisodeMetrics] = []
    results: dict[int, SimulationResult] = {}
    for entry in entries:
        run = entry["run"]
        agent = MATD3Agent(config, total_regular_steps=1)
        payload = agent.load_checkpoint(
            manifest_file.parent / entry["best_checkpoint"], load_optimizers=False
        )
        run_output = (
            output_path / "test"
            if len(entries) == 1
            else output_path / "test" / f"run_{run:02d}"
        )
        results[run], comparison = _evaluate_actor_and_baseline(
            agent=agent,
            input_path=input_path,
            output_path=run_output,
            scenario=scenario,
            period="test",
            episode=int(payload["episode"]),
            gamma=config.gamma,
            run=run,
        )
        metrics.extend(comparison)
    _write_metrics(output_path / "learning_test_metrics.csv", metrics)
    _write_run_summary(output_path / "learning_test_summary.csv", metrics)
    selected = results.get(manifest.get("selected_run"), results[entries[0]["run"]])
    write_results(output_path, selected)
    return selected
