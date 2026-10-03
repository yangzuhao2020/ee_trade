"""Command-line entry point for the electricity-market simulator."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .config import load_market_settings
from .errors import SimulationError
from .plotting import generate_learning_plots, generate_plots
from .simulation import run_simulation
from .training import evaluate_learning_scenario, train_learning_scenario

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the lightweight single-node electricity-market simulator."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("examples/input/example_01b"),
        help="Directory containing config.yaml and the V1/V2 CSV input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/example_01b"),
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument(
        "--scenario",
        default="base",
        help="Scenario key in config.yaml (for example 'base' or a V2 EOM scenario).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV results only; skip post-simulation PNG generation.",
    )
    parser.add_argument(
        "--learning-mode",
        choices=("train", "evaluate"),
        help=(
            "For a Version 5 learning scenario, train a policy or evaluate the "
            "best saved policy. Defaults to train."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint to resume for training or load for evaluation. Evaluation "
            "defaults to OUTPUT_DIR/checkpoints/best.pt."
        ),
    )
    parser.add_argument(
        "--training-episodes",
        type=int,
        help="Override learning_config.training_episodes for this training run.",
    )
    parser.add_argument(
        "--training-runs",
        type=int,
        help=(
            "Number of independent learning runs. Defaults to 3; use 1 only for "
            "resume or quick diagnostic runs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    learning_metrics_path: Path | None = None
    try:
        settings = load_market_settings(
            arguments.input_dir / "config.yaml", scenario=arguments.scenario
        )
        is_learning = (settings.learning_config is not None
                       and settings.learning_config.learning_mode)
        # 存在学习配置 且学习模式不为None，说明是第五版学习场景
        learning_episode_label = "Actor evaluation"

        if is_learning: # 对应第五版学习场景，用于训练和评估机组报价。
            learning_mode = arguments.learning_mode or "train"
            if (
                learning_mode == "evaluate"
                and arguments.checkpoint is not None
                and arguments.checkpoint.name != "best.pt"
            ):
                learning_episode_label = (
                    f"Actor evaluation ({arguments.checkpoint.stem})"
                )
            else:
                learning_episode_label = "Best Actor evaluation"

            if learning_mode == "train":
                result = train_learning_scenario(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    scenario=arguments.scenario,
                    checkpoint_path=arguments.checkpoint,
                    training_episodes=arguments.training_episodes,
                    independent_runs=arguments.training_runs,
                )
                learning_metrics_path = arguments.output_dir / "learning_metrics.csv"
            else:
                if (
                    arguments.training_episodes is not None
                    or arguments.training_runs is not None
                ):
                    raise SimulationError(
                        "--training-episodes and --training-runs can only be used with "
                        "--learning-mode train."
                    )
                result = evaluate_learning_scenario(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    scenario=arguments.scenario,
                    checkpoint_path=arguments.checkpoint,
                )
                learning_metrics_path = (
                    arguments.output_dir / "learning_evaluation_metrics.csv"
                )
        else: # 对应 1~4 版场景，直接运行模拟
            result = run_simulation(
                input_dir=arguments.input_dir,
                output_dir=arguments.output_dir,
                scenario=arguments.scenario,
            )
    except SimulationError as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 2

    if not arguments.no_plots:
        try:
            generate_plots(
                result,
                arguments.output_dir / "plots",
                learning_episode_label=learning_episode_label,
            )
            if learning_metrics_path is not None:
                generate_learning_plots(
                    learning_metrics_path, arguments.output_dir / "plots"
                )
        except Exception as exc:
            # CSV output is already durable at this point. Plot failures must
            # not change a successful simulation into a lost result.
            print(
                f"Simulation completed, but PNG generation failed: {exc}",
                file=sys.stderr,
            )

    print(
        f"Simulation complete: {len(result.market_results)} delivery products written to "
        f"{arguments.output_dir}."
    )
    return 0
