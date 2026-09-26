"""Command-line entry point for the electricity-market simulator."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import load_market_settings
from .errors import SimulationError
from .plotting import generate_plots
from .simulation import run_simulation


def _plot_opening(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--plot-opening must be an ISO-like date and time."
        ) from exc


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
        "--plot-opening",
        type=_plot_opening,
        help=(
            "For complex clearing, render detailed plots for this market opening "
            "(for example '2019-01-01 00:00'). Defaults to the first opening."
        ),
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
    try:
        settings = load_market_settings(
            arguments.input_dir / "config.yaml", scenario=arguments.scenario
        )
        is_learning = (
            settings.learning_config is not None
            and settings.learning_config.learning_mode
        )
        if is_learning:
            from .training import (
                evaluate_learning_scenario,
                train_learning_scenario,
            )

            learning_mode = arguments.learning_mode or "train"
            if learning_mode == "train":
                result = train_learning_scenario(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    scenario=arguments.scenario,
                    checkpoint_path=arguments.checkpoint,
                    training_episodes=arguments.training_episodes,
                    independent_runs=arguments.training_runs,
                )
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
        else:
            if (
                arguments.learning_mode is not None
                or arguments.checkpoint is not None
                or arguments.training_episodes is not None
                or arguments.training_runs is not None
            ):
                raise SimulationError(
                    "Learning command-line options require a Version 5 learning scenario."
                )
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
                opening_time=arguments.plot_opening,
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
