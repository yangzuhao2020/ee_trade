"""Command-line entry point for the electricity-market simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .errors import SimulationError
from .plotting import generate_plots
from .simulation import run_simulation


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
        help="Scenario in config.yaml: 'base' or 'base_with_exchanges'.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV results only; skip post-simulation PNG generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
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
            generate_plots(result, arguments.output_dir / "plots")
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
