"""电力市场仿真器的命令行入口。"""

import argparse
import sys
from pathlib import Path
from .config import load_market_settings
from .errors import SimulationError
from .plotting import generate_learning_plots, generate_plots
from .simulation import run_simulation
from .deep_learning.test import evaluate_learning_scenario, run_learning_test
from .deep_learning.training import train_learning_scenario


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
        choices=("train", "evaluate", "test"),
        help=(
            "For a Version 5 learning scenario: train (train period updates, "
            "validation period selects best.pt), evaluate one checkpoint on the "
            "train or validation period, or test every run's best.pt on the "
            "held-out test period. Defaults to train."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Checkpoint to resume for training or load for evaluation (defaults to "
            "OUTPUT_DIR/checkpoints/best.pt). For test, the runs.json manifest or "
            "its directory (defaults to OUTPUT_DIR/checkpoints/runs.json)."
        ),
    )
    parser.add_argument(
        "--period",
        choices=("train", "validation"),
        help="Data period for --learning-mode evaluate. Defaults to validation.",
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

        if is_learning: # 对应第五版学习场景，用于训练、评估和测试机组报价。
            learning_mode = arguments.learning_mode or "train"
            if arguments.period is not None and learning_mode != "evaluate":
                raise SimulationError("--period 只能与 --learning-mode evaluate 一起使用。")
            if learning_mode != "train" and (
                arguments.training_episodes is not None
                or arguments.training_runs is not None
            ):
                raise SimulationError(
                    "--training-episodes 和 --training-runs 只能与 "
                    "--learning-mode train 一起使用。"
                )
            evaluation_period = arguments.period or "validation"
            if learning_mode == "train":
                learning_episode_label = "Best Actor (validation period)"
            elif learning_mode == "test":
                learning_episode_label = "Selected Actor (test period)"
            elif (
                arguments.checkpoint is not None
                and arguments.checkpoint.name != "best.pt"
            ):
                learning_episode_label = (
                    f"Actor evaluation ({arguments.checkpoint.stem}, "
                    f"{evaluation_period} period)"
                )
            else:
                learning_episode_label = (
                    f"Best Actor evaluation ({evaluation_period} period)"
                )

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
            elif learning_mode == "test":
                result = run_learning_test(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    scenario=arguments.scenario,
                    manifest_path=arguments.checkpoint,
                )
                learning_metrics_path = arguments.output_dir / "learning_test_metrics.csv"
            else:
                result = evaluate_learning_scenario(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    scenario=arguments.scenario,
                    checkpoint_path=arguments.checkpoint,
                    period=evaluation_period,
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
        print(f"仿真失败：{exc}", file=sys.stderr)
        return 2

    if not arguments.no_plots:
        try:
            generate_plots(result, arguments.output_dir / "plots",
                learning_episode_label=learning_episode_label,)
            if learning_metrics_path is not None:
                generate_learning_plots(learning_metrics_path,
                                        arguments.output_dir / "plots")
        except Exception as exc:
            # 此时 CSV 结果已经保存到磁盘。
            # 即使绘图失败，也应保留已成功完成的仿真结果。
            print(f"仿真已完成，但 PNG 图片生成失败：{exc}",file=sys.stderr,)

    print(
        f"仿真完成：已将 {len(result.market_results)} 个交付时段的结果写入 "
        f"{arguments.output_dir}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
