"""示例场景仿真结果、学习回调、警告及 CSV 输出的回归快照记录与对比。
在项目根目录运行：
    python -B tests/eom_regression.py baseline ARCHIVE
    python -B tests/eom_regression.py compare ARCHIVE
快照与常规输出分开保存；场景配置调整仅在测试内存中生效。"""

import argparse
from datetime import datetime, timedelta
from math import tanh
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import yaml

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from electricity_market_sim.config import load_market_settings
from electricity_market_sim.reporting import write_results
from electricity_market_sim.simulation import simulate


DATASETS = (
    "example_01a", "example_01b", "example_01c", "example_01h",
    "example_03", "example_02a",
)


def cases():
    for dataset in DATASETS:
        input_path = ROOT / "examples" / "input" / dataset
        config = yaml.safe_load((input_path / "config.yaml").read_text(encoding="utf-8"))
        for scenario in config:
            yield dataset, scenario, input_path, None
        if dataset == "example_02a":
            # Upstream YAML lacks mandatory V5 fields. Exercise its market data
            # with explicit test-only settings without altering the input files.
            # The train period reproduces the scenario's original delivery hours;
            # validation and test follow it and are not simulated here.
            for scenario in ("base", "tiny"):
                fixture = dict(config[scenario])
                start = datetime.fromisoformat(str(fixture["start_date"]))
                end = datetime.fromisoformat(str(fixture["end_date"]))
                day = timedelta(days=1)
                fixture["learning_config"] = {
                    **fixture["learning_config"],
                    "replay_buffer_size": 10000,
                    "exploration_noise_std": 0.2,
                    "action_noise_schedule": "linear",
                    "tau": 0.005,
                    "policy_delay": 2,
                    "target_policy_noise": 0.2,
                    "target_noise_clip": 0.5,
                    "periods": {
                        "train": {"start": start + timedelta(hours=1), "end": end},
                        "validation": {"start": end, "end": end + day},
                        "test": {"start": end + day, "end": end + 2 * day},
                    },
                }
                with patch("electricity_market_sim.config.loader.yaml.safe_load",
                           return_value={scenario: fixture}):
                    settings = load_market_settings(input_path / "config.yaml", scenario)
                yield dataset, scenario + "_deterministic_fixture", input_path, settings


def capture(input_path, scenario, fixture_settings):
    trace = []
    previous_reward = 0.0

    def action(state):
        chosen = (tanh(state[0] + previous_reward / 100),
                  tanh(state[-1] - previous_reward / 100))
        trace.append(("action", state, chosen))
        return chosen

    def transition(value):
        nonlocal previous_reward
        previous_reward = value.reward
        trace.append(("transition", value))

    kwargs = {}
    try:
        settings = fixture_settings or load_market_settings(input_path / "config.yaml", scenario)
        if settings.learning_config and settings.learning_config.learning_mode:
            kwargs = {"learning_action_provider": action,
                      "learning_transition_consumer": transition,
                      "learning_period": "train"}
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with patch("electricity_market_sim.simulation.load_market_settings",
                       return_value=settings):
                result = simulate(input_path, scenario, **kwargs)
        return {"result": result, "trace": trace,
                "warnings": [str(item.message) for item in recorded]}
    except Exception as exc:
        return {"error": (type(exc).__name__, str(exc))}


def run(mode, archive):
    archive = Path(archive)
    failures = []
    counts = {"success": 0, "error": 0}
    for dataset, scenario, input_path, fixture in cases():
        name = dataset + "/" + scenario
        snapshot = capture(input_path, scenario, fixture)
        expected_dir = archive / dataset / scenario
        if mode == "baseline":
            expected_dir.mkdir(parents=True, exist_ok=False)
            (expected_dir / "snapshot.pkl").write_bytes(pickle.dumps(snapshot))
            if "result" in snapshot:
                write_results(expected_dir, snapshot["result"])
        else:
            expected = pickle.loads((expected_dir / "snapshot.pkl").read_bytes())
            if snapshot != expected:
                failures.append(name + ": result/callback/warning mismatch")
            if "result" in snapshot:
                with TemporaryDirectory(prefix="ee_trade_csv_compare_") as output:
                    write_results(Path(output), snapshot["result"])
                    actual_files = {p.name: p.read_bytes() for p in Path(output).glob("*.csv")}
                    expected_files = {p.name: p.read_bytes() for p in expected_dir.glob("*.csv")}
                    if actual_files != expected_files:
                        differing = sorted(k for k in actual_files.keys() | expected_files.keys()
                                           if actual_files.get(k) != expected_files.get(k))
                        failures.append(name + ": CSV mismatch " + ", ".join(differing))
        if "error" in snapshot:
            counts["error"] += 1
            print(name, "existing error:", *snapshot["error"], flush=True)
        else:
            counts["success"] += 1
            print(name, len(snapshot["result"].market_results), "products OK", flush=True)
    print(mode, counts, flush=True)
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "compare"))
    parser.add_argument("archive")
    args = parser.parse_args()
    run(args.mode, args.archive)
