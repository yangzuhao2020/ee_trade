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
from electricity_market_sim.config.loader import _LEARNING_PARAMETER_FIELDS
from electricity_market_sim.reporting import write_results
from electricity_market_sim.simulation import simulate


DATASETS = (
    "example_01a", "example_01b", "example_01c", "example_01h",
    "example_03", "example_02a",
)

# Fully pinned learning settings for the deterministic fixtures: tuning the shared
# matd3.yaml or the scenario YAML must not change their snapshots.
_FIXTURE_LEARNING = {
    "learning_mode": True,
    "trained_policies_save_path": None, "trained_policies_load_path": None,
    "algorithm": "matd3", "actor_architecture": "mlp", "device": "cpu",
    "learning_rate": 0.001, "replay_buffer_size": 10000, "gamma": 0.99,
    "validation_episodes_interval": 5, "max_bid_price": 100,
    "exploration_noise_std": 0.2, "action_noise_schedule": "linear",
    "noise_sigma": 0.1, "noise_scale": 1, "noise_dt": 1, "tau": 0.005,
    "policy_delay": 2, "target_policy_noise": 0.2, "target_noise_clip": 0.5,
}
_FIXTURE_SCENARIOS = {
    "base": ((datetime(2019, 3, 1), datetime(2019, 3, 31)), {
        "training_episodes": 100, "episodes_collecting_initial_experience": 5,
        "train_freq": "100h", "gradient_steps": 10, "batch_size": 128,
    }),
    "tiny": ((datetime(2019, 1, 1), datetime(2019, 1, 5)), {
        "training_episodes": 10, "episodes_collecting_initial_experience": 3,
        "train_freq": "24h", "gradient_steps": 24, "batch_size": 64,
    }),
}


def cases():
    for dataset in DATASETS:
        input_path = ROOT / "examples" / "input" / dataset
        config = yaml.safe_load((input_path / "config.yaml").read_text(encoding="utf-8"))
        for scenario in config:
            yield dataset, scenario, input_path, None
        if dataset == "example_02a":
            # Exercise delivery hours with test-only settings without altering
            # the shared defaults or input files.
            # The train period reproduces the scenario's original delivery hours;
            # validation and test follow it and are not simulated here.
            for scenario, ((start, end), parameters) in _FIXTURE_SCENARIOS.items():
                fixture = dict(config[scenario])
                day = timedelta(days=1)
                fixture["learning_config"] = {
                    **_FIXTURE_LEARNING,
                    **parameters,
                    "periods": {
                        "train": {"start": start + timedelta(hours=1), "end": end},
                        "validation": {"start": end, "end": end + day},
                        "test": {"start": end + day, "end": end + 2 * day},
                    },
                }
                unpinned = _LEARNING_PARAMETER_FIELDS - fixture["learning_config"].keys()
                if unpinned:
                    raise AssertionError(
                        "Deterministic fixtures must pin every learning parameter: "
                        + ", ".join(sorted(unpinned))
                    )
                original_safe_load = yaml.safe_load
                scene_text = (input_path / "config.yaml").read_text(encoding="utf-8")
                def scenario_safe_load(text):
                    return {scenario: fixture} if text == scene_text else original_safe_load(text)
                with patch("electricity_market_sim.config.loader.yaml.safe_load",
                           side_effect=scenario_safe_load):
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
