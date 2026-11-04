"""V5 信息边界、数据时段划分、预测需求解耦及 16 维模型训练、选模与独立测试的端到端测试。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import os
from dataclasses import replace
from io import StringIO
from datetime import datetime, timedelta
from math import cos, pi, sin
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from electricity_market_sim.config import load_market_settings, parse_duration
from electricity_market_sim.deep_learning.agent import MATD3Agent, _periods_payload
from electricity_market_sim.deep_learning.learning import (
    LearningEpisodeSession, build_learning_observation,
)
from electricity_market_sim.deep_learning.test import (
    evaluate_learning_scenario, run_learning_test,
)
from electricity_market_sim.deep_learning.training import train_learning_scenario
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.main import main
from electricity_market_sim.plotting import generate_learning_plots
from electricity_market_sim.simulation import (
    learning_period_settings, market_openings, simulate_learning_episode,
)
from electricity_market_sim.simulators.base_market import BaseMarket
from electricity_market_sim.simulators.base_market.forecasts import (
    _forecast_starts, _naive_price_forecasts,
)
from electricity_market_sim.simulators.eom import EomExtensionInputs


START = datetime(2024, 1, 1)
HOUR = timedelta(hours=1)
PRICE_PREDICTOR = (
    "electricity_market_sim.simulators.base_market.market._naive_price_forecasts"
)
# 默认划分（相对 START 的交付小时）：训练 [1,4)、验证 [4,6)、测试 [6,8)。
DEFAULT_PERIODS = {"train": (1, 4), "validation": (4, 6), "test": (6, 8)}
PERIOD_HOURS = {"train": [1, 2, 3], "validation": [4, 5], "test": [6, 7]}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_scenario(
    directory, *, rival_cost=20.0, rival_capacity=10.0,
    rival_strategy="powerplant_energy_naive", storage=False, profile_hours=8,
    demand_change_from=None, periods=DEFAULT_PERIODS, eom_start=None,
):
    """生成独立小场景，不依赖 examples 联接。
    periods 为 None 时不写时段配置；数值为相对 START 的交付小时。
    """
    directory.mkdir(parents=True, exist_ok=True)
    learning_config = {
        "learning_mode": True, "algorithm": "matd3", "learning_rate": 0.001,
        "device": "cpu",
        "training_episodes": 3, "episodes_collecting_initial_experience": 1,
        "replay_buffer_size": 64, "batch_size": 2, "gamma": 0.99,
        "train_freq": "1h", "gradient_steps": 1,
        "validation_episodes_interval": 1, "exploration_noise_std": 0.2,
        "noise_sigma": 0.1, "noise_scale": 1.0, "noise_dt": 1.0,
        "action_noise_schedule": "linear", "tau": 0.005, "policy_delay": 2,
        "target_policy_noise": 0.2, "target_noise_clip": 0.5,
        "max_bid_price": 200.0,
    }
    if periods is not None:
        learning_config["periods"] = {
            name: {
                "start": (START + start * HOUR).isoformat(sep=" "),
                "end": (START + end * HOUR).isoformat(sep=" "),
            }
            for name, (start, end) in periods.items()
        }
    eom = {
        "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
        "opening_frequency": "1h", "opening_duration": "1h",
        "maximum_bid_price": 3000.0, "minimum_bid_price": -500.0,
        "market_mechanism": "pay_as_clear",
    }
    if eom_start is not None:
        eom["start_date"] = eom_start.isoformat(sep=" ")
    config = {
        "start_date": START.isoformat(sep=" "),
        "end_date": (START + 4 * HOUR).isoformat(sep=" "),
        "time_step": "1h",
        "seed": 7,
        "learning_config": learning_config,
        "markets_config": {"EOM": eom},
    }
    (directory / "config.yaml").write_text(
        yaml.safe_dump({"base": config}), encoding="utf-8"
    )
    plant_fields = (
        "name", "technology", "bidding_EOM", "fuel_type", "emission_factor",
        "max_power", "min_power", "efficiency", "additional_cost", "unit_operator",
    )
    write_csv(directory / "powerplant_units.csv", plant_fields, [
        dict(zip(plant_fields, (
            "pp_6", "gas", "powerplant_energy_learning", "gas", 0.0,
            10.0, 1.0, 0.5, 0.0, "Operator-RL",
        ))),
        dict(zip(plant_fields, (
            "rival", "hard coal", rival_strategy, "rival_fuel", 0.0,
            rival_capacity, 1.0, 1.0, 0.0, "competitor",
        ))),
    ])
    write_csv(directory / "demand_units.csv", (
        "name", "technology", "bidding_EOM", "max_power", "min_power", "unit_operator",
    ), [{
        "name": "load", "technology": "inflex_demand",
        "bidding_EOM": "demand_energy_naive", "max_power": 100.0,
        "min_power": 0.0, "unit_operator": "demand_operator",
    }])
    write_csv(directory / "demand_df.csv", ("datetime", "load"), [
        {"datetime": (START + i * HOUR).isoformat(sep=" "),
         "load": 17.0 if demand_change_from is not None and i >= demand_change_from else 7.0}
        for i in range(profile_hours)
    ])
    write_csv(directory / "fuel_prices_df.csv", ("fuel", "gas", "rival_fuel", "co2"), [
        {"fuel": "price", "gas": 20.0, "rival_fuel": rival_cost, "co2": 0.0}
    ])
    if storage:
        write_csv(directory / "storage_units.csv", (
            "name", "technology", "bidding_EOM", "unit_operator",
            "max_power_charge", "max_power_discharge", "efficiency_charge",
            "efficiency_discharge", "min_soc", "max_soc", "capacity", "initial_soc",
            "additional_cost_charge", "additional_cost_discharge",
        ), [{
            "name": "battery", "technology": "battery",
            "bidding_EOM": "storage_energy_heuristic_flexable",
            "unit_operator": "storage_operator", "max_power_charge": 1.0,
            "max_power_discharge": 1.0, "efficiency_charge": 0.9,
            "efficiency_discharge": 0.9, "min_soc": 0.0, "max_soc": 1.0,
            "capacity": 4.0, "initial_soc": 0.5,
            "additional_cost_charge": 0.0, "additional_cost_discharge": 0.0,
        }])
    return directory


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_bytes(root):
    return {path: path.read_bytes() for path in sorted(Path(root).rglob("*.pt"))}


def collect_states(path, period="train"):
    """以固定动作运行一个时段，并确认全程不调用电价预测。"""
    states = []
    transitions = []

    def action(state):
        states.append(state)
        return (0.75, 0.75)

    with patch(PRICE_PREDICTOR, side_effect=AssertionError("Unexpected price forecast")):
        result = simulate_learning_episode(
            path, action, transition_consumer=transitions.append, period=period
        )
    return states, transitions, result


class LearningConfigurationContracts(unittest.TestCase):
    def test_duration_parser_preserves_units_and_rejects_zero_after_rounding(self):
        for value, expected in (("15m", timedelta(minutes=15)),
                                ("15min", timedelta(minutes=15)),
                                ("0.5h", timedelta(minutes=30)),
                                ("1d", timedelta(days=1))):
            with self.subTest(value=value):
                self.assertEqual(parse_duration(value, "duration"), expected)
        for unit in ("m", "min", "h", "d"):
            for magnitude in ("0", "0.000000000001"):
                with self.subTest(unit=unit, magnitude=magnitude):
                    with self.assertRaisesRegex(InputValidationError, "duration"):
                        parse_duration(magnitude + unit, "duration")

    def test_market_opening_durations_reject_zero_after_rounding(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            eom = scene["base"]["markets_config"]["EOM"]
            for field in ("opening_frequency", "opening_duration"):
                with self.subTest(field=field):
                    eom[field] = "0.000000000001h"
                    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                    with self.assertRaisesRegex(InputValidationError, field):
                        load_market_settings(path)
                    eom[field] = "1h"

    def test_seed_accepts_null_and_nonnegative_integers(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            for value in (None, 0, 7):
                with self.subTest(value=value):
                    scene["base"]["seed"] = value
                    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                    self.assertEqual(load_market_settings(path).seed, value)
            for value in (-1, True, False, 1.5, "7"):
                with self.subTest(value=value):
                    scene["base"]["seed"] = value
                    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                    with self.assertRaisesRegex(InputValidationError, "seed"):
                        load_market_settings(path)

    def test_shared_defaults_are_loaded_relative_to_loader_and_scene_overrides_them(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)).resolve()
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            periods = scene["base"]["learning_config"]["periods"]
            scene["base"]["learning_config"] = {"learning_mode": True, "periods": periods}
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(path)
                config = load_market_settings(config_path).learning_config
            finally:
                os.chdir(previous)
            self.assertEqual(config.device, "cpu")
            self.assertEqual(config.training_episodes, 30)
            self.assertEqual(config.replay_buffer_size, 100000)
            self.assertEqual(config.train_frequency_steps, 100)
            scene["base"]["learning_config"].update(device="cuda:0", gamma=0.8)
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            config = load_market_settings(config_path).learning_config
            self.assertEqual(config.device, "cuda:0")
            self.assertEqual(config.gamma, 0.8)

    def test_nonlearning_and_disabled_scenes_do_not_read_shared_file(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"]["learning_mode"] = False
            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with patch("electricity_market_sim.config.loader._load_learning_defaults",
                       side_effect=AssertionError("Unexpected shared config read")):
                self.assertFalse(load_market_settings(path).learning_config.learning_mode)
                scene["base"]["learning_config"] = {"learning_mode": False}
                path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                with self.assertRaisesRegex(InputValidationError, "缺少必填字段"):
                    load_market_settings(path)
                del scene["base"]["learning_config"]
                path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                self.assertIsNone(load_market_settings(path).learning_config)

    def test_shared_file_errors_and_scene_only_fields_are_reported(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input") / "config.yaml"
            shared = root / "shared.yaml"
            with patch("electricity_market_sim.config.loader._MATD3_DEFAULTS_PATH", shared):
                with self.assertRaisesRegex(InputValidationError, "shared.yaml"):
                    load_market_settings(path)
                for content in ("learning_config: [", "[]"):
                    shared.write_text(content, encoding="utf-8")
                    with self.subTest(content=content), self.assertRaisesRegex(InputValidationError, "shared.yaml"):
                        load_market_settings(path)
                for field in ("learning_mode", "periods",
                              "trained_policies_load_path", "trained_policies_save_path"):
                    shared.write_text(yaml.safe_dump({"learning_config": {field: None}}), encoding="utf-8")
                    with self.subTest(field=field), self.assertRaisesRegex(InputValidationError, "场景专属字段"):
                        load_market_settings(path)

    def test_unknown_fields_warn_without_changing_known_values(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"]["devcie"] = "cuda:0"
            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "devcie") as caught:
                config = load_market_settings(path).learning_config
            self.assertEqual(config.device, "cpu")
            self.assertIn(str(path), str(caught.warning))
            self.assertIn("'base'", str(caught.warning))
            self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve())

    def test_unknown_shared_fields_warn_with_shared_path_at_caller(self):
        from electricity_market_sim.config import loader
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input") / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            for algorithm, path_name in (("matd3", "_MATD3_DEFAULTS_PATH"),
                                         ("grpo", "_GRPO_DEFAULTS_PATH")):
                with self.subTest(algorithm=algorithm):
                    scene["base"]["learning_config"].update(algorithm=algorithm, training_steps=72)
                    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                    shared = root / f"{algorithm}.yaml"
                    defaults = yaml.safe_load(getattr(loader, path_name).read_text(encoding="utf-8"))
                    defaults["learning_config"]["extra_option"] = 1
                    shared.write_text(yaml.safe_dump(defaults), encoding="utf-8")
                    with patch.object(loader, path_name, shared), \
                            self.assertWarnsRegex(UserWarning, "extra_option") as caught:
                        load_market_settings(path)
                    self.assertIn(str(shared), str(caught.warning))
                    self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve())

    def test_null_parameters_are_rejected_instead_of_overriding_defaults(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            for field in ("device", "max_bid_price", "tau"):
                scene["base"]["learning_config"][field] = None
            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "null.*device, max_bid_price, tau"):
                load_market_settings(path)

    def test_integer_learning_parameters_reject_coercion_and_out_of_range_values(self):
        fields = {
            "matd3": {
                "training_episodes": 1, "episodes_collecting_initial_experience": 0,
                "replay_buffer_size": 1, "batch_size": 1, "gradient_steps": 1,
                "validation_episodes_interval": 1, "policy_delay": 1, "training_steps": 0,
            },
            "grpo": {
                "training_steps": 1, "group_size": 2, "sampling_workers": 1,
                "update_epochs": 1, "minibatch_size": 1, "validation_groups_interval": 1,
            },
        }
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            original = scene["base"]["learning_config"].copy()
            for algorithm, minima in fields.items():
                for field, minimum in minima.items():
                    for value in (2.9, 2.0, True, False, "2", minimum - 1):
                        with self.subTest(algorithm=algorithm, field=field, value=value):
                            scene["base"]["learning_config"] = {
                                **original, "algorithm": algorithm, "training_steps": 72,
                                field: value,
                            }
                            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                            with self.assertRaisesRegex(InputValidationError, field):
                                load_market_settings(path)

    def test_integer_learning_parameters_accept_zero_and_minimum_group_size(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            scene["base"]["learning_config"].update(
                episodes_collecting_initial_experience=0, training_steps=0,
            )
            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            config = load_market_settings(path).learning_config
            self.assertEqual(config.initial_experience_episodes, 0)
            self.assertEqual(config.training_steps, 0)
            scene["base"]["learning_config"].update(
                algorithm="grpo", training_steps=1, group_size=2, sampling_workers=1,
                update_epochs=1, minibatch_size=1, validation_groups_interval=1,
            )
            path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            config = load_market_settings(path).learning_config
            self.assertEqual(config.group_size, 2)
            self.assertEqual(config.training_steps, 1)
            self.assertEqual(config.validation_interval, 1)

    def test_product_count_requires_a_positive_integer(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary)) / "config.yaml"
            scene = yaml.safe_load(path.read_text(encoding="utf-8"))
            for value in (1.9, 1.0, True, False, "1", 0, -1):
                with self.subTest(value=value):
                    scene["base"]["markets_config"]["EOM"]["products"][0]["count"] = value
                    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
                    with self.assertRaisesRegex(InputValidationError, "count"):
                        load_market_settings(path)

    def test_field_registry_matches_shared_defaults(self):
        from electricity_market_sim.config import loader
        defaults = yaml.safe_load(loader._MATD3_DEFAULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(defaults["learning_config"]), loader._LEARNING_PARAMETER_FIELDS)
        grpo_defaults = loader._load_learning_defaults(loader._GRPO_DEFAULTS_PATH)
        self.assertLessEqual(grpo_defaults.keys(), loader._LEARNING_FIELDS)


class LearningCoverageContracts(unittest.TestCase):
    def test_test_period_gap_is_rejected_before_training_writes_checkpoints(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            rows = read_rows(path / "demand_df.csv")
            # 中间缺少测试小时 6；最后的小时 7 仍存在，不能只检查首尾范围。
            write_csv(path / "demand_df.csv", ("datetime", "load"), rows[:6] + rows[7:])
            with self.assertRaisesRegex(InputValidationError, "test.*demand_df.csv/load"):
                train_learning_scenario(path, root / "output", independent_runs=1)
            self.assertFalse((root / "output").exists())

    def test_preflight_checks_metadata_without_parsing_values_or_forecasting(self):
        from electricity_market_sim.deep_learning.data_coverage import validate_learning_data_coverage
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            rows = read_rows(path / "demand_df.csv")
            rows[6]["load"] = "not-a-number"
            write_csv(path / "demand_df.csv", ("datetime", "load"), rows)
            settings = load_market_settings(path / "config.yaml")
            with patch(PRICE_PREDICTOR, side_effect=AssertionError("Unexpected forecast")):
                validate_learning_data_coverage(path, settings)

    def test_optional_availability_checks_all_periods_when_provided(self):
        from electricity_market_sim.deep_learning.data_coverage import validate_learning_data_coverage
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            write_csv(path / "availability_df.csv", ("datetime", "pp_6"), [
                {"datetime": (START + hour * HOUR).isoformat(sep=" "), "pp_6": 1}
                for hour in range(6)
            ])
            with self.assertRaisesRegex(InputValidationError, "test.*availability_df.csv/pp_6"):
                validate_learning_data_coverage(path, load_market_settings(path / "config.yaml"))

    def test_hourly_coverage_uses_available_samples_and_rejects_empty_cells(self):
        from electricity_market_sim.deep_learning.data_coverage import _hourly_coverage
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiles.csv"
            rows = [{"datetime": (START + HOUR + timedelta(minutes=15)).isoformat(sep=" "), "load": "7"}]
            write_csv(path, ("datetime", "load"), rows)
            self.assertEqual(_hourly_coverage(path, ("load",)), {"load": {START + HOUR}})
            rows[0]["load"] = ""
            write_csv(path, ("datetime", "load"), rows)
            with self.assertRaisesRegex(InputValidationError, "missing data"):
                _hourly_coverage(path, ("load",))

    def test_enabled_exchange_requires_test_period_coverage(self):
        from electricity_market_sim.deep_learning.data_coverage import validate_learning_data_coverage
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            settings = replace(load_market_settings(path / "config.yaml"), exchange_units_file="exchange_units.csv")
            write_csv(path / "exchange_units.csv", (
                "name", "unit_operator", "bidding_EOM", "price_import", "price_export",
            ), [{"name": "grid", "unit_operator": "exchange", "bidding_EOM": "exchange_energy_naive",
                 "price_import": 50, "price_export": 20}])
            write_csv(path / "exchanges_df.csv", ("datetime", "grid_import", "grid_export"), [
                {"datetime": (START + hour * HOUR).isoformat(sep=" "), "grid_import": 1, "grid_export": 0}
                for hour in range(6)
            ])
            with self.assertRaisesRegex(InputValidationError, "test.*exchanges_df.csv/grid_import"):
                validate_learning_data_coverage(path, settings)


class LearningMarketContracts(unittest.TestCase):
    collect_states = staticmethod(collect_states)

    def test_transitions_precede_next_action_and_finalization_does_not_repeat_them(self):
        events = []
        finalize = LearningEpisodeSession.finalize

        def action(state):
            events.append("action")
            return (0.75, 0.75)

        def transition(value):
            events.append("terminal" if value.done else "transition")

        def finalize_twice(session):
            result = finalize(session)
            self.assertEqual(finalize(session), result)
            return result

        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            with patch.object(LearningEpisodeSession, "finalize", autospec=True,
                              side_effect=finalize_twice):
                simulate_learning_episode(path, action, transition_consumer=transition,
                                          period="train")
        self.assertEqual(events, [
            "action", "transition", "action", "transition", "action", "terminal",
        ])

    def test_nonconsecutive_learning_hours_are_rejected(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            config_path = path / "config.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            scene["base"]["markets_config"]["EOM"]["opening_frequency"] = "2h"
            config_path.write_text(yaml.safe_dump(scene), encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "continuous hourly"):
                self.collect_states(path)

    def test_learning_runs_without_price_prediction_and_receives_only_past_prices(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            states, transitions, result = self.collect_states(path)
        self.assertEqual(len(states), 3)
        self.assertTrue(all(len(state) == 16 for state in states))
        self.assertEqual(states[0][:12], (0.0,) * 12)
        self.assertEqual(states[0][14:], (0.0, 0.2))
        self.assertEqual(states[1][:12], (0.0,) * 11 + (0.1,))
        self.assertEqual(states[2][:12], (0.0,) * 10 + (0.1, 0.1))
        self.assertEqual(transitions[0].next_state, states[1])
        self.assertEqual(transitions[-1].next_state, (0.0,) * 16)
        self.assertTrue(transitions[-1].done)
        self.assertEqual(len(result.market_results), 3)

    def test_thermal_rival_parameters_affect_feedback_but_not_initial_observation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, _, original = self.collect_states(make_scenario(root / "baseline"))
            for name, change in (
                ("cost", {"rival_cost": 80.0}),
                ("capacity", {"rival_capacity": 2.0}),
            ):
                with self.subTest(parameter=name):
                    states, _, result = self.collect_states(make_scenario(root / name, **change))
                    self.assertEqual(states[0], baseline[0])
                    self.assertNotEqual(
                        result.market_results[0].clearing_price_eur_per_mwh,
                        original.market_results[0].clearing_price_eur_per_mwh,
                    )
                    self.assertEqual(
                        states[1][11], result.market_results[0].clearing_price_eur_per_mwh / 200.0
                    )
                    self.assertNotEqual(states[1][11], baseline[1][11])

    def test_price_history_rolls_over_with_previous_dispatch_and_storage(self):
        with TemporaryDirectory() as temporary:
            for storage in (False, True):
                with self.subTest(storage=storage):
                    path = make_scenario(
                        Path(temporary) / str(storage), storage=storage,
                        demand_change_from=1, profile_hours=21,
                        periods={"train": (1, 17), "validation": (17, 19), "test": (19, 21)},
                    )
                    states = []

                    def action(state):
                        index = len(states)
                        states.append(state)
                        return (0.15 + 0.02 * index, 0.3 + 0.02 * index)

                    result = simulate_learning_episode(path, action, period="train")
                    for index, state in enumerate(states):
                        history = result.market_results[max(0, index - 12):index]
                        expected = (0.0,) * (12 - len(history)) + tuple(
                            market.clearing_price_eur_per_mwh / 200.0 for market in history
                        )
                        self.assertEqual(state[:12], expected)
                        previous_power = sum(
                            item.accepted_power_mw
                            for item in result.market_results[index - 1].offers
                            if item.offer.unit_name == "pp_6"
                        ) if index else 0.0
                        self.assertEqual(state[14], previous_power / 10.0)
                    self.assertEqual(len(states), 16)
                    self.assertNotEqual(states[-1][:12], states[-2][:12])

    def test_future_demand_changes_are_visible_only_after_the_changed_market_clears(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, _, original = self.collect_states(make_scenario(root / "baseline"))
            for k in (0, 1):
                with self.subTest(changed_step=k):
                    path = make_scenario(root / f"changed_{k}", demand_change_from=k + 1)
                    states, _, result = self.collect_states(path)
                    self.assertEqual(states[:k + 1], baseline[:k + 1])
                    self.assertEqual(result.market_results[:k], original.market_results[:k])
                    self.assertEqual(result.market_results[k].clearing_price_eur_per_mwh, 150.0)
                    self.assertEqual(original.market_results[k].clearing_price_eur_per_mwh, 20.0)
                    self.assertEqual(states[k + 1][11], 150.0 / 200.0)
                    self.assertEqual(states[k + 1][14], 7.0 / 10.0)
                    self.assertNotEqual(states[k + 1], baseline[k + 1])

    def test_only_delivery_data_is_required_and_unused_future_demand_has_no_effect(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = self.collect_states(make_scenario(root / "short", profile_hours=4))
            extended = self.collect_states(make_scenario(
                root / "extended", profile_hours=18, demand_change_from=4,
            ))
            self.assertEqual(short, extended)
            path = make_scenario(root / "missing", profile_hours=3)
            with self.assertRaisesRegex(InputValidationError, "demand_df.csv/load"):
                self.collect_states(path)

    def test_calendar_encoding_and_history_window(self):
        def observation(delivery):
            return build_learning_observation(
                delivery_start=delivery,
                historical_clearing_prices_eur_per_mwh=range(15),
                previous_power_mw=5.0, rated_power_mw=10.0,
                marginal_cost_eur_per_mwh=40.0, price_scale_eur_per_mwh=200.0,
            )

        for hour, expected in ((0, (0.0, 1.0)), (6, (1.0, 0.0)),
                               (12, (0.0, -1.0)), (18, (-1.0, 0.0))):
            with self.subTest(hour=hour):
                state = observation(START + hour * HOUR)
                self.assertEqual(state[:12], tuple(value / 200.0 for value in range(3, 15)))
                self.assertEqual(state[14:], (0.5, 0.2))
                for actual, wanted in zip(state[12:14], expected):
                    self.assertAlmostEqual(actual, wanted)
                tomorrow = observation(START + (hour + 24) * HOUR)
                for actual, wanted in zip(tomorrow[12:14], state[12:14]):
                    self.assertAlmostEqual(actual, wanted)
        before_midnight = observation(START + 23.5 * HOUR)
        after_midnight = observation(START + 24.5 * HOUR)
        self.assertLess(before_midnight[12], 0.0)
        self.assertGreater(after_midnight[12], 0.0)
        self.assertAlmostEqual(before_midnight[12], -after_midnight[12])
        self.assertAlmostEqual(before_midnight[13], after_midnight[13])

    def test_heuristic_and_storage_participants_still_enable_price_prediction(self):
        with TemporaryDirectory() as temporary:
            for name, options in (
                ("heuristic", {"rival_strategy": "powerplant_energy_heuristic_flexable"}),
                ("storage", {"storage": True}),
            ):
                with self.subTest(participant=name):
                    path = make_scenario(Path(temporary) / name, profile_hours=18, **options)
                    with patch(PRICE_PREDICTOR, wraps=_naive_price_forecasts) as forecast:
                        result = simulate_learning_episode(
                            path, lambda state: (0.2, 0.2), period="train"
                        )
                    forecast.assert_called_once()
                    self.assertEqual(len(result.market_results), 3)

    def test_extension_price_window_is_explicit_and_separate_from_data_window(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary), profile_hours=18)
            settings = load_market_settings(path / "config.yaml")
            market = BaseMarket(path, settings, tuple(market_openings(settings)))
            with patch(PRICE_PREDICTOR, wraps=_naive_price_forecasts) as forecast:
                market.prepare(input_loader=lambda _: EomExtensionInputs(
                    required_data_offsets=(0, 10), price_forecast_offsets=(0, 2),
                ))
            forecast.assert_called_once()
            self.assertEqual(set(market.price_forecasts), {
                START + i * HOUR for i in range(1, 6)
            })


class LearningPeriodContracts(unittest.TestCase):
    def test_periods_are_required_ordered_disjoint_and_aligned(self):
        cases = (
            ("missing", None, "learning_config.periods"),
            ("overlap", {**DEFAULT_PERIODS, "validation": (3, 6)}, "不能重叠"),
            ("order", {"train": (4, 6), "validation": (1, 4), "test": (6, 8)}, "不能重叠"),
            ("empty", {**DEFAULT_PERIODS, "test": (6, 6)}, "end 必须晚于 start"),
            ("unaligned", {**DEFAULT_PERIODS, "test": (6.5, 8)}, "对齐"),
            ("incomplete", {"train": (1, 4), "validation": (4, 6)}, "恰好包含"),
        )
        with TemporaryDirectory() as temporary:
            for name, periods, message in cases:
                with self.subTest(case=name):
                    path = make_scenario(Path(temporary) / name, periods=periods)
                    with self.assertRaisesRegex(InputValidationError, message):
                        load_market_settings(path / "config.yaml")

    def test_each_period_delivers_exactly_its_own_hours_regardless_of_start_dates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, options in (
                ("plain", {}),
                ("distracting_eom_start", {"eom_start": START + 10 * HOUR}),
            ):
                path = make_scenario(root / name, **options)
                delivered = {}
                for period, hours in PERIOD_HOURS.items():
                    with self.subTest(case=name, period=period):
                        states, _, result = collect_states(path, period)
                        starts = [market.delivery_start for market in result.market_results]
                        self.assertEqual(starts, [START + hour * HOUR for hour in hours])
                        self.assertTrue(all(
                            market.delivery_end - market.delivery_start == HOUR
                            for market in result.market_results
                        ))
                        angle = 2 * pi * hours[0] / 24
                        self.assertAlmostEqual(states[0][12], sin(angle))
                        self.assertAlmostEqual(states[0][13], cos(angle))
                        self.assertEqual(states[0][:12], (0.0,) * 12)
                        delivered[period] = set(starts)
                self.assertFalse(delivered["train"] & delivered["validation"])
                self.assertFalse(delivered["validation"] & delivered["test"])
                self.assertEqual(
                    sorted(delivered["train"] | delivered["validation"] | delivered["test"]),
                    [START + hour * HOUR for hour in range(1, 8)],
                )

    def test_learning_simulation_requires_an_explicit_period(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            for period in (None, "all"):
                with self.subTest(period=period):
                    with self.assertRaisesRegex(InputValidationError, "learning period"):
                        simulate_learning_episode(path, lambda state: (0.2, 0.2), period=period)

    def test_strict_horizon_clips_forecast_windows_before_the_period_end(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            settings = load_market_settings(path / "config.yaml")
            period_settings = learning_period_settings(settings, "train")
            deliveries = [START + hour * HOUR for hour in PERIOD_HOURS["train"]]
            clipped = _forecast_starts(
                deliveries, settings=period_settings, requires_price_forecast=True,
                extra_forecast_offsets=(0, 10), requires_storage_price_forecast=True,
                hard_forecast_horizon=True,
            )
            self.assertTrue(clipped)
            self.assertTrue(all(start < period_settings.end for start in clipped))
            unclipped = _forecast_starts(
                deliveries, settings=period_settings, requires_price_forecast=True,
                extra_forecast_offsets=(0, 10), requires_storage_price_forecast=True,
            )
            self.assertTrue(any(start > period_settings.end for start in unclipped))

    def test_heuristic_opponents_cannot_see_beyond_the_current_period(self):
        """启发式对手的预测窗口在时段边界截断，后续时段的真实数据不改变前一时段结果。"""
        def run(path, period):
            states = []
            result = simulate_learning_episode(
                path, lambda state: (states.append(state), (0.75, 0.75))[1],
                period=period,
            )
            return states, result.market_results

        options = {
            "rival_strategy": "powerplant_energy_heuristic_flexable",
            "profile_hours": 16,
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = {
                period: run(make_scenario(root / "baseline", **options), period)
                for period in PERIOD_HOURS
            }
            for name, change in (
                ("validation", {"demand_change_from": 4}),
                ("test", {"demand_change_from": 6}),
            ):
                with self.subTest(changed_period=name):
                    path = make_scenario(root / name, **options, **change)
                    states, train_result = run(path, "train")
                    self.assertEqual(train_result, baseline["train"][1])
                    self.assertEqual(states, baseline["train"][0])
                    _, validation_result = run(path, "validation")
                    if name == "validation":
                        self.assertNotEqual(validation_result, baseline["validation"][1])
                    else:
                        self.assertEqual(validation_result, baseline["validation"][1])
                    _, test_result = run(path, "test")
                    self.assertNotEqual(test_result, baseline["test"][1])

    def test_mixed_scenario_needs_only_the_current_periods_data(self):
        """混合场景按时段运行时，只要求该时段自身的数据。"""
        with TemporaryDirectory() as temporary:
            path = make_scenario(
                Path(temporary),
                rival_strategy="powerplant_energy_heuristic_flexable",
                profile_hours=4,
            )
            simulate_learning_episode(path, lambda state: (0.75, 0.75), period="train")
            with self.assertRaisesRegex(InputValidationError, "demand_df.csv/load"):
                simulate_learning_episode(path, lambda state: (0.75, 0.75), period="validation")


class LearningTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_nondefault_price_scale_preserves_cost_centered_initial_exploration(self):
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary))
            config = replace(
                load_market_settings(path / "config.yaml").learning_config,
                exploration_noise_std=0.0,
            )
            agent = MATD3Agent(config, total_regular_steps=1, seed=7)
            states = []

            def action(state):
                states.append(state)
                return agent.exploration_action(state, initial_experience=True)

            with patch(PRICE_PREDICTOR, side_effect=AssertionError("Unexpected price forecast")):
                result = simulate_learning_episode(path, action, period="train")
        self.assertEqual(states[0][-1], 40.0 / config.max_bid_price)
        self.assertEqual(states[1][11], 20.0 / config.max_bid_price)
        prices = [
            cleared.offer.bid_price_eur_per_mwh
            for market in result.market_results for cleared in market.offers
            if cleared.offer.unit_name == "pp_6"
        ]
        self.assertEqual(prices, [40.0] * 6)

    def test_train_save_and_evaluate_without_external_forecasts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            with patch(PRICE_PREDICTOR, side_effect=AssertionError("Unexpected price forecast")):
                result = train_learning_scenario(path, output, independent_runs=1)
                best = output / "checkpoints" / "best.pt"
                payload = torch.load(best, map_location="cpu", weights_only=True)
                config = load_market_settings(path / "config.yaml").learning_config
                self.assertEqual(set(payload), {
                    "checkpoint_version", "algorithm", "state_version", "observation_size", "action_size",
                    "episode", "validation_total_reward", "periods", "max_bid_price",
                    "seed", "run", "actor",
                })
                self.assertEqual(payload["checkpoint_version"], 2)
                self.assertEqual(payload["algorithm"], "matd3")
                self.assertEqual(payload["state_version"], "v5-observation-16")
                self.assertEqual(payload["observation_size"], 16)
                self.assertEqual(payload["periods"], _periods_payload(config))
                self.assertEqual(payload["actor"]["fc1.weight"].shape[1], 16)
                self.assertEqual(list((output / "checkpoints").glob("*.pt")), [best])
                self.assertEqual(
                    [step.delivery_start for step in result.learning_steps],
                    [START + hour * HOUR for hour in PERIOD_HOURS["validation"]],
                )
                self.assertFalse((output / "test").exists())
                before = best.read_bytes()
                evaluated = evaluate_learning_scenario(path, root / "evaluation", checkpoint_path=best)
                self.assertEqual(evaluated.learning_steps, result.learning_steps)
                trained_period = evaluate_learning_scenario(
                    path, root / "evaluation_train", checkpoint_path=best, period="train"
                )
                self.assertEqual(
                    [step.delivery_start for step in trained_period.learning_steps],
                    [START + hour * HOUR for hour in PERIOD_HOURS["train"]],
                )
                self.assertEqual(best.read_bytes(), before)
            with self.assertRaisesRegex(InputValidationError, "--learning-mode test"):
                evaluate_learning_scenario(path, output, checkpoint_path=best, period="test")

    def test_best_checkpoint_is_selected_by_validation_total_reward(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            train_learning_scenario(path, root / "output", training_episodes=5, independent_runs=1)
            rows = [
                row for row in read_rows(root / "output" / "learning_metrics.csv")
                if row["phase"] == "validation" and int(row["episode"]) > 1
            ]
            best_row = max(rows, key=lambda row: float(row["total_reward"]))
            payload = torch.load(
                root / "output" / "checkpoints" / "best.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(payload["episode"], int(best_row["episode"]))
            self.assertAlmostEqual(
                payload["validation_total_reward"], float(best_row["total_reward"])
            )
            self.assertEqual(
                [int(row["episode"]) for row in rows].index(payload["episode"]),
                [float(row["total_reward"]) for row in rows].index(
                    float(best_row["total_reward"])
                ),
            )

    def test_training_weights_do_not_depend_on_test_period_values(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = []
            for name, options in (
                ("baseline", {}),
                ("changed_test_demand", {"demand_change_from": 6}),
            ):
                path = make_scenario(root / name, **options)
                output = root / f"{name}_output"
                train_learning_scenario(path, output, independent_runs=1)
                payload = torch.load(
                    output / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
                )
                weights.append((payload["episode"], payload["actor"]))
            for episode, actor in weights[1:]:
                self.assertEqual(episode, weights[0][0])
                self.assertEqual(actor.keys(), weights[0][1].keys())
                for key in actor:
                    self.assertTrue(torch.equal(actor[key], weights[0][1][key]), key)

    def test_training_preflight_rejects_missing_test_period(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input", profile_hours=6)
            with self.assertRaisesRegex(InputValidationError, "test.*demand_df.csv/load"):
                train_learning_scenario(path, root / "output", independent_runs=1)

    def test_command_line_overrides_episode_count_and_prints_effective_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main([
                    "--input-dir", str(path), "--output-dir", str(output),
                    "--training-episodes", "4", "--training-runs", "1", "--no-plots",
                ])
            self.assertEqual(code, 0)
            episodes = [
                int(row["episode"]) for row in read_rows(output / "learning_metrics.csv")
                if row["phase"] in {"initial_experience", "train"}
            ]
            self.assertEqual(episodes, [1, 2, 3, 4])
            self.assertTrue((output / "checkpoints" / "best.pt").is_file())
            self.assertIn("cpu", stdout.getvalue())
            for period in DEFAULT_PERIODS:
                self.assertIn(period + ": [", stdout.getvalue())

    def test_independent_test_command_evaluates_every_run_without_changing_models(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=3)
            manifest = json.loads(
                (output / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            self.assertEqual([entry["run"] for entry in manifest["runs"]], [1, 2, 3])
            self.assertEqual([entry["seed"] for entry in manifest["runs"]], [7, 8, 9])
            selected = max(manifest["runs"], key=lambda entry: entry["validation_total_reward"])
            self.assertEqual(manifest["selected_run"], selected["run"])
            self.assertFalse(any(
                row["phase"].startswith("test") for row in read_rows(output / "learning_metrics.csv")
            ))
            before = checkpoint_bytes(output / "checkpoints")
            training_plots = generate_learning_plots(
                output / "learning_metrics.csv", output / "plots"
            )
            self.assertEqual(
                [plot.name for plot in training_plots],
                ["learning_curves.png", "learning_validation_comparison.png"],
            )

            result = run_learning_test(path, output)

            self.assertEqual(checkpoint_bytes(output / "checkpoints"), before)
            self.assertEqual(
                [step.delivery_start for step in result.learning_steps],
                [START + hour * HOUR for hour in PERIOD_HOURS["test"]],
            )
            rows = read_rows(output / "learning_test_metrics.csv")
            self.assertEqual(
                sorted((int(row["run"]), row["phase"]) for row in rows),
                sorted((run, phase) for run in (1, 2, 3)
                       for phase in ("test_actor", "test_baseline")),
            )
            summary = {row["phase"]: row for row in read_rows(output / "learning_test_summary.csv")}
            self.assertEqual(set(summary), {"test_actor", "test_baseline"})
            for phase, row in summary.items():
                values = [float(item["total_reward"]) for item in rows if item["phase"] == phase]
                self.assertEqual(int(row["run_count"]), 3)
                self.assertAlmostEqual(float(row["total_reward_mean"]), fmean(values))
            for run in (1, 2, 3):
                self.assertTrue((output / "test" / f"run_{run:02d}" / "baseline").is_dir())
            plots = generate_learning_plots(
                output / "learning_test_metrics.csv", output / "plots"
            )
            self.assertEqual([plot.name for plot in plots], ["learning_test_comparison.png"])
            self.assertEqual(
                sorted(plot.name for plot in (output / "plots").glob("learning_*.png")),
                ["learning_curves.png", "learning_test_comparison.png",
                 "learning_validation_comparison.png"],
            )

    def test_retraining_replaces_run_records_and_training_progress(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=3, training_episodes=4)
            train_learning_scenario(path, output, independent_runs=1, training_episodes=3)
            manifest = json.loads((output / "checkpoints" / "runs.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["run"] for entry in manifest["runs"]], [1])
            self.assertEqual(manifest["runs"][0]["best_checkpoint"], "best.pt")
            self.assertEqual(
                [int(row["episode"]) for row in read_rows(output / "learning_progress.csv")],
                [1, 2, 3],
            )
            episodes = [
                int(row["episode"]) for row in read_rows(output / "learning_metrics.csv")
                if row["phase"] in {"initial_experience", "train"}
            ]
            self.assertEqual(episodes, [1, 2, 3])

    def test_training_cli_does_not_accept_an_evaluation_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                code = main([
                    "--input-dir", str(path), "--output-dir", str(root / "output"),
                    "--learning-mode", "train", "--checkpoint", str(root / "old.pt"),
                    "--no-plots",
                ])
            self.assertEqual(code, 2)
            self.assertIn("--checkpoint 仅用于评估或测试", stderr.getvalue())
            self.assertFalse((root / "output").exists())

    def test_command_line_trains_then_tests_separately(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            common = ["--input-dir", str(path), "--output-dir", str(output), "--no-plots"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main([*common, "--training-runs", "1"]), 0)
                self.assertFalse((output / "learning_test_metrics.csv").exists())
                self.assertEqual(
                    main([*common, "--learning-mode", "test", "--period", "train"]), 2
                )
                self.assertEqual(main([*common, "--learning-mode", "test"]), 0)
                self.assertEqual(len(read_rows(output / "learning_test_metrics.csv")), 2)
                with self.assertRaises(SystemExit):
                    main([*common, "--learning-mode", "evaluate", "--period", "test"])

    def test_changed_periods_reject_existing_checkpoints_and_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=1)
            make_scenario(root / "input", periods={**DEFAULT_PERIODS, "test": (6, 7)})
            config = load_market_settings(path / "config.yaml").learning_config
            agent = MATD3Agent(config, total_regular_steps=1)
            with self.assertRaisesRegex(InputValidationError, "时段与当前"):
                agent.load_checkpoint(output / "checkpoints" / "best.pt")
            with self.assertRaisesRegex(InputValidationError, "时段与当前"):
                run_learning_test(path, output)

    def test_old_and_other_dimensional_checkpoints_are_rejected(self):
        # 旧版本与旧维度的检查点不做迁移，统一按不兼容拒绝。
        with TemporaryDirectory() as temporary:
            path = make_scenario(Path(temporary) / "input")
            config = load_market_settings(path / "config.yaml").learning_config
            agent = MATD3Agent(config, total_regular_steps=1, seed=7)
            checkpoint = Path(temporary) / "incompatible.pt"
            metadata = {
                "checkpoint_version": 2, "algorithm": "matd3", "state_version": "v5-observation-16",
                "observation_size": 16, "action_size": 2,
            }
            for payload in (
                {**metadata, "state_version": "v5-observation-38"},
                {**metadata, "state_version": "v5-observation-26"},
                {**metadata, "observation_size": 38},
                {**metadata, "observation_size": 26},
                {**metadata, "observation_size": 14},
                {**metadata, "checkpoint_version": 1},
                {key: value for key, value in metadata.items() if key != "algorithm"},
                {},
                {**metadata, "state_version": "unknown"},
                {**metadata, "observation_size": None},
                {**metadata, "observation_size": "26"},
            ):
                with self.subTest(payload=payload):
                    torch.save(payload, checkpoint)
                    with self.assertRaisesRegex(
                        InputValidationError, "incompatible with Version 5"
                    ):
                        agent.load_checkpoint(checkpoint)

if __name__ == "__main__":
    unittest.main()
