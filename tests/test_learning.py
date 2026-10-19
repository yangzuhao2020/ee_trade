"""V5 信息边界、数据时段划分、预测需求解耦及 16 维模型训练、选模与独立测试的端到端测试。"""

from contextlib import redirect_stderr, redirect_stdout
import csv
import json
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

from electricity_market_sim.config import load_market_settings
from electricity_market_sim.deep_learning.agent import MATD3Agent, _periods_payload
from electricity_market_sim.deep_learning.learning import build_learning_observation
from electricity_market_sim.deep_learning.learning_metrics import _metrics
from electricity_market_sim.deep_learning.test import (
    evaluate_learning_scenario, run_learning_test,
)
from electricity_market_sim.deep_learning.training import (
    _restore_best_checkpoint, train_learning_scenario,
)
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


class LearningMarketContracts(unittest.TestCase):
    collect_states = staticmethod(collect_states)

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

    def test_train_save_resume_and_evaluate_without_external_forecasts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            with patch(PRICE_PREDICTOR, side_effect=AssertionError("Unexpected price forecast")):
                result = train_learning_scenario(path, output, independent_runs=1)
                latest = output / "checkpoints" / "latest.pt"
                best = output / "checkpoints" / "best.pt"
                payload = torch.load(latest, map_location="cpu", weights_only=True)
                config = load_market_settings(path / "config.yaml").learning_config
                self.assertEqual(payload["checkpoint_version"], 2)
                self.assertEqual(payload["state_version"], "v5-observation-16")
                self.assertEqual(payload["observation_size"], 16)
                self.assertEqual(payload["periods"], _periods_payload(config))
                self.assertEqual(payload["actor"]["fc1.weight"].shape[1], 16)
                self.assertEqual(payload["critics"]["q1.0.weight"].shape[1], 18)
                # 经验池只含训练时段：3 个回合 × 3 个训练交付小时。
                self.assertEqual(payload["replay"]["states"].shape, (9, 16))
                self.assertEqual(payload["replay"]["next_states"].shape, (9, 16))
                self.assertNotIn("load_base_mw", payload)
                self.assertNotIn("validation_discounted_reward", payload)
                self.assertGreater(payload["gradient_updates"], 0)
                self.assertTrue(payload["actor_optimizer"]["state"])
                self.assertEqual(
                    [step.delivery_start for step in result.learning_steps],
                    [START + hour * HOUR for hour in PERIOD_HOURS["validation"]],
                )
                self.assertFalse((output / "test").exists())
                restored = MATD3Agent(config, total_regular_steps=6)
                restored.load_checkpoint(latest, load_optimizers=True)
                self.assertEqual(restored.replay.size, 9)
                self.assertEqual(restored.gradient_updates, payload["gradient_updates"])
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
                resumed_output = root / "resumed"
                resumed = train_learning_scenario(
                    path, resumed_output, checkpoint_path=latest,
                    training_episodes=4, independent_runs=1,
                )
                resumed_payload = torch.load(
                    resumed_output / "checkpoints" / "latest.pt",
                    map_location="cpu", weights_only=True,
                )
                self.assertEqual(resumed_payload["episode"], 4)
                self.assertEqual(resumed_payload["replay"]["states"].shape, (12, 16))
                self.assertGreater(resumed_payload["gradient_updates"], payload["gradient_updates"])
                reevaluated = evaluate_learning_scenario(
                    path, root / "reevaluation",
                    checkpoint_path=resumed_output / "checkpoints" / "best.pt",
                )
                self.assertEqual(reevaluated.learning_steps, resumed.learning_steps)
                self.assertEqual(best.read_bytes(), before)
            with self.assertRaisesRegex(InputValidationError, "--learning-mode test"):
                evaluate_learning_scenario(path, root / "peek", checkpoint_path=best, period="test")

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

    def test_training_never_reads_the_test_period(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = []
            for name, options in (
                ("baseline", {}),
                ("changed_test_demand", {"demand_change_from": 6}),
                ("missing_test_demand", {"profile_hours": 6}),
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
            with self.assertRaisesRegex(InputValidationError, "demand_df.csv/load"):
                run_learning_test(
                    root / "missing_test_demand", root / "missing_test_demand_output"
                )

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

    def test_resumed_run_keeps_its_number_and_updates_the_manifest(self):
        """恢复 run_02/latest.pt 沿用编号 2，清单合并更新，其余运行保持不变。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=3)
            manifest_before = json.loads(
                (output / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            checkpoints_before = checkpoint_bytes(output / "checkpoints")

            train_learning_scenario(
                path, output,
                checkpoint_path=output / "checkpoints" / "run_02" / "latest.pt",
                training_episodes=4, independent_runs=1,
            )

            manifest_after = json.loads(
                (output / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            entries = {entry["run"]: entry for entry in manifest_after["runs"]}
            self.assertEqual(sorted(entries), [1, 2, 3])
            self.assertEqual(entries[1], manifest_before["runs"][0])
            self.assertEqual(entries[3], manifest_before["runs"][2])
            self.assertEqual(entries[2]["seed"], manifest_before["runs"][1]["seed"])
            self.assertEqual(entries[2]["best_checkpoint"], "run_02/best.pt")
            self.assertGreaterEqual(
                entries[2]["validation_total_reward"],
                manifest_before["runs"][1]["validation_total_reward"],
            )
            payload = torch.load(
                output / "checkpoints" / "run_02" / "latest.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(payload["episode"], 4)
            self.assertEqual(payload["run"], 2)
            # 未恢复运行的检查点不受影响；根目录 best.pt 与入选运行一致。
            checkpoints_after = checkpoint_bytes(output / "checkpoints")
            for file_path, data in checkpoints_before.items():
                if file_path.parent.name in {"run_01", "run_03"}:
                    self.assertEqual(checkpoints_after[file_path], data)
            selected = manifest_after["selected_run"]
            self.assertEqual(
                (output / "checkpoints" / "best.pt").read_bytes(),
                (output / "checkpoints" / entries[selected]["best_checkpoint"]).read_bytes(),
            )
            # 独立测试仍覆盖全部三个运行。
            run_learning_test(path, output)
            rows = read_rows(output / "learning_test_metrics.csv")
            self.assertEqual({int(row["run"]) for row in rows}, {1, 2, 3})

    def test_appended_resumed_run_wins_a_tie_with_a_higher_run_number(self):
        """清单 [1, 3] 追加恢复运行 2 后，2、3 奖励并列最高时应选 2。"""
        def controlled_metrics(result, *, run, phase, episode, gamma):
            metrics = _metrics(
                result, run=run, phase=phase, episode=episode, gamma=gamma
            )
            if phase == "validation":
                return replace(metrics, total_reward=1.0 if run == 1 else 2.0)
            return metrics

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            with patch(
                "electricity_market_sim.deep_learning.training._metrics",
                side_effect=controlled_metrics,
            ):
                train_learning_scenario(path, output, independent_runs=3)
                manifest_path = output / "checkpoints" / "runs.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["runs"] = [entry for entry in manifest["runs"] if entry["run"] != 2]
                manifest["selected_run"] = 3
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                train_learning_scenario(
                    path, output,
                    checkpoint_path=output / "checkpoints" / "run_02" / "latest.pt",
                    training_episodes=4, independent_runs=1,
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([entry["run"] for entry in manifest["runs"]], [1, 2, 3])
            self.assertEqual(
                [entry["validation_total_reward"] for entry in manifest["runs"]],
                [1.0, 2.0, 2.0],
            )
            self.assertEqual(manifest["selected_run"], 2)
            self.assertEqual(
                (output / "checkpoints" / "best.pt").read_bytes(),
                (output / "checkpoints" / "run_02" / "best.pt").read_bytes(),
            )

    def test_resumed_run_into_a_fresh_directory_preserves_its_number(self):
        """把 run_02/latest.pt 恢复到全新目录，运行编号仍保留为 2。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=3)
            resumed = root / "resumed"
            train_learning_scenario(
                path, resumed,
                checkpoint_path=output / "checkpoints" / "run_02" / "latest.pt",
                training_episodes=4, independent_runs=1,
            )
            manifest = json.loads(
                (resumed / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["runs"]), 1)
            entry = manifest["runs"][0]
            self.assertEqual(entry["run"], 2)
            self.assertEqual(entry["best_checkpoint"], "run_02/best.pt")
            self.assertEqual(manifest["selected_run"], 2)
            payload = torch.load(
                resumed / "checkpoints" / "run_02" / "latest.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(payload["episode"], 4)
            self.assertEqual(payload["run"], 2)
            self.assertEqual(
                (resumed / "checkpoints" / "best.pt").read_bytes(),
                (resumed / "checkpoints" / "run_02" / "best.pt").read_bytes(),
            )

    def test_single_run_layout_converts_to_multi_run_on_resume(self):
        """单运行清单指向根 best.pt；恢复 run 2 时原模型先迁移到 run_01/。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_scenario(root / "input")
            output = root / "output"
            train_learning_scenario(path, output, independent_runs=1)
            manifest_before = json.loads(
                (output / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_before["runs"][0]["best_checkpoint"], "best.pt")
            first_best = (output / "checkpoints" / "best.pt").read_bytes()
            first_latest = (output / "checkpoints" / "latest.pt").read_bytes()

            other = root / "other"
            train_learning_scenario(path, other, independent_runs=3)
            train_learning_scenario(
                path, output,
                checkpoint_path=other / "checkpoints" / "run_02" / "latest.pt",
                training_episodes=4, independent_runs=1,
            )

            manifest = json.loads(
                (output / "checkpoints" / "runs.json").read_text(encoding="utf-8")
            )
            entries = {entry["run"]: entry for entry in manifest["runs"]}
            self.assertEqual(sorted(entries), [1, 2])
            self.assertEqual(entries[1]["best_checkpoint"], "run_01/best.pt")
            self.assertEqual(entries[2]["best_checkpoint"], "run_02/best.pt")
            # 原模型迁入 run_01/ 且内容不变，两个条目指向不同的模型文件。
            self.assertEqual(
                (output / "checkpoints" / "run_01" / "best.pt").read_bytes(),
                first_best,
            )
            self.assertEqual(
                (output / "checkpoints" / "run_01" / "latest.pt").read_bytes(),
                first_latest,
            )
            self.assertNotEqual(
                (output / "checkpoints" / "run_01" / "best.pt").read_bytes(),
                (output / "checkpoints" / "run_02" / "best.pt").read_bytes(),
            )
            # 根 best.pt 只保存入选运行检查点的副本。
            selected = manifest["selected_run"]
            self.assertEqual(
                (output / "checkpoints" / "best.pt").read_bytes(),
                (output / "checkpoints" / entries[selected]["best_checkpoint"]).read_bytes(),
            )
            run_learning_test(path, output)
            rows = read_rows(output / "learning_test_metrics.csv")
            self.assertEqual({int(row["run"]) for row in rows}, {1, 2})

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
                agent.load_checkpoint(output / "checkpoints" / "best.pt", load_optimizers=False)
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
                "checkpoint_version": 2, "state_version": "v5-observation-16",
                "observation_size": 16, "action_size": 2,
            }
            for payload in (
                {**metadata, "state_version": "v5-observation-38"},
                {**metadata, "state_version": "v5-observation-26"},
                {**metadata, "observation_size": 38},
                {**metadata, "observation_size": 26},
                {**metadata, "observation_size": 14},
                {**metadata, "checkpoint_version": 1},
                [], {},
                {**metadata, "state_version": "unknown"},
                {**metadata, "observation_size": None},
                {**metadata, "observation_size": "26"},
            ):
                with self.subTest(payload=payload):
                    torch.save(payload, checkpoint)
                    with self.assertRaisesRegex(
                        InputValidationError, "incompatible with Version 5"
                    ):
                        agent.load_checkpoint(checkpoint, load_optimizers=False)

    def test_best_checkpoint_restore_requires_matching_price_scale_and_periods(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_market_settings(
                make_scenario(root / "input") / "config.yaml"
            ).learning_config
            periods = _periods_payload(config)
            companion = root / "best.pt"
            destination = root / "resumed" / "best.pt"
            destination.parent.mkdir()
            metadata = {
                "checkpoint_version": 2, "state_version": "v5-observation-16",
                "observation_size": 16, "action_size": 2,
                "validation_total_reward": 1.0, "seed": 7, "max_bid_price": 200.0,
                "periods": periods,
            }
            torch.save(metadata, companion)
            torch.save({**metadata, "max_bid_price": 100.0}, destination)
            arguments = dict(
                restored_path=root / "latest.pt", destination=destination,
                expected_reward=1.0, expected_seed=7, expected_max_bid_price=200.0,
                expected_periods=periods,
            )
            _restore_best_checkpoint(**arguments)
            self.assertEqual(destination.read_bytes(), companion.read_bytes())
            for changed in (
                {"max_bid_price": 100.0},
                {"periods": {**periods, "test": periods["validation"]}},
            ):
                with self.subTest(changed=list(changed)):
                    for checkpoint in (companion, destination):
                        torch.save({**metadata, **changed}, checkpoint)
                    with self.assertRaisesRegex(InputValidationError, "no matching best.pt"):
                        _restore_best_checkpoint(**arguments)


if __name__ == "__main__":
    unittest.main()
