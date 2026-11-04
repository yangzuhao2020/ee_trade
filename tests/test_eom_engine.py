"""共享市场事件时序、扩展报价替换、学习动作处理及结果输出的行为测试。"""

import csv
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from electricity_market_sim.bidding import PlantRuntimeState
from electricity_market_sim.clearing_algorithm import clear_pay_as_clear
from electricity_market_sim.deep_learning.learning import LearningEpisodeSession
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.models import (
    DemandBid,
    DemandUnit,
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    PowerPlant,
    SimulationResult,
    StorageDispatchResult,
    SupplyOffer,
)
from electricity_market_sim.reporting import write_results
from electricity_market_sim.simulators.base_market.forecasts import _opening_coverage_issues
from electricity_market_sim.simulators.base_market.plants import (
    _apply_startup_costs, _offers_for_opening, _record_runtime_states,
)
from electricity_market_sim.simulators.eom import simulate_eom_market
from UI import charts


START = datetime(2024, 1, 1)
HOUR = timedelta(hours=1)
SETTINGS = MarketSettings(START, START + 4 * HOUR, HOUR, HOUR, HOUR,
                          HOUR, 1, HOUR, 100.0, -100.0, "pay_as_clear")
PLANT = PowerPlant("plant", "operator", "gas", "powerplant_energy_naive",
                   "gas", 0.2, 1.0, 0.0, 0.5, 0.0, start_cost_eur=17.0)


def orders(start, name="plant", quantity=1.0, segment="single"):
    end = start + HOUR
    bid = DemandBid("load", "operator", start, end, 1.0, 100.0)
    offer = SupplyOffer(name, "operator", "gas", start, end, quantity,
                        quantity, 10.0, 10.0,
                        offer_id=name + "::" + segment, offer_segment=segment)
    return bid, offer


class TraceExtension:
    def __init__(self, before_opening):
        self.settle_before_opening = before_opening
        self.startup_costs = []

    def record_delivery_start(self, result):
        self.startup_costs.append(result.offers[0].startup_cost_eur)

    def finalize(self, result):
        return result


class TraceMarket:
    def __init__(self, before_opening=None):
        self.settings = SETTINGS
        self.events = []
        self.extension = (None if before_opening is None else
                          TraceExtension(before_opening))
        self.valid_openings = tuple(
            MarketOpening(START + i * HOUR,
                          ((START + (i + 1) * HOUR, START + (i + 2) * HOUR),))
            for i in range(3)
        )
        self.runtime_states = {PLANT.name: PlantRuntimeState.initially_off()}
        self.observed_costs = []

    def clear_opening(self, opening, scheduled_results):
        self.events.append(("open", opening.opening_time))
        if scheduled_results:
            self.observed_costs.append(next(iter(scheduled_results.values())).offers[0].startup_cost_eur)
        bid, offer = orders(opening.products[0][0])
        return (clear_pay_as_clear([bid], [offer]),)

    def start_delivery(self, result):
        self.events.append(("start", result.delivery_start))
        return _apply_startup_costs(result, (PLANT,), self.runtime_states)

    def end_delivery(self, result):
        self.events.append(("end", result.delivery_end))
        _record_runtime_states(result, self.runtime_states)
        return []


class EventContracts(unittest.TestCase):
    def test_conventional_and_industrial_open_before_settling_starts(self):
        for before_opening in (None, False):
            with self.subTest(extension=before_opening):
                market = TraceMarket(before_opening)
                result = simulate_eom_market(market)
                at_two = [kind for kind, time in market.events
                          if time == START + 2 * HOUR]
                self.assertEqual(at_two, ["end", "open", "start"])
                self.assertEqual(market.observed_costs, [0.0, 17.0])
                self.assertEqual(len(result.market_results), 3)

    def test_learning_settles_starts_before_requesting_the_next_action(self):
        market = TraceMarket(True)
        simulate_eom_market(market)
        at_two = [kind for kind, time in market.events
                  if time == START + 2 * HOUR]
        self.assertEqual(at_two, ["end", "start", "open"])
        self.assertEqual(market.observed_costs, [17.0, 17.0])

    def test_start_callback_receives_startup_cost_and_delivery_ends_commit_state(self):
        market = TraceMarket(True)
        result = simulate_eom_market(market)
        self.assertEqual(market.extension.startup_costs, [17.0, 0.0, 0.0])
        self.assertEqual(market.runtime_states[PLANT.name].elapsed_periods, 3)
        self.assertEqual([item.offers[0].startup_cost_eur for item in result.market_results],
                         market.extension.startup_costs)


class SupplyHookContracts(unittest.TestCase):
    def quote(self, replacement):
        normal = replace(PLANT, name="normal")

        def supply(plant, products, powers, cost_at, scheduled):
            return replacement if plant.name == PLANT.name else None

        return _offers_for_opening(
            opening=MarketOpening(START, ((START + HOUR, START + 2 * HOUR),)),
            plants=(PLANT, normal), runtime_states={}, availability_profiles={},
            marginal_costs={PLANT.name: 10.0, normal.name: 10.0},
            fuel_price_profiles=None, price_forecasts={},
            exchange_unit=None, exchange_schedules={},
            extension=SimpleNamespace(offers_for_plant=supply), scheduled_results={},
        )

    def test_extension_replaces_only_its_plant_without_duplicate_supply(self):
        minimum = orders(START + HOUR, segment="minimum", quantity=0.25)[1]
        flexible = orders(START + HOUR, segment="flexible", quantity=0.75)[1]
        offers = self.quote((minimum, flexible))
        self.assertEqual([item.identifier for item in offers],
                         ["plant::minimum", "plant::flexible",
                          "normal::single::2024-01-01T01:00:00"])
        self.assertEqual(sum(item.offered_energy_mwh for item in offers), 2.0)

    def test_empty_replacement_suppresses_supply_but_none_keeps_normal_quotes(self):
        self.assertEqual([item.unit_name for item in self.quote(())], ["normal"])
        self.assertEqual([item.unit_name for item in self.quote(None)], ["plant", "normal"])

    def test_extension_only_strategy_is_never_interpreted_as_a_block_bid(self):
        plant = replace(PLANT, bidding_strategy="powerplant_energy_learning")
        with self.assertRaisesRegex(InputValidationError, "supply-side market extension"):
            _offers_for_opening(
                opening=MarketOpening(START, ((START + HOUR, START + 2 * HOUR),)),
                plants=(plant,), runtime_states={plant.name: PlantRuntimeState.initially_off()},
                availability_profiles={}, marginal_costs={plant.name: 10.0},
                fuel_price_profiles=None, price_forecasts={},
                exchange_unit=None, exchange_schedules={},
            )

    def test_extension_prediction_window_participates_in_opening_preflight(self):
        delivery = START + HOUR
        for window in (
            {"required_data_offsets": (0, 1)},
            {"price_forecast_offsets": (0, 1)},
        ):
            with self.subTest(window=window):
                issues = _opening_coverage_issues(
                    opening=MarketOpening(START, ((delivery, delivery + HOUR),)),
                    settings=SETTINGS, plants=(PLANT,),
                    demand_units=(DemandUnit("load", "operator"),),
                    demand_profiles={"load": {delivery: 1.0}}, availability_profiles={},
                    exchange_unit=None, exchange_profiles={}, requires_price_forecast=False,
                    **window,
                )
                self.assertEqual(issues,
                                 ["报价预测数据缺少 demand_df.csv/load: 2024-01-01 02:00"])


class LearningActionContracts(unittest.TestCase):
    def session(self, action, bounded, transitions=None):
        return LearningEpisodeSession(
            plant=replace(PLANT, min_power_mw=0.25),
            action_provider=lambda state: action,
            enforce_action_bounds=bounded,
            transition_consumer=None if transitions is None else transitions.append,
        )

    def quote(self, session, power=1.0):
        return session.offers_for_product(
            delivery_start=START + HOUR, delivery_end=START + 2 * HOUR,
            available_power_mw=power, marginal_cost_eur_per_mwh=10.0,
            scheduled_results={},
        )

    def test_experience_keeps_action_order_while_offer_prices_are_sorted(self):
        cases = (
            (True, (1.7, -2.5), (1.0, -1.0), (-100.0, 100.0)),
            (False, (1.7, -2.5), (1.7, -2.5), (-250.0, 170.0)),
            (True, (0.8, -0.3), (0.8, -0.3), (-30.0, 80.0)),
            (False, (0.8, -0.3), (0.8, -0.3), (-30.0, 80.0)),
        )
        for bounded, action, executed, prices in cases:
            with self.subTest(bounded=bounded, action=action):
                callbacks = []
                session = self.session(action, bounded, callbacks)
                offers = self.quote(session)
                self.assertEqual(tuple(item.bid_price_eur_per_mwh for item in offers), prices)
                self.assertEqual(tuple(item.offered_power_mw for item in offers), (0.25, 0.75))
                bid = DemandBid("load", "operator", START + HOUR, START + 2 * HOUR,
                                1.0, 400.0)
                session.record_result(clear_pay_as_clear([bid], list(offers)))
                steps, transitions = session.finalize()
                self.assertEqual(steps[0].action, executed)
                self.assertEqual(transitions[0].action, executed)
                self.assertTrue(transitions[0].done)
                self.assertEqual(callbacks, list(transitions))

    def test_invalid_actions_fail_in_both_modes_even_without_available_power(self):
        invalid = (
            ((0.0,), r"Learning action must have shape \(2,\)."),
            ((0.0, 1.0, 2.0), r"Learning action must have shape \(2,\)."),
            (((0.0, 1.0),), r"Learning action must have shape \(2,\)."),
            ((float("nan"), 0.0), "Learning action must contain only finite values."),
            ((0.0, float("inf")), "Learning action must contain only finite values."),
        )
        for bounded in (True, False):
            for action, message in invalid:
                for power in (0.0, 1.0):
                    with self.subTest(bounded=bounded, action=action, power=power):
                        with self.assertRaisesRegex(ValueError, message):
                            self.quote(self.session(action, bounded), power)

    def test_valid_action_produces_no_offer_below_minimum_power(self):
        for bounded in (True, False):
            with self.subTest(bounded=bounded):
                self.assertEqual(self.quote(self.session((1.7, -2.5), bounded), 0.0), ())


class DeliveryIntervalContracts(unittest.TestCase):
    def test_demand_bid_rejects_non_positive_delivery_duration(self):
        for end in (START, START - HOUR):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, "交付时长"):
                DemandBid("load", "operator", START, end, 1.0, 100.0)

    def test_clearing_result_rejects_non_positive_delivery_duration(self):
        for end in (START, START - HOUR):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, "交付时长"):
                MarketClearingResult(START, end, 0.0, 0.0, 0.0, 0.0, 0.0, (), (), None)


class ReportingContracts(unittest.TestCase):
    def test_optional_tables_left_by_an_earlier_run_are_removed(self):
        bid, offer = orders(START)
        result = SimulationResult(SETTINGS, (clear_pay_as_clear([bid], [offer]),))
        stale = ("exchange_results.csv", "storage_results.csv", "offer_results.csv",
                 "household_results.csv", "household_flexibility_results.csv",
                 "industry_results.csv", "industry_flexibility_results.csv",
                 "industry_optimization_windows.csv", "learning_results.csv")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for name in stale:
                (output / name).write_text("stale\n", encoding="utf-8")
            write_results(output, result)
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["demand_results.csv", "market_results.csv",
                              "operator_results.csv", "unit_results.csv"])

    def test_unaccepted_storage_charge_is_not_unserved_demand(self):
        bid, offer = orders(START)
        elastic = DemandBid("elastic", "operator", START, START + HOUR, 0.5, 5.0,
                            demand_type="elastic_load")
        charge = DemandBid("storage", "operator", START, START + HOUR, 2.0, 5.0,
                           bid_type="storage_charge")
        market = clear_pay_as_clear([bid, elastic, charge], [offer])
        self.assertAlmostEqual(market.requested_storage_charge_mwh, 2.0)
        self.assertAlmostEqual(market.unaccepted_storage_charge_mwh, 2.0)
        self.assertAlmostEqual(market.unserved_demand_mwh, 0.5)
        with TemporaryDirectory() as directory:
            write_results(Path(directory), SimulationResult(SETTINGS, (market,)))
            with (Path(directory) / "market_results.csv").open(encoding="utf-8") as file:
                (row,) = csv.DictReader(file)
        self.assertAlmostEqual(float(row["unserved_total_demand_mwh"]), 0.5)
        self.assertAlmostEqual(float(row["unaccepted_storage_charge_mwh"]), 2.0)
        self.assertAlmostEqual(
            float(row["requested_demand_energy_mwh"]),
            float(row["cleared_energy_mwh"]) + 0.5 + 2.0,
        )

    def test_operator_results_include_storage_cash_flow(self):
        bid, offer = orders(START)
        market = clear_pay_as_clear([bid], [offer])
        storage = StorageDispatchResult(
            START, START, START + HOUR, "storage", "operator", "battery",
            50.0, 49.0, 100.0, 2.0, 2.0, 40.0, 1.0, 1.0, 60.0, 50.0,
            20.0, 30.0, 0.56, 0.28,
        )
        result = SimulationResult(SETTINGS, (market,), (storage,))
        generation = market.offers[0]
        with TemporaryDirectory() as directory:
            write_results(Path(directory), result)
            with (Path(directory) / "operator_results.csv").open(encoding="utf-8") as file:
                (row,) = csv.DictReader(file)
        self.assertAlmostEqual(float(row["accepted_energy_mwh"]),
                               generation.accepted_energy_mwh + 1.0)
        self.assertAlmostEqual(float(row["accepted_charge_mwh"]), 2.0)
        self.assertAlmostEqual(float(row["revenue_eur"]), generation.revenue_eur + 30.0)
        self.assertAlmostEqual(float(row["purchase_cost_eur"]), 20.0)
        self.assertAlmostEqual(float(row["variable_cost_eur"]),
                               generation.variable_cost_eur + 0.84)
        self.assertAlmostEqual(float(row["total_cost_eur"]),
                               generation.total_cost_eur + 20.84)
        self.assertAlmostEqual(float(row["profit_eur"]),
                               generation.profit_eur + storage.net_cash_flow_eur)


class StorageChargeCharts(unittest.TestCase):
    def setUp(self):
        bid, offer = orders(START)
        household = DemandBid("household", "operator", START, START + HOUR, 0.5, 100.0,
                              demand_type="household_load")
        charge = DemandBid("storage", "operator", START, START + HOUR, 2.0, 5.0,
                           bid_type="storage_charge")
        storage = StorageDispatchResult(
            START, START, START + HOUR, "storage", "operator", "battery",
            50.0, 50.0, 100.0, 2.0, 0.0, 4.3, 0.0, 0.0, None, 10.0,
            0.0, 0.0, 0.0, 0.0,
        )
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        write_results(Path(directory.name), SimulationResult(
            SETTINGS, (clear_pay_as_clear([bid, household, charge], [offer]),), (storage,)))
        self.tables = charts.load_run(Path(directory.name))

    def test_market_overview_only_shows_household_requested_demand(self):
        figure = charts.market_overview(self.tables)
        self.assertNotIn("总申报需求（不含储能充电）", [trace.name for trace in figure.data])
        self.assertNotIn("已成交的储能充电需求", [trace.name for trace in figure.data])
        self.assertNotIn("未成交的储能充电需求",
                         [item.text for item in figure.layout.annotations])
        (requested,) = [trace for trace in figure.data
                        if trace.name == "居民申报需求"]
        self.assertEqual(list(requested.y), [0.5])

    def test_market_overview_without_households_keeps_market_demand(self):
        bid, offer = orders(START, quantity=2.0)
        elastic = DemandBid("elastic", "operator", START, START + HOUR, 0.5, 50.0,
                            demand_type="elastic_load")
        charge = DemandBid("storage", "operator", START, START + HOUR, 2.0, 5.0,
                           bid_type="storage_charge")
        with TemporaryDirectory() as directory:
            write_results(Path(directory), SimulationResult(
                SETTINGS, (clear_pay_as_clear([bid, elastic, charge], [offer]),)))
            figure = charts.market_overview(charts.load_run(Path(directory)))
        traces = {trace.name: trace for trace in figure.data}
        self.assertNotIn("居民申报需求", traces)
        self.assertNotIn("已成交的居民用电需求", traces)
        self.assertEqual(list(traces["总申报需求（不含储能充电）"].y), [1.5])
        self.assertEqual(list(traces["已满足的刚性负荷"].y), [1.0])
        self.assertEqual(list(traces["已成交的弹性需求"].y), [0.5])
        self.assertIn("未满足的刚性负荷", [item.text for item in figure.layout.annotations])

    def test_single_storage_power_is_colored_by_direction(self):
        figure = charts.storage_dispatch(self.tables)
        power = [trace for trace in figure.data if trace.yaxis == "y2"]
        self.assertEqual([trace.name for trace in power], ["放电", "充电"])
        self.assertNotEqual(power[0].line.color, power[1].line.color)

    def test_market_summary_still_balances_with_storage_charge(self):
        demand, supply = charts.market_summary(self.tables).data
        self.assertEqual(list(demand.y), list(supply.y))


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _write_household_scenario(directory, battery_power_mw, *, min_soc=0, initial_soc=0,
                              loss_rate=""):
    """一天的 15 分钟居民场景：12:00 前有零价供给，之后只有 50 EUR/MWh 的机组。"""
    directory = Path(directory)
    (directory / "config.yaml").write_text(
        "base:\n"
        "  start_date: 2024-01-01 00:00\n"
        "  end_date: 2024-01-02 00:00\n"
        "  time_step: 15min\n"
        "  markets_config:\n"
        "    EOM:\n"
        "      operator: EOM_operator\n"
        "      product_type: energy\n"
        "      products:\n"
        "        - duration: 15min\n"
        "          count: 96\n"
        "          first_delivery: 15min\n"
        "      opening_frequency: 24h\n"
        "      opening_duration: 15min\n"
        "      volume_unit: MWh\n"
        "      maximum_bid_volume: 100000\n"
        "      maximum_bid_price: 3000\n"
        "      minimum_bid_price: -500\n"
        "      price_unit: EUR/MWh\n"
        "      market_mechanism: pay_as_bid\n",
        encoding="utf-8",
    )
    times = [datetime(2024, 1, 1) + index * timedelta(minutes=15) for index in range(97)]
    cheap = [1.0 if time.hour < 12 else 0.0 for time in times]
    _write_csv(directory / "powerplant_units.csv",
               ["name", "technology", "bidding_EOM", "fuel_type", "emission_factor",
                "max_power", "min_power", "efficiency", "additional_cost", "unit_operator"],
               [["cheap", "solar", "powerplant_energy_naive", "renewable", 0, 10, 0, 1, 0, "pv"],
                ["expensive", "lignite", "powerplant_energy_naive", "lignite", 0, 100, 0, 1, 0,
                 "utility"]])
    _write_csv(directory / "availability_df.csv", ["datetime", "cheap"],
               [[time.isoformat(sep=" "), value] for time, value in zip(times, cheap)])
    _write_csv(directory / "fuel_prices_df.csv", ["datetime", "lignite", "co2"],
               [[time.isoformat(sep=" "), 50, 0] for time in times])
    _write_csv(directory / "demand_units.csv",
               ["name", "technology", "bidding_EOM", "max_power", "min_power", "node",
                "unit_operator", "price"],
               [["demand_EOM", "inflex_demand", "demand_energy_naive", 1000, 0, "north",
                 "eom", 100]])
    _write_csv(directory / "demand_df.csv", ["datetime", "demand_EOM"],
               [[time.isoformat(sep=" "), 5] for time in times])
    _write_csv(directory / "forecasts_df.csv", ["datetime", "price_EOM", "A360_heat_demand"],
               [[time.isoformat(sep=" "), 0 if value else 50, 2]
                for time, value in zip(times, cheap)])
    _write_csv(directory / "residential_dsm_units.csv",
               ["name", "unit_type", "technology", "node", "bidding_EOM", "unit_operator",
                "objective", "flexibility_measure", "cost_tolerance", "max_power",
                "min_power", "ramp_up", "ramp_down", "min_operating_time", "cop",
                "capacity", "min_soc", "initial_soc", "storage_loss_rate",
                "efficiency_charge", "efficiency_discharge", "max_charging_rate",
                "max_discharging_rate", "is_prosumer"],
               [["A360", "building", "heat_pump", "north", "household_energy_optimization",
                 "dsm", "min_variable_cost", "cost_based_load_shift", 10, 5, 0, 5, 5, 0, 2,
                 "", "", "", "", "", "", "", "", "No"],
                ["A360", "building", "generic_storage", "north", "", "", "", "", "", "", "",
                 battery_power_mw, battery_power_mw, "", "", 1, min_soc, initial_soc,
                 loss_rate, 1, 1, battery_power_mw, battery_power_mw, ""]])
    return directory


class HouseholdSavingContracts(unittest.TestCase):
    def test_positive_solar_quote_is_used_in_optimized_and_baseline_payments(self):
        from electricity_market_sim.simulation import simulate

        with TemporaryDirectory() as directory:
            path = _write_household_scenario(directory, 0.5)
            plants_path = path / "powerplant_units.csv"
            with plants_path.open(encoding="utf-8", newline="") as stream:
                plants = list(csv.DictReader(stream))
            for plant in plants:
                plant["bid_price"] = 20 if plant["name"] == "cheap" else ""
            _write_csv(plants_path, list(plants[0]), [list(row.values()) for row in plants])
            forecast_path = path / "forecasts_df.csv"
            with forecast_path.open(encoding="utf-8", newline="") as stream:
                forecasts = list(csv.DictReader(stream))
            for row in forecasts:
                if float(row["price_EOM"]) == 0:
                    row["price_EOM"] = 20
            _write_csv(forecast_path, list(forecasts[0]),
                       [list(row.values()) for row in forecasts])
            result = simulate(path)
        summary = self.summary(result)
        self.assertTrue(summary.comparable)
        # 首个产品从 00:15 开始：低价时段 11.75 h，高价时段 12 h。
        self.assertAlmostEqual(summary.baseline_payment_eur, 835)
        self.assertAlmostEqual(summary.payment_eur, 805)
        self.assertAlmostEqual(summary.actual_saving_eur, 30)

    def simulate(self, battery_power_mw, **battery):
        from electricity_market_sim.simulation import simulate

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return simulate(_write_household_scenario(directory.name, battery_power_mw, **battery))

    def summary(self, result):
        from electricity_market_sim.models import summarize_household_savings

        (summary,) = summarize_household_savings(
            result.household_results, result.household_cost_results)
        return summary

    def test_baseline_without_battery_ignores_self_discharge(self):
        result = self.simulate(0.5, min_soc=0.5, initial_soc=0.5, loss_rate=0.01)
        self.assertTrue(all(abs(cost.baseline_soc_after - 0.5) < 1e-9
                            for cost in result.household_cost_results))
        summary = self.summary(result)
        self.assertTrue(summary.comparable)
        self.assertGreaterEqual(summary.final_soc_difference, -1e-9)

    def test_battery_saves_by_shifting_purchases_to_cheap_periods(self):
        result = self.simulate(0.5)
        costs = result.household_cost_results
        self.assertEqual(len(costs), len(result.household_results))
        payment = sum(cost.payment_eur for cost in costs)
        baseline = sum(cost.baseline_payment_eur for cost in costs)
        saving = sum(cost.actual_saving_eur for cost in costs)
        self.assertAlmostEqual(saving, baseline - payment)
        # 无损电池在零价时段充满 1 MWh，在 50 EUR/MWh 时段全部放出。
        self.assertAlmostEqual(saving, 50.0, places=6)
        summary = self.summary(result)
        self.assertTrue(summary.comparable)
        self.assertAlmostEqual(summary.actual_saving_eur, saving)
        self.assertIn("对照可比", summary.describe())
        self.assertAlmostEqual(sum(cost.forecast_saving_eur for cost in costs), 50.0, places=6)
        demand_payment = sum(
            cleared.payment_eur
            for market in result.market_results
            for cleared in market.demand_bids
            if cleared.bid.unit_name == "A360"
        )
        self.assertAlmostEqual(payment, demand_payment)
        for cost in costs:
            if cost.accepted_grid_energy_mwh > 0:
                self.assertAlmostEqual(
                    cost.actual_price_eur_per_mwh,
                    0.0 if cost.delivery_start.hour < 12 else 50.0,
                )

    def test_without_battery_the_optimized_run_equals_the_baseline(self):
        result = self.simulate(0.0)
        for cost in result.household_cost_results:
            self.assertAlmostEqual(cost.payment_eur, cost.baseline_payment_eur)
            self.assertAlmostEqual(cost.accepted_grid_energy_mwh,
                                   cost.baseline_grid_energy_mwh)
            self.assertAlmostEqual(cost.actual_saving_eur, 0.0)
            self.assertAlmostEqual(cost.forecast_saving_eur, 0.0)

    def test_household_results_csv_and_plots_include_savings(self):
        from electricity_market_sim.plotting import generate_plots

        result = self.simulate(0.5)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_results(output, result)
            with (output / "household_results.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                list(rows[0])[-10:],
                ["accepted_grid_energy_mwh", "actual_price_eur_per_mwh", "payment_eur",
                 "baseline_grid_energy_mwh", "baseline_payment_eur", "actual_saving_eur",
                 "forecast_saving_eur", "baseline_unmet_electricity_mwh",
                 "baseline_unmet_heat_mwh_th", "baseline_soc_after"],
            )
            self.assertAlmostEqual(sum(float(row["actual_saving_eur"]) for row in rows),
                                   50.0, places=4)
            names = {path.name for path in generate_plots(result, output / "plots")}
        self.assertTrue({"household_dispatch.png", "household_dispatch_zoom.png",
                         "household_savings.png"} <= names)

    def test_dashboard_shows_household_dispatch_and_savings(self):
        result = self.simulate(0.5)
        with TemporaryDirectory() as directory:
            write_results(Path(directory), result)
            tables = charts.load_run(Path(directory))
        titles = [figure.layout.title.text for figure in charts.build_charts(tables)]
        self.assertIn("居民调度", titles)
        (savings,) = [title for title in titles if title.startswith("居民购电成本与节省")]
        self.assertNotIn("对照不可比", savings)
        figure = charts.household_savings(tables)
        self.assertIn("实际节省 50.00 欧元", figure.layout.annotations[-1].text)

        tables["household"].loc[0, "unmet_heat_mwh_th"] = 0.25
        figure = charts.household_savings(tables)
        self.assertIn("对照不可比", figure.layout.title.text)
        self.assertIn("缺热量逐时段不一致 0.250000", figure.layout.annotations[-1].text)

    def test_offsetting_unmet_heat_in_different_periods_is_not_comparable(self):
        result = self.simulate(0.5)
        dispatch = list(result.household_results)
        costs = list(result.household_cost_results)
        # 00:15 只有基准缺热，00:45 只有优化方案缺热；合计净差为 0，期末 SOC 不变。
        costs[0] = replace(costs[0], baseline_unmet_heat_mwh_th=0.25)
        dispatch[2] = replace(dispatch[2], unmet_heat_mwh_th=0.25)
        result = replace(result, household_results=tuple(dispatch),
                         household_cost_results=tuple(costs))
        summary = self.summary(result)
        self.assertAlmostEqual(summary.unmet_heat_difference_mwh_th, 0.0)
        self.assertAlmostEqual(summary.unmet_heat_mismatch_mwh_th, 0.5)
        self.assertFalse(summary.comparable)
        with TemporaryDirectory() as directory:
            write_results(Path(directory), result)
            figure = charts.household_savings(charts.load_run(Path(directory)))
        self.assertIn("对照不可比", figure.layout.title.text)
        self.assertIn("缺热量逐时段不一致 0.500000", figure.layout.annotations[-1].text)

    def test_dashboard_tolerates_results_without_comparison_columns(self):
        result = self.simulate(0.5)
        with TemporaryDirectory() as directory:
            write_results(Path(directory), result)
            tables = charts.load_run(Path(directory))
        cases = {
            "上一版省钱结果": ["baseline_unmet_electricity_mwh", "baseline_unmet_heat_mwh_th",
                         "baseline_soc_after"],
            "无成本列的旧结果": list(tables["household"].columns[16:]),
        }
        for name, columns in cases.items():
            with self.subTest(name):
                old = {**tables, "household": tables["household"].drop(columns=columns)}
                figures = {figure.layout.title.text: figure for figure in charts.build_charts(old)}
                (savings,) = [figure for title, figure in figures.items()
                              if title.startswith("居民购电成本与节省")]
                self.assertIn("请重新运行仿真生成", savings.layout.annotations[0].text)
                self.assertIn("居民调度", figures)


class HouseholdComparabilityContracts(unittest.TestCase):
    """支付更少但服务不同或消耗了电池存量时，不能判为同等服务下的节省。"""

    QUARTER = timedelta(minutes=15)

    def pair(self, *, payment, unmet_heat=0.0, soc=(1.0, 1.0), baseline_unmet_heat=0.0):
        from electricity_market_sim.models import HouseholdCostResult, HouseholdDispatchResult

        start = START
        dispatch = HouseholdDispatchResult(
            start, start + self.QUARTER, "A360", 50.0, 3.0, 0.0, 1.5, 1.5, 0.0, 0.0,
            soc[0], soc[1], 0.0, unmet_heat,
        )
        cost = HouseholdCostResult(
            start, start + self.QUARTER, "A360", payment / 50.0, payment,
            0.375, 18.75, 0.0, 0.0, baseline_unmet_heat, 1.0,
        )
        return (dispatch,), (cost,)

    def summary(self, dispatch, costs):
        from electricity_market_sim.models import summarize_household_savings

        (summary,) = summarize_household_savings(dispatch, costs)
        return summary

    def test_less_heat_supplied_is_not_a_comparable_saving(self):
        summary = self.summary(*self.pair(payment=12.5, unmet_heat=0.25))
        self.assertAlmostEqual(summary.actual_saving_eur, 6.25)
        self.assertAlmostEqual(summary.unmet_heat_difference_mwh_th, 0.25)
        self.assertFalse(summary.comparable)
        self.assertIn("对照不可比", summary.describe())

    def test_consuming_initial_battery_energy_is_not_a_comparable_saving(self):
        summary = self.summary(*self.pair(payment=12.5, soc=(1.0, 0.875)))
        self.assertAlmostEqual(summary.actual_saving_eur, 6.25)
        self.assertAlmostEqual(summary.final_soc_difference, -0.125)
        self.assertFalse(summary.comparable)

    def test_unmet_heat_in_different_periods_does_not_offset(self):
        later = START + self.QUARTER
        first_dispatch, first_cost = self.pair(payment=18.75, baseline_unmet_heat=0.25)
        second_dispatch, second_cost = self.pair(payment=12.5, unmet_heat=0.25)
        second_dispatch = (replace(second_dispatch[0], delivery_start=later,
                                   delivery_end=later + self.QUARTER),)
        second_cost = (replace(second_cost[0], delivery_start=later,
                               delivery_end=later + self.QUARTER),)
        summary = self.summary(first_dispatch + second_dispatch, first_cost + second_cost)
        self.assertAlmostEqual(summary.unmet_heat_difference_mwh_th, 0.0)
        self.assertAlmostEqual(summary.unmet_heat_mismatch_mwh_th, 0.5)
        self.assertFalse(summary.comparable)

    def test_equal_service_and_soc_is_comparable(self):
        summary = self.summary(*self.pair(payment=12.5, unmet_heat=0.25,
                                          baseline_unmet_heat=0.25))
        self.assertTrue(summary.comparable)

    def test_savings_plot_marks_non_comparable_runs(self):
        from electricity_market_sim.plotting.common import _load_matplotlib
        from electricity_market_sim.plotting.household import _plot_household_savings

        pyplot, dates = _load_matplotlib()
        with TemporaryDirectory() as directory:
            path = _plot_household_savings(
                pyplot, dates, *self.pair(payment=12.5, unmet_heat=0.25),
                Path(directory) / "household_savings.png",
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
