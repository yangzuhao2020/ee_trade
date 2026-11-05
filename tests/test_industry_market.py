"""第四版独立运行流程的事件顺序、待交付产量及部分成交测试。"""

from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.models import (
    DemandUnit, IndustrialDevice, IndustrialUnit, MarketOpening,
    MarketSettings, PowerPlant,
)
from electricity_market_sim.simulators.base_market import BaseMarket
from electricity_market_sim.simulators.industry_market import simulate_v4_market


START = datetime(2024, 1, 1)
HOUR = timedelta(hours=1)
SETTINGS = MarketSettings(
    START, START + 4 * HOUR, HOUR, HOUR, HOUR, 1, HOUR,
    1000.0, -100.0, "pay_as_clear", industrial_dsm_units_file="industry.csv",
)
PLANT = PowerPlant(
    "plant", "operator", "gas", "powerplant_energy_heuristic_flexable",
    "gas", 0.2, 10.0, 0.0, 1.0, 0.0, start_cost_eur=17.0,
)
UNIT = IndustrialUnit(
    name="steel", operator="operator",
    cost_tolerance_percent=20.0, demand_t=2.0, load_profile_deviation=0.1,
    look_ahead=3 * HOUR,
    commit_horizon=HOUR, forecast_price_column="steel_price",
    electrolyser=IndustrialDevice("electrolyser", 2.0, 0.0, 2.0, 2.0, efficiency=1.0),
    dri_plant=IndustrialDevice(
        "dri_plant", 1.0, 0.0, 1.0, 1.0,
        specific_electricity_consumption=1.0, specific_hydrogen_consumption=1.0,
        specific_iron_ore_consumption=1.0,
    ),
    eaf=IndustrialDevice(
        "eaf", 1.0, 0.0, 1.0, 1.0,
        specific_dri_demand=1.0, specific_electricity_consumption=1.0,
    ),
)


class IndustryMarketContracts(unittest.TestCase):
    def continuous_unit(self, *, demand_t=1.5, minimum_power=0.2):
        devices = {
            name: replace(getattr(UNIT, name), min_power_mw=minimum_power,
                          ramp_up_mw=0.25, ramp_down_mw=0.25)
            for name in ("electrolyser", "dri_plant", "eaf")
        }
        return replace(UNIT, demand_t=demand_t, commit_horizon=3 * HOUR, **devices)

    def run_market(self, *, delay=HOUR, availability=None, events=None, unit=UNIT,
                   product_count=3, load_profile=None):
        settings = replace(SETTINGS, first_delivery=delay, end=START + delay + product_count * HOUR)
        openings = tuple(
            MarketOpening(
                START + index * HOUR,
                ((START + delay + index * HOUR, START + delay + (index + 1) * HOUR),),
            )
            for index in range(product_count)
        )
        times = [START + index * HOUR for index in range(24)]
        profiles = {
            "steel_price": {time: 50.0 for time in times},
            "steel_normalized_load_profile": {time: 1.0 for time in times},
            "electricity_price": {time: 50.0 for time in times},
        }
        profiles["steel_normalized_load_profile"].update(load_profile or {})
        prices = {
            time: {"gas": 10.0, "co2": 0.0, "iron ore": 1.0, "lime": 1.0}
            for time in times
        }
        base_module = "electricity_market_sim.simulators.base_market.market."
        scenario_module = "electricity_market_sim.simulators.industry_market."
        loaders = {
            base_module + "load_powerplants": (PLANT,),
            base_module + "load_storage_units": (),
            base_module + "load_demand_units": (DemandUnit("load", "operator"),),
            base_module + "load_hourly_demand_profiles": {"load": {t: 0.25 for t in times}},
            base_module + "load_hourly_availability_profiles": (
                {} if availability is None else {"plant": {t: availability.get(t, 1.0) for t in times}}
            ),
            scenario_module + "load_industrial_units": (unit,),
            scenario_module + "load_aligned_fuel_price_profiles": prices,
        }
        with ExitStack() as stack:
            for name, value in loaders.items():
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch(
                scenario_module + "load_aligned_time_series_profiles",
                side_effect=lambda path, columns, duration: {name: profiles[name] for name in columns},
            ))
            if events is not None:
                clear_opening = BaseMarket.clear_opening
                start_delivery = BaseMarket.start_delivery
                end_delivery = BaseMarket.end_delivery

                def clear(market, opening, scheduled, **kwargs):
                    events.append(("open", opening.opening_time))
                    return clear_opening(market, opening, scheduled, **kwargs)

                def start(market, result):
                    events.append(("start", result.delivery_start))
                    return start_delivery(market, result)

                def end(market, result):
                    events.append(("end", result.delivery_end))
                    return end_delivery(market, result)

                stack.enter_context(patch.object(BaseMarket, "clear_opening", clear))
                stack.enter_context(patch.object(BaseMarket, "start_delivery", start))
                stack.enter_context(patch.object(BaseMarket, "end_delivery", end))
            return simulate_v4_market(Path("unused"), settings, openings)

    def test_delivery_ends_before_opening_and_startup_costs_are_kept(self):
        events = []
        result = self.run_market(events=events)
        at_two = [kind for kind, time in events if time == START + 2 * HOUR]
        self.assertEqual(at_two, ["end", "open", "start"])
        self.assertEqual([sum(offer.startup_cost_eur for offer in item.offers)
                          for item in result.market_results], [17.0, 0.0, 0.0])
        self.assertEqual(len(result.industry_results), 3)
        self.assertAlmostEqual(sum(item.actual_steel_output_t for item in result.industry_results), 2.0)

    def test_later_openings_account_for_cleared_but_undelivered_production(self):
        result = self.run_market(delay=2 * HOUR)
        self.assertEqual(len(result.industry_optimization_windows), 3)
        self.assertAlmostEqual(sum(item.plan.planned_steel_output_t for item in result.industry_results), 2.0)
        self.assertAlmostEqual(sum(item.actual_steel_output_t for item in result.industry_results), 2.0)

    def test_small_task_is_spread_across_commits_instead_of_completed_early(self):
        unit = replace(UNIT, demand_t=0.6)
        for delay in (HOUR, 2 * HOUR):
            with self.subTest(first_delivery=delay):
                result = self.run_market(delay=delay, unit=unit)
                first = result.industry_results[0]
                self.assertGreaterEqual(first.plan.planned_steel_output_t, 0.18 - 1e-7)
                self.assertLessEqual(first.plan.planned_steel_output_t, 0.22 + 1e-7)
                self.assertTrue(all(item.actual_steel_output_t > 0
                                    for item in result.industry_results))
                self.assertAlmostEqual(sum(item.actual_steel_output_t
                                           for item in result.industry_results), unit.demand_t)

    def test_partial_first_delivery_is_recovered_in_later_plans(self):
        result = self.run_market(availability={START + HOUR: 0.05})
        first = result.industry_results[0]
        self.assertLess(first.actual_steel_output_t, first.plan.planned_steel_output_t)
        self.assertAlmostEqual(sum(item.actual_steel_output_t for item in result.industry_results), 2.0)

    def test_continuous_load_respects_minimums_and_ramps_inside_a_commit(self):
        unit = self.continuous_unit()
        for delay in (HOUR, 2 * HOUR):
            with self.subTest(first_delivery=delay):
                result = self.run_market(delay=delay, unit=unit)
                previous = 0.6
                for row in result.industry_results:
                    self.assertGreaterEqual(row.actual_grid_power_mw, 0.6 - 1e-7)
                    self.assertLessEqual(abs(row.actual_grid_power_mw - previous), 0.75 + 1e-7)
                    previous = row.actual_grid_power_mw
                self.assertAlmostEqual(sum(row.actual_steel_output_t
                                           for row in result.industry_results), unit.demand_t)

    def test_minimum_load_is_not_dropped_to_fit_an_insufficient_task(self):
        with self.assertRaises(InputValidationError):
            self.run_market(unit=self.continuous_unit(demand_t=0.3))

    def test_continuous_production_starts_running_when_minimum_exceeds_ramp(self):
        unit = self.continuous_unit(minimum_power=0.4)
        unit = replace(unit, **{
            name: replace(getattr(unit, name), ramp_up_mw=0.1, ramp_down_mw=0.1)
            for name in ("electrolyser", "dri_plant", "eaf")
        })
        result = self.run_market(unit=unit)
        previous = 1.2
        for row in result.industry_results:
            self.assertGreaterEqual(row.actual_grid_power_mw, 1.2 - 1e-7)
            self.assertLessEqual(abs(row.actual_grid_power_mw - previous), 0.3 + 1e-7)
            previous = row.actual_grid_power_mw
        self.assertAlmostEqual(sum(row.actual_steel_output_t
                                   for row in result.industry_results), unit.demand_t)

    def test_zero_reference_commit_keeps_minimum_production(self):
        unit = self.continuous_unit(demand_t=2.0)
        profile = {
            START + (index + 1) * HOUR: value
            for index, value in enumerate((10, 10, 10, 0, 0, 0, 10, 10, 10))
        }
        result = self.run_market(unit=unit, product_count=9, load_profile=profile)
        for row in result.industry_results:
            self.assertGreaterEqual(row.actual_grid_power_mw, 0.6 - 1e-7)
        for row in result.industry_results:
            self.assertGreaterEqual(row.plan.minimum_power_mw, 0.6 - 1e-7)
        self.assertAlmostEqual(sum(row.actual_steel_output_t
                                   for row in result.industry_results[3:6]), 0.6)
        self.assertAlmostEqual(sum(row.actual_steel_output_t
                                   for row in result.industry_results), unit.demand_t)

    def test_future_commit_limits_reserve_time_to_ramp_down(self):
        unit = self.continuous_unit(demand_t=2.0, minimum_power=0.01)
        profile = {
            START + (index + 1) * HOUR: value
            for index, value in enumerate((10, 10, 10, 1, 1, 1, 10, 10, 10))
        }
        result = self.run_market(unit=unit, product_count=9, load_profile=profile)
        rows = result.industry_results
        first = sum(row.actual_steel_output_t for row in rows[:3])
        middle = sum(row.actual_steel_output_t for row in rows[3:6])
        self.assertLessEqual(middle, 0.03 + (unit.demand_t - first - 0.06) * 3 / 33 * 1.1 + 1e-7)
        previous = 0.03
        for row in rows:
            self.assertGreaterEqual(row.actual_grid_power_mw, 0.03 - 1e-7)
            self.assertLessEqual(abs(row.actual_grid_power_mw - previous), 0.75 + 1e-7)
            previous = row.actual_grid_power_mw
        self.assertAlmostEqual(sum(row.actual_steel_output_t for row in rows), unit.demand_t)

    def test_partial_fill_below_minimum_production_power_is_reported(self):
        with self.assertRaisesRegex(InputValidationError, "低于最低生产功率"):
            self.run_market(unit=self.continuous_unit(), availability={START + HOUR: 0.05})

    def test_actual_delivery_ramp_violation_is_reported(self):
        with self.assertRaisesRegex(InputValidationError, "实际功率变化超过爬坡限制"):
            self.run_market(unit=self.continuous_unit(minimum_power=0.0),
                            availability={START + 3 * HOUR: 0.0})

    def test_final_delivery_shortfall_is_reported(self):
        with self.assertRaisesRegex(InputValidationError, "仿真结束时.*需求尚未完成"):
            self.run_market(availability={START + 3 * HOUR: 0.0})


if __name__ == "__main__":
    unittest.main()
