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
    START, START + 4 * HOUR, HOUR, HOUR, HOUR, HOUR, 1, HOUR,
    1000.0, -100.0, "pay_as_clear", industrial_dsm_units_file="industry.csv",
)
PLANT = PowerPlant(
    "plant", "operator", "gas", "powerplant_energy_heuristic_flexable",
    "gas", 0.2, 10.0, 0.0, 1.0, 0.0, start_cost_eur=17.0,
)
UNIT = IndustrialUnit(
    name="steel", operator="operator", bidding_strategy="industry_energy_optimization",
    objective="min_variable_cost", flexibility_measure="cost_based_load_shift",
    cost_tolerance_percent=20.0, demand_t=2.0, load_profile_deviation=0.1,
    horizon_mode="rolling_horizon", look_ahead=3 * HOUR,
    commit_horizon=HOUR, rolling_step=HOUR, forecast_price_column="steel_price",
    electrolyser=IndustrialDevice("electrolyser", "", 2.0, 0.0, 2.0, 2.0, efficiency=1.0),
    dri_plant=IndustrialDevice(
        "dri_plant", "hydrogen", 1.0, 0.0, 1.0, 1.0,
        specific_electricity_consumption=1.0, specific_hydrogen_consumption=1.0,
        specific_iron_ore_consumption=1.0,
    ),
    eaf=IndustrialDevice(
        "eaf", "", 1.0, 0.0, 1.0, 1.0,
        specific_dri_demand=1.0, specific_electricity_consumption=1.0,
    ),
)


class IndustryMarketContracts(unittest.TestCase):
    def run_market(self, *, delay=HOUR, availability=None, events=None):
        settings = replace(SETTINGS, first_delivery=delay, end=START + delay + 3 * HOUR)
        openings = tuple(
            MarketOpening(
                START + index * HOUR,
                ((START + delay + index * HOUR, START + delay + (index + 1) * HOUR),),
            )
            for index in range(3)
        )
        times = [START + index * HOUR for index in range(24)]
        profiles = {
            "steel_price": {time: 50.0 for time in times},
            "steel_normalized_load_profile": {time: 1.0 for time in times},
            "electricity_price": {time: 50.0 for time in times},
        }
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
            scenario_module + "load_industrial_units": (UNIT,),
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
        self.assertAlmostEqual(sum(item.planned_steel_output_t for item in result.industry_results), 2.0)
        self.assertAlmostEqual(sum(item.actual_steel_output_t for item in result.industry_results), 2.0)

    def test_partial_first_delivery_is_recovered_in_later_plans(self):
        result = self.run_market(availability={START + HOUR: 0.05})
        first = result.industry_results[0]
        self.assertLess(first.actual_steel_output_t, first.planned_steel_output_t)
        self.assertAlmostEqual(sum(item.actual_steel_output_t for item in result.industry_results), 2.0)

    def test_final_delivery_shortfall_is_reported(self):
        with self.assertRaisesRegex(InputValidationError, "仿真结束时.*需求尚未完成"):
            self.run_market(availability={START + 3 * HOUR: 0.0})


if __name__ == "__main__":
    unittest.main()
