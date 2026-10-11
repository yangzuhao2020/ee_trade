"""Behavior contracts for shared events and extension-controlled plant offers."""

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

from electricity_market_sim.bidding import PlantRuntimeState
from electricity_market_sim.clearing import clear_pay_as_clear
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.learning import LearningEpisodeSession
from electricity_market_sim.market_models import DemandBid, SupplyOffer
from electricity_market_sim.models import DemandUnit, MarketOpening, MarketSettings, PowerPlant
from electricity_market_sim.simulators.base_market.forecasts import _opening_coverage_issues
from electricity_market_sim.simulators.base_market.plants import (
    _apply_startup_costs, _offers_for_opening, _record_runtime_states,
)
from electricity_market_sim.simulators.eom import simulate_eom_market


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
    def __init__(self, before_opening, events):
        self.settle_before_opening = before_opening
        self.events = events
        self.startup_costs = []

    def record_delivery_start(self, result):
        self.startup_costs.append(result.offers[0].startup_cost_eur)

    def record_delivery(self, result):
        self.events.append(("extension_end", result.delivery_end))

    def finalize(self, result):
        return result


class TraceMarket:
    def __init__(self, before_opening=None):
        self.settings = SETTINGS
        self.events = []
        self.extension = (None if before_opening is None else
                          TraceExtension(before_opening, self.events))
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
                          if time == START + 2 * HOUR and kind != "extension_end"]
                self.assertEqual(at_two, ["end", "open", "start"])
                self.assertEqual(market.observed_costs, [0.0, 17.0])
                self.assertEqual(len(result.market_results), 3)

    def test_learning_settles_starts_before_requesting_the_next_action(self):
        market = TraceMarket(True)
        simulate_eom_market(market)
        at_two = [kind for kind, time in market.events
                  if time == START + 2 * HOUR and kind != "extension_end"]
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
        issues = _opening_coverage_issues(
            opening=MarketOpening(START, ((delivery, delivery + HOUR),)),
            settings=SETTINGS, plants=(PLANT,),
            demand_units=(DemandUnit("load", "operator"),),
            demand_profiles={"load": {delivery: 1.0}}, availability_profiles={},
            exchange_unit=None, exchange_profiles={}, requires_price_forecast=False,
            extra_forecast_offsets=(0, 1),
        )
        self.assertEqual(issues,
                         ["报价预测数据缺少 demand_df.csv/load: 2024-01-01 02:00"])


class LearningActionContracts(unittest.TestCase):
    def session(self, action, bounded, transitions=None):
        forecasts = {START + i * HOUR: 1.0 for i in range(16)}
        return LearningEpisodeSession(
            plant=replace(PLANT, min_power_mw=0.25),
            action_provider=lambda state: action,
            residual_load_forecasts_mw=forecasts,
            price_forecasts_eur_per_mwh=forecasts,
            load_base_mw=1.0,
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


if __name__ == "__main__":
    unittest.main()
