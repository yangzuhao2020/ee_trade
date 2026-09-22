import csv
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from electricity_market_sim.clearing import clear_pay_as_bid
from electricity_market_sim.config import load_market_settings
from electricity_market_sim.household import (
    dispatch_household,
    evaluate_household_flexibility,
    optimize_household,
)
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.loader import (
    load_household_units,
    load_time_series_profiles,
)
from electricity_market_sim.models import (
    DemandBid,
    HouseholdPlan,
    HouseholdUnit,
    MarketSettings,
    SimulationResult,
    SupplyOffer,
)
from electricity_market_sim.plotting import generate_plots
from electricity_market_sim.reporting import write_results
from electricity_market_sim.simulation import run_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_INPUT = PROJECT_ROOT / "examples" / "input" / "example_01h"


class PayAsBidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2024, 6, 1, 0, 15)
        self.end = self.start + timedelta(minutes=15)

    def demand(
        self, name: str, energy: float, price: float, demand_type: str
    ) -> DemandBid:
        return DemandBid(
            unit_name=name,
            operator=f"operator-{name}",
            delivery_start=self.start,
            delivery_end=self.end,
            volume_mwh=energy,
            price_eur_per_mwh=price,
            bid_id=f"bid-{name}",
            demand_type=demand_type,
        )

    def offer(self, name: str, energy: float, price: float) -> SupplyOffer:
        return SupplyOffer(
            unit_name=name,
            operator=f"operator-{name}",
            technology="test",
            delivery_start=self.start,
            delivery_end=self.end,
            offered_power_mw=energy / 0.25,
            offered_energy_mwh=energy,
            bid_price_eur_per_mwh=price,
            marginal_cost_eur_per_mwh=price,
            offer_id=f"offer-{name}",
        )

    def test_matches_divisible_offers_at_each_seller_price(self) -> None:
        result = clear_pay_as_bid(
            [
                self.demand("A360", 5, 3000, "household_load"),
                self.demand("demand_EOM", 3, 25, "inelastic_load"),
            ],
            [self.offer("cheap", 4, 20), self.offer("expensive", 4, 30)],
        )

        offers = {item.offer.unit_name: item for item in result.offers}
        demands = {item.bid.unit_name: item for item in result.demand_bids}
        self.assertIsNone(result.clearing_price_eur_per_mwh)
        self.assertEqual(result.pricing_method, "pay_as_bid")
        self.assertAlmostEqual(offers["cheap"].accepted_energy_mwh, 4)
        self.assertAlmostEqual(offers["expensive"].accepted_energy_mwh, 1)
        self.assertAlmostEqual(demands["A360"].accepted_energy_mwh, 5)
        self.assertAlmostEqual(demands["A360"].payment_eur, 110)
        self.assertAlmostEqual(demands["A360"].clearing_price_eur_per_mwh, 22)
        self.assertAlmostEqual(demands["demand_EOM"].accepted_energy_mwh, 0)
        self.assertAlmostEqual(result.unserved_load_mwh, 3)
        self.assertEqual(
            [trade.trade_price_eur_per_mwh for trade in result.trades], [20, 30]
        )
        self.assertAlmostEqual(
            sum(trade.payment_eur for trade in result.trades),
            result.transaction_value_eur,
        )

    def test_order_volumes_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.demand("invalid", -1, 100, "inelastic_load")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.offer("invalid", -1, 20)

    def test_reported_trade_payments_reconcile_after_decimal_rounding(self) -> None:
        result = clear_pay_as_bid(
            [self.demand("buyer", 2, 1, "inelastic_load")],
            [
                self.offer("seller-1", 1, 0.0000006),
                self.offer("seller-2", 1, 0.0000007),
            ],
        )
        settings = MarketSettings(
            start=self.start,
            end=self.end,
            time_step=timedelta(minutes=15),
            opening_frequency=timedelta(days=1),
            opening_duration=timedelta(minutes=15),
            product_duration=timedelta(minutes=15),
            product_count=1,
            first_delivery=timedelta(minutes=15),
            maximum_bid_price=3000,
            minimum_bid_price=-500,
            market_mechanism="pay_as_bid",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_results(output, SimulationResult(settings, (result,)))
            with (output / "trade_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                trade_rows = list(csv.DictReader(file))
            with (output / "demand_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                demand_row = next(csv.DictReader(file))

        reported_trade_payment = sum(
            (Decimal(row["payment_eur"]) for row in trade_rows),
            Decimal(0),
        )
        self.assertEqual(Decimal(demand_row["payment_eur"]), reported_trade_payment)
        for row in trade_rows:
            expected_payment = (
                Decimal(row["trade_energy_mwh"])
                * Decimal(row["trade_price_eur_per_mwh"])
            ).quantize(Decimal("0.000000000001"))
            self.assertEqual(Decimal(row["payment_eur"]), expected_payment)

    def test_pay_as_bid_plots_use_the_trade_price_path(self) -> None:
        result = clear_pay_as_bid(
            [self.demand("A360", 2, 3000, "household_load")],
            [self.offer("seller-1", 1, 20), self.offer("seller-2", 1, 30)],
        )
        settings = MarketSettings(
            start=self.start - timedelta(minutes=15),
            end=self.end,
            time_step=timedelta(minutes=15),
            opening_frequency=timedelta(days=1),
            opening_duration=timedelta(minutes=15),
            product_duration=timedelta(minutes=15),
            product_count=1,
            first_delivery=timedelta(minutes=15),
            maximum_bid_price=3000,
            minimum_bid_price=-500,
            market_mechanism="pay_as_bid",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = generate_plots(
                SimulationResult(settings=settings, market_results=(result,)),
                Path(temporary_directory),
            )

            self.assertEqual(
                {path.name for path in paths},
                {
                    "market_overview.png",
                    "market_summary.png",
                    "dispatch_by_unit.png",
                    "operator_profit.png",
                    "pay_as_bid_first_product.png",
                },
            )
            for path in paths:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class HouseholdTradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.household = HouseholdUnit(
            name="A360",
            operator="household-operator",
            node="north",
            bidding_strategy="household_energy_optimization",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            cost_tolerance_percent=10,
            is_prosumer=False,
            fixed_power_mw=0,
            heat_pump_max_power_mw=5,
            heat_pump_min_power_mw=0,
            heat_pump_ramp_up_mw=5,
            heat_pump_ramp_down_mw=5,
            cop=2,
            battery_capacity_mwh=1,
            battery_min_soc=0,
            battery_max_soc=1,
            battery_initial_soc=0.5,
            battery_efficiency_charge=0.9,
            battery_efficiency_discharge=0.9,
            battery_max_charge_power_mw=1,
            battery_max_discharge_power_mw=1,
            battery_ramp_up_mw=1,
            battery_ramp_down_mw=1,
        )
        self.start = datetime(2024, 6, 1, 0, 15)

    def products(self, count: int) -> tuple[tuple[datetime, datetime], ...]:
        return tuple(
            (
                self.start + index * timedelta(minutes=15),
                self.start + (index + 1) * timedelta(minutes=15),
            )
            for index in range(count)
        )

    def test_constant_price_does_not_create_battery_cycles(self) -> None:
        products = self.products(8)
        prices = {start: 100 for start, _ in products}
        heat = {start: 2 for start, _ in products}

        plans = optimize_household(
            self.household,
            products,
            prices,
            heat,
            self.household.initial_energy_mwh,
        )

        self.assertTrue(
            all(plan.planned_battery_charge_power_mw == 0 for plan in plans)
        )
        self.assertTrue(
            all(plan.planned_battery_discharge_power_mw == 0 for plan in plans)
        )
        self.assertAlmostEqual(plans[-1].planned_soc_after, 0.5)

    def test_low_then_high_price_charges_then_discharges(self) -> None:
        products = self.products(8)
        prices = {
            start: (10 if index < 4 else 100)
            for index, (start, _) in enumerate(products)
        }
        heat = {start: 2 for start, _ in products}

        plans = optimize_household(
            self.household,
            products,
            prices,
            heat,
            self.household.initial_energy_mwh,
        )

        self.assertGreater(
            sum(plan.planned_battery_charge_power_mw for plan in plans[:4]), 0
        )
        self.assertGreater(
            sum(plan.planned_battery_discharge_power_mw for plan in plans[4:]), 0
        )
        self.assertTrue(
            all(
                not (
                    plan.planned_battery_charge_power_mw > 1e-8
                    and plan.planned_battery_discharge_power_mw > 1e-8
                )
                for plan in plans
            )
        )
        self.assertGreaterEqual(plans[-1].planned_soc_after + 1e-8, 0.5)

    def test_cost_tolerant_flexibility_contains_the_baseline(self) -> None:
        products = self.products(8)
        prices = {
            start: (10 if index < 4 else 100)
            for index, (start, _) in enumerate(products)
        }
        heat = {start: 2 for start, _ in products}
        plans = optimize_household(
            self.household,
            products,
            prices,
            heat,
            self.household.initial_energy_mwh,
        )

        flexibility = evaluate_household_flexibility(
            self.household,
            products,
            prices,
            heat,
            self.household.initial_energy_mwh,
            plans,
        )

        self.assertEqual(len(flexibility), len(plans))
        self.assertTrue(
            all(
                bounds.minimum_grid_power_mw
                <= plan.planned_grid_power_mw + 1e-8
                and bounds.maximum_grid_power_mw + 1e-8
                >= plan.planned_grid_power_mw
                for bounds, plan in zip(flexibility, plans, strict=True)
            )
        )
        self.assertTrue(
            any(
                bounds.maximum_grid_power_mw
                - bounds.minimum_grid_power_mw
                > 1e-8
                for bounds in flexibility
            )
        )

    def test_partial_purchase_cancels_charge_and_prioritizes_heat(self) -> None:
        start, end = self.products(1)[0]
        plan = HouseholdPlan(
            delivery_start=start,
            delivery_end=end,
            unit_name=self.household.name,
            forecast_price_eur_per_mwh=10,
            heat_demand_mw_th=2,
            fixed_power_mw=0,
            planned_grid_power_mw=2,
            planned_heat_pump_power_mw=1,
            planned_battery_charge_power_mw=1,
            planned_battery_discharge_power_mw=0,
            planned_soc_after=0.725,
        )

        enough_for_heat = dispatch_household(
            self.household,
            self.start - timedelta(minutes=15),
            (plan,),
            {start: 0.25},
            self.household.initial_energy_mwh,
        )[0]
        insufficient_for_heat = dispatch_household(
            self.household,
            self.start - timedelta(minutes=15),
            (plan,),
            {start: 0.125},
            self.household.initial_energy_mwh,
        )[0]

        self.assertAlmostEqual(enough_for_heat.heat_pump_power_mw, 1)
        self.assertAlmostEqual(enough_for_heat.battery_charge_power_mw, 0)
        self.assertAlmostEqual(enough_for_heat.unmet_heat_mwh_th, 0)
        self.assertAlmostEqual(insufficient_for_heat.heat_pump_power_mw, 0.5)
        self.assertAlmostEqual(insufficient_for_heat.battery_charge_power_mw, 0)
        self.assertAlmostEqual(insufficient_for_heat.unmet_heat_mwh_th, 0.25)
        self.assertAlmostEqual(insufficient_for_heat.soc_after, 0.5)

    def test_next_opening_inherits_device_power_for_ramp_constraints(self) -> None:
        household = replace(
            self.household,
            battery_ramp_up_mw=0.1,
            battery_ramp_down_mw=0.1,
        )
        products = self.products(1)
        start, _ = products[0]
        plans = optimize_household(
            household,
            products,
            {start: 100},
            {start: 2},
            household.initial_energy_mwh,
            initial_heat_pump_power_mw=1,
            initial_battery_charge_power_mw=0.5,
        )

        self.assertGreaterEqual(
            plans[0].planned_battery_charge_power_mw + 1e-8,
            0.4,
        )
        self.assertLessEqual(
            plans[0].planned_battery_charge_power_mw,
            0.6 + 1e-8,
        )
        dispatch = dispatch_household(
            household,
            self.start - timedelta(minutes=15),
            plans,
            {start: plans[0].planned_grid_power_mw * 0.25},
            household.initial_energy_mwh,
            initial_heat_pump_power_mw=1,
            initial_battery_charge_power_mw=0.5,
        )[0]
        self.assertAlmostEqual(
            dispatch.battery_charge_power_mw,
            plans[0].planned_battery_charge_power_mw,
        )

    def test_storage_loss_is_loaded_and_applied_by_plan_and_dispatch(self) -> None:
        with (EXAMPLE_INPUT / "residential_dsm_units.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None
        for row in rows:
            if row["technology"] == "generic_storage":
                row["storage_loss_rate"] = "0.04"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "residential_dsm_units.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            loaded = load_household_units(path)[0]
        self.assertEqual(loaded.battery_loss_rate, 0.04)

        household = replace(self.household, battery_loss_rate=0.04)
        start, end = self.products(1)[0]
        plans = optimize_household(
            household,
            ((start, end),),
            {start: 100},
            {start: 2},
            household.initial_energy_mwh,
            initial_heat_pump_power_mw=1,
        )
        self.assertGreater(plans[0].planned_battery_charge_power_mw, 0)
        self.assertGreaterEqual(plans[0].planned_soc_after + 1e-8, 0.5)

        idle_plan = HouseholdPlan(
            delivery_start=start,
            delivery_end=end,
            unit_name=household.name,
            forecast_price_eur_per_mwh=100,
            heat_demand_mw_th=0,
            fixed_power_mw=0,
            planned_grid_power_mw=0,
            planned_heat_pump_power_mw=0,
            planned_battery_charge_power_mw=0,
            planned_battery_discharge_power_mw=0,
            planned_soc_after=0.5 * (1 - 0.04) ** 0.25,
        )
        dispatched = dispatch_household(
            household,
            self.start - timedelta(minutes=15),
            (idle_plan,),
            {start: 0},
            household.initial_energy_mwh,
        )[0]
        self.assertAlmostEqual(
            dispatched.soc_after,
            0.5 * (1 - 0.04) ** 0.25,
        )

        protected_household = replace(
            household,
            battery_min_soc=0.5,
            battery_initial_soc=0.5,
        )
        protected_plan = optimize_household(
            protected_household,
            ((start, end),),
            {start: 100},
            {start: 2},
            protected_household.initial_energy_mwh,
            initial_heat_pump_power_mw=1,
        )[0]
        protected_dispatch = dispatch_household(
            protected_household,
            self.start - timedelta(minutes=15),
            (protected_plan,),
            {start: protected_plan.planned_grid_power_mw * 0.25},
            protected_household.initial_energy_mwh,
            initial_heat_pump_power_mw=1,
        )[0]
        self.assertGreaterEqual(protected_dispatch.soc_after + 1e-8, 0.5)

        with self.assertRaisesRegex(
            InputValidationError,
            "cannot cover battery losses while maintaining min_soc",
        ):
            dispatch_household(
                protected_household,
                self.start - timedelta(minutes=15),
                (idle_plan,),
                {start: 0},
                protected_household.initial_energy_mwh,
            )

    def test_missing_milp_solver_reports_a_clear_error(self) -> None:
        start, end = self.products(1)[0]
        with patch("electricity_market_sim.household.milp", None):
            with self.assertRaisesRegex(
                InputValidationError,
                "requires SciPy with scipy.optimize.milp",
            ):
                optimize_household(
                    self.household,
                    ((start, end),),
                    {start: 100},
                    {start: 2},
                    self.household.initial_energy_mwh,
                )

    def test_dispatch_rejects_accepted_energy_that_devices_cannot_absorb(self) -> None:
        household = replace(
            self.household,
            heat_pump_ramp_up_mw=0.2,
            heat_pump_ramp_down_mw=0.2,
            battery_max_charge_power_mw=0,
            battery_max_discharge_power_mw=0,
            battery_ramp_up_mw=0,
            battery_ramp_down_mw=0,
        )
        plans = tuple(
            HouseholdPlan(
                delivery_start=start,
                delivery_end=end,
                unit_name=household.name,
                forecast_price_eur_per_mwh=100,
                heat_demand_mw_th=2,
                fixed_power_mw=0,
                planned_grid_power_mw=1,
                planned_heat_pump_power_mw=1,
                planned_battery_charge_power_mw=0,
                planned_battery_discharge_power_mw=0,
                planned_soc_after=0.5,
            )
            for start, end in self.products(2)
        )
        accepted = {
            plans[0].delivery_start: 0,
            plans[1].delivery_start: 0.25,
        }

        with self.assertRaisesRegex(
            InputValidationError,
            "cannot be allocated to the heat pump or battery",
        ):
            dispatch_household(
                household,
                self.start - timedelta(minutes=15),
                plans,
                accepted,
                household.initial_energy_mwh,
                initial_heat_pump_power_mw=1,
            )


class VersionThreeInputTests(unittest.TestCase):
    def test_pay_as_bid_config_rejects_more_than_96_products(self) -> None:
        config = (EXAMPLE_INPUT / "config.yaml").read_text(encoding="utf-8")
        config = config.replace("count: 96", "count: 97")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.yaml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "at most 96"):
                load_market_settings(path, scenario="eom")

    def test_unused_demand_columns_are_reported_and_ignored(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profiles = load_time_series_profiles(
                EXAMPLE_INPUT / "demand_df.csv",
                ("demand_EOM",),
                warn_unused_columns=True,
            )

        self.assertEqual(set(profiles), {"demand_EOM"})
        self.assertEqual(len(caught), 1)
        self.assertIn("A361_building_load_profile", str(caught[0].message))

    def test_invalid_household_cost_tolerance_has_a_clear_error(self) -> None:
        with (EXAMPLE_INPUT / "residential_dsm_units.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None
        for row in rows:
            if row["cost_tolerance"]:
                row["cost_tolerance"] = "invalid"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "residential_dsm_units.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                InputValidationError,
                "cost_tolerance must be numeric",
            ):
                load_household_units(path)


class VersionThreeIntegrationTests(unittest.TestCase):
    def test_example_01h_runs_and_reconciles_payments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = run_simulation(EXAMPLE_INPUT, output, scenario="eom")

            self.assertEqual(len(result.market_results), 2782)
            self.assertEqual(len(result.household_results), 2782)
            self.assertEqual(len(result.household_flexibility_results), 2782)
            first = result.market_results[0]
            bids = {item.bid.unit_name: item for item in first.demand_bids}
            offer = first.offers[0]
            self.assertEqual(first.duration_hours, 0.25)
            self.assertIsNone(first.clearing_price_eur_per_mwh)
            self.assertAlmostEqual(bids["demand_EOM"].bid.volume_mwh, 25)
            self.assertAlmostEqual(bids["demand_EOM"].bid.price_eur_per_mwh, 62.88)
            self.assertAlmostEqual(bids["A360"].bid.volume_mwh, 0.375)
            self.assertAlmostEqual(bids["A360"].bid.price_eur_per_mwh, 3000)
            self.assertAlmostEqual(offer.offer.offered_energy_mwh, 280)
            self.assertAlmostEqual(offer.accepted_power_mw, 101.5)
            self.assertLess(offer.accepted_power_mw, 560)
            self.assertEqual(result.household_results[0].heat_pump_power_mw, 1.5)
            self.assertEqual(result.household_results[0].unmet_heat_mwh_th, 0)

            with (output / "trade_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                trade_rows = list(csv.DictReader(file))
            with (output / "demand_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                demand_rows = list(csv.DictReader(file))
            with (output / "market_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                market_rows = list(csv.DictReader(file))
            with (output / "household_flexibility_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                flexibility_rows = list(csv.DictReader(file))

            payments: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
            for row in trade_rows:
                payments[(row["opening_id"], row["buyer_bid_id"])] += Decimal(
                    row["payment_eur"]
                )
            for row in demand_rows:
                self.assertEqual(row["side"], "buy")
                self.assertEqual(
                    Decimal(row["payment_eur"]),
                    payments[(row["opening_id"], row["bid_id"])],
                )
            self.assertTrue(
                all(row["clearing_price_eur_per_mwh"] == "" for row in market_rows)
            )
            self.assertEqual(len(flexibility_rows), 2782)
            self.assertTrue(
                all(
                    float(row["minimum_grid_power_mw"])
                    <= float(row["maximum_grid_power_mw"])
                    for row in flexibility_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
