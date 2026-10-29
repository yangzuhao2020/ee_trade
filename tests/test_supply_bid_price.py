"""固定售电报价的输入兼容性，以及报价与成本分别结算。"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from electricity_market_sim.bidding import naive_offer
from electricity_market_sim.clearing_algorithm import clear_pay_as_bid
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.inputs import load_powerplants
from electricity_market_sim.models import DemandBid


class FixedSupplyBidPriceTests(unittest.TestCase):
    def load_plant(self, **columns):
        row = {
            "name": "solar", "technology": "solar",
            "bidding_EOM": "powerplant_energy_naive", "fuel_type": "renewable",
            "emission_factor": 0, "max_power": 50, "min_power": 0,
            "efficiency": 1, "additional_cost": 0, "unit_operator": "pv",
            **columns,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "powerplant_units.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            return load_powerplants(path)[0]

    def offer(self, plant):
        start = datetime(2024, 6, 1)
        cost = plant.marginal_cost({"co2": 25})
        return naive_offer(plant, start, start + timedelta(minutes=15), 4, cost)

    def test_missing_or_blank_price_keeps_cost_based_offer(self):
        for columns in ({}, {"bid_price": ""}, {"bid_price": "  "}):
            with self.subTest(columns=columns):
                plant = self.load_plant(additional_cost=7, **columns)
                self.assertIsNone(plant.bid_price_eur_per_mwh)
                self.assertEqual(self.offer(plant).bid_price_eur_per_mwh, 7)

    def test_fixed_price_changes_payment_without_inflating_cost(self):
        plant = self.load_plant(bid_price=20)
        offer = self.offer(plant)
        buyer = DemandBid("A360", "dsm", offer.delivery_start,
                          offer.delivery_end, 1, 3000)
        result = clear_pay_as_bid([buyer], [offer])
        self.assertEqual(offer.marginal_cost_eur_per_mwh, 0)
        self.assertEqual(result.demand_bids[0].payment_eur, 20)
        self.assertEqual(result.offers[0].variable_cost_eur, 0)
        self.assertEqual(result.offers[0].profit_eur, 20)

    def test_zero_and_negative_prices_are_explicit_quotes(self):
        for price in (0, -5):
            with self.subTest(price=price):
                plant = self.load_plant(additional_cost=7, bid_price=price)
                self.assertEqual(self.offer(plant).bid_price_eur_per_mwh, price)
                self.assertEqual(self.offer(plant).marginal_cost_eur_per_mwh, 7)

    def test_invalid_prices_are_rejected(self):
        for price in ("nan", "inf", "not-a-number"):
            with self.subTest(price=price):
                with self.assertRaises(InputValidationError):
                    self.load_plant(bid_price=price)

    def test_fixed_price_is_not_silently_ignored_by_other_strategies(self):
        with self.assertRaisesRegex(InputValidationError, "bid_price is only supported"):
            self.load_plant(bid_price=20, bidding_EOM="powerplant_energy_heuristic_flexable")


if __name__ == "__main__":
    unittest.main()
