"""Input and clearing contracts: python -m unittest discover -s tests -v."""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from electricity_market_sim.bidding import available_power_mw
from electricity_market_sim.clearing.algorithms import (
    clear_complex_opening,
    clear_pay_as_bid,
    clear_pay_as_clear,
)
from electricity_market_sim.errors import InputValidationError
from electricity_market_sim.inputs.csv_loader import (
    load_aligned_time_series_profiles,
    load_fuel_prices,
    load_hourly_availability_profiles,
    load_hourly_demand_profiles,
    load_hourly_exchange_profiles,
)
from electricity_market_sim.market_models import DemandBid, SupplyOffer
from electricity_market_sim.models import (
    DemandUnit, ExchangeUnit, PowerPlant, StorageClearingContext,
)


START = datetime(2024, 1, 1)
HOUR = timedelta(hours=1)
PLANT = PowerPlant("plant", "operator", "gas", "powerplant_energy_naive",
                   "gas", 0.2, 10.0, 0.0, 0.5, 0.0)
DEMAND = DemandUnit("load", "operator", profile_column="load")
EXCHANGE = ExchangeUnit("grid", "operator", 0.0, 100.0)


def bid(name="load", volume=2.0, price=100.0, **kwargs):
    return DemandBid(name, "operator", START, START + HOUR, volume, price, **kwargs)


def offer(name="plant", volume=1.0, price=10.0, **kwargs):
    return SupplyOffer(name, "operator", "gas", START, START + HOUR,
                       volume, volume, price, price, **kwargs)


class CsvContracts(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.csv"

    def write(self, text, encoding="utf-8"):
        self.path.write_text(text, encoding=encoding)
        return self.path

    def test_bom_and_plain_utf8_have_identical_values(self):
        cases = (
            (lambda p: load_hourly_demand_profiles(p, (DEMAND,)),
             "datetime,load\n2024-01-01 00:00,2\n2024-01-01 00:45,10\n",
             {"load": {START: 6.0}}),
            (lambda p: load_hourly_availability_profiles(p, (PLANT,)),
             "datetime,plant\n2024-01-01 00:00,0.2\n2024-01-01 00:45,0.8\n",
             {"plant": {START: 0.5}}),
            (lambda p: load_hourly_exchange_profiles(p, EXCHANGE)[START].import_power_mw,
             "datetime,grid_import,grid_export\n2024-01-01 00:00,2,3\n"
             "2024-01-01 00:45,10,7\n", 6.0),
            (load_fuel_prices, "fuel,gas,co2\nprice,30,80\n",
             {"gas": 30.0, "co2": 80.0}),
        )
        for load, text, expected in cases:
            for encoding in ("utf-8", "utf-8-sig"):
                with self.subTest(text=text.splitlines()[0], encoding=encoding):
                    self.assertEqual(load(self.write(text, encoding)), expected)

    def test_hourly_samples_are_equal_weighted_and_missing_hours_stay_missing(self):
        path = self.write("datetime,load\n2024-01-01 00:00,2\n"
                          "2024-01-01 00:45,10\n2024-01-01 02:15,4\n")
        self.assertEqual(load_hourly_demand_profiles(path, (DEMAND,)),
                         {"load": {START: 6.0, START + 2 * HOUR: 4.0}})

    def test_aligned_profiles_still_discard_partial_buckets(self):
        rows = "datetime,load\n" + "".join(
            f"{START + timedelta(minutes=15 * i)},{i}\n" for i in range(6)
        )
        path = self.write(rows)
        self.assertEqual(load_aligned_time_series_profiles(path, ("load",), HOUR),
                         {"load": {START: 1.5}})
        self.assertEqual(load_hourly_demand_profiles(path, (DEMAND,)),
                         {"load": {START: 1.5, START + HOUR: 4.5}})

    def test_aligned_profiles_still_reject_irregular_cadence(self):
        path = self.write("datetime,load\n2024-01-01 00:00,2\n"
                          "2024-01-01 00:15,3\n2024-01-01 00:45,10\n")
        with self.assertRaisesRegex(InputValidationError, "regular cadence"):
            load_aligned_time_series_profiles(path, ("load",), HOUR)

    def test_invalid_hourly_input_still_fails_with_or_without_bom(self):
        cases = (
            ("datetime,load\n2024-01-01 00:00,-1\n", "demand cannot be negative"),
            ("datetime,load\n2024-01-01 00:00,nan\n", "must be finite"),
            ("datetime,load\n2024-01-01 00:00,x\n", "must be numeric"),
            ("datetime,load\n2024-01-01 00:00,1\n2024-01-01 00:00,2\n",
             "duplicate timestamp"),
        )
        for text, message in cases:
            for encoding in ("utf-8", "utf-8-sig"):
                with self.subTest(message=message, encoding=encoding):
                    with self.assertRaisesRegex(InputValidationError, message):
                        load_hourly_demand_profiles(self.write(text, encoding), (DEMAND,))

    def test_missing_availability_error_does_not_require_complete_hour(self):
        with self.assertRaises(KeyError) as error:
            available_power_mw(PLANT, START, {"plant": {}})
        self.assertEqual(error.exception.args[0],
                         "availability_df.csv has no hourly profile for "
                         "2024-01-01 00:00 and plant 'plant'.")


class ClearingContracts(unittest.TestCase):
    def test_validation_precedence_is_specific_to_each_algorithm(self):
        first = bid()
        second = replace(first, delivery_start=START + HOUR, delivery_end=START + 2 * HOUR)
        with self.assertRaisesRegex(ValueError, "Demand bid identifiers must be unique"):
            clear_pay_as_clear([first, second], [])
        with self.assertRaisesRegex(ValueError, "must share a product"):
            clear_pay_as_bid([first, second], [])

    def test_supply_id_error_keeps_algorithm_specific_detail(self):
        cases = (
            (clear_pay_as_clear,
             "Supply offer identifiers must be unique within a product; "
             "unit names must be unique when no offer_id is supplied."),
            (clear_pay_as_bid, "Supply offer identifiers must be unique within a product."),
        )
        for clear, expected in cases:
            with self.subTest(algorithm=clear.__name__):
                with self.assertRaises(ValueError) as error:
                    clear([], [offer(), offer()])
                self.assertEqual(str(error.exception), expected)

    def test_empty_market_errors_are_preserved(self):
        for clear, message in (
            (clear_pay_as_clear, "Cannot clear an empty market."),
            (clear_pay_as_bid, "Cannot clear an empty market."),
            (clear_complex_opening, "Cannot clear an empty complex market opening."),
        ):
            with self.subTest(algorithm=clear.__name__):
                with self.assertRaises(ValueError) as error:
                    clear([], [])
                self.assertEqual(str(error.exception), message)

    def test_statistics_distinguish_inelastic_load_export_and_flexible_demand(self):
        bids = [
            bid("inelastic", price=600),
            bid("elastic", price=500, demand_type="elastic_load"),
            bid("household", price=400, demand_type="household_load"),
            bid("industry", price=300, demand_type="industrial_load"),
            bid("export", price=200, bid_type="export"),
            bid("storage", price=100, bid_type="storage_charge"),
        ]
        for clear in (clear_pay_as_clear, clear_pay_as_bid, clear_complex_opening):
            with self.subTest(algorithm=clear.__name__):
                kwargs = {}
                if clear is clear_complex_opening:
                    kwargs["storage_contexts"] = (
                        StorageClearingContext("storage", 0.0, 0.0, 10.0, 1.0, 1.0),
                    )
                result = clear(bids, [offer()], **kwargs)
                if isinstance(result, tuple):
                    result = result[0]
                self.assertEqual(result.requested_demand_mwh, 12.0)
                self.assertEqual(result.cleared_energy_mwh, 1.0)
                self.assertEqual(result.unserved_load_mwh, 1.0)
                self.assertEqual(result.unfulfilled_export_mwh, 2.0)
                self.assertEqual(result.demand_total("unserved_energy_mwh"), 11.0)
                self.assertEqual(result.demand_total("bid.volume_mwh",
                                 demand_type="industrial_load", bid_type="local_load"), 2.0)
                self.assertEqual(result.demand_total("bid.volume_mwh",
                                 demand_type="industrial_load", bid_type="export"), 0)

    def test_simple_algorithms_keep_their_settlement_rules(self):
        bids = [bid(volume=3)]
        offers = [offer("cheap", volume=1, price=10), offer("dear", volume=3, price=20)]
        uniform = clear_pay_as_clear(bids, offers)
        discriminatory = clear_pay_as_bid(bids, offers)
        self.assertEqual(uniform.clearing_price_eur_per_mwh, 20)
        self.assertEqual(uniform.demand_bids[0].payment_eur, 60)
        self.assertIsNone(discriminatory.clearing_price_eur_per_mwh)
        self.assertEqual(discriminatory.demand_bids[0].payment_eur, 50)
        self.assertEqual([trade.trade_price_eur_per_mwh for trade in discriminatory.trades],
                         [10, 20])

    def test_complex_still_clears_multiple_products(self):
        bids = [bid()]
        offers = [offer()]
        bids.append(replace(bid(), delivery_start=START + HOUR, delivery_end=START + 2 * HOUR))
        offers.append(replace(offer(name="other"), delivery_start=START + HOUR,
                              delivery_end=START + 2 * HOUR))
        results = clear_complex_opening(bids, offers)
        self.assertEqual([result.delivery_start for result in results], [START, START + HOUR])
        self.assertEqual([result.unserved_load_mwh for result in results], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
