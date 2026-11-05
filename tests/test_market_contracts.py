"""CSV 输入与市场出清行为的契约测试。
运行方式：python -B -m unittest discover -s tests -v"""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from electricity_market_sim.bidding import available_power_mw
from electricity_market_sim.clearing_algorithm import (
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
from electricity_market_sim.models import (
    DemandBid,
    DemandUnit,
    ExchangeUnit,
    PowerPlant,
    StorageUnit,
    SupplyOffer,
)


START = datetime(2024, 1, 1)
HOUR = timedelta(hours=1)
PLANT = PowerPlant("plant", "operator", "gas", "powerplant_energy_naive",
                   "gas", 0.2, 10.0, 0.0, 0.5, 0.0)
DEMAND = DemandUnit("load", "operator", profile_column="load")
EXCHANGE = ExchangeUnit("grid", "operator", 0.0, 100.0)
STORAGE = StorageUnit(
    "storage", "operator", "battery",
    10.0, 10.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0,
)


def bid(name="load", volume=2.0, price=100.0, **kwargs):
    return DemandBid(name, "operator", START, START + HOUR, volume, price, **kwargs)


def offer(name="plant", volume=1.0, price=10.0, **kwargs):
    return SupplyOffer(name, "operator", "gas", START, START + HOUR,
                       volume, price, price, **kwargs)


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
            (lambda p: load_hourly_exchange_profiles(p, EXCHANGE)[START][0],
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

    def test_missing_availability_time_raises_key_error(self):
        with self.assertRaises(KeyError) as error:
            available_power_mw(PLANT, START, {"plant": {}})
        self.assertEqual(error.exception.args[0], START)


class ClearingContracts(unittest.TestCase):
    def test_minimum_acceptance_ratio_follows_offer_type(self):
        for kind, expected in (("SB", 0.0), ("LB", 0.0), ("BB", 1.0)):
            with self.subTest(kind=kind):
                self.assertEqual(offer(bid_type=kind).min_acceptance_ratio, expected)

    def test_storage_clearing_uses_projected_energy_instead_of_initial_soc(self):
        for energy, expected in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(projected_energy=energy):
                (result,) = clear_complex_opening(
                    [bid(volume=1.0)],
                    [offer("storage", price=10.0, offer_type="storage_discharge")],
                    storages=((STORAGE, energy),),
                )
                self.assertAlmostEqual(result.cleared_energy_mwh, expected)

    def test_clearing_result_compares_supply_and_demand_independently(self):
        result = clear_pay_as_clear([bid()], [offer()])
        demand = replace(result.demand_bids[0], accepted_energy_mwh=0.0)
        with self.assertRaisesRegex(ValueError, "买单成交电量"):
            replace(result, demand_bids=(demand,))

    def test_demand_category_determines_market_role_and_settlement(self):
        for category, role in (
            ("inelastic_load", "local_load"), ("elastic_load", "local_load"),
            ("household_load", "local_load"), ("industrial_load", "local_load"),
            ("export", "export"), ("storage_charge", "storage_charge"),
        ):
            with self.subTest(category=category):
                demand = bid(demand_type=category)
                self.assertEqual(demand.bid_type, role)
                result = clear_pay_as_bid([demand], [offer()])
                self.assertEqual(result.demand_bids[0].payment_eur, 10.0)
        with self.assertRaisesRegex(ValueError, "demand_type"):
            bid(demand_type="unknown")

    def test_small_trades_keep_their_energy_price_and_payment(self):
        for quantity in (5e-8, 1e-7, 2e-7):
            for clear, marginal_unit in (
                (clear_pay_as_clear, "plant"),
                (clear_pay_as_bid, None),
            ):
                with self.subTest(quantity=quantity, algorithm=clear.__name__):
                    result = clear([bid(volume=quantity)],
                                   [offer(volume=quantity, price=50.0)])
                    self.assertEqual(result.cleared_energy_mwh, quantity)
                    self.assertEqual(result.marginal_unit_name, marginal_unit)
                    self.assertEqual(len(result.trades), 1)
                    self.assertEqual(result.trades[0].trade_energy_mwh, quantity)
                    self.assertEqual(result.trades[0].trade_price_eur_per_mwh, 50.0)
                    self.assertEqual(result.offers[0].revenue_eur, quantity * 50.0)
                    self.assertEqual(result.demand_bids[0].payment_eur, quantity * 50.0)

    def test_duplicate_demand_ids_are_rejected(self):
        for clear in (clear_pay_as_clear, clear_pay_as_bid):
            with self.subTest(algorithm=clear.__name__):
                with self.assertRaisesRegex(ValueError, "Demand bid identifiers must be unique"):
                    clear([bid(), bid()], [])

    def test_duplicate_supply_ids_are_rejected(self):
        for clear in (clear_pay_as_clear, clear_pay_as_bid):
            with self.subTest(algorithm=clear.__name__):
                with self.assertRaises(ValueError) as error:
                    clear([], [offer(), offer()])
                self.assertEqual(str(error.exception),
                                 "Supply offer identifiers must be unique within a product.")

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
            bid("export", price=200, demand_type="export"),
            bid("storage", price=100, demand_type="storage_charge"),
        ]
        for clear in (clear_pay_as_clear, clear_pay_as_bid, clear_complex_opening):
            with self.subTest(algorithm=clear.__name__):
                kwargs = {}
                if clear is clear_complex_opening:
                    kwargs["storages"] = ((STORAGE, 0.0),)
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

    def test_trade_records_preserve_partial_matches_and_settlement(self):
        bids = [bid("high", volume=2), bid("low", volume=2, price=30)]
        offers = [offer("cheap", volume=1, price=-10),
                  offer("dear", volume=2, price=20), offer("unused", price=200)]
        for clear, prices, payments, revenues in (
            (clear_pay_as_clear, [20, 20, 20], [40, 20], [20, 40, 0]),
            (clear_pay_as_bid, [-10, 20, 20], [10, 20], [-10, 40, 0]),
        ):
            with self.subTest(algorithm=clear.__name__):
                result = clear(bids, offers)
                self.assertEqual(result.cleared_energy_mwh, 3)
                self.assertEqual(
                    [(t.buyer_bid_id, t.seller_offer_id, t.trade_energy_mwh)
                     for t in result.trades],
                    [("high", "cheap", 1), ("high", "dear", 1), ("low", "dear", 1)],
                )
                self.assertEqual({(t.delivery_start, t.delivery_end) for t in result.trades},
                                 {(START, START + HOUR)})
                self.assertEqual([t.trade_price_eur_per_mwh for t in result.trades], prices)
                self.assertEqual([d.accepted_energy_mwh for d in result.demand_bids], [2, 1])
                self.assertEqual([s.accepted_energy_mwh for s in result.offers], [1, 2, 0])
                self.assertEqual([d.payment_eur for d in result.demand_bids], payments)
                self.assertEqual([s.revenue_eur for s in result.offers], revenues)

    def test_unmatched_orders_generate_no_trade_records(self):
        for demand, supply in (
            (bid(volume=0), offer()),
            (bid(), offer(volume=0)),
            (bid(price=5), offer(price=10)),
        ):
            for clear in (clear_pay_as_clear, clear_pay_as_bid):
                with self.subTest(algorithm=clear.__name__, demand=demand, supply=supply):
                    result = clear([demand], [supply])
                    self.assertEqual(result.trades, ())
                    self.assertEqual(result.cleared_energy_mwh, 0)
                    self.assertEqual(result.demand_bids[0].payment_eur, 0)
                    self.assertEqual(result.offers[0].revenue_eur, 0)

    def test_complex_still_clears_multiple_products(self):
        bids = [bid()]
        offers = [offer()]
        bids.append(replace(bid(), delivery_start=START + HOUR, delivery_end=START + 2 * HOUR))
        offers.append(replace(offer(name="other"), delivery_start=START + HOUR,
                              delivery_end=START + 2 * HOUR))
        results = clear_complex_opening(bids, offers)
        self.assertEqual([result.delivery_start for result in results], [START, START + HOUR])
        self.assertEqual([result.unserved_load_mwh for result in results], [1.0, 1.0])
        for result in results:
            self.assertEqual(result.clearing_price_eur_per_mwh, 100)
            self.assertEqual([c.clearing_price_eur_per_mwh
                              for c in (*result.offers, *result.demand_bids)], [100, 100])


if __name__ == "__main__":
    unittest.main()
