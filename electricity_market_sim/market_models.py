"""Market orders, trades, and clearing-result data structures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite


_ENERGY_TOLERANCE_MWH = 1e-7

__all__ = [
    "ClearedDemandBid",
    "ClearedSupplyOffer",
    "DemandBid",
    "MarketClearingResult",
    "SupplyOffer",
    "Trade",
]


@dataclass(frozen=True)
class DemandBid:
    """A buy order. Its volume is always a non-negative energy quantity."""

    unit_name: str
    operator: str
    delivery_start: datetime
    delivery_end: datetime
    volume_mwh: float
    price_eur_per_mwh: float
    bid_type: str = "local_load"
    bid_id: str | None = None
    # ``bid_type`` retains the market-side role used by the V1 exchange logic.
    # ``demand_type`` distinguishes reliability load from price-responsive load
    # so rejected elastic bids are not reported as involuntary curtailment.
    demand_type: str | None = None

    def __post_init__(self) -> None:
        if self.delivery_end <= self.delivery_start:
            raise ValueError("A demand bid must have a positive delivery duration.")
        if not isfinite(self.volume_mwh) or self.volume_mwh < 0:
            raise ValueError("Demand bid volume must be a non-negative finite value.")
        if not isfinite(self.price_eur_per_mwh):
            raise ValueError("Demand bid price must be finite.")
        if self.bid_type not in {"local_load", "export", "storage_charge"}:
            raise ValueError(
                "Demand bids must be local_load, export, or storage_charge bids."
            )
        demand_type = self.demand_type
        if demand_type is None:
            if self.bid_type == "export":
                demand_type = "export"
            elif self.bid_type == "storage_charge":
                demand_type = "storage_charge"
            else:
                demand_type = "inelastic_load"
            object.__setattr__(self, "demand_type", demand_type)
        if demand_type not in {
            "inelastic_load",
            "elastic_load",
            "household_load",
            "industrial_load",
            "export",
            "storage_charge",
        }:
            raise ValueError(
                "demand_type must be inelastic_load, elastic_load, household_load, "
                "industrial_load, export, or storage_charge."
            )
        if (self.bid_type == "export") != (demand_type == "export"):
            raise ValueError("Export demand bids must use demand_type='export'.")
        if (self.bid_type == "storage_charge") != (
            demand_type == "storage_charge"
        ):
            raise ValueError(
                "Storage charge bids must use demand_type='storage_charge'."
            )

    @property
    def identifier(self) -> str:
        """Return a stable order identifier for detailed demand reporting."""

        return self.bid_id or self.unit_name

    @property
    def side(self) -> str:
        return "buy"


@dataclass(frozen=True)
class SupplyOffer:
    """A sell order. Energy is used for clearing; power is retained for reporting."""

    unit_name: str
    operator: str
    technology: str
    delivery_start: datetime
    delivery_end: datetime
    offered_power_mw: float
    offered_energy_mwh: float
    bid_price_eur_per_mwh: float
    marginal_cost_eur_per_mwh: float
    offer_type: str = "power_plant"
    # A plant can submit more than one offer in V2.  ``unit_name`` remains the
    # physical-unit identifier; ``offer_id`` identifies one market order.
    offer_id: str | None = None
    offer_segment: str = "single"
    # Complex-clearing metadata. ``offer_id`` identifies one product leg;
    # ``bid_id`` identifies a potentially multi-product market order.
    bid_id: str | None = None
    bid_type: str = "SB"
    min_acceptance_ratio: float = 0.0
    parent_bid_id: str | None = None

    def __post_init__(self) -> None:
        """Enforce the deliberately narrow V2 minimum-acceptance contract."""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("A supply offer must have a positive delivery duration.")
        if not isfinite(self.offered_power_mw) or self.offered_power_mw < 0:
            raise ValueError("Offered power must be a non-negative finite value.")
        if not isfinite(self.offered_energy_mwh) or self.offered_energy_mwh < 0:
            raise ValueError("Offered energy must be a non-negative finite value.")
        if not isfinite(self.bid_price_eur_per_mwh):
            raise ValueError("Supply offer price must be finite.")
        if not isfinite(self.marginal_cost_eur_per_mwh):
            raise ValueError("Supply marginal cost must be finite.")
        if self.bid_type not in {"SB", "BB", "LB"}:
            raise ValueError(f"Unsupported bid_type {self.bid_type!r}.")
        if not isfinite(self.min_acceptance_ratio):
            raise ValueError("min_acceptance_ratio must be finite.")
        if self.bid_type == "BB" and self.min_acceptance_ratio != 1.0:
            raise ValueError("Version two supports BB orders only with MAR=1.")
        if self.bid_type in {"SB", "LB"} and self.min_acceptance_ratio != 0.0:
            raise ValueError(
                "Version two supports SB and LB orders only with MAR=0."
            )

    @property
    def identifier(self) -> str:
        """Return the unique order key, retaining V1's unit-name default."""

        return self.offer_id or self.unit_name

    @property
    def complex_identifier(self) -> str:
        """Return the order identity shared by legs of a block bid."""

        return self.bid_id or self.identifier

    @property
    def side(self) -> str:
        return "sell"


@dataclass(frozen=True)
class ClearedSupplyOffer:
    """The clearing outcome for one offer, including zero-acceptance outcomes."""

    offer: SupplyOffer
    accepted_energy_mwh: float
    clearing_price_eur_per_mwh: float
    startup_cost_eur: float = 0.0

    @property
    def duration_hours(self) -> float:
        return (
            self.offer.delivery_end - self.offer.delivery_start
        ).total_seconds() / 3600

    @property
    def accepted_power_mw(self) -> float:
        return self.accepted_energy_mwh / self.duration_hours

    @property
    def revenue_eur(self) -> float:
        return self.accepted_energy_mwh * self.clearing_price_eur_per_mwh

    @property
    def variable_cost_eur(self) -> float:
        return self.accepted_energy_mwh * self.offer.marginal_cost_eur_per_mwh

    @property
    def total_cost_eur(self) -> float:
        """Variable cost plus a possible whole-unit startup cost."""
        return self.variable_cost_eur + self.startup_cost_eur

    @property
    def profit_eur(self) -> float:
        return self.revenue_eur - self.total_cost_eur


@dataclass(frozen=True)
class ClearedDemandBid:
    """The cleared quantity and settlement payment for one demand bid."""

    bid: DemandBid
    accepted_energy_mwh: float
    clearing_price_eur_per_mwh: float

    @property
    def duration_hours(self) -> float:
        return (self.bid.delivery_end - self.bid.delivery_start).total_seconds() / 3600

    @property
    def accepted_power_mw(self) -> float:
        return self.accepted_energy_mwh / self.duration_hours

    @property
    def unserved_energy_mwh(self) -> float:
        return self.bid.volume_mwh - self.accepted_energy_mwh

    @property
    def payment_eur(self) -> float:
        return self.accepted_energy_mwh * self.clearing_price_eur_per_mwh


@dataclass(frozen=True)
class Trade:
    """One immutable buyer-seller match in a pay-as-bid product."""

    delivery_start: datetime
    delivery_end: datetime
    buyer_bid_id: str
    seller_offer_id: str
    trade_energy_mwh: float
    trade_price_eur_per_mwh: float

    def __post_init__(self) -> None:
        if self.delivery_end <= self.delivery_start:
            raise ValueError("A trade must have a positive delivery duration.")
        if self.trade_energy_mwh <= 0:
            raise ValueError("Trade energy must be positive.")
        if not isfinite(self.trade_price_eur_per_mwh):
            raise ValueError("Trade price must be finite.")

    @property
    def payment_eur(self) -> float:
        return self.trade_energy_mwh * self.trade_price_eur_per_mwh


@dataclass(frozen=True)
class MarketClearingResult:
    """One delivery product's cleared market state."""

    delivery_start: datetime
    delivery_end: datetime
    requested_demand_mwh: float
    cleared_energy_mwh: float
    unserved_load_mwh: float
    unfulfilled_export_mwh: float
    clearing_price_eur_per_mwh: float | None
    offers: tuple[ClearedSupplyOffer, ...]
    demand_bids: tuple[ClearedDemandBid, ...]
    marginal_unit_name: str | None
    pricing_method: str = "merit_order"
    opening_time: datetime | None = None
    trades: tuple[Trade, ...] = ()

    def __post_init__(self) -> None:
        """Guard the physical and financial quantities used by reporting and plots."""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("A market product must have a positive delivery duration.")
        if self.opening_time is not None and self.opening_time > self.delivery_start:
            raise ValueError("A market opening cannot occur after delivery starts.")
        if self.pricing_method == "pay_as_bid":
            if self.clearing_price_eur_per_mwh is not None:
                raise ValueError("Pay-as-bid markets do not have one clearing price.")
        elif self.clearing_price_eur_per_mwh is None or not isfinite(
            self.clearing_price_eur_per_mwh
        ):
            raise ValueError("The clearing price must be finite.")
        if any(
            value < -_ENERGY_TOLERANCE_MWH
            for value in (
                self.requested_demand_mwh,
                self.cleared_energy_mwh,
                self.unserved_load_mwh,
                self.unfulfilled_export_mwh,
            )
        ):
            raise ValueError("Market energy quantities cannot be negative.")
        if any(
            cleared.offer.delivery_start != self.delivery_start
            or cleared.offer.delivery_end != self.delivery_end
            for cleared in self.offers
        ) or any(
            cleared.bid.delivery_start != self.delivery_start
            or cleared.bid.delivery_end != self.delivery_end
            for cleared in self.demand_bids
        ):
            raise ValueError("All cleared bids and offers must share the market product.")
        if any(
            cleared.accepted_energy_mwh < -_ENERGY_TOLERANCE_MWH
            or cleared.accepted_energy_mwh
            > cleared.offer.offered_energy_mwh + _ENERGY_TOLERANCE_MWH
            for cleared in self.offers
        ) or any(
            cleared.accepted_energy_mwh < -_ENERGY_TOLERANCE_MWH
            or cleared.accepted_energy_mwh
            > cleared.bid.volume_mwh + _ENERGY_TOLERANCE_MWH
            for cleared in self.demand_bids
        ):
            raise ValueError("Accepted energy must be between zero and the submitted volume.")
        if self.pricing_method != "pay_as_bid" and any(
            not isclose(
                cleared.clearing_price_eur_per_mwh,
                self.clearing_price_eur_per_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            )
            for cleared in (*self.offers, *self.demand_bids)
        ):
            raise ValueError("All cleared orders must use the market clearing price.")
        offer_ids = [cleared.offer.identifier for cleared in self.offers]
        if len(offer_ids) != len(set(offer_ids)):
            raise ValueError(
                "Supply offer identifiers must be unique within a product; "
                "unit names must be unique when no offer_id is supplied."
            )
        if any(
            cleared.offer.offer_type
            not in {"power_plant", "import", "storage_discharge"}
            for cleared in self.offers
        ):
            raise ValueError(
                "Supply offers must be power_plant, import, or storage_discharge offers."
            )

        requested_demand = sum(cleared.bid.volume_mwh for cleared in self.demand_bids)
        if not isclose(
            self.requested_demand_mwh,
            requested_demand,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "Requested demand energy must equal the sum of demand bids."
            )
        accepted_energy = sum(offer.accepted_energy_mwh for offer in self.offers)
        if not isclose(
            accepted_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "The sum of accepted supply energy must equal cleared energy."
            )
        accepted_demand_energy = sum(
            demand.accepted_energy_mwh for demand in self.demand_bids
        )
        if not isclose(
            accepted_demand_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "The sum of accepted demand energy must equal cleared energy."
            )
        expected_unserved_load = sum(
            demand.unserved_energy_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "inelastic_load"
        )
        if not isclose(
            self.unserved_load_mwh,
            expected_unserved_load,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "Unserved load must equal the unmet local-load demand energy."
            )
        expected_unfulfilled_export = sum(
            demand.unserved_energy_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "export"
        )
        if not isclose(
            self.unfulfilled_export_mwh,
            expected_unfulfilled_export,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "Unfulfilled export must equal the unmet export demand energy."
            )

        if self.pricing_method not in {"merit_order", "dual", "pay_as_bid"}:
            raise ValueError("pricing_method must be merit_order, dual, or pay_as_bid.")

        if self.pricing_method == "pay_as_bid" or self.trades:
            self._validate_trades()
        offer_names_set = {cleared.offer.unit_name for cleared in self.offers}
        if self.pricing_method == "merit_order" and self.cleared_energy_mwh > _ENERGY_TOLERANCE_MWH:
            if self.marginal_unit_name is None:
                raise ValueError("A cleared market must identify its marginal unit.")
            if self.marginal_unit_name not in offer_names_set:
                raise ValueError("The marginal unit must belong to the cleared offers.")
            marginal_offers = [
                offer
                for offer in self.offers
                if offer.offer.unit_name == self.marginal_unit_name
                and offer.accepted_energy_mwh > _ENERGY_TOLERANCE_MWH
                and isclose(
                    offer.offer.bid_price_eur_per_mwh,
                    self.clearing_price_eur_per_mwh,
                    rel_tol=0.0,
                    abs_tol=_ENERGY_TOLERANCE_MWH,
                )
            ]
            if not marginal_offers:
                raise ValueError(
                    "The marginal unit must have an accepted offer at the clearing price."
                )
        elif self.pricing_method == "merit_order" and self.marginal_unit_name is not None:
            raise ValueError("An uncleared market cannot have a marginal unit.")

    @property
    def duration_hours(self) -> float:
        """Delivery duration used to convert energy quantities into power."""

        return (self.delivery_end - self.delivery_start).total_seconds() / 3600

    @property
    def requested_demand_power_mw(self) -> float:
        return self.requested_demand_mwh / self.duration_hours

    @property
    def cleared_power_mw(self) -> float:
        return self.cleared_energy_mwh / self.duration_hours

    @property
    def unserved_demand_mwh(self) -> float:
        """All unaccepted demand, including price-responsive elastic bids."""

        return sum(demand.unserved_energy_mwh for demand in self.demand_bids)

    @property
    def unaccepted_elastic_demand_mwh(self) -> float:
        return sum(
            demand.unserved_energy_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "elastic_load"
        )

    @property
    def requested_inelastic_demand_mwh(self) -> float:
        return sum(
            demand.bid.volume_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "inelastic_load"
        )

    @property
    def cleared_inelastic_demand_mwh(self) -> float:
        return sum(
            demand.accepted_energy_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "inelastic_load"
        )

    @property
    def requested_elastic_demand_mwh(self) -> float:
        return sum(
            demand.bid.volume_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "elastic_load"
        )

    @property
    def cleared_elastic_demand_mwh(self) -> float:
        return sum(
            demand.accepted_energy_mwh
            for demand in self.demand_bids
            if demand.bid.demand_type == "elastic_load"
        )

    @property
    def requested_local_demand_mwh(self) -> float:
        return sum(
            demand.bid.volume_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "local_load"
        )

    @property
    def cleared_local_demand_mwh(self) -> float:
        return sum(
            demand.accepted_energy_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "local_load"
        )

    @property
    def cleared_export_mwh(self) -> float:
        return sum(
            demand.accepted_energy_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "export"
        )

    @property
    def cleared_import_mwh(self) -> float:
        return sum(
            offer.accepted_energy_mwh
            for offer in self.offers
            if offer.offer.offer_type == "import"
        )

    @property
    def net_exchange_mwh(self) -> float:
        """Positive values are net imports into the local market."""

        return self.cleared_import_mwh - self.cleared_export_mwh

    @property
    def import_revenue_eur(self) -> float:
        return sum(
            offer.revenue_eur
            for offer in self.offers
            if offer.offer.offer_type == "import"
        )

    @property
    def export_payment_eur(self) -> float:
        return sum(
            demand.payment_eur
            for demand in self.demand_bids
            if demand.bid.bid_type == "export"
        )

    @property
    def exchange_cash_flow_eur(self) -> float:
        """Positive values are net market receipts for the Exchange participant."""

        return self.import_revenue_eur - self.export_payment_eur

    @property
    def transaction_value_eur(self) -> float:
        if self.pricing_method == "pay_as_bid":
            return sum(trade.payment_eur for trade in self.trades)
        assert self.clearing_price_eur_per_mwh is not None
        return self.cleared_energy_mwh * self.clearing_price_eur_per_mwh

    @property
    def average_trade_price_eur_per_mwh(self) -> float | None:
        if not self.trades:
            return None
        return sum(trade.payment_eur for trade in self.trades) / sum(
            trade.trade_energy_mwh for trade in self.trades
        )

    def _validate_trades(self) -> None:
        """Reconcile immutable trades with accepted order quantities and prices."""

        if any(
            trade.delivery_start != self.delivery_start
            or trade.delivery_end != self.delivery_end
            for trade in self.trades
        ):
            raise ValueError("All trades must share the market product.")
        trade_energy = sum(trade.trade_energy_mwh for trade in self.trades)
        if not isclose(
            trade_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("Trade energy must equal cleared market energy.")

        offers = {cleared.offer.identifier: cleared for cleared in self.offers}
        bids = {cleared.bid.identifier: cleared for cleared in self.demand_bids}
        traded_by_offer: dict[str, float] = defaultdict(float)
        traded_by_bid: dict[str, float] = defaultdict(float)
        payment_by_bid: dict[str, float] = defaultdict(float)
        for trade in self.trades:
            if trade.seller_offer_id not in offers:
                raise ValueError(f"Unknown seller offer {trade.seller_offer_id!r}.")
            if trade.buyer_bid_id not in bids:
                raise ValueError(f"Unknown buyer bid {trade.buyer_bid_id!r}.")
            offer = offers[trade.seller_offer_id].offer
            bid = bids[trade.buyer_bid_id].bid
            expected_trade_price = (
                offer.bid_price_eur_per_mwh
                if self.pricing_method == "pay_as_bid"
                else self.clearing_price_eur_per_mwh
            )
            assert expected_trade_price is not None
            if not isclose(
                trade.trade_price_eur_per_mwh,
                expected_trade_price,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError(
                    "Trade price must match the market settlement rule."
                )
            if trade.trade_price_eur_per_mwh > bid.price_eur_per_mwh + _ENERGY_TOLERANCE_MWH:
                raise ValueError("A trade price cannot exceed the buyer bid price.")
            traded_by_offer[trade.seller_offer_id] += trade.trade_energy_mwh
            traded_by_bid[trade.buyer_bid_id] += trade.trade_energy_mwh
            payment_by_bid[trade.buyer_bid_id] += trade.payment_eur

        for identifier, cleared in offers.items():
            accepted = traded_by_offer[identifier]
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("Trade quantities must reconcile to seller acceptance.")
            if self.pricing_method == "pay_as_bid":
                expected_price = (
                    cleared.offer.bid_price_eur_per_mwh
                    if accepted > _ENERGY_TOLERANCE_MWH
                    else 0.0
                )
                if not isclose(
                    cleared.clearing_price_eur_per_mwh,
                    expected_price,
                    rel_tol=0.0,
                    abs_tol=_ENERGY_TOLERANCE_MWH,
                ):
                    raise ValueError(
                        "A pay-as-bid seller must settle at its own offer price."
                    )
        for identifier, cleared in bids.items():
            accepted = traded_by_bid[identifier]
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("Trade quantities must reconcile to buyer acceptance.")
            if not isclose(
                payment_by_bid[identifier],
                cleared.payment_eur,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("Trade payments must reconcile to buyer payment.")
