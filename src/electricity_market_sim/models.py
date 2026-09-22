"""Small, explicit data structures used by the electricity-market simulator.
例如：
- 市场配置：MarketSettings
- 发电机组与需求单元：PowerPlant、DemandUnit
- 买单和卖单：DemandBid、SupplyOffer
- 单时段出清结果：MarketClearingResult
- 完整仿真结果：SimulationResult
它还提供成交电量转成交功率、收入、成本、利润等基础计算，并检查能量守恒。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isclose, isfinite


_ENERGY_TOLERANCE_MWH = 1e-7


@dataclass(frozen=True)
class MarketSettings:
    """The EOM settings shared by simple and complex V2 market openings."""

    start: datetime
    end: datetime
    time_step: timedelta
    opening_frequency: timedelta
    opening_duration: timedelta
    product_duration: timedelta
    product_count: int
    first_delivery: timedelta
    maximum_bid_price: float
    minimum_bid_price: float
    market_mechanism: str
    market_id: str = "EOM"
    exchange_units_file: str | None = None
    additional_fields: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class MarketOpening:
    """All delivery products offered together at one market opening."""

    opening_time: datetime
    products: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class DemandUnit:
    """An EOM demand participant, optionally generated from elasticity inputs."""

    name: str
    operator: str
    bidding_strategy: str = "demand_energy_naive"
    profile_column: str | None = None
    max_power_mw: float | None = None
    elasticity: float | None = None
    elasticity_model: str | None = None
    max_price_eur_per_mwh: float | None = None
    num_bids: int | None = None
    price_eur_per_mwh: float | None = None

    @property
    def is_elastic(self) -> bool:
        return self.bidding_strategy == "demand_energy_heuristic_elastic"


@dataclass(frozen=True)
class ExchangeUnit:
    """The single virtual participant representing planned imports and exports."""

    name: str
    operator: str
    price_import_eur_per_mwh: float
    price_export_eur_per_mwh: float


@dataclass(frozen=True)
class ExchangeSchedule:
    """One delivery hour's positive planned import and export powers."""

    import_power_mw: float
    export_power_mw: float


@dataclass(frozen=True)
class PowerPlant:
    """A dispatchable plant and the parameters used by its bidding strategy."""

    name: str
    operator: str
    technology: str
    bidding_strategy: str
    fuel_type: str
    emission_factor: float
    max_power_mw: float
    min_power_mw: float
    efficiency: float
    additional_cost_eur_per_mwh: float
    # These optional V2 parameters retain the V2 example defaults when their
    # columns are absent from ``powerplant_units.csv``.
    start_cost_eur: float = 0.0
    min_operating_time_hours: float = 1.0
    min_down_time_hours: float = 1.0

    def marginal_cost(self, fuel_prices: dict[str, float]) -> float:
        """Return the ASSUME-compatible variable marginal cost.

        Input emission factors are interpreted per unit of fuel input, hence both
        fuel and CO2 costs are divided by the electrical efficiency.
        """

        # Variable renewables in the supplied V2 examples have no fuel-price
        # column; their fuel component is conventionally zero.
        fuel_price = fuel_prices.get(self.fuel_type, 0.0)
        co2_price = fuel_prices["co2"]
        return (
            (fuel_price + co2_price * self.emission_factor) / self.efficiency
            + self.additional_cost_eur_per_mwh
        )


@dataclass(frozen=True)
class StorageUnit:
    """A storage participant and the physical limits used by its EOM strategy."""

    name: str
    operator: str
    technology: str
    bidding_strategy: str
    max_power_charge_mw: float
    max_power_discharge_mw: float
    efficiency_charge: float
    efficiency_discharge: float
    min_soc: float
    max_soc: float
    capacity_mwh: float
    initial_soc: float
    additional_cost_charge_eur_per_mwh: float
    additional_cost_discharge_eur_per_mwh: float
    natural_inflow_mw: float = 0.0

    @property
    def min_energy_mwh(self) -> float:
        return self.min_soc * self.capacity_mwh

    @property
    def max_energy_mwh(self) -> float:
        return self.max_soc * self.capacity_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.capacity_mwh


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
            "export",
            "storage_charge",
        }:
            raise ValueError(
                "demand_type must be inelastic_load, elastic_load, household_load, "
                "export, or storage_charge."
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
    buyer_name: str
    buyer_bid_id: str
    seller_name: str
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

        if self.pricing_method == "pay_as_bid":
            self._validate_pay_as_bid_trades()
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
    def requested_export_mwh(self) -> float:
        return sum(
            demand.bid.volume_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "export"
        )

    @property
    def cleared_export_mwh(self) -> float:
        return sum(
            demand.accepted_energy_mwh
            for demand in self.demand_bids
            if demand.bid.bid_type == "export"
        )

    @property
    def offered_import_mwh(self) -> float:
        return sum(
            offer.offer.offered_energy_mwh
            for offer in self.offers
            if offer.offer.offer_type == "import"
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

    def _validate_pay_as_bid_trades(self) -> None:
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
        traded_by_offer: dict[str, float] = {}
        traded_by_bid: dict[str, float] = {}
        payment_by_bid: dict[str, float] = {}
        for trade in self.trades:
            if trade.seller_offer_id not in offers:
                raise ValueError(f"Unknown seller offer {trade.seller_offer_id!r}.")
            if trade.buyer_bid_id not in bids:
                raise ValueError(f"Unknown buyer bid {trade.buyer_bid_id!r}.")
            offer = offers[trade.seller_offer_id].offer
            bid = bids[trade.buyer_bid_id].bid
            if trade.seller_name != offer.unit_name or trade.buyer_name != bid.unit_name:
                raise ValueError("Trade participant names must match their order IDs.")
            if not isclose(
                trade.trade_price_eur_per_mwh,
                offer.bid_price_eur_per_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("Pay-as-bid trade price must equal the seller offer price.")
            if trade.trade_price_eur_per_mwh > bid.price_eur_per_mwh + _ENERGY_TOLERANCE_MWH:
                raise ValueError("A trade price cannot exceed the buyer bid price.")
            traded_by_offer[trade.seller_offer_id] = (
                traded_by_offer.get(trade.seller_offer_id, 0.0)
                + trade.trade_energy_mwh
            )
            traded_by_bid[trade.buyer_bid_id] = (
                traded_by_bid.get(trade.buyer_bid_id, 0.0)
                + trade.trade_energy_mwh
            )
            payment_by_bid[trade.buyer_bid_id] = (
                payment_by_bid.get(trade.buyer_bid_id, 0.0) + trade.payment_eur
            )

        for identifier, cleared in offers.items():
            accepted = traded_by_offer.get(identifier, 0.0)
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("Trade quantities must reconcile to seller acceptance.")
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
            accepted = traded_by_bid.get(identifier, 0.0)
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("Trade quantities must reconcile to buyer acceptance.")
            if not isclose(
                payment_by_bid.get(identifier, 0.0),
                cleared.payment_eur,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("Trade payments must reconcile to buyer payment.")


@dataclass(frozen=True)
class HouseholdUnit:
    """One building participant with a heat pump and an internal battery."""

    name: str
    operator: str
    node: str
    bidding_strategy: str
    objective: str
    flexibility_measure: str
    cost_tolerance_percent: float
    is_prosumer: bool
    fixed_power_mw: float
    heat_pump_max_power_mw: float
    heat_pump_min_power_mw: float
    heat_pump_ramp_up_mw: float
    heat_pump_ramp_down_mw: float
    cop: float
    battery_capacity_mwh: float
    battery_min_soc: float
    battery_max_soc: float
    battery_initial_soc: float
    battery_efficiency_charge: float
    battery_efficiency_discharge: float
    battery_max_charge_power_mw: float
    battery_max_discharge_power_mw: float
    battery_ramp_up_mw: float
    battery_ramp_down_mw: float
    battery_loss_rate: float = 0.0
    bid_price_eur_per_mwh: float = 3000.0

    @property
    def initial_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_initial_soc

    @property
    def min_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_min_soc

    @property
    def max_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_max_soc


@dataclass(frozen=True)
class HouseholdPlan:
    """A household's forecast-based schedule for one delivery product."""

    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    planned_heat_pump_power_mw: float
    planned_battery_charge_power_mw: float
    planned_battery_discharge_power_mw: float
    planned_soc_after: float


@dataclass(frozen=True)
class HouseholdDispatchResult:
    """Actual building operation after the corresponding buy order clears."""

    opening_time: datetime
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    heat_pump_power_mw: float
    battery_charge_power_mw: float
    battery_discharge_power_mw: float
    soc_before: float
    soc_after: float
    unmet_electricity_mwh: float
    unmet_heat_mwh_th: float


@dataclass(frozen=True)
class HouseholdFlexibilityResult:
    """Cost-tolerant grid-power bounds for one planned household product."""

    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    minimum_grid_power_mw: float
    maximum_grid_power_mw: float


@dataclass(frozen=True)
class SimulationResult:
    """The complete, in-memory outcome of one simulation run."""

    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
    storage_results: tuple[StorageDispatchResult, ...] = ()
    household_results: tuple[HouseholdDispatchResult, ...] = ()
    household_flexibility_results: tuple[HouseholdFlexibilityResult, ...] = ()


@dataclass(frozen=True)
class StorageClearingContext:
    """Physical storage data required by one joint complex-clearing opening."""

    unit_name: str
    initial_energy_mwh: float
    min_energy_mwh: float
    max_energy_mwh: float
    efficiency_charge: float
    efficiency_discharge: float


@dataclass(frozen=True)
class StorageDispatchResult:
    """One storage unit's accepted dispatch and SOC transition for one product."""

    opening_time: datetime
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    operator: str
    technology: str
    energy_before_mwh: float
    energy_after_mwh: float
    capacity_mwh: float
    offered_charge_mwh: float
    accepted_charge_mwh: float
    charge_bid_price_eur_per_mwh: float | None
    offered_discharge_mwh: float
    accepted_discharge_mwh: float
    discharge_bid_price_eur_per_mwh: float | None
    clearing_price_eur_per_mwh: float
    charge_payment_eur: float
    discharge_revenue_eur: float
    additional_charge_cost_eur: float
    additional_discharge_cost_eur: float

    @property
    def soc_before(self) -> float:
        return self.energy_before_mwh / self.capacity_mwh

    @property
    def soc_after(self) -> float:
        return self.energy_after_mwh / self.capacity_mwh

    @property
    def net_cash_flow_eur(self) -> float:
        return (
            self.discharge_revenue_eur
            - self.charge_payment_eur
            - self.additional_charge_cost_eur
            - self.additional_discharge_cost_eur
        )
