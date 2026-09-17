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

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isclose, isfinite


_ENERGY_TOLERANCE_MWH = 1e-7


@dataclass(frozen=True)
class MarketSettings:
    """The subset of YAML market settings supported by version one."""

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

@dataclass(frozen=True)
class DemandUnit:
    """An inflexible demand participant backed by one CSV profile column."""

    name: str
    operator: str
    profile_column: str


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
class DemandBid:
    """A buy order. Its volume is always a non-negative energy quantity."""

    unit_name: str
    operator: str
    delivery_start: datetime
    delivery_end: datetime
    volume_mwh: float
    price_eur_per_mwh: float
    bid_type: str = "local_load"


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

    @property
    def identifier(self) -> str:
        """Return the unique order key, retaining V1's unit-name default."""

        return self.offer_id or self.unit_name


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
class MarketClearingResult:
    """One delivery product's cleared market state."""

    delivery_start: datetime
    delivery_end: datetime
    requested_demand_mwh: float
    cleared_energy_mwh: float
    unserved_load_mwh: float
    unfulfilled_export_mwh: float
    clearing_price_eur_per_mwh: float
    offers: tuple[ClearedSupplyOffer, ...]
    demand_bids: tuple[ClearedDemandBid, ...]
    marginal_unit_name: str | None

    def __post_init__(self) -> None:
        """Guard the physical and financial quantities used by reporting and plots."""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("A market product must have a positive delivery duration.")
        if not isfinite(self.clearing_price_eur_per_mwh):
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
        if any(
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
            cleared.bid.bid_type not in {"local_load", "export"}
            for cleared in self.demand_bids
        ):
            raise ValueError("Demand bids must be local_load or export bids.")
        if any(
            cleared.offer.offer_type not in {"power_plant", "import"}
            for cleared in self.offers
        ):
            raise ValueError("Supply offers must be power_plant or import offers.")

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
            if demand.bid.bid_type == "local_load"
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

        offer_names_set = {cleared.offer.unit_name for cleared in self.offers}
        if self.cleared_energy_mwh > _ENERGY_TOLERANCE_MWH:
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
        elif self.marginal_unit_name is not None:
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
        """All unmet demand: local unserved load plus unfulfilled exports."""

        return self.unserved_load_mwh + self.unfulfilled_export_mwh

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
        return self.cleared_energy_mwh * self.clearing_price_eur_per_mwh


@dataclass(frozen=True)
class SimulationResult:
    """The complete, in-memory outcome of one simulation run."""

    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
