"""Small, explicit data structures used by the first-version simulator.
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
from math import isclose


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

    @property
    def product_duration_hours(self) -> float:
        return self.product_duration.total_seconds() / 3600


@dataclass(frozen=True)
class DemandUnit:
    """An inflexible demand participant backed by one CSV profile column."""

    name: str
    operator: str
    profile_column: str


@dataclass(frozen=True)
class PowerPlant:
    """A dispatchable plant using the naïve marginal-cost bidding strategy."""

    name: str
    operator: str
    technology: str
    fuel_type: str
    emission_factor: float
    max_power_mw: float
    min_power_mw: float
    efficiency: float
    additional_cost_eur_per_mwh: float

    def marginal_cost(self, fuel_prices: dict[str, float]) -> float:
        """Return the ASSUME-compatible variable marginal cost.

        Input emission factors are interpreted per unit of fuel input, hence both
        fuel and CO2 costs are divided by the electrical efficiency.
        """

        fuel_price = fuel_prices[self.fuel_type]
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


@dataclass(frozen=True)
class ClearedSupplyOffer:
    """The clearing outcome for one offer, including zero-acceptance outcomes."""

    offer: SupplyOffer
    accepted_energy_mwh: float
    clearing_price_eur_per_mwh: float

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
    def profit_eur(self) -> float:
        return self.revenue_eur - self.variable_cost_eur


@dataclass(frozen=True)
class MarketClearingResult:
    """One delivery product's cleared market state."""

    delivery_start: datetime
    delivery_end: datetime
    requested_demand_mwh: float
    cleared_energy_mwh: float
    unserved_load_mwh: float
    clearing_price_eur_per_mwh: float
    offers: tuple[ClearedSupplyOffer, ...]
    marginal_unit_name: str | None

    def __post_init__(self) -> None:
        """Guard the physical and financial quantities used by reporting and plots."""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("A market product must have a positive delivery duration.")
        if any(
            value < -_ENERGY_TOLERANCE_MWH
            for value in (
                self.requested_demand_mwh,
                self.cleared_energy_mwh,
                self.unserved_load_mwh,
            )
        ):
            raise ValueError("Market energy quantities cannot be negative.")
        if not isclose(
            self.requested_demand_mwh,
            self.cleared_energy_mwh + self.unserved_load_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError(
                "Demand energy must equal cleared energy plus unserved load."
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
        offer_names = {offer.offer.unit_name for offer in self.offers}
        if self.cleared_energy_mwh > _ENERGY_TOLERANCE_MWH:
            if self.marginal_unit_name is None:
                raise ValueError("A cleared market must identify its marginal unit.")
            if self.marginal_unit_name not in offer_names:
                raise ValueError("The marginal unit must belong to the cleared offers.")
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
    def unserved_load_power_mw(self) -> float:
        return self.unserved_load_mwh / self.duration_hours

    @property
    def transaction_value_eur(self) -> float:
        return self.cleared_energy_mwh * self.clearing_price_eur_per_mwh


@dataclass(frozen=True)
class SimulationResult:
    """The complete, in-memory outcome of one simulation run."""

    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
