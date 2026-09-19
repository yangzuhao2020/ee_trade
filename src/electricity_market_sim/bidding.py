"""V2 generation availability, state tracking, and bidding strategies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .models import PowerPlant, SupplyOffer

_POWER_TOLERANCE_MW = 1e-9
_FORECAST_HOURS = 12


@dataclass
class PlantRuntimeState:
    """The sequential state needed by ``powerplant_energy_heuristic_flexable``.

    The V2 contract uses an initially-off unit with zero output.  Its initial
    downtime is already sufficient to permit an immediate startup.  Ramping,
    CHP commitments, and minimum-up/down enforcement are deliberately outside
    this version's dispatch constraints, but their state parameters are kept
    explicit here so the bid logic has a stable extension point.
    """

    is_running: bool
    power_mw: float
    down_time_hours: float
    elapsed_periods: int = 0
    operating_periods: int = 0

    @classmethod
    def initially_off(cls, plant: PowerPlant) -> PlantRuntimeState:
        return cls(
            is_running=False,
            power_mw=0.0,
            down_time_hours=max(plant.min_down_time_hours, 1.0),
        )

    @property
    def average_operating_time(self) -> float:
        """V2's historical on-period share (not a continuous run duration)."""

        if self.elapsed_periods == 0:
            return 0.0
        return self.operating_periods / self.elapsed_periods

    def record_dispatch(self, accepted_power_mw: float, duration_hours: float) -> None:
        """Advance the state after the associated delivery period ends."""

        self.elapsed_periods += 1
        self.power_mw = accepted_power_mw if accepted_power_mw > _POWER_TOLERANCE_MW else 0.0
        self.is_running = self.power_mw > _POWER_TOLERANCE_MW
        if self.is_running:
            self.operating_periods += 1
            self.down_time_hours = 0.0
        else:
            self.down_time_hours += duration_hours


def available_power_mw(
    plant: PowerPlant,
    delivery_start: datetime,
    availability_profiles: dict[str, dict[datetime, float]],
) -> float:
    """Return a plant's capacity after its optional availability factor."""

    profile = availability_profiles.get(plant.name)
    if profile is None:
        return plant.max_power_mw
    try:
        return plant.max_power_mw * profile[delivery_start]
    except KeyError as exc:
        raise KeyError(
            f"availability_df.csv has no complete hourly profile for "
            f"{delivery_start.isoformat(sep=' ', timespec='minutes')} and "
            f"plant {plant.name!r}."
        ) from exc


def naive_offer(
    plant: PowerPlant,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power: float,
    marginal_cost: float,
) -> SupplyOffer:
    """Build V1's one-part marginal-cost offer with V2 availability applied."""

    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
    offer_id = f"{plant.name}::single::{delivery_start.isoformat()}"
    return SupplyOffer(
        unit_name=plant.name,
        operator=plant.operator,
        technology=plant.technology,
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        offered_power_mw=available_power,
        offered_energy_mwh=available_power * duration_hours,
        bid_price_eur_per_mwh=marginal_cost,
        marginal_cost_eur_per_mwh=marginal_cost,
        offer_id=offer_id,
        bid_id=offer_id,
    )


def heuristic_flexible_offers(
    plant: PowerPlant,
    state: PlantRuntimeState,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power: float,
    marginal_cost: float,
    price_forecast: dict[datetime, float],
) -> list[SupplyOffer]:
    """Split an eligible plant into inflexible and flexible V2 offers.

    The flexible tranche is always offered at marginal cost.  An off plant
    distributes its startup cost across the minimum-stable tranche.  A running
    plant may bid that tranche at zero when the specified 13-point outlook says
    that retaining operation has non-negative future value.
    """

    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
    # A unit that cannot physically reach its minimum stable output does not
    # submit an offer.  This is distinct from clearing: a submitted minimum
    # tranche may still be partially accepted by the market.
    if available_power < plant.min_power_mw:
        return []

    inflexible_power = plant.min_power_mw
    flexible_power = available_power - inflexible_power

    if inflexible_power > _POWER_TOLERANCE_MW and state.is_running:
        future_prices = [
            price_forecast[delivery_start + timedelta(hours=hour)]
            for hour in range(_FORECAST_HOURS + 1)
        ]
        future_expected_profit = sum(price - marginal_cost for price in future_prices)
        # 简单预测未来是否盈利。
        current_forecast = future_prices[0]
        inflexible_price = (
            0.0
            if current_forecast < marginal_cost and future_expected_profit >= 0.0
            else marginal_cost
        ) # 机组已经处于运行状态时，最低稳定出力段 inflexible 应该报多少钱。
    elif inflexible_power > _POWER_TOLERANCE_MW:
        expected_operating_time = max(
            state.average_operating_time,
            plant.min_operating_time_hours,
        )
        inflexible_price = marginal_cost + plant.start_cost_eur / (
            expected_operating_time * inflexible_power
        )

    offers: list[SupplyOffer] = []
    timestamp = delivery_start.isoformat()
    if inflexible_power > _POWER_TOLERANCE_MW:
        offers.append(
            SupplyOffer(
            unit_name=plant.name,
            operator=plant.operator,
            technology=plant.technology,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            offered_power_mw=inflexible_power,
            offered_energy_mwh=inflexible_power * duration_hours,
            bid_price_eur_per_mwh=inflexible_price,
            marginal_cost_eur_per_mwh=marginal_cost,
            offer_id=f"{plant.name}::inflexible::{timestamp}",
            offer_segment="inflexible",
            bid_id=f"{plant.name}::inflexible::{timestamp}",
            )
        )
    if flexible_power > _POWER_TOLERANCE_MW:
        offers.append(
            SupplyOffer(
            unit_name=plant.name,
            operator=plant.operator,
            technology=plant.technology,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            offered_power_mw=flexible_power,
            offered_energy_mwh=flexible_power * duration_hours,
            bid_price_eur_per_mwh=marginal_cost,
            marginal_cost_eur_per_mwh=marginal_cost,
            offer_id=f"{plant.name}::flexible::{timestamp}",
            offer_segment="flexible",
            bid_id=f"{plant.name}::flexible::{timestamp}",
            )
        )
    return offers


def heuristic_block_offers(
    plant: PowerPlant,
    state: PlantRuntimeState,
    products: tuple[tuple[datetime, datetime], ...],
    available_powers_mw: dict[datetime, float],
    marginal_cost: float,
    price_forecast: dict[datetime, float],
    opening_time: datetime,
    linked: bool,
) -> list[SupplyOffer]:
    """Create one minimum-output BB and its hourly flexible legs for an opening.

    ``state`` is deliberately not advanced while the offers are built: V2 prices
    all products of one day-ahead opening from the gate-closure state snapshot.
    """

    inflexible_legs: list[SupplyOffer] = []
    flexible_legs: list[SupplyOffer] = []
    for delivery_start, delivery_end in products:
        hourly = heuristic_flexible_offers(
            plant,
            state,
            delivery_start,
            delivery_end,
            available_powers_mw[delivery_start],
            marginal_cost,
            price_forecast,
        )
        for offer in hourly:
            if offer.offer_segment == "inflexible":
                inflexible_legs.append(offer)
            else:
                flexible_legs.append(offer)

    opening_id = opening_time.isoformat()
    parent_bid_id = f"{plant.name}::BB::{opening_id}"
    has_parent = bool(inflexible_legs)
    offers: list[SupplyOffer] = []
    if has_parent:
        total_energy = sum(leg.offered_energy_mwh for leg in inflexible_legs)
        block_price = sum(
            leg.bid_price_eur_per_mwh * leg.offered_energy_mwh
            for leg in inflexible_legs
        ) / total_energy
        offers.extend(
            replace(
                leg,
                offer_id=f"{parent_bid_id}::{leg.delivery_start.isoformat()}",
                bid_id=parent_bid_id,
                bid_type="BB",
                min_acceptance_ratio=1.0,
                parent_bid_id=None,
                bid_price_eur_per_mwh=block_price,
                offer_segment="block_inflexible",
            )
            for leg in inflexible_legs
        )

    for leg in flexible_legs:
        child_id = f"{plant.name}::{'LB' if linked and has_parent else 'SB'}::{opening_id}::{leg.delivery_start.isoformat()}"
        offers.append(
            replace(
                leg,
                offer_id=child_id,
                bid_id=child_id,
                bid_type="LB" if linked and has_parent else "SB",
                parent_bid_id=parent_bid_id if linked and has_parent else None,
                offer_segment="linked_flexible" if linked and has_parent else "flexible",
            )
        )
    return offers
