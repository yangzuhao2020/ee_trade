"""Rigid/elastic demand orders and planned Exchange imports/exports."""

from __future__ import annotations

from datetime import datetime

from ...errors import InputValidationError
from ...market_models import DemandBid, SupplyOffer
from ...models import ExchangeSchedule, ExchangeUnit
from ...time_utils import format_timestamp, hours_between


def _profile_power(
    profiles: dict[str, dict[datetime, float]],
    unit_name: str,
    delivery_start: datetime,
    source_name: str,
) -> float:
    try:
        return profiles[unit_name][delivery_start]
    except KeyError as exc:
        raise InputValidationError(
            f"{source_name} has no hourly profile for "
            f"{format_timestamp(delivery_start)} and unit {unit_name!r}."
        ) from exc

def _elastic_demand_bids(
    demand_unit,
    delivery_start: datetime,
    delivery_end: datetime,
) -> list[DemandBid]:
    """Discretise V2's isoelastic demand curve into descending-value buy bids."""

    assert demand_unit.max_power_mw is not None
    assert demand_unit.elasticity is not None
    assert demand_unit.max_price_eur_per_mwh is not None
    assert demand_unit.num_bids is not None
    duration_hours = hours_between(delivery_start, delivery_end)
    maximum_power = demand_unit.max_power_mw
    maximum_price = demand_unit.max_price_eur_per_mwh
    first_power = maximum_power * maximum_price**demand_unit.elasticity
    incremental_power = (maximum_power - first_power) / (demand_unit.num_bids - 1)
    timestamp = delivery_start.isoformat()
    bids: list[DemandBid] = []
    for index in range(demand_unit.num_bids):
        if index == 0:
            power = first_power
            price = maximum_price
        else:
            power = incremental_power
            cumulative_power = first_power + index * incremental_power
            price = (cumulative_power / maximum_power) ** (1.0 / demand_unit.elasticity)
        bids.append(
            DemandBid(
                unit_name=demand_unit.name,
                operator=demand_unit.operator,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                volume_mwh=power * duration_hours,
                price_eur_per_mwh=price,
                bid_id=f"{demand_unit.name}::elastic::{timestamp}::{index + 1}",
                demand_type="elastic_load",
            )
        )
    return bids

def _demand_bids_for_product(
    *,
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    delivery_start: datetime,
    delivery_end: datetime,
    maximum_bid_price: float,
    include_elastic: bool = True,
) -> tuple[list[DemandBid], ExchangeSchedule | None]:
    """Create local-load, elastic-load, and optional exchange demand orders."""

    duration_hours = hours_between(delivery_start, delivery_end)
    demand_bids: list[DemandBid] = []
    for demand_unit in demand_units:
        if demand_unit.is_elastic:
            if include_elastic:
                demand_bids.extend(
                    _elastic_demand_bids(demand_unit, delivery_start, delivery_end)
                )
            continue
        demand_bids.append(
            DemandBid(
                unit_name=demand_unit.name,
                operator=demand_unit.operator,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                volume_mwh=_profile_power(
                    demand_profiles,
                    demand_unit.name,
                    delivery_start,
                    "demand_df.csv",
                )
                * duration_hours,
                price_eur_per_mwh=maximum_bid_price,
                bid_id=f"{demand_unit.name}::load::{delivery_start.isoformat()}",
            )
        )
    if exchange_unit is None:
        return demand_bids, None
    try:
        exchange_schedule = exchange_profiles[delivery_start]
    except KeyError as exc:
        raise InputValidationError(
            "exchanges_df.csv has no hourly profile for "
            f"{format_timestamp(delivery_start)}."
        ) from exc
    demand_bids.append(
        DemandBid(
            unit_name=exchange_unit.name,
            operator=exchange_unit.operator,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            volume_mwh=exchange_schedule.export_power_mw * duration_hours,
            price_eur_per_mwh=exchange_unit.price_export_eur_per_mwh,
            bid_type="export",
            bid_id=f"{exchange_unit.name}::export::{delivery_start.isoformat()}",
            demand_type="export",
        )
    )
    return demand_bids, exchange_schedule

def _exchange_import_offer(
    exchange_unit: ExchangeUnit,
    exchange_schedule: ExchangeSchedule,
    delivery_start: datetime,
    delivery_end: datetime,
) -> SupplyOffer:
    duration_hours = hours_between(delivery_start, delivery_end)
    identifier = f"{exchange_unit.name}::import::{delivery_start.isoformat()}"
    return SupplyOffer(
        unit_name=exchange_unit.name,
        operator=exchange_unit.operator,
        technology="exchange",
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        offered_power_mw=exchange_schedule.import_power_mw,
        offered_energy_mwh=exchange_schedule.import_power_mw * duration_hours,
        bid_price_eur_per_mwh=exchange_unit.price_import_eur_per_mwh,
        marginal_cost_eur_per_mwh=0.0,
        offer_type="import",
        offer_id=identifier,
        bid_id=identifier,
    )
