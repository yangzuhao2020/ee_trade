"""Marginal-cost price forecasts and complete-opening input preflight."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from ...bidding import naive_offer
from ...clearing import clear_pay_as_clear
from ...models import ExchangeSchedule, ExchangeUnit, MarketOpening, MarketSettings, PowerPlant
from ...time_utils import format_timestamp
from .demand import _demand_bids_for_product, _exchange_import_offer
from .plants import _available_power

_FORECAST_HOURS = 12


def _naive_price_forecasts(
    *,
    products: list[tuple[datetime, datetime]],
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
    marginal_costs: dict[str, float],
    availability_profiles: dict[str, dict[datetime, float]],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    forecast_starts: set[datetime] | None = None,
) -> dict[datetime, float]:
    """Calculate naïve EOM prices needed by plant and storage strategies."""

    if forecast_starts is None:
        forecast_starts = {
            delivery_start + timedelta(hours=hour)
            for delivery_start, _ in products
            for hour in range(_FORECAST_HOURS + 1)
        }
    prices: dict[datetime, float] = {}
    for delivery_start in sorted(forecast_starts):
        delivery_end = delivery_start + settings.product_duration
        demand_bids, exchange_schedule = _demand_bids_for_product(
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            maximum_bid_price=settings.maximum_bid_price,
            include_elastic=False,
        )
        offers = [
            naive_offer(
                plant,
                delivery_start,
                delivery_end,
                _available_power(plant, delivery_start, availability_profiles),
                marginal_costs[plant.name],
            )
            for plant in plants
        ]
        if exchange_unit is not None and exchange_schedule is not None:
            offers.append(
                _exchange_import_offer(
                    exchange_unit, exchange_schedule, delivery_start, delivery_end
                )
            )
        prices[delivery_start] = clear_pay_as_clear(
            demand_bids, offers
        ).clearing_price_eur_per_mwh
    return prices

def _forecast_starts(
    delivery_starts: Iterable[datetime],
    *,
    settings: MarketSettings,
    requires_price_forecast: bool = False,
    extra_forecast_offsets: tuple[int, ...] = (),
    requires_storage_price_forecast: bool = False,
) -> set[datetime]:
    """Forecast timestamps each enabled strategy family needs for deliveries.

    Plants look _FORECAST_HOURS + 1 hours ahead. Extensions declare their
    offsets explicitly; storage adds a symmetric window clipped to the horizon.
    """

    starts: set[datetime] = set()
    for delivery_start in delivery_starts:
        if requires_price_forecast:
            starts.update(
                delivery_start + timedelta(hours=offset)
                for offset in range(_FORECAST_HOURS + 1)
            )
        starts.update(
            delivery_start + timedelta(hours=offset)
            for offset in extra_forecast_offsets
        )
        if requires_storage_price_forecast:
            starts.update(
                delivery_start + timedelta(hours=offset)
                for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1)
                if settings.start
                <= delivery_start + timedelta(hours=offset)
                <= settings.end
            )
    return starts

def _missing_profile_times(
    profile: dict[datetime, object] | None, required_times: set[datetime]
) -> list[datetime]:
    return sorted(
        time for time in required_times if profile is None or time not in profile
    )

def _opening_coverage_issues(
    *,
    opening: MarketOpening,
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    availability_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    requires_price_forecast: bool,
    extra_forecast_offsets: tuple[int, ...] = (),
    requires_storage_price_forecast: bool = False,
) -> list[str]:
    """Return every known reason an opening cannot be quoted completely."""

    issues: list[str] = []
    delivery_times = {delivery_start for delivery_start, _ in opening.products}
    beyond_end = [
        (delivery_start, delivery_end)
        for delivery_start, delivery_end in opening.products
        if delivery_end > settings.end
    ]
    if beyond_end:
        issues.append(
            "交付时段超出仿真结束时间: "
            + ", ".join(
                f"{format_timestamp(start)}–{format_timestamp(end)}" for start, end in beyond_end
            )
        )
    forecast_times = _forecast_starts(
        delivery_times,
        settings=settings,
        requires_price_forecast=requires_price_forecast,
        extra_forecast_offsets=extra_forecast_offsets,
        requires_storage_price_forecast=requires_storage_price_forecast,
    )
    required_times = delivery_times | forecast_times

    def add_missing_issue(source: str, missing: list[datetime]) -> None:
        if not missing:
            return
        kind = (
            "报价预测数据"
            if any(time in forecast_times for time in missing)
            else "交付数据"
        )
        issues.append(
            f"{kind}缺少 {source}: " + ", ".join(format_timestamp(time) for time in missing)
        )

    for demand_unit in demand_units:
        if demand_unit.is_elastic:
            continue
        missing = _missing_profile_times(
            demand_profiles.get(demand_unit.name), required_times
        )
        add_missing_issue(f"demand_df.csv/{demand_unit.name}", missing)
    for plant in plants:
        if plant.name not in availability_profiles:
            continue
        missing = _missing_profile_times(
            availability_profiles[plant.name], required_times
        )
        add_missing_issue(f"availability_df.csv/{plant.name}", missing)
    if exchange_unit is not None:
        missing = _missing_profile_times(exchange_profiles, required_times)
        add_missing_issue("exchanges_df.csv", missing)
    return issues
