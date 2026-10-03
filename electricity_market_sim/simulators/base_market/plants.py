"""Conventional plant quoting, startup settlement and delivered runtime state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from ...bidding import (
    PlantRuntimeState, available_power_mw, heuristic_block_offers,
    heuristic_flexible_offers, naive_offer,
)
from ...errors import InputValidationError
from ...market_models import MarketClearingResult, SupplyOffer
from ...models import ExchangeSchedule, ExchangeUnit, MarketOpening, MarketSettings, PowerPlant
from ...time_utils import format_timestamp
from .demand import _exchange_import_offer

if TYPE_CHECKING:
    from ..eom import EomMarketExtension, ScheduledProductKey

_POWER_TOLERANCE_MW = 1e-9
_LINKED_REQUIRED_FIELDS = frozenset({"bid_type", "min_acceptance_ratio", "parent_bid_id"})


def _available_power(
    plant: PowerPlant,
    delivery_start: datetime,
    availability_profiles: dict[str, dict[datetime, float]],
) -> float:
    try:
        return available_power_mw(plant, delivery_start, availability_profiles)
    except KeyError as exc:
        raise InputValidationError(str(exc)) from exc

def _validate_strategy_configuration(
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
) -> None:
    """Reject linked-bid configurations before availability or bidding is read."""

    linked_plants = [
        plant.name
        for plant in plants
        if plant.bidding_strategy == "powerplant_energy_heuristic_linked"
    ]
    if not linked_plants:
        return
    if settings.market_mechanism != "complex_clearing":
        raise InputValidationError(
            "powerplant_energy_heuristic_linked requires "
            "market_mechanism: complex_clearing."
        )
    missing_fields = sorted(_LINKED_REQUIRED_FIELDS - settings.additional_fields)
    if missing_fields:
        raise InputValidationError(
            "powerplant_energy_heuristic_linked requires "
            "markets_config.EOM.additional_fields to include: "
            + ", ".join(missing_fields)
            + "."
        )

def _apply_startup_costs(
    market_result: MarketClearingResult,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, PlantRuntimeState],
) -> MarketClearingResult:
    """Attach a startup cost when an accepted product begins delivery."""

    plants_by_name = {plant.name: plant for plant in plants}
    accepted_by_plant: dict[str, float] = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.offer_type == "power_plant":
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    started_names = {
        name
        for name, state in runtime_states.items()
        if not state.is_running and accepted_by_plant[name] > _POWER_TOLERANCE_MW
    }
    if not started_names:
        return market_result
    has_inflexible_offer = {
        cleared.offer.unit_name
        for cleared in market_result.offers
        if cleared.offer.offer_segment
        in {"inflexible", "block_inflexible", "learning_minimum"}
    }
    applied_names: set[str] = set()
    cleared_offers = []
    for cleared in market_result.offers:
        offer = cleared.offer
        startup_cost = 0.0
        if (
            offer.unit_name in started_names
            and offer.unit_name not in applied_names
            and offer.offer_type == "power_plant"
            and (
                offer.offer_segment
                in {"inflexible", "block_inflexible", "learning_minimum"}
                or offer.unit_name not in has_inflexible_offer
            )
        ):
            startup_cost = plants_by_name[offer.unit_name].start_cost_eur
            applied_names.add(offer.unit_name)
        cleared_offers.append(replace(cleared, startup_cost_eur=startup_cost))
    return replace(market_result, offers=tuple(cleared_offers))

def _record_runtime_states(
    market_result: MarketClearingResult,
    runtime_states: dict[str, PlantRuntimeState],
) -> None:
    """Commit actual dispatch only after the product's delivery period ends."""

    accepted_by_plant: dict[str, float] = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.unit_name in runtime_states:
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    for plant_name, state in runtime_states.items():
        state.record_dispatch(
            accepted_by_plant[plant_name] / market_result.duration_hours
        )

def _offers_for_opening(
    *,
    opening: MarketOpening,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, PlantRuntimeState],
    availability_profiles: dict[str, dict[datetime, float]],
    marginal_costs: dict[str, float],
    fuel_price_profiles: dict[datetime, dict[str, float]] | None,
    price_forecasts: dict[datetime, float],
    exchange_unit: ExchangeUnit | None,
    exchange_schedules: dict[datetime, ExchangeSchedule | None],
    extension: EomMarketExtension | None = None,
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] | None = None,
) -> list[SupplyOffer]:
    """Build an entire opening using its common gate-closure state snapshot."""

    offers: list[SupplyOffer] = []
    for plant in plants:
        available_powers = {
            delivery_start: _available_power(
                plant, delivery_start, availability_profiles
            )
            for delivery_start, _ in opening.products
        }

        def marginal_cost_at(
            delivery_start: datetime, current_plant: PowerPlant = plant
        ) -> float:
            if fuel_price_profiles is None:
                return marginal_costs[current_plant.name]
            try:
                prices = fuel_price_profiles[delivery_start]
            except KeyError as exc:
                raise InputValidationError(
                    "fuel_prices_df.csv has no complete product profile for "
                    f"{format_timestamp(delivery_start)}."
                ) from exc
            return current_plant.marginal_cost(prices)

        extension_offers = (
            extension.offers_for_plant(
                plant, opening.products, available_powers, marginal_cost_at,
                scheduled_results if scheduled_results is not None else {},
            )
            if extension is not None else None
        )
        if extension_offers is not None:
            offers.extend(extension_offers)
            continue

        if plant.bidding_strategy == "powerplant_energy_naive":
            offers.extend(
                naive_offer(
                    plant,
                    start,
                    end,
                    available_powers[start],
                    marginal_cost_at(start),
                )
                for start, end in opening.products
            )
        elif plant.bidding_strategy == "powerplant_energy_heuristic_flexable":
            for start, end in opening.products:
                offers.extend(
                    heuristic_flexible_offers(
                        plant,
                        runtime_states[plant.name],
                        start,
                        end,
                        available_powers[start],
                        marginal_cost_at(start),
                        price_forecasts,
                    )
                )
        elif plant.bidding_strategy in {
            "powerplant_energy_heuristic_block", "powerplant_energy_heuristic_linked"
        }:
            marginal_cost = marginal_cost_at(opening.products[0][0])
            offers.extend(
                heuristic_block_offers(
                    plant,
                    runtime_states[plant.name],
                    opening.products,
                    available_powers,
                    marginal_cost,
                    price_forecasts,
                    opening.opening_time,
                    linked=(
                        plant.bidding_strategy == "powerplant_energy_heuristic_linked"
                    ),
                )
            )
        else:
            raise InputValidationError(
                f"{plant.bidding_strategy} requires a supply-side market extension."
            )
    if exchange_unit is not None:
        for delivery_start, delivery_end in opening.products:
            schedule = exchange_schedules[delivery_start]
            assert schedule is not None
            offers.append(
                _exchange_import_offer(
                    exchange_unit, schedule, delivery_start, delivery_end
                )
            )
    return offers
