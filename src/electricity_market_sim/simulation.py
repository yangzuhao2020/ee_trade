"""Orchestration from scenario files through V2 bids, clearing, and reporting."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .bidding import (
    PlantRuntimeState,
    available_power_mw,
    heuristic_flexible_offers,
    naive_offer,
)
from .clearing import clear_pay_as_clear
from .config import load_market_settings
from .errors import InputValidationError
from .loader import (
    load_demand_units,
    load_exchange_unit,
    load_fuel_prices,
    load_hourly_availability_profiles,
    load_hourly_demand_profiles,
    load_hourly_exchange_profiles,
    load_powerplants,
    validate_fuel_coverage,
)
from .models import (
    ClearedSupplyOffer,
    DemandBid,
    ExchangeSchedule,
    ExchangeUnit,
    MarketClearingResult,
    MarketSettings,
    PowerPlant,
    SimulationResult,
    SupplyOffer,
)
from .reporting import write_results


_POWER_TOLERANCE_MW = 1e-9


def delivery_products(settings: MarketSettings) -> list[tuple[datetime, datetime]]:
    """Create only products whose delivery end does not exceed ``end_date``."""

    products: list[tuple[datetime, datetime]] = []
    opening_time = settings.start
    while True:
        delivery_start = opening_time + settings.first_delivery
        delivery_end = delivery_start + settings.product_duration
        if delivery_end > settings.end:
            return products
        products.append((delivery_start, delivery_end))
        opening_time += settings.opening_frequency


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
            f"{source_name} has no complete hourly profile for "
            f"{delivery_start.isoformat(sep=' ', timespec='minutes')}"
            f" and unit {unit_name!r}."
        ) from exc


def _demand_bids_for_product(
    *,
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    delivery_start: datetime,
    delivery_end: datetime,
    maximum_bid_price: float,
) -> tuple[list[DemandBid], ExchangeSchedule | None]:
    """Create V1-compatible local-load and optional Exchange demand orders."""

    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
    demand_bids = [
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
        )
        for demand_unit in demand_units
    ]
    if exchange_unit is None:
        return demand_bids, None

    try:
        exchange_schedule = exchange_profiles[delivery_start]
    except KeyError as exc:
        raise InputValidationError(
            "exchanges_df.csv has no complete hourly profile for "
            f"{delivery_start.isoformat(sep=' ', timespec='minutes')}."
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
        )
    )
    return demand_bids, exchange_schedule


def _exchange_import_offer(
    exchange_unit: ExchangeUnit,
    exchange_schedule: ExchangeSchedule,
    delivery_start: datetime,
    delivery_end: datetime,
) -> SupplyOffer:
    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
    return SupplyOffer(
        unit_name=exchange_unit.name,
        operator=exchange_unit.operator,
        technology="exchange",
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        offered_power_mw=exchange_schedule.import_power_mw,
        offered_energy_mwh=exchange_schedule.import_power_mw * duration_hours,
        bid_price_eur_per_mwh=exchange_unit.price_import_eur_per_mwh,
        # Import price is a bid priority, not an external procurement cost.
        marginal_cost_eur_per_mwh=0.0,
        offer_type="import",
    )


def _available_power(
    plant: PowerPlant,
    delivery_start: datetime,
    availability_profiles: dict[str, dict[datetime, float]],
) -> float:
    try:
        return available_power_mw(plant, delivery_start, availability_profiles)
    except KeyError as exc:
        raise InputValidationError(str(exc)) from exc


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
) -> dict[datetime, float]:
    """Calculate the internal 13-point-naïve EOM price forecast required by V2."""

    forecast_starts = {
        delivery_start + timedelta(hours=hour)
        for delivery_start, _ in products
        for hour in range(13)
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
                    exchange_unit,
                    exchange_schedule,
                    delivery_start,
                    delivery_end,
                )
            )
        prices[delivery_start] = clear_pay_as_clear(
            demand_bids, offers
        ).clearing_price_eur_per_mwh
    return prices


def _validate_offer_prices(
    offers: list[SupplyOffer], settings: MarketSettings
) -> None:
    for offer in offers:
        if not (
            settings.minimum_bid_price
            <= offer.bid_price_eur_per_mwh
            <= settings.maximum_bid_price
        ):
            raise InputValidationError(
                f"Bid price for {offer.identifier!r} "
                f"({offer.bid_price_eur_per_mwh:.6f} EUR/MWh) is outside "
                "the configured bid-price limits."
            )


def _apply_startup_costs(
    market_result: MarketClearingResult,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, PlantRuntimeState],
) -> MarketClearingResult:
    """Attach one complete startup cost to each formerly-off started unit."""

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
        if cleared.offer.offer_segment == "inflexible"
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
                offer.offer_segment == "inflexible"
                or offer.unit_name not in has_inflexible_offer
            )
        ):
            startup_cost = plants_by_name[offer.unit_name].start_cost_eur
            applied_names.add(offer.unit_name)
        cleared_offers.append(
            ClearedSupplyOffer(
                offer=offer,
                accepted_energy_mwh=cleared.accepted_energy_mwh,
                clearing_price_eur_per_mwh=cleared.clearing_price_eur_per_mwh,
                startup_cost_eur=startup_cost,
            )
        )
    return MarketClearingResult(
        delivery_start=market_result.delivery_start,
        delivery_end=market_result.delivery_end,
        requested_demand_mwh=market_result.requested_demand_mwh,
        cleared_energy_mwh=market_result.cleared_energy_mwh,
        unserved_load_mwh=market_result.unserved_load_mwh,
        unfulfilled_export_mwh=market_result.unfulfilled_export_mwh,
        clearing_price_eur_per_mwh=market_result.clearing_price_eur_per_mwh,
        offers=tuple(cleared_offers),
        demand_bids=market_result.demand_bids,
        marginal_unit_name=market_result.marginal_unit_name,
    )


def _record_runtime_states(
    market_result: MarketClearingResult,
    runtime_states: dict[str, PlantRuntimeState],
) -> None:
    accepted_by_plant: dict[str, float] = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.unit_name in runtime_states:
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    for plant_name, state in runtime_states.items():
        state.record_dispatch(
            accepted_by_plant[plant_name] / market_result.duration_hours,
            market_result.duration_hours,
        )


def simulate(
    input_dir: str | Path,
    scenario: str = "base",
) -> SimulationResult:
    """Calculate one V1- or V2-compatible scenario without writing files."""

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    plants = load_powerplants(input_path / "powerplant_units.csv")
    demand_units = load_demand_units(input_path / "demand_units.csv")
    fuel_prices = load_fuel_prices(input_path / "fuel_prices_df.csv")
    validate_fuel_coverage(plants, fuel_prices)
    demand_profiles = load_hourly_demand_profiles(
        input_path / "demand_df.csv", demand_units
    )
    availability_profiles = load_hourly_availability_profiles(
        input_path / "availability_df.csv", plants
    )

    exchange_unit = None
    exchange_profiles: dict[datetime, ExchangeSchedule] = {}
    if settings.exchange_units_file is not None:
        exchange_unit = load_exchange_unit(input_path / settings.exchange_units_file)
        existing_names = {plant.name for plant in plants} | {
            demand_unit.name for demand_unit in demand_units
        }
        if exchange_unit.name in existing_names:
            raise InputValidationError(
                f"Exchange name {exchange_unit.name!r} must not duplicate another unit name."
            )
        for label, price in (
            ("import", exchange_unit.price_import_eur_per_mwh),
            ("export", exchange_unit.price_export_eur_per_mwh),
        ):
            if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
                raise InputValidationError(
                    f"Exchange {label} price ({price:.6f} EUR/MWh) is outside "
                    "the configured bid-price limits."
                )
        exchange_profiles = load_hourly_exchange_profiles(
            input_path / "exchanges_df.csv", exchange_unit
        )

    marginal_costs = {plant.name: plant.marginal_cost(fuel_prices) for plant in plants}
    for plant in plants:
        marginal_cost = marginal_costs[plant.name]
        if not settings.minimum_bid_price <= marginal_cost <= settings.maximum_bid_price:
            raise InputValidationError(
                f"Marginal cost for {plant.name!r} ({marginal_cost:.6f} EUR/MWh) "
                "is outside the configured bid-price limits."
            )

    products = delivery_products(settings)
    heuristic_plants = tuple(
        plant
        for plant in plants
        if plant.bidding_strategy == "powerplant_energy_heuristic_flexable"
    )
    price_forecasts: dict[datetime, float] = {}
    if heuristic_plants:
        price_forecasts = _naive_price_forecasts(
            products=products,
            settings=settings,
            plants=plants,
            marginal_costs=marginal_costs,
            availability_profiles=availability_profiles,
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
        )
    runtime_states = {
        plant.name: PlantRuntimeState.initially_off(plant) for plant in heuristic_plants
    }

    results: list[MarketClearingResult] = []
    for delivery_start, delivery_end in products:
        demand_bids, exchange_schedule = _demand_bids_for_product(
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            maximum_bid_price=settings.maximum_bid_price,
        )
        offers: list[SupplyOffer] = []
        for plant in plants:
            available_power = _available_power(
                plant, delivery_start, availability_profiles
            )
            marginal_cost = marginal_costs[plant.name]
            if plant.bidding_strategy == "powerplant_energy_naive":
                offers.append(
                    naive_offer(
                        plant,
                        delivery_start,
                        delivery_end,
                        available_power,
                        marginal_cost,
                    )
                )
            else:
                offers.extend(
                    heuristic_flexible_offers(
                        plant,
                        runtime_states[plant.name],
                        delivery_start,
                        delivery_end,
                        available_power,
                        marginal_cost,
                        price_forecasts,
                    )
                )
        if exchange_unit is not None and exchange_schedule is not None:
            offers.append(
                _exchange_import_offer(
                    exchange_unit,
                    exchange_schedule,
                    delivery_start,
                    delivery_end,
                )
            )
        _validate_offer_prices(offers, settings)
        market_result = clear_pay_as_clear(demand_bids, offers)
        market_result = _apply_startup_costs(market_result, plants, runtime_states)
        _record_runtime_states(market_result, runtime_states)
        results.append(market_result)

    return SimulationResult(settings=settings, market_results=tuple(results))


def run_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
) -> SimulationResult:
    """Run a simulation and persist the CSV results before optional plotting."""

    result = simulate(input_dir=input_dir, scenario=scenario)
    write_results(Path(output_dir), result)
    return result
