"""V3 household optimization and divisible pay-as-bid market simulation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..bidding import naive_offer
from ..clearing import clear_pay_as_bid, validate_demand_prices, validate_offer_prices
from ..errors import InputValidationError
from ..household import (
    dispatch_household,
    evaluate_household_flexibility,
    optimize_household,
)
from ..inputs import (
    load_demand_units,
    load_exact_availability_profiles,
    load_fuel_price_profiles,
    load_household_units,
    load_powerplants,
    load_time_series_profiles,
)
from ..market_models import DemandBid, MarketClearingResult, SupplyOffer
from ..models import MarketOpening, MarketSettings, SimulationResult


_POWER_TOLERANCE_MW = 1e-9


def _timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


def simulate_household_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
) -> SimulationResult:
    """Run the V3 15-minute household and pay-as-bid trading path."""

    plants = load_powerplants(input_path / "powerplant_units.csv")
    unsupported_plants = [
        plant.name
        for plant in plants
        if plant.bidding_strategy != "powerplant_energy_naive"
    ]
    if unsupported_plants:
        raise InputValidationError(
            "pay_as_bid currently supports divisible powerplant_energy_naive "
            "offers only: "
            + ", ".join(unsupported_plants)
            + "."
        )
    demand_units = load_demand_units(input_path / "demand_units.csv")
    if any(unit.is_elastic for unit in demand_units):
        raise InputValidationError(
            "pay_as_bid household scenarios do not yet support elastic demand units."
        )
    households = load_household_units(input_path / "residential_dsm_units.csv")
    if not households:
        raise InputValidationError(
            "pay_as_bid requires residential_dsm_units.csv with at least one household."
        )
    participant_names = [
        *(plant.name for plant in plants),
        *(unit.name for unit in demand_units),
        *(household.name for household in households),
    ]
    duplicates = sorted(
        name for name in set(participant_names) if participant_names.count(name) > 1
    )
    if duplicates:
        raise InputValidationError(
            "Participant names must be unique across plants, demand, and households: "
            + ", ".join(duplicates)
            + "."
        )

    demand_profiles = load_time_series_profiles(
        input_path / "demand_df.csv",
        tuple(
            unit.profile_column
            for unit in demand_units
            if unit.profile_column is not None
        ),
        warn_unused_columns=True,
    )
    forecast_columns = ("price_EOM",) + tuple(
        f"{household.name}_heat_demand" for household in households
    )
    forecasts = load_time_series_profiles(
        input_path / "forecasts_df.csv", forecast_columns
    )
    fuel_prices = load_fuel_price_profiles(input_path / "fuel_prices_df.csv")
    availability = load_exact_availability_profiles(
        input_path / "availability_df.csv", plants
    )
    available_fuels = set(next(iter(fuel_prices.values()))) if fuel_prices else set()
    missing_fuels = sorted(
        {
            plant.fuel_type
            for plant in plants
            if plant.fuel_type != "renewable"
        }
        - available_fuels
    )
    if missing_fuels:
        raise InputValidationError(
            "fuel_prices_df.csv is missing prices for: " + ", ".join(missing_fuels)
        )

    household_energy = {
        household.name: household.initial_energy_mwh for household in households
    }
    household_heat_pump_power = {household.name: 0.0 for household in households}
    household_charge_power = {household.name: 0.0 for household in households}
    household_discharge_power = {household.name: 0.0 for household in households}
    market_results: list[MarketClearingResult] = []
    household_results = []
    household_flexibility_results = []
    cleared_products: set[tuple[datetime, datetime]] = set()

    for opening in openings:
        if any(product in cleared_products for product in opening.products):
            raise ValueError("A delivery product cannot be quoted in two openings.")

        plans_by_household = {
            household.name: optimize_household(
                household,
                opening.products,
                forecasts["price_EOM"],
                forecasts[f"{household.name}_heat_demand"],
                household_energy[household.name],
                initial_heat_pump_power_mw=household_heat_pump_power[household.name],
                initial_battery_charge_power_mw=household_charge_power[household.name],
                initial_battery_discharge_power_mw=household_discharge_power[
                    household.name
                ],
            )
            for household in households
        }
        flexibility_by_household = {
            household.name: evaluate_household_flexibility(
                household,
                opening.products,
                forecasts["price_EOM"],
                forecasts[f"{household.name}_heat_demand"],
                household_energy[household.name],
                plans_by_household[household.name],
                initial_heat_pump_power_mw=household_heat_pump_power[household.name],
                initial_battery_charge_power_mw=household_charge_power[household.name],
                initial_battery_discharge_power_mw=household_discharge_power[
                    household.name
                ],
            )
            for household in households
        }
        for household in households:
            household_flexibility_results.extend(
                flexibility_by_household[household.name]
            )
        plan_by_product = {
            (plan.unit_name, plan.delivery_start): plan
            for plans in plans_by_household.values()
            for plan in plans
        }

        opening_results: list[MarketClearingResult] = []
        for delivery_start, delivery_end in opening.products:
            duration_hours = (
                delivery_end - delivery_start
            ).total_seconds() / 3600
            demand_bids: list[DemandBid] = []
            for unit in demand_units:
                assert unit.profile_column is not None
                try:
                    power = demand_profiles[unit.profile_column][delivery_start]
                except KeyError as exc:
                    raise InputValidationError(
                        f"demand_df.csv is missing {unit.profile_column!r} at "
                        f"{_timestamp(delivery_start)}."
                    ) from exc
                if power < 0:
                    raise InputValidationError("Demand power cannot be negative.")
                demand_bids.append(
                    DemandBid(
                        unit_name=unit.name,
                        operator=unit.operator,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        volume_mwh=power * duration_hours,
                        price_eur_per_mwh=(
                            unit.price_eur_per_mwh
                            if unit.price_eur_per_mwh is not None
                            else settings.maximum_bid_price
                        ),
                        bid_id=f"{unit.name}::load::{delivery_start.isoformat()}",
                        demand_type="inelastic_load",
                    )
                )

            for household in households:
                plan = plan_by_product[(household.name, delivery_start)]
                volume = plan.planned_grid_power_mw * duration_hours
                if volume <= _POWER_TOLERANCE_MW:
                    continue
                demand_bids.append(
                    DemandBid(
                        unit_name=household.name,
                        operator=household.operator,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        volume_mwh=volume,
                        price_eur_per_mwh=household.bid_price_eur_per_mwh,
                        bid_id=(
                            f"{household.name}::household::{delivery_start.isoformat()}"
                        ),
                        demand_type="household_load",
                    )
                )

            try:
                product_fuel_prices = fuel_prices[delivery_start]
            except KeyError as exc:
                raise InputValidationError(
                    f"fuel_prices_df.csv is missing {_timestamp(delivery_start)}."
                ) from exc
            offers: list[SupplyOffer] = []
            for plant in plants:
                plant_profile = availability.get(plant.name)
                if plant_profile is None:
                    available_power = plant.max_power_mw
                else:
                    try:
                        available_power = (
                            plant.max_power_mw * plant_profile[delivery_start]
                        )
                    except KeyError as exc:
                        raise InputValidationError(
                            f"availability_df.csv is missing {plant.name!r} at "
                            f"{_timestamp(delivery_start)}."
                        ) from exc
                marginal_cost = plant.marginal_cost(product_fuel_prices)
                offers.append(
                    naive_offer(
                        plant,
                        delivery_start,
                        delivery_end,
                        available_power,
                        marginal_cost,
                    )
                )

            validate_demand_prices(demand_bids, settings)
            validate_offer_prices(offers, settings)
            result = clear_pay_as_bid(demand_bids, offers)
            opening_results.append(
                replace(result, opening_time=opening.opening_time)
            )

        results_by_start = {
            result.delivery_start: result for result in opening_results
        }
        for household in households:
            accepted_grid_energy: dict[datetime, float] = {}
            for delivery_start, _ in opening.products:
                cleared_bid = next(
                    (
                        cleared
                        for cleared in results_by_start[delivery_start].demand_bids
                        if cleared.bid.unit_name == household.name
                        and cleared.bid.demand_type == "household_load"
                    ),
                    None,
                )
                accepted_grid_energy[delivery_start] = (
                    0.0 if cleared_bid is None else cleared_bid.accepted_energy_mwh
                )
            dispatch = dispatch_household(
                household,
                opening.opening_time,
                plans_by_household[household.name],
                accepted_grid_energy,
                household_energy[household.name],
                initial_heat_pump_power_mw=household_heat_pump_power[household.name],
                initial_battery_charge_power_mw=household_charge_power[household.name],
                initial_battery_discharge_power_mw=household_discharge_power[
                    household.name
                ],
            )
            household_results.extend(dispatch)
            if dispatch:
                household_energy[household.name] = (
                    dispatch[-1].soc_after * household.battery_capacity_mwh
                )
                household_heat_pump_power[household.name] = dispatch[
                    -1
                ].heat_pump_power_mw
                household_charge_power[household.name] = dispatch[
                    -1
                ].battery_charge_power_mw
                household_discharge_power[household.name] = dispatch[
                    -1
                ].battery_discharge_power_mw

        market_results.extend(opening_results)
        cleared_products.update(opening.products)

    return SimulationResult(
        settings=settings,
        market_results=tuple(
            sorted(market_results, key=lambda result: result.delivery_start)
        ),
        household_results=tuple(
            sorted(
                household_results,
                key=lambda result: (result.delivery_start, result.unit_name),
            )
        ),
        household_flexibility_results=tuple(
            sorted(
                household_flexibility_results,
                key=lambda result: (result.delivery_start, result.unit_name),
            )
        ),
    )
