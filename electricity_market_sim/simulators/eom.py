"""Shared EOM market engine for conventional and extended simulations."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Protocol

from ..bidding import (
    PlantRuntimeState,
    StorageRuntimeState,
    available_power_mw,
    heuristic_block_offers,
    heuristic_flexible_offers,
    naive_offer,
    storage_heuristic_orders,
)
from ..clearing import (
    clear_complex_opening,
    clear_pay_as_clear,
    validate_demand_prices,
    validate_offer_prices,
)
from ..errors import InputValidationError
from ..inputs import (
    load_demand_units,
    load_exchange_unit,
    load_fuel_prices,
    load_hourly_availability_profiles,
    load_hourly_demand_profiles,
    load_hourly_exchange_profiles,
    load_powerplants,
    load_storage_units,
    validate_fuel_coverage,
)
from ..learning import (
    ActionProvider,
    LearningEpisodeSession,
    TransitionConsumer,
    calculate_load_base,
)
from ..market_models import (
    ClearedSupplyOffer,
    DemandBid,
    MarketClearingResult,
    SupplyOffer,
)
from ..models import (
    ExchangeSchedule,
    ExchangeUnit,
    MarketOpening,
    MarketSettings,
    PowerPlant,
    SimulationResult,
    StorageClearingContext,
    StorageDispatchResult,
    StorageUnit,
)

_POWER_TOLERANCE_MW = 1e-9
_FORECAST_HOURS = 12
_LINKED_REQUIRED_FIELDS = frozenset(
    {"bid_type", "min_acceptance_ratio", "parent_bid_id"}
)
ScheduledProductKey = tuple[datetime, datetime, datetime]


@dataclass(frozen=True)
class EomExtensionInputs:
    """Additional market inputs supplied by an optional simulator extension."""

    participant_names: tuple[str, ...]
    fuel_prices: dict[str, float]
    fuel_price_profiles: dict[datetime, dict[str, float]] | None = None
    market_price_forecasts: dict[datetime, float] | None = None


class EomMarketExtension(Protocol):
    """Hooks used to add participants without duplicating the EOM event loop."""

    def prepare(self, *, requires_market_price_forecast: bool) -> EomExtensionInputs:
        """Load extension inputs after the base market participants are known."""

    def initialize(self, openings: tuple[MarketOpening, ...]) -> None:
        """Initialize state after incomplete market openings have been removed."""

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[DemandBid, ...]:
        """Return extra demand bids for one opening."""

    def record_delivery(self, result: MarketClearingResult) -> None:
        """Commit a completed delivery to the extension state."""

    def finalize(self, result: SimulationResult) -> SimulationResult:
        """Validate and attach extension-specific output to the result."""


def _timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


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
            f"{_timestamp(delivery_start)} and unit {unit_name!r}."
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
    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
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

    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
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
            "exchanges_df.csv has no complete hourly profile for "
            f"{_timestamp(delivery_start)}."
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
    duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
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
    requires_learning_forecast: bool = False,
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
                f"{_timestamp(start)}–{_timestamp(end)}" for start, end in beyond_end
            )
        )
    forecast_times: set[datetime] = set()
    if requires_price_forecast:
        forecast_times |= {
            delivery_start + timedelta(hours=offset)
            for delivery_start in delivery_times
            for offset in range(_FORECAST_HOURS + 1)
        }
    if requires_learning_forecast:
        forecast_times |= {
            delivery_start + timedelta(hours=offset)
            for delivery_start in delivery_times
            for offset in range(_FORECAST_HOURS)
        }
    if requires_storage_price_forecast:
        forecast_times |= {
            delivery_start + timedelta(hours=offset)
            for delivery_start in delivery_times
            for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1)
            if settings.start
            <= delivery_start + timedelta(hours=offset)
            <= settings.end
        }
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
            f"{kind}缺少 {source}: " + ", ".join(_timestamp(time) for time in missing)
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
        pricing_method=market_result.pricing_method,
        opening_time=market_result.opening_time,
        trades=market_result.trades,
    )


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
            accepted_by_plant[plant_name] / market_result.duration_hours,
            market_result.duration_hours,
        )


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


def _learning_plant(
    settings: MarketSettings, plants: tuple[PowerPlant, ...]
) -> PowerPlant | None:
    """Validate and return Version 5's sole learning participant."""

    learning_plants = tuple(
        plant
        for plant in plants
        if plant.bidding_strategy == "powerplant_energy_learning"
    )
    if not learning_plants:
        if (
            settings.learning_config is not None
            and settings.learning_config.learning_mode
        ):
            raise InputValidationError(
                "learning_mode requires exactly one powerplant_energy_learning unit."
            )
        return None
    if len(learning_plants) != 1:
        raise InputValidationError(
            "Version 5 requires exactly one powerplant_energy_learning unit."
        )
    if settings.learning_config is None or not settings.learning_config.learning_mode:
        raise InputValidationError(
            "powerplant_energy_learning requires learning_config.learning_mode: true."
        )
    if settings.product_count != 1 or settings.product_duration != timedelta(hours=1):
        raise InputValidationError(
            "Version 5 requires one 1-hour product per EOM opening."
        )
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "Version 5 requires EOM market_mechanism: pay_as_clear."
        )
    max_bid_price = settings.learning_config.max_bid_price
    if (
        settings.minimum_bid_price > -max_bid_price
        or settings.maximum_bid_price < max_bid_price
    ):
        raise InputValidationError(
            "The EOM price limits must contain the learning action range "
            f"[-{max_bid_price:g}, {max_bid_price:g}]."
        )
    return learning_plants[0]


def _residual_load_forecasts(
    *,
    forecast_times: set[datetime],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    plants: tuple[PowerPlant, ...],
    availability_profiles: dict[str, dict[datetime, float]],
) -> dict[datetime, float]:
    """Return demand minus forecast wind/solar production for each hour."""

    variable_renewables = tuple(
        plant
        for plant in plants
        if any(label in plant.technology.lower() for label in ("wind", "solar"))
    )
    residual: dict[datetime, float] = {}
    for forecast_time in sorted(forecast_times):
        demand_mw = sum(
            _profile_power(demand_profiles, unit.name, forecast_time, "demand_df.csv")
            for unit in demand_units
            if not unit.is_elastic
        )
        renewable_mw = sum(
            _available_power(plant, forecast_time, availability_profiles)
            for plant in variable_renewables
        )
        residual[forecast_time] = demand_mw - renewable_mw
    return residual


def _validate_storage_configuration(
    settings: MarketSettings,
    storages: tuple[StorageUnit, ...],
) -> None:
    """Validate the clearing assumptions used by the storage implementation."""

    if not storages:
        return

    product_horizon = settings.product_duration * settings.product_count
    if settings.opening_frequency < product_horizon:
        raise InputValidationError(
            "Storage market openings must not overlap: EOM opening_frequency "
            f"({settings.opening_frequency}) must be at least the complete "
            f"product horizon ({product_horizon}). Overlapping storage "
            "commitments are not supported yet."
        )

    if settings.product_count > 1 and settings.market_mechanism != "complex_clearing":
        raise InputValidationError(
            "Storage participating in a multi-product opening requires "
            "market_mechanism: complex_clearing so accepted quantities obey "
            "cross-period SOC constraints."
        )
    if settings.market_mechanism == "complex_clearing" and any(
        storage.natural_inflow_mw > _POWER_TOLERANCE_MW for storage in storages
    ):
        raise InputValidationError(
            "Non-zero storage natural_inflow is currently supported for "
            "single-product pay_as_clear openings only."
        )


def _storage_market_quantities(
    market_result: MarketClearingResult,
    storage_name: str,
) -> tuple[float, float, float | None, float, float, float | None]:
    """Return offered/accepted charge and discharge data for one product."""

    charge_orders = [
        cleared
        for cleared in market_result.demand_bids
        if cleared.bid.unit_name == storage_name
        and cleared.bid.demand_type == "storage_charge"
    ]
    discharge_orders = [
        cleared
        for cleared in market_result.offers
        if cleared.offer.unit_name == storage_name
        and cleared.offer.offer_type == "storage_discharge"
    ]
    offered_charge = sum(item.bid.volume_mwh for item in charge_orders)
    accepted_charge = sum(item.accepted_energy_mwh for item in charge_orders)
    offered_discharge = sum(item.offer.offered_energy_mwh for item in discharge_orders)
    accepted_discharge = sum(item.accepted_energy_mwh for item in discharge_orders)
    charge_price = charge_orders[0].bid.price_eur_per_mwh if charge_orders else None
    discharge_price = (
        discharge_orders[0].offer.bid_price_eur_per_mwh if discharge_orders else None
    )
    return (
        offered_charge,
        accepted_charge,
        charge_price,
        offered_discharge,
        accepted_discharge,
        discharge_price,
    )


def _project_storage_energies(
    *,
    opening: MarketOpening,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
    scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
) -> dict[str, float]:
    """Project delivered state through accepted commitments before first delivery."""

    first_delivery = opening.products[0][0]
    pending_before_delivery = sorted(
        (
            result
            for result in scheduled_results.values()
            if result.delivery_end > opening.opening_time
            and result.delivery_end <= first_delivery
        ),
        key=lambda result: (result.delivery_end, result.delivery_start),
    )
    projected: dict[str, float] = {}
    for storage in storages:
        energy = storage_states[storage.name].energy_mwh
        for market_result in pending_before_delivery:
            (
                _,
                accepted_charge,
                _,
                _,
                accepted_discharge,
                _,
            ) = _storage_market_quantities(market_result, storage.name)
            energy = min(
                storage.max_energy_mwh,
                energy + storage.natural_inflow_mw * market_result.duration_hours,
            )
            energy += accepted_charge * storage.efficiency_charge
            energy -= accepted_discharge / storage.efficiency_discharge
        tolerance = 1e-7
        if not (
            storage.min_energy_mwh - tolerance
            <= energy
            <= storage.max_energy_mwh + tolerance
        ):
            raise InputValidationError(
                f"Accepted commitments project storage {storage.name!r} outside "
                f"its SOC bounds before {_timestamp(first_delivery)}."
            )
        projected[storage.name] = min(
            storage.max_energy_mwh, max(storage.min_energy_mwh, energy)
        )
    return projected


def _storage_orders_for_opening(
    *,
    opening: MarketOpening,
    storages: tuple[StorageUnit, ...],
    initial_energies_mwh: dict[str, float],
    price_forecasts: dict[datetime, float],
    forecast_start: datetime,
    forecast_end: datetime,
) -> tuple[
    list[DemandBid],
    list[SupplyOffer],
    tuple[StorageClearingContext, ...],
]:
    """Generate storage orders and matching physical clearing contexts."""

    charge_bids: list[DemandBid] = []
    discharge_offers: list[SupplyOffer] = []
    contexts: list[StorageClearingContext] = []
    for storage in storages:
        initial_energy = initial_energies_mwh[storage.name]
        charges, discharges = storage_heuristic_orders(
            storage,
            initial_energy,
            opening.products,
            price_forecasts,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        charge_bids.extend(charges)
        discharge_offers.extend(discharges)
        contexts.append(
            StorageClearingContext(
                unit_name=storage.name,
                initial_energy_mwh=initial_energy,
                min_energy_mwh=storage.min_energy_mwh,
                max_energy_mwh=storage.max_energy_mwh,
                efficiency_charge=storage.efficiency_charge,
                efficiency_discharge=storage.efficiency_discharge,
            )
        )
    return charge_bids, discharge_offers, tuple(contexts)


def _record_storage_states(
    market_result: MarketClearingResult,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
) -> list[StorageDispatchResult]:
    """Apply delivered storage dispatch and produce auditable SOC records."""

    opening_time = market_result.opening_time or market_result.delivery_start
    records: list[StorageDispatchResult] = []
    for storage in storages:
        (
            offered_charge,
            accepted_charge,
            charge_price,
            offered_discharge,
            accepted_discharge,
            discharge_price,
        ) = _storage_market_quantities(market_result, storage.name)
        before, after = storage_states[storage.name].record_dispatch(
            storage,
            accepted_charge,
            accepted_discharge,
            market_result.duration_hours,
        )
        records.append(
            StorageDispatchResult(
                opening_time=opening_time,
                delivery_start=market_result.delivery_start,
                delivery_end=market_result.delivery_end,
                unit_name=storage.name,
                operator=storage.operator,
                technology=storage.technology,
                energy_before_mwh=before,
                energy_after_mwh=after,
                capacity_mwh=storage.capacity_mwh,
                offered_charge_mwh=offered_charge,
                accepted_charge_mwh=accepted_charge,
                charge_bid_price_eur_per_mwh=charge_price,
                offered_discharge_mwh=offered_discharge,
                accepted_discharge_mwh=accepted_discharge,
                discharge_bid_price_eur_per_mwh=discharge_price,
                clearing_price_eur_per_mwh=(market_result.clearing_price_eur_per_mwh),
                charge_payment_eur=(
                    accepted_charge * market_result.clearing_price_eur_per_mwh
                ),
                discharge_revenue_eur=(
                    accepted_discharge * market_result.clearing_price_eur_per_mwh
                ),
                additional_charge_cost_eur=(
                    accepted_charge * storage.additional_cost_charge_eur_per_mwh
                ),
                additional_discharge_cost_eur=(
                    accepted_discharge * storage.additional_cost_discharge_eur_per_mwh
                ),
            )
        )
    return records


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
    learning_session: LearningEpisodeSession | None = None,
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
                    f"{_timestamp(delivery_start)}."
                ) from exc
            return current_plant.marginal_cost(prices)

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
        elif plant.bidding_strategy == "powerplant_energy_learning":
            if learning_session is None or scheduled_results is None:
                raise InputValidationError(
                    "powerplant_energy_learning requires a learning episode policy."
                )
            for start, end in opening.products:
                offers.extend(
                    learning_session.offers_for_product(
                        delivery_start=start,
                        delivery_end=end,
                        available_power_mw=available_powers[start],
                        marginal_cost_eur_per_mwh=marginal_cost_at(start),
                        scheduled_results=scheduled_results,
                    )
                )
        else:
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


def _clear_opening(
    *,
    opening: MarketOpening,
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, PlantRuntimeState],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    availability_profiles: dict[str, dict[datetime, float]],
    marginal_costs: dict[str, float],
    fuel_price_profiles: dict[datetime, dict[str, float]] | None,
    price_forecasts: dict[datetime, float],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    storages: tuple[StorageUnit, ...] = (),
    storage_initial_energies_mwh: dict[str, float] | None = None,
    additional_demand_bids: tuple[DemandBid, ...] = (),
    learning_session: LearningEpisodeSession | None = None,
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] | None = None,
) -> tuple[MarketClearingResult, ...]:
    """Quote and clear one opening from its gate-closure state snapshot."""

    all_demand_bids: list[DemandBid] = []
    exchange_schedules: dict[datetime, ExchangeSchedule | None] = {}
    for delivery_start, delivery_end in opening.products:
        demand_bids, schedule = _demand_bids_for_product(
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            maximum_bid_price=settings.maximum_bid_price,
        )
        all_demand_bids.extend(demand_bids)
        exchange_schedules[delivery_start] = schedule
    all_demand_bids.extend(additional_demand_bids)

    offers = _offers_for_opening(
        opening=opening,
        plants=plants,
        runtime_states=runtime_states,
        availability_profiles=availability_profiles,
        marginal_costs=marginal_costs,
        fuel_price_profiles=fuel_price_profiles,
        price_forecasts=price_forecasts,
        exchange_unit=exchange_unit,
        exchange_schedules=exchange_schedules,
        learning_session=learning_session,
        scheduled_results=scheduled_results,
    )
    storage_contexts: tuple[StorageClearingContext, ...] = ()
    if storages:
        if storage_initial_energies_mwh is None:
            raise ValueError("Storage initial energies are required for storage bids.")
        storage_bids, storage_offers, storage_contexts = _storage_orders_for_opening(
            opening=opening,
            storages=storages,
            initial_energies_mwh=storage_initial_energies_mwh,
            price_forecasts=price_forecasts,
            forecast_start=settings.start,
            forecast_end=settings.end,
        )
        all_demand_bids.extend(storage_bids)
        offers.extend(storage_offers)

    validate_demand_prices(all_demand_bids, settings)
    validate_offer_prices(offers, settings)
    if settings.market_mechanism == "complex_clearing":
        return clear_complex_opening(
            all_demand_bids,
            offers,
            storage_contexts=storage_contexts,
        )
    if any(offer.bid_type != "SB" for offer in offers):
        raise InputValidationError(
            "Block and linked EOM strategies require market_mechanism: "
            "complex_clearing."
        )
    return tuple(
        clear_pay_as_clear(
            [
                bid
                for bid in all_demand_bids
                if (bid.delivery_start, bid.delivery_end) == (start, end)
            ],
            [
                offer
                for offer in offers
                if (offer.delivery_start, offer.delivery_end) == (start, end)
            ],
        )
        for start, end in opening.products
    )


def simulate_eom_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
    *,
    extension: EomMarketExtension | None = None,
    learning_action_provider: ActionProvider | None = None,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_load_base_mw: float | None = None,
    learning_enforce_action_bounds: bool = True,
) -> SimulationResult:
    """Run the shared EOM event loop with an optional participant extension."""

    plants = load_powerplants(input_path / "powerplant_units.csv")
    _validate_strategy_configuration(settings, plants)
    learning_plant = _learning_plant(settings, plants)
    if learning_plant is not None and learning_action_provider is None:
        raise InputValidationError(
            "A learning_action_provider is required for a Version 5 market episode."
        )
    provided_load_base: float | None = None
    if learning_load_base_mw is not None:
        try:
            provided_load_base = float(learning_load_base_mw)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "The saved learning load base must be numeric."
            ) from exc
        if not isfinite(provided_load_base) or provided_load_base <= 0:
            raise InputValidationError(
                "The saved learning load base must be a positive finite value."
            )
        if learning_plant is None:
            raise InputValidationError(
                "A saved learning load base requires a learning plant."
            )
    storages = load_storage_units(input_path / "storage_units.csv")
    _validate_storage_configuration(settings, storages)
    demand_units = load_demand_units(input_path / "demand_units.csv")
    runtime_plants = tuple(
        plant for plant in plants if plant.bidding_strategy != "powerplant_energy_naive"
    )
    requires_price_forecast = any(
        plant.bidding_strategy != "powerplant_energy_learning"
        and plant.min_power_mw > _POWER_TOLERANCE_MW
        for plant in runtime_plants
    )
    requires_learning_forecast = learning_plant is not None
    requires_storage_price_forecast = bool(storages)
    requires_market_price_forecast = (
        requires_price_forecast
        or requires_learning_forecast
        or requires_storage_price_forecast
    )
    extension_inputs = (
        extension.prepare(requires_market_price_forecast=requires_market_price_forecast)
        if extension is not None
        else EomExtensionInputs(
            participant_names=(),
            fuel_prices=load_fuel_prices(input_path / "fuel_prices_df.csv"),
        )
    )
    participant_names = [
        *(plant.name for plant in plants),
        *(storage.name for storage in storages),
        *(unit.name for unit in demand_units),
        *extension_inputs.participant_names,
    ]
    if len(participant_names) != len(set(participant_names)):
        duplicates = sorted(
            {name for name in participant_names if participant_names.count(name) > 1}
        )
        raise InputValidationError(
            "Participant names must be unique across all market units: "
            + ", ".join(duplicates)
            + "."
        )
    fuel_prices = extension_inputs.fuel_prices
    fuel_price_profiles = extension_inputs.fuel_price_profiles
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
        existing_names = (
            {plant.name for plant in plants}
            | {storage.name for storage in storages}
            | {unit.name for unit in demand_units}
            | set(extension_inputs.participant_names)
        )
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
                    f"Exchange {label} price ({price:.6f} EUR/MWh) is outside the configured bid-price limits."
                )
        exchange_profiles = load_hourly_exchange_profiles(
            input_path / "exchanges_df.csv", exchange_unit
        )

    marginal_costs = {plant.name: plant.marginal_cost(fuel_prices) for plant in plants}
    cost_times = (
        sorted({start for opening in openings for start, _ in opening.products})
        if fuel_price_profiles is not None
        else (None,)
    )
    for plant in plants:
        for cost_time in cost_times:
            if cost_time is None:
                marginal_cost = marginal_costs[plant.name]
            else:
                try:
                    marginal_cost = plant.marginal_cost(fuel_price_profiles[cost_time])
                except KeyError as exc:
                    raise InputValidationError(
                        "fuel_prices_df.csv has no complete product profile for "
                        f"{_timestamp(cost_time)}."
                    ) from exc
            if not (
                settings.minimum_bid_price
                <= marginal_cost
                <= settings.maximum_bid_price
            ):
                time_detail = (
                    "" if cost_time is None else f" at {_timestamp(cost_time)}"
                )
                raise InputValidationError(
                    f"Marginal cost for {plant.name!r}{time_detail} "
                    f"({marginal_cost:.6f} EUR/MWh) is outside the configured "
                    "bid-price limits."
                )

    valid_openings: list[MarketOpening] = []
    for opening in openings:
        issues = _opening_coverage_issues(
            opening=opening,
            settings=settings,
            plants=plants,
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            availability_profiles=availability_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
            requires_price_forecast=requires_price_forecast,
            requires_learning_forecast=requires_learning_forecast,
            requires_storage_price_forecast=requires_storage_price_forecast,
        )
        if issues:
            if learning_plant is not None:
                raise InputValidationError(
                    f"Learning opening {_timestamp(opening.opening_time)} is incomplete: "
                    + "; ".join(issues)
                )
            warnings.warn(
                f"跳过市场开放 {_timestamp(opening.opening_time)}："
                + "; ".join(issues),
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        valid_openings.append(opening)

    price_forecasts: dict[datetime, float] = {}
    if requires_market_price_forecast:
        if extension_inputs.market_price_forecasts is not None:
            price_forecasts = extension_inputs.market_price_forecasts
        else:
            forecast_starts: set[datetime] = set()
            for opening in valid_openings:
                for delivery_start, _ in opening.products:
                    if requires_price_forecast:
                        forecast_starts.update(
                            delivery_start + timedelta(hours=offset)
                            for offset in range(_FORECAST_HOURS + 1)
                        )
                    if requires_learning_forecast:
                        forecast_starts.update(
                            delivery_start + timedelta(hours=offset)
                            for offset in range(_FORECAST_HOURS)
                        )
                    if requires_storage_price_forecast:
                        forecast_starts.update(
                            delivery_start + timedelta(hours=offset)
                            for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1)
                            if settings.start
                            <= delivery_start + timedelta(hours=offset)
                            <= settings.end
                        )
            price_forecasts = _naive_price_forecasts(
                products=[
                    product
                    for opening in valid_openings
                    for product in opening.products
                ],
                settings=settings,
                plants=plants,
                marginal_costs=marginal_costs,
                availability_profiles=availability_profiles,
                demand_units=demand_units,
                demand_profiles=demand_profiles,
                exchange_unit=exchange_unit,
                exchange_profiles=exchange_profiles,
                forecast_starts=forecast_starts,
            )
    learning_session: LearningEpisodeSession | None = None
    if learning_plant is not None:
        training_times = sorted(
            {start for opening in valid_openings for start, _ in opening.products}
        )
        forecast_times = {
            start + timedelta(hours=offset)
            for start in training_times
            for offset in range(_FORECAST_HOURS)
        }
        residual_forecasts = _residual_load_forecasts(
            forecast_times=forecast_times,
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            plants=plants,
            availability_profiles=availability_profiles,
        )
        if provided_load_base is None:
            load_base = calculate_load_base(
                [residual_forecasts[start] for start in training_times]
            )
        else:
            load_base = provided_load_base
        assert learning_action_provider is not None
        assert settings.learning_config is not None
        learning_session = LearningEpisodeSession(
            plant=learning_plant,
            action_provider=learning_action_provider,
            residual_load_forecasts_mw=residual_forecasts,
            price_forecasts_eur_per_mwh=price_forecasts,
            load_base_mw=load_base,
            transition_consumer=learning_transition_consumer,
            enforce_action_bounds=learning_enforce_action_bounds,
            price_scale_eur_per_mwh=settings.learning_config.max_bid_price,
        )
    runtime_states = {
        plant.name: PlantRuntimeState.initially_off(plant) for plant in runtime_plants
    }
    storage_states = {
        storage.name: StorageRuntimeState.from_initial_soc(storage)
        for storage in storages
    }
    if extension is not None:
        extension.initialize(tuple(valid_openings))

    # Clearing is a forward market event; its products are not dispatched at
    # gate closure.  Keep cleared products on a physical-time queue so a later
    # opening observes only deliveries that have already ended.
    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in valid_openings:
        openings_by_time[opening.opening_time].append(opening)

    # Keys retain the opening identity in case different openings have products
    # with equal delivery timestamps.
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] = {}
    pending_starts: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    pending_ends: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    results: list[MarketClearingResult] = []
    storage_results: list[StorageDispatchResult] = []

    def finalize_product_starts(keys: list[ScheduledProductKey]) -> None:
        for key in keys:
            market_result = _apply_startup_costs(
                scheduled_results[key], plants, runtime_states
            )
            scheduled_results[key] = market_result
            results.append(market_result)
            if learning_session is not None:
                learning_session.record_result(market_result)

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])

        # A product ending at this instant belongs to the past for every market
        # that opens now.  Only these completed deliveries may affect its bids.
        for key in pending_ends.pop(event_time, []):
            _record_runtime_states(scheduled_results[key], runtime_states)
            storage_results.extend(
                _record_storage_states(scheduled_results[key], storages, storage_states)
            )
            if extension is not None:
                extension.record_delivery(scheduled_results[key])

        starting_keys = pending_starts.pop(event_time, [])
        if learning_session is not None:
            # Finalize the reward before requesting the next action.  Once the
            # next state is built, a trainer may update the Actor first.
            finalize_product_starts(starting_keys)

        for opening in openings_by_time.pop(event_time, []):
            additional_demand_bids = (
                extension.bids_for_products(opening.products, scheduled_results)
                if extension is not None
                else ()
            )
            storage_initial_energies = _project_storage_energies(
                opening=opening,
                storages=storages,
                storage_states=storage_states,
                scheduled_results=scheduled_results,
            )
            opening_results = _clear_opening(
                opening=opening,
                settings=settings,
                plants=plants,
                runtime_states=runtime_states,
                demand_units=demand_units,
                demand_profiles=demand_profiles,
                availability_profiles=availability_profiles,
                marginal_costs=marginal_costs,
                fuel_price_profiles=fuel_price_profiles,
                price_forecasts=price_forecasts,
                exchange_unit=exchange_unit,
                exchange_profiles=exchange_profiles,
                storages=storages,
                storage_initial_energies_mwh=storage_initial_energies,
                additional_demand_bids=additional_demand_bids,
                learning_session=learning_session,
                scheduled_results=scheduled_results,
            )
            for market_result in opening_results:
                market_result = replace(
                    market_result, opening_time=opening.opening_time
                )
                key = (
                    opening.opening_time,
                    market_result.delivery_start,
                    market_result.delivery_end,
                )
                scheduled_results[key] = market_result
                pending_starts[market_result.delivery_start].append(key)
                pending_ends[market_result.delivery_end].append(key)

        if learning_session is None:
            # Preserve the established V1--V4 event order.
            finalize_product_starts(starting_keys)

    learning_steps = ()
    learning_transitions = ()
    learning_load_base_mw = None
    if learning_session is not None:
        learning_steps, learning_transitions = learning_session.finalize()
        learning_load_base_mw = learning_session.load_base_mw

    result = SimulationResult(
        settings=settings,
        market_results=tuple(
            sorted(
                results,
                key=lambda result: (
                    result.delivery_start,
                    result.opening_time or result.delivery_start,
                ),
            )
        ),
        storage_results=tuple(
            sorted(
                storage_results,
                key=lambda result: (
                    result.delivery_start,
                    result.opening_time,
                    result.unit_name,
                ),
            )
        ),
        learning_steps=learning_steps,
        learning_transitions=learning_transitions,
        learning_load_base_mw=learning_load_base_mw,
    )
    return extension.finalize(result) if extension is not None else result
