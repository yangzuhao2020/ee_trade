"""Storage bids, projected commitments and delivered SOC bookkeeping."""

from __future__ import annotations

from datetime import datetime

from ...bidding import StorageRuntimeState, storage_heuristic_orders
from ...errors import InputValidationError
from ...market_models import DemandBid, MarketClearingResult, SupplyOffer
from ...models import (
    MarketOpening, MarketSettings, StorageClearingContext,
    StorageDispatchResult, StorageUnit,
)
from ...time_utils import format_timestamp

_POWER_TOLERANCE_MW = 1e-9


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
                f"its SOC bounds before {format_timestamp(first_delivery)}."
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
