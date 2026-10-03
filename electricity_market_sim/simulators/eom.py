"""Shared event engine: openings, delivery starts/ends and participant extensions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ..market_models import DemandBid, MarketClearingResult, SupplyOffer
from ..models import MarketOpening, PowerPlant, SimulationResult, StorageDispatchResult

if TYPE_CHECKING:
    from .base_market import BaseMarket

ScheduledProductKey = tuple[datetime, datetime, datetime]


@dataclass(frozen=True)
class EomExtensionInputs:
    """Extra inputs and forecast requirements declared before opening preflight."""

    participant_names: tuple[str, ...] = ()
    fuel_prices: dict[str, float] | None = None
    fuel_price_profiles: dict[datetime, dict[str, float]] | None = None
    market_price_forecasts: dict[datetime, float] | None = None
    forecast_offsets: tuple[int, ...] = ()
    incomplete_opening_error: str | None = None


class EomMarketExtension(Protocol):
    """Connect extra demand or replacement plant bids to the shared event engine."""

    settle_before_opening: bool

    def prepare(self, market: BaseMarket) -> EomExtensionInputs:
        """Inspect loaded conventional participants and declare extra inputs."""

    def initialize(self, market: BaseMarket) -> None:
        """Initialize after preflight, price forecasts and runtime states are ready."""

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[DemandBid, ...]:
        """Return extra demand orders for the opening."""

    def offers_for_plant(
        self,
        plant: PowerPlant,
        products: tuple[tuple[datetime, datetime], ...],
        available_powers: dict[datetime, float],
        marginal_cost_at: Callable[[datetime], float],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[SupplyOffer, ...] | None:
        """Replace one plant's bids; None leaves conventional quoting in charge."""

    def record_delivery_start(self, result: MarketClearingResult) -> None:
        """Receive the delivered product after its startup costs are attached."""

    def record_delivery(self, result: MarketClearingResult) -> None:
        """Commit an ended delivery to the extension's state."""

    def finalize(self, result: SimulationResult) -> SimulationResult:
        """Attach extension-specific results after all events have completed."""


def simulate_eom_market(market: BaseMarket) -> SimulationResult:
    """Run a prepared base market with the extension's declared event ordering."""

    extension = market.extension
    settle_before_opening = extension is not None and extension.settle_before_opening
    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening)

    # Preserve the opening identity even when delivery timestamps coincide.
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] = {}
    pending_starts: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    pending_ends: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    results: list[MarketClearingResult] = []
    storage_results: list[StorageDispatchResult] = []

    def finalize_product_starts(keys: list[ScheduledProductKey]) -> None:
        for key in keys:
            result = market.start_delivery(scheduled_results[key])
            scheduled_results[key] = result
            results.append(result)
            if extension is not None:
                extension.record_delivery_start(result)

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])

        # Only ended deliveries affect bids at the current opening.
        for key in pending_ends.pop(event_time, []):
            storage_results.extend(market.end_delivery(scheduled_results[key]))
            if extension is not None:
                extension.record_delivery(scheduled_results[key])

        starting_keys = pending_starts.pop(event_time, [])
        if settle_before_opening:
            finalize_product_starts(starting_keys)

        for opening in openings_by_time.pop(event_time, []):
            for result in market.clear_opening(opening, scheduled_results):
                result = replace(result, opening_time=opening.opening_time)
                key = (opening.opening_time, result.delivery_start, result.delivery_end)
                scheduled_results[key] = result
                pending_starts[result.delivery_start].append(key)
                pending_ends[result.delivery_end].append(key)

        if not settle_before_opening:
            finalize_product_starts(starting_keys)

    result = SimulationResult(
        settings=market.settings,
        market_results=tuple(sorted(
            results,
            key=lambda item: (item.delivery_start, item.opening_time or item.delivery_start),
        )),
        storage_results=tuple(sorted(
            storage_results,
            key=lambda item: (item.delivery_start, item.opening_time, item.unit_name),
        )),
    )
    return extension.finalize(result) if extension is not None else result
