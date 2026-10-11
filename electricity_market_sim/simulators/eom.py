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
    """按时间顺序模拟一个市场从开市、出清、开始交付到结束交付的完整过程."""
    extension = market.extension # 服务于 4~5 版本
    settle_before_opening = extension is not None and extension.settle_before_opening
    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening) # 按照它们的开市时间分组保存

    # Preserve the opening identity even when delivery timestamps coincide.
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] = {} # 暂存待交付的结果
    pending_starts: dict[datetime, list[ScheduledProductKey]] = defaultdict(list) # 按开始时间安排交付
    pending_ends: dict[datetime, list[ScheduledProductKey]] = defaultdict(list) # 按结束时间安排结束
    results: list[MarketClearingResult] = [] # 最终市场结果
    storage_results: list[StorageDispatchResult] = [] # 最终储能调度结果
    """
    市场出清结果
        ↓
    scheduled_results：暂存待交付的结果 （“市场出清”和“实际交付”是两回事，因此要暂存。）
        ↓
    pending_starts：按开始时间安排交付
    pending_ends：按结束时间安排结束
        ↓
    results：最终市场结果
    storage_results：最终储能调度结果
    """

    def finalize_product_starts(keys: list[ScheduledProductKey]) -> None:
        # keys 是当前时刻待开始交付的产品标识，每个标识包含开市时间、交付开始时间和结束时间。
        for key in keys:
            # 根据机组此前的运行状态，为本次交付中需要启动的机组补计启动成本。
            result = market.start_delivery(scheduled_results[key])
            # 将补计成本后的结果写回，供后续交付结束事件及扩展模块使用。
            scheduled_results[key] = result
            # 纳入最终市场结果，此时结果已包含本次交付应计的启动成本。
            results.append(result)
            if extension is not None:
                # 通知扩展模块：V5 据此记录学习结果并计算奖励，V4 为无需处理的空实现。
                extension.record_delivery_start(result)

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])

        # 处理当前时间点所有“应该结束交付”的产品，并记录由此产生的储能结果和扩展状态更新。
        # 只有已结束的交割才能影响本次开市的报价，因此必须先于开市处理。
        for key in pending_ends.pop(event_time, []):
            storage_results.extend(market.end_delivery(scheduled_results[key]))
            if extension is not None:
                extension.record_delivery(scheduled_results[key])

        starting_keys = pending_starts.pop(event_time, [])
        if settle_before_opening:
            finalize_product_starts(starting_keys)

        # 处理当前时刻的所有开市事件，对产品进行市场出清，并把成交产品登记到未来的“开始交付”和“结束交付”事件队列。
        for opening in openings_by_time.pop(event_time, []):
            for result in market.clear_opening(opening, scheduled_results):
                result = replace(result, opening_time=opening.opening_time)
                key = (opening.opening_time, result.delivery_start, result.delivery_end)
                scheduled_results[key] = result
                pending_starts[result.delivery_start].append(key)
                pending_ends[result.delivery_end].append(key)
        
        if not settle_before_opening:
            finalize_product_starts(starting_keys)

    result = SimulationResult(settings=market.settings,
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
