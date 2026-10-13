"""第一、二、四、五版共用的能量市场事件引擎和参与者扩展接口。
按时间处理开市、交付开始与结束事件，并调用扩展完成额外报价、状态记录及结果汇总。"""

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
    """市场扩展提供的额外输入和预测需求，供开市前的数据检查使用。"""
    participant_names: tuple[str, ...] = ()  # 额外参与者名称，用于检查市场单位名称是否重复。
    fuel_prices: dict[str, float] | None = None  # 扩展提供的燃料价格，优先于基础市场从 CSV 加载的价格。
    fuel_price_profiles: dict[datetime, dict[str, float]] | None = None  # 各时段的燃料价格，用于计算边际成本。
    market_price_forecasts: dict[datetime, float] | None = None  # 扩展提供的电价预测，优先于基础市场自行生成的预测。
    forecast_offsets: tuple[int, ...] = ()  # 相对交付开始时间的小时偏移，指定扩展额外需要的预测时间点。
    incomplete_opening_error: str | None = None  # 设置后，开市数据不完整时以此说明报错；默认警告并跳过开市。


class EomMarketExtension(Protocol):
    """Connect extra demand or replacement plant bids to the shared event engine."""

    settle_before_opening: bool

    def prepare(self, market: BaseMarket) -> EomExtensionInputs:
        """检查已加载的常规参与者，并声明扩展所需的额外输入。"""

    def initialize(self, market: BaseMarket) -> None:
        """在预检、价格预测和运行状态都准备就绪后进行初始化。"""

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[DemandBid, ...]:
        """返回本次开市的额外需求订单。"""

    def offers_for_plant(
        self,
        plant: PowerPlant,
        products: tuple[tuple[datetime, datetime], ...],
        available_powers: dict[datetime, float],
        marginal_cost_at: Callable[[datetime], float],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[SupplyOffer, ...] | None:
        """替换某台机组的报价；返回 None 时仍由常规报价逻辑负责。"""

    def record_delivery_start(self, result: MarketClearingResult) -> None:
        """在启动成本计入之后，接收开始交付的产品结果。"""

    def record_delivery(self, result: MarketClearingResult) -> None:
        """将已结束的交付写入扩展模块的状态。"""

    def finalize(self, result: SimulationResult) -> SimulationResult:
        """所有事件完成后，附加扩展模块特有的结果。"""


def simulate_eom_market(market: BaseMarket) -> SimulationResult:
    """按时间顺序模拟一个市场从开市、出清、开始交付到结束交付的完整过程."""
    extension = market.extension # 服务于 4~5 版本
    settle_before_opening = extension is not None and extension.settle_before_opening
    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening) # 按照它们的开市时间分组保存

    # Preserve the opening identity even when delivery timestamps coincide.
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] = {} # 保存已出清产品结果，供交付事件及后续报价查询
    pending_starts: dict[datetime, list[ScheduledProductKey]] = defaultdict(list) # 按开始时间安排交付
    pending_ends: dict[datetime, list[ScheduledProductKey]] = defaultdict(list) # 按结束时间安排结束
    results: list[MarketClearingResult] = [] # 最终市场结果
    storage_results: list[StorageDispatchResult] = [] # 最终储能调度结果

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
    
    """
    到达开市时间，调用 market.clear_opening(opening, scheduled_results)
        ↓
    根据需求预测、机组可用出力、成本、运行状态及报价策略生成买单和卖单，
    并加入已启用的储能、外部电力交换和扩展报价。
        ↓
    按 market_mechanism 执行出清：
        pay_as_clear：逐产品撮合买卖订单，计算成交电量和统一出清价。
        complex_clearing：联合求解本次开市的多个产品及其跨时段约束。
        ↓
    返回各产品的 MarketClearingResult，并补充开市时间 opening_time。
    scheduled_results：按（开市时间、交付开始时间、交付结束时间）保存产品结果；
    已交付的结果也保留，供交付事件及后续报价查询。
        ↓
    pending_starts / pending_ends：分别按交付开始 / 结束时间登记产品标识。
        ├─ 开始交付：start_delivery() 补计机组启动成本，
        │              将更新后的结果写回 scheduled_results，并追加到 results。
        └─ 结束交付：end_delivery() 更新机组运行状态和储能电量，
                       将储能调度结果追加到 storage_results。
    所有事件完成后，将 results 和 storage_results 汇总为 SimulationResult。
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
