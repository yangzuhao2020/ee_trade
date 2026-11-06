"""第一、二、五版共用的能量市场事件引擎。
按时间处理开市、交付开始与结束事件，并向学习会话记录交付反馈。"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from ..models import SimulationResult


ScheduledProductKey = tuple[datetime, datetime, datetime]


def simulate_eom_market(market):
    """按时间顺序模拟一个市场从开市、出清、开始交付到结束交付的完整过程."""
    learning_session = market.learning_session
    settle_before_opening = learning_session is not None
    openings_by_time= defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening) # 按照它们的开市时间分组保存

    # Preserve the opening identity even when delivery timestamps coincide.
    scheduled_results = {} # 仅保存尚未结束交付的产品结果。
    pending_starts = defaultdict(list) # 按开始时间安排交付
    pending_ends = defaultdict(list) # 按结束时间安排结束
    results = [] # 最终市场结果
    storage_results = [] # 最终储能调度结果

    def finalize_product_starts(keys):
        # 处理当前开始交付的产品，补计机组启动成本、更新并汇总市场结果，并在启用学习时记录交付反馈。
        for key in keys:
            result = market.start_delivery(scheduled_results[key])
            scheduled_results[key] = result
            results.append(result)
            if learning_session is not None:
                learning_session.record_result(result)

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])
        # 处理当前时间点所有“应该结束交付”的产品，更新常规状态并记录储能结果。
        # 只有已结束的交割才能影响本次开市的报价，因此必须先于开市处理。
        for key in pending_ends.pop(event_time, []):
            storage_results.extend(market.end_delivery(scheduled_results.pop(key)))

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
    并加入已启用的储能、外部电力交换和学习报价。
        ↓
    按 market_mechanism 执行出清：
        pay_as_clear：逐产品撮合买卖订单，计算成交电量和统一出清价。
        complex_clearing：联合求解本次开市的多个产品及其跨时段约束。
        ↓
    返回各产品的 MarketClearingResult，并补充开市时间 opening_time。
    scheduled_results：按（开市时间、交付开始时间、交付结束时间）保存产品结果；
    交付结束后移除该产品，完整历史由 results 保存。
        ↓
    pending_starts / pending_ends：分别按交付开始 / 结束时间登记产品标识。
        ├─ 开始交付：start_delivery() 补计机组启动成本，
        │              将更新后的结果写回 scheduled_results，并追加到 results。
        └─ 结束交付：end_delivery() 更新机组运行状态和储能电量，
                       将储能调度结果追加到 storage_results。
    所有事件完成后，将 results 和 storage_results 汇总为 SimulationResult。
    """

    return SimulationResult(settings=market.settings,
        market_results=tuple(sorted(
            results,
            key=lambda item: (item.delivery_start, item.opening_time),
        )),
        storage_results=tuple(sorted(
            storage_results,
            key=lambda item: (item.delivery_start, item.opening_time, item.unit.name),
        )),
    )
