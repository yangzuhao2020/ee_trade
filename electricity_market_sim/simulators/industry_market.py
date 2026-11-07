"""第四版工业需求响应场景的仿真入口。
在入口直接组织工业优化、出清及交付事件，与第三版一样不使用共用事件引擎。"""

import warnings
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from ..errors import InputValidationError
from ..industry import IndustryRollingCoordinator, missing_industrial_price_names
from ..inputs import (
    load_aligned_fuel_price_profiles,
    load_aligned_time_series_profiles,
    load_industrial_units,
)
from ..models import (
    MarketOpening,
    MarketSettings,
    SimulationResult,
)
from .base_market import BaseMarket


def _load_industry_inputs(market, units):
    """按产品时长对齐工业预测和价格，返回时序数据。"""
    required_forecasts = tuple(
        sorted(
            ({"electricity_price"} if market.requires_market_price_forecast else set())
            | {unit.forecast_price_column for unit in units}
            | {f"{unit.name}_normalized_load_profile" for unit in units}
        )
    )
    profiles = load_aligned_time_series_profiles(
        market.input_path / "forecasts_df.csv",
        required_forecasts,
        market.settings.product_duration,
    )
    fuel_price_profiles = load_aligned_fuel_price_profiles(
        market.input_path / "fuel_prices_df.csv",
        market.settings.product_duration,
    )
    if not fuel_price_profiles:
        raise InputValidationError(
            "fuel_prices_df.csv 中没有完整交付时段的价格数据。")

    for missing_price in missing_industrial_price_names(fuel_price_profiles):
        warnings.warn(
            f"fuel_prices_df.csv 中缺少原料 {missing_price!r} 的价格数据；"
            "工业优化将按文档规定使用默认价格 0。",
            RuntimeWarning,
            stacklevel=2,
        )
    return profiles, fuel_price_profiles


def simulate_v4_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
) -> SimulationResult:
    """直接组织 V4 的数据准备、工业购电计划、市场出清与实际交付。
    同一时刻按“交付结束 → 开市优化及出清 → 交付开始”处理，
    机组报价、储能订单、出清及常规设备状态仍由 BaseMarket 负责。
    """
    # 1. 准备基础市场和工业输入，建立滚动生产协调器。
    if settings.industrial_dsm_units_file is None:
        raise InputValidationError(
            "第四版场景要求在 config.yaml 中配置 industrial_dsm_units。")
    units = load_industrial_units(input_path / settings.industrial_dsm_units_file)
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "第四版工业滚动优化目前仅支持统一出清价机制"
            "（market_mechanism: pay_as_clear）。")
    market = BaseMarket(input_path, settings, openings)
    profiles, fuel_price_profiles = _load_industry_inputs(market, units)
    market.prepare(
        participant_names=tuple(unit.name for unit in units),
        fuel_price_profiles=fuel_price_profiles,
        market_price_forecasts=(
            profiles["electricity_price"] if market.requires_market_price_forecast else None),
    )
    product_list = [
        product
        for opening in market.valid_openings
        for product in opening.products
    ]
    
    if len(product_list) != len(set(product_list)):
        raise InputValidationError(
            "工业滚动优化要求不同 EOM 开市不能包含相同的交付时段，"
            "以确保每个交付时段只承诺一次。"
        )
    coordinator = IndustryRollingCoordinator(
        units=units,
        all_products=tuple(sorted(product_list)), # 所有可安排的交付时段
        price_forecasts={
            unit.name: profiles[unit.forecast_price_column] for unit in units},
        normalized_load_profiles={
            unit.name: profiles[f"{unit.name}_normalized_load_profile"]
            for unit in units}, # 每家工厂对应的参考负荷曲线
        fuel_price_profiles=fuel_price_profiles,
        maximum_bid_price=settings.maximum_bid_price,
    )

    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening)
        
    scheduled_results = {}
    pending_starts = defaultdict(list)
    pending_ends = defaultdict(list)
    market_results = []
    storage_results = []

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])

        # 取出当前时间对应的列表，并从字典里删除这一项，表示这批结束事件已经取出处理。如果当前时间没有交付结束，就返回空列表 []，循环执行零次。
        for key in pending_ends.pop(event_time, []):
            result = scheduled_results.pop(key)
            storage_results.extend(market.end_delivery(result))
            coordinator.record_delivery(result) # 更新钢铁厂的实际生产情况

        starting_keys = pending_starts.pop(event_time, [])
        for opening in openings_by_time.pop(event_time, []):
            # 3. 优化或复用已承诺的工业生产计划，并生成购电订单。
            industrial_bids = tuple(
                coordinator.bids_for_products(opening.products, scheduled_results)) # TODO: 经过了优化的。
            # 4. 复用基础市场生成常规订单、储能订单，并与工业买单一起出清。
            opening_results = market.clear_opening(
                opening, scheduled_results, additional_demand_bids=industrial_bids
            )
            for result in opening_results:
                result = replace(result, opening_time=opening.opening_time)
                key = (opening.opening_time, result.delivery_start, result.delivery_end)
                scheduled_results[key] = result
                pending_starts[result.delivery_start].append(key)
                pending_ends[result.delivery_end].append(key)

        # 5. 开始交付时补计机组启动成本，实际产量留到交付结束再记账。
        for key in starting_keys:
            result = market.start_delivery(scheduled_results[key])
            scheduled_results[key] = result
            market_results.append(result)

    # 6. 检查生产任务是否完成，并汇总市场、储能和工业结果。
    coordinator.validate_completion()
    return SimulationResult(
        settings=settings,
        market_results=tuple(sorted(
            market_results,
            key=lambda item: (item.delivery_start, item.opening_time),
        )),
        storage_results=tuple(sorted(
            storage_results,
            key=lambda item: (item.delivery_start, item.opening_time, item.unit.name),
        )),
        industry_results=tuple(sorted(
            coordinator.dispatch_results,
            key=lambda item: (item.delivery_start, item.unit_name),
        )),
        industry_optimization_windows=tuple(sorted(
            coordinator.window_results,
            key=lambda item: (item.optimization_start, item.unit_name),
        )),
    )
