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
    IndustrialUnit,
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    SimulationResult,
    StorageDispatchResult,
)
from .base_market import BaseMarket
from .eom import (
    EomExtensionInputs,
    ScheduledProductKey,
)


class _IndustryInputs:
    """加载工业输入，并向基础市场提供参与者、燃料价格和预测数据。"""

    def __init__(
        self,
        input_path: Path,
        settings: MarketSettings,
        units: tuple[IndustrialUnit, ...],
    ) -> None:
        self.input_path = input_path
        self.settings = settings
        self.units = units
        self.profiles: dict[str, dict[datetime, float]] = {}
        self.fuel_price_profiles: dict[datetime, dict[str, float]] = {}

    def load(self, market: BaseMarket) -> EomExtensionInputs:
        requires_market_price_forecast = market.requires_market_price_forecast
        external_forecast_columns = (
            {"electricity_price"} if requires_market_price_forecast else set()
        )
        required_forecasts = tuple(
            sorted(
                external_forecast_columns
                | {unit.forecast_price_column for unit in self.units}
                | {f"{unit.name}_normalized_load_profile" for unit in self.units}
            )
        )
        self.profiles = load_aligned_time_series_profiles(
            self.input_path / "forecasts_df.csv",
            required_forecasts,
            self.settings.product_duration,
        )
        self.fuel_price_profiles = load_aligned_fuel_price_profiles(
            self.input_path / "fuel_prices_df.csv",
            self.settings.product_duration,
        )
        if not self.fuel_price_profiles:
            raise InputValidationError(
                "fuel_prices_df.csv 中没有完整交付时段的价格数据。")

        for missing_price in missing_industrial_price_names(self.fuel_price_profiles):
            warnings.warn(
                f"fuel_prices_df.csv 中缺少原料 {missing_price!r} 的价格数据；"
                "工业优化将按文档规定使用默认价格 0。",
                RuntimeWarning,
                stacklevel=2,
            )
        market_price_forecasts = (
            self.profiles["electricity_price"]
            if requires_market_price_forecast else None)
        return EomExtensionInputs(
            participant_names=tuple(unit.name for unit in self.units),
            fuel_prices=next(iter(self.fuel_price_profiles.values())),
            fuel_price_profiles=self.fuel_price_profiles,
            market_price_forecasts=market_price_forecasts,
        )


def _load_and_validate_industrial_units(
    input_path: Path, settings: MarketSettings
) -> tuple[IndustrialUnit, ...]:
    if settings.industrial_dsm_units_file is None:
        raise InputValidationError(
            "第四版场景要求在 config.yaml 中配置 industrial_dsm_units。")
    units = load_industrial_units(input_path / settings.industrial_dsm_units_file)
    if not units:
        raise InputValidationError(
            "industrial_dsm_units 指定的文件中必须至少包含一家钢铁厂。")
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "第四版工业滚动优化目前仅支持统一出清价机制"
            "（market_mechanism: pay_as_clear）。")
    for unit in units:
        for field_name, horizon in (
            ("look_ahead_horizon", unit.look_ahead),
            ("commit_horizon", unit.commit_horizon),
            ("rolling_step", unit.rolling_step),
        ):
            ratio = horizon.total_seconds() / settings.product_duration.total_seconds()
            if abs(ratio - round(ratio)) > 1e-9:
                raise InputValidationError(
                    f"工业参数 {field_name}（单元 {unit.name!r}）必须是 "
                    "EOM 交付时段时长的整数倍。"
                )
    return units


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
    units = _load_and_validate_industrial_units(input_path, settings)
    inputs = _IndustryInputs(input_path, settings, units)
    market = BaseMarket(input_path, settings, openings)
    market.prepare(input_loader=inputs.load)
    product_list = [
        product
        for opening in market.valid_openings
        for product in opening.products
        if product[1] <= settings.end
    ]
    if len(product_list) != len(set(product_list)):
        raise InputValidationError(
            "工业滚动优化要求不同 EOM 开市不能包含相同的交付时段，"
            "以确保每个交付时段只承诺一次。"
        )
    coordinator = IndustryRollingCoordinator(
        units=units,
        all_products=tuple(sorted(product_list)),
        price_forecasts={
            unit.name: inputs.profiles[unit.forecast_price_column] for unit in units
        },
        normalized_load_profiles={
            unit.name: inputs.profiles[f"{unit.name}_normalized_load_profile"]
            for unit in units
        },
        fuel_price_profiles=inputs.fuel_price_profiles,
        maximum_bid_price=settings.maximum_bid_price,
    )

    openings_by_time: dict[datetime, list[MarketOpening]] = defaultdict(list)
    for opening in market.valid_openings:
        openings_by_time[opening.opening_time].append(opening)
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] = {}
    pending_starts: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    pending_ends: dict[datetime, list[ScheduledProductKey]] = defaultdict(list)
    market_results: list[MarketClearingResult] = []
    storage_results: list[StorageDispatchResult] = []

    while openings_by_time or pending_starts or pending_ends:
        event_time = min([*openings_by_time, *pending_starts, *pending_ends])

        # 2. 先记录已结束的交付，实际钢产量和设备状态供本次开市使用。
        for key in pending_ends.pop(event_time, []):
            result = scheduled_results[key]
            storage_results.extend(market.end_delivery(result))
            coordinator.record_delivery(result)

        starting_keys = pending_starts.pop(event_time, [])
        for opening in openings_by_time.pop(event_time, []):
            # 3. 优化或复用已承诺的工业生产计划，并生成购电订单。
            industrial_bids = tuple(
                coordinator.bids_for_products(opening.products, scheduled_results)
            )
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
            key=lambda item: (item.delivery_start, item.opening_time or item.delivery_start),
        )),
        storage_results=tuple(sorted(
            storage_results,
            key=lambda item: (item.delivery_start, item.opening_time, item.unit_name),
        )),
        industry_results=tuple(sorted(
            coordinator.dispatch_results,
            key=lambda item: (item.delivery_start, item.unit_name),
        )),
        industry_flexibility_results=tuple(sorted(
            coordinator.flexibility_results,
            key=lambda item: (item.delivery_start, item.unit_name),
        )),
        industry_optimization_windows=tuple(sorted(
            coordinator.window_results,
            key=lambda item: (item.optimization_start, item.unit_name),
        )),
    )
