"""第一、二、四、五版共用的基础市场：加载参与者、准备预测、组织出清并更新交付状态。"""

import warnings
from datetime import datetime
from pathlib import Path
from ...bidding import PlantRuntimeState, StorageRuntimeState
from ...clearing import (
    clear_complex_opening, clear_pay_as_clear,
    validate_demand_prices, validate_offer_prices,
)
from ...errors import InputValidationError
from ...inputs import (
    load_demand_units, load_exchange_unit, load_fuel_prices,
    load_hourly_availability_profiles, load_hourly_demand_profiles,
    load_hourly_exchange_profiles, load_powerplants, load_storage_units,
)
from ...market_models import DemandBid, MarketClearingResult
from ...models import (
    ExchangeSchedule, ExchangeUnit, MarketOpening, MarketSettings, PowerPlant,
    StorageClearingContext, StorageDispatchResult, StorageUnit,
)
from ...utils import format_timestamp, validate_fuel_coverage, validate_unique_names
from ..eom import EomExtensionInputs, EomMarketExtension, ScheduledProductKey
from .demand import _demand_bids_for_product
from .forecasts import _forecast_starts, _naive_price_forecasts, _opening_coverage_issues
from .plants import (
    _apply_startup_costs, _offers_for_opening,
    _record_runtime_states, _validate_strategy_configuration,
)
from .storage import (
    _project_storage_energies, _record_storage_states,
    _storage_orders_for_opening, _validate_storage_configuration,
)

_POWER_TOLERANCE_MW = 1e-9
_FORECAST_STRATEGIES = frozenset({
    "powerplant_energy_heuristic_flexable",
    "powerplant_energy_heuristic_block",
    "powerplant_energy_heuristic_linked",
})


class BaseMarket:
    """Load participants, prepare prices, quote openings and commit deliveries."""

    def __init__(self, input_path: Path, settings: MarketSettings,
                 openings: tuple[MarketOpening, ...]) -> None:
        self.input_path = input_path
        self.settings = settings
        self.openings = openings
        self.extension: EomMarketExtension | None = None

    def prepare(self, extension: EomMarketExtension | None = None) -> None:
        input_path, settings = self.input_path, self.settings
        self.extension = extension
        self.plants = load_powerplants(input_path / "powerplant_units.csv")
        _validate_strategy_configuration(settings, self.plants)
        self.storages = load_storage_units(input_path / "storage_units.csv")
        _validate_storage_configuration(settings, self.storages)
        self.demand_units = load_demand_units(input_path / "demand_units.csv")
        # 读取发电机组、储能和需求侧的数据，并检查它们是否符合当前市场规则。
        self.runtime_plants = tuple(
            plant for plant in self.plants
            if plant.bidding_strategy != "powerplant_energy_naive"
        ) # 排除使用 powerplant_energy_naive 报价策略的机组。
        self.requires_price_forecast = any(
            plant.bidding_strategy in _FORECAST_STRATEGIES
            and plant.min_power_mw > _POWER_TOLERANCE_MW
            for plant in self.runtime_plants
        ) # 判断是否存在至少一个需要预测电价的发电机组。
        self.requires_storage_price_forecast = bool(self.storages)
        self.requires_market_price_forecast = (
            self.requires_price_forecast or self.requires_storage_price_forecast)
        self.demand_profiles = load_hourly_demand_profiles(
            input_path / "demand_df.csv", self.demand_units) # 读取需求侧的负荷预测数据。
        self.availability_profiles = load_hourly_availability_profiles(
            input_path / "availability_df.csv", self.plants) # 读取发电机组的可用性预测数据。
        extension_inputs = (
            extension.prepare(self) if extension is not None else EomExtensionInputs())
        self.extension_inputs = extension_inputs
        self.requires_market_price_forecast |= bool(extension_inputs.forecast_offsets)
        # 扩展模块的预测需求合并到市场的预测开关中
        self.fuel_prices = (
            extension_inputs.fuel_prices
            if extension_inputs.fuel_prices is not None
            else load_fuel_prices(input_path / "fuel_prices_df.csv")
        ) # 读取燃料价格数据，优先使用扩展模块提供的燃料价格。
        self.fuel_price_profiles = extension_inputs.fuel_price_profiles
        validate_fuel_coverage(self.plants, self.fuel_prices)

        # 外部电力交换初始化说明：
        # 1. 仅在配置 exchange_units_file 时启用；未配置时，交换单元为 None，时序数据为空字典。
        # 2. 从配置文件读取唯一的交换参与者，包括名称、运营方及进口/出口报价（EUR/MWh）。
        # 3. 外部电网名称与其他参与者一起，由下方的 validate_unique_names 统一检查重名。
        # 先加载可选的外部电网参与者，再统一检查所有市场参与者的名称是否重名。
        self.exchange_unit = None
        if settings.exchange_units_file is not None:
            self.exchange_unit = load_exchange_unit(input_path / settings.exchange_units_file)
        participant_names = [
            *(plant.name for plant in self.plants),
            *(storage.name for storage in self.storages),
            *(unit.name for unit in self.demand_units),
            *extension_inputs.participant_names,
        ]
        if self.exchange_unit is not None:
            participant_names.append(self.exchange_unit.name)
        validate_unique_names(participant_names, "all market units")

        self.exchange_profiles: dict[datetime, ExchangeSchedule] = {}
        if self.exchange_unit is not None:
            for label, price in (
                ("import", self.exchange_unit.price_import_eur_per_mwh),
                ("export", self.exchange_unit.price_export_eur_per_mwh),
            ):
                if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
                    raise InputValidationError(
                        f"Exchange {label} price ({price:.6f} EUR/MWh) is outside the configured bid-price limits."
                    )
            self.exchange_profiles = load_hourly_exchange_profiles(
                input_path / "exchanges_df.csv", self.exchange_unit
            )
        # 4. 分别检查进口和出口报价是否位于市场最低、最高报价之间（含边界）；价格越界时报错。
        # 5. 从 exchanges_df.csv 读取“参与者名称_import”和“参与者名称_export”列，功率必须非负。
        #    每小时的可用样本取等权平均，保存为“小时起点 -> ExchangeSchedule”的映射，功率单位为 MW。
        # 6. 后续生成订单时，进口电力作为本地市场的供给卖单，出口电力作为本地市场的需求买单。

        self.marginal_costs = {
            plant.name: plant.marginal_cost(self.fuel_prices) for plant in self.plants
        }
        self._validate_marginal_costs()
        self._prepare_openings_and_forecasts()
        self.runtime_states = {
            plant.name: PlantRuntimeState.initially_off() for plant in self.runtime_plants
        }
        self.storage_states = {
            storage.name: StorageRuntimeState.from_initial_soc(storage)
            for storage in self.storages
        }
        if extension is not None:
            extension.initialize(self)

    def _validate_marginal_costs(self) -> None:
        """检查各机组的边际成本是否处于市场报价上下限内（含边界），单位为 EUR/MWh。
        使用静态燃料价格时，每台机组只检查一次预先计算的边际成本；
        使用分时燃料价格时，按所有开市产品的交付开始时间，去重后逐时计算并检查。
        """
        # 分时模式提取交付开始时间（不是开市时间），用集合去重后排序；
        # 静态模式用 (None,) 作为仅含一个占位值的元组，让每台机组只检查一次。
        cost_times = (
            sorted({start for opening in self.openings for start, _ in opening.products})
            if self.fuel_price_profiles is not None else (None,)
        )
        for plant in self.plants:
            for cost_time in cost_times:
                if cost_time is None:
                    # 静态模式直接使用初始化时已计算好的机组边际成本。
                    marginal_cost = self.marginal_costs[plant.name]
                else:
                    try:
                        # 分时模式取该交付时段的价格，重新计算燃料、CO2 和附加成本。
                        marginal_cost = plant.marginal_cost(self.fuel_price_profiles[cost_time])
                    except KeyError as exc:
                        # 将缺少时段或价格键的错误转换为输入校验异常，并保留原始异常链。
                        raise InputValidationError(
                            "fuel_prices_df.csv has no complete product profile for "
                            f"{format_timestamp(cost_time)}."
                        ) from exc
                # 最低价和最高价均允许；越界时报告机组名称、成本，以及分时模式下的交付时间。
                if not self.settings.minimum_bid_price <= marginal_cost <= self.settings.maximum_bid_price:
                    time_detail = "" if cost_time is None else f" at {format_timestamp(cost_time)}"
                    raise InputValidationError(
                        f"Marginal cost for {plant.name!r}{time_detail} "
                        f"({marginal_cost:.6f} EUR/MWh) is outside the configured bid-price limits."
                    )

    def _prepare_openings_and_forecasts(self) -> None:
        """筛选可用的开市，并准备机组、储能及扩展报价策略所需的电价预测。
        检查各开市的交付时段是否越界，以及交付和预测窗口所需的数据是否齐全。
        发现问题时，按扩展配置选择抛出 InputValidationError，或发出警告并跳过该次开市。
        将通过检查的开市保存到 self.valid_openings，保持原有顺序。
        需要电价预测时，优先使用扩展提供的预测，否则根据有效开市生成朴素出清价格预测。
        预测保存为“时段开始时间 -> 电价（EUR/MWh）”的 self.price_forecasts；无需预测时为空字典。
        """
        inputs, settings = self.extension_inputs, self.settings
        valid_openings = []
        for opening in self.openings:
            # 检查产品交付范围、刚性负荷、已提供的机组可用率及启用的电力交换数据。
            # 检查时间还包含机组、储能和扩展策略所需的预测窗口；issues 列出发现的问题。
            issues = _opening_coverage_issues(
                opening=opening,
                settings=settings,
                plants=self.plants,
                demand_units=self.demand_units,
                demand_profiles=self.demand_profiles,
                availability_profiles=self.availability_profiles,
                exchange_unit=self.exchange_unit,
                exchange_profiles=self.exchange_profiles,
                requires_price_forecast=self.requires_price_forecast,
                extra_forecast_offsets=inputs.forecast_offsets,
                requires_storage_price_forecast=self.requires_storage_price_forecast,
            )
            if issues:
                if inputs.incomplete_opening_error is not None:
                    # 扩展要求数据完整时立即报错，用指定的说明及问题列表解释原因。
                    raise InputValidationError(
                        f"{inputs.incomplete_opening_error} {format_timestamp(opening.opening_time)} is incomplete: "
                        + "; ".join(issues)
                    )
                # 默认发出警告后跳过该次开市；stacklevel=2 将警告位置指向本函数的调用处。
                warnings.warn(
                    f"跳过市场开放 {format_timestamp(opening.opening_time)}：" + "; ".join(issues),
                    RuntimeWarning, stacklevel=2,
                )
                continue
            # 只有通过检查的开市才会参与后续报价和出清。
            valid_openings.append(opening)

        self.valid_openings = tuple(valid_openings)
        # 没有策略需要电价预测时，保留空字典。
        self.price_forecasts: dict[datetime, float] = {}
        if self.requires_market_price_forecast:
            if inputs.market_price_forecasts is not None:
                # 优先直接使用扩展模块提供的预测，不在这里重新计算。
                self.price_forecasts = inputs.market_price_forecasts
            else:
                # 根据有效产品的交付开始时间，汇总机组、储能及扩展策略所需的预测时间点并去重。
                forecast_starts = _forecast_starts(
                    (start for opening in valid_openings for start, _ in opening.products),
                    settings=settings,
                    requires_price_forecast=self.requires_price_forecast,
                    extra_forecast_offsets=inputs.forecast_offsets,
                    requires_storage_price_forecast=self.requires_storage_price_forecast,
                )
                # 用机组边际成本和可用出力生成朴素卖单，结合刚性负荷及电力交换，
                # 在所需时段按统一出清价机制模拟出清，将出清电价作为预测值。
                self.price_forecasts = _naive_price_forecasts(
                    products=[product for opening in valid_openings for product in opening.products],
                    settings=settings,
                    plants=self.plants,
                    marginal_costs=self.marginal_costs,
                    availability_profiles=self.availability_profiles,
                    demand_units=self.demand_units,
                    demand_profiles=self.demand_profiles,
                    exchange_unit=self.exchange_unit,
                    exchange_profiles=self.exchange_profiles,
                    forecast_starts=forecast_starts,
                )

    def clear_opening(self, opening: MarketOpening,
                      scheduled_results: dict[ScheduledProductKey, MarketClearingResult]
                      ) -> tuple[MarketClearingResult, ...]:
        # 对本次开市的全部产品组织报价并出清，返回各产品的 MarketClearingResult 元组。
        # opening 指定开市时间及交付时段；scheduled_results 保存此前已出清的产品结果。
        # 获取扩展模块提供的额外需求报价（例如 V4 的工业用电需求）；没有扩展时使用空元组。
        additional_demand_bids = (
            self.extension.bids_for_products(opening.products, scheduled_results)
            if self.extension is not None else ())

        # 以当前已记录的储能电量为基础，计入首个产品开始交付前会结束的已成交充放电等影响。
        # 得到各储能单元在首个交付时段开始时的预计电量（MWh），供储能报价使用。
        # 此处仅预测电量，实际储能状态由交付结束事件更新。
        storage_initial_energies = _project_storage_energies(
            opening=opening,
            storages=self.storages,
            storage_states=self.storage_states,
            scheduled_results=scheduled_results,
        )

        # 由公共出清函数生成并汇总需求、发电、储能及扩展报价，按配置的市场机制执行出清。
        # 返回的是市场出清结果；交付开始时需要计入的机组启动成本由 start_delivery 后续补计。
        return _clear_opening(
            # 本次开市信息与市场规则。
            opening=opening,
            settings=self.settings,
            # 发电机组参数及当前已记录的运行状态。
            plants=self.plants,
            runtime_states=self.runtime_states,
            # 常规需求单元及其随时间变化的用电需求。
            demand_units=self.demand_units,
            demand_profiles=self.demand_profiles,
            # 机组可用出力、发电成本数据及报价策略所需的电价预测。
            availability_profiles=self.availability_profiles,
            marginal_costs=self.marginal_costs,
            fuel_price_profiles=self.fuel_price_profiles,
            price_forecasts=self.price_forecasts,
            # 外部电力交换单元及各时段的交换计划。
            exchange_unit=self.exchange_unit,
            exchange_profiles=self.exchange_profiles,
            # 储能参数及预测的交付初始电量，用于生成充放电报价和构建物理约束。
            storages=self.storages,
            storage_initial_energies_mwh=storage_initial_energies,
            # 额外需求报价及扩展接口；此前出清结果也供扩展发电报价策略使用。
            additional_demand_bids=additional_demand_bids,
            extension=self.extension,
            scheduled_results=scheduled_results,
        )

    def start_delivery(self, result: MarketClearingResult) -> MarketClearingResult:
        return _apply_startup_costs(result, self.plants, self.runtime_states)

    def end_delivery(self, result: MarketClearingResult) -> list[StorageDispatchResult]:
        _record_runtime_states(result, self.runtime_states)
        return _record_storage_states(result, self.storages, self.storage_states)


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
    extension: EomMarketExtension | None = None,
    scheduled_results: dict[ScheduledProductKey, MarketClearingResult] | None = None,
) -> tuple[MarketClearingResult, ...]:
    """根据开市时已知的机组运行状态、预计储能电量和预测数据，为本次开市的各产品生成报价。
    汇总需求买单和供给卖单（含外部电力交换、储能及扩展报价），检查报价是否符合价格规则，
    再按配置的市场机制逐产品出清或联合出清，返回各产品的市场出清结果元组。
    """

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
        ) # 本地刚性负荷买单与弹性负荷买单
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
        extension=extension,
        scheduled_results=scheduled_results,
    ) # 机组卖单
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

    validate_demand_prices(all_demand_bids, settings) # 验证买单报价
    validate_offer_prices(offers, settings) # 验证卖单报价
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
    clearing_results = []

    for start, end in opening.products:
        # 1. 收集当前产品的买单。
        product_bids = []
        for bid in all_demand_bids:
            if (bid.delivery_start, bid.delivery_end) == (start, end):
                product_bids.append(bid)

        # 2. 收集当前产品的卖单。
        product_offers = []
        for offer in offers:
            if (offer.delivery_start, offer.delivery_end) == (start, end):
                product_offers.append(offer)

        # 3. 执行出清，保存当前产品的结果。
        result = clear_pay_as_clear(product_bids, product_offers)
        clearing_results.append(result)

    return tuple(clearing_results)
