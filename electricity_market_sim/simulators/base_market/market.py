"""Prepare and operate the conventional market shared by V1/V2, V4 and V5."""

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
        validate_unique_names(
            [
                *(plant.name for plant in self.plants),
                *(storage.name for storage in self.storages),
                *(unit.name for unit in self.demand_units),
                *extension_inputs.participant_names,
            ],
            "all market units",
        )
        self.fuel_prices = (
            extension_inputs.fuel_prices
            if extension_inputs.fuel_prices is not None
            else load_fuel_prices(input_path / "fuel_prices_df.csv")
        ) # 读取燃料价格数据，优先使用扩展模块提供的燃料价格。
        self.fuel_price_profiles = extension_inputs.fuel_price_profiles
        validate_fuel_coverage(self.plants, self.fuel_prices)
        
        self.exchange_unit = None
        self.exchange_profiles: dict[datetime, ExchangeSchedule] = {}
        if settings.exchange_units_file is not None:
            self.exchange_unit = load_exchange_unit(input_path / settings.exchange_units_file)
            existing_names = (
                {plant.name for plant in self.plants}
                | {storage.name for storage in self.storages}
                | {unit.name for unit in self.demand_units}
                | set(extension_inputs.participant_names)
            )
            if self.exchange_unit.name in existing_names:
                raise InputValidationError(
                    f"Exchange name {self.exchange_unit.name!r} must not duplicate another unit name."
                )
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
        cost_times = (
            sorted({start for opening in self.openings for start, _ in opening.products})
            if self.fuel_price_profiles is not None else (None,)
        )
        for plant in self.plants:
            for cost_time in cost_times:
                if cost_time is None:
                    marginal_cost = self.marginal_costs[plant.name]
                else:
                    try:
                        marginal_cost = plant.marginal_cost(self.fuel_price_profiles[cost_time])
                    except KeyError as exc:
                        raise InputValidationError(
                            "fuel_prices_df.csv has no complete product profile for "
                            f"{format_timestamp(cost_time)}."
                        ) from exc
                if not self.settings.minimum_bid_price <= marginal_cost <= self.settings.maximum_bid_price:
                    time_detail = "" if cost_time is None else f" at {format_timestamp(cost_time)}"
                    raise InputValidationError(
                        f"Marginal cost for {plant.name!r}{time_detail} "
                        f"({marginal_cost:.6f} EUR/MWh) is outside the configured bid-price limits."
                    )

    def _prepare_openings_and_forecasts(self) -> None:
        inputs, settings = self.extension_inputs, self.settings
        valid_openings = []
        for opening in self.openings:
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
                    raise InputValidationError(
                        f"{inputs.incomplete_opening_error} {format_timestamp(opening.opening_time)} is incomplete: "
                        + "; ".join(issues)
                    )
                warnings.warn(
                    f"跳过市场开放 {format_timestamp(opening.opening_time)}：" + "; ".join(issues),
                    RuntimeWarning, stacklevel=2,
                )
                continue
            valid_openings.append(opening)
        self.valid_openings = tuple(valid_openings)
        self.price_forecasts: dict[datetime, float] = {}
        if self.requires_market_price_forecast:
            if inputs.market_price_forecasts is not None:
                self.price_forecasts = inputs.market_price_forecasts
            else:
                forecast_starts = _forecast_starts(
                    (start for opening in valid_openings for start, _ in opening.products),
                    settings=settings,
                    requires_price_forecast=self.requires_price_forecast,
                    extra_forecast_offsets=inputs.forecast_offsets,
                    requires_storage_price_forecast=self.requires_storage_price_forecast,
                )
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
            if self.extension is not None else ()
        )

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
