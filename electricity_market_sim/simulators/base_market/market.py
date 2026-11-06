"""第一、二、四、五版共用的基础市场：加载参与者、准备预测、组织出清并更新交付状态。"""

import warnings
from collections import defaultdict
from dataclasses import replace
from ...bidding import storage_heuristic_orders
from ...clearing_algorithm import (
    clear_complex_opening, clear_pay_as_clear, validate_order_prices,
)
from ...errors import InputValidationError
from ...inputs import (
    load_demand_units, load_exchange_unit, load_fuel_prices,
    load_hourly_availability_profiles, load_hourly_demand_profiles,
    load_hourly_exchange_profiles, load_powerplants, load_storage_units,
)
from ...utils import format_timestamp, validate_fuel_coverage, validate_unique_names
from .demand import _demand_bids_for_product
from .forecasts import _forecast_starts, _naive_price_forecasts, _opening_coverage_issues
from .plants import (
    _offers_for_opening,
    _record_runtime_states, _validate_strategy_configuration,
)
from .storage import (
    StorageRuntimeState,
    _project_storage_energies, _record_storage_states,
    _validate_storage_configuration,
)

_POWER_TOLERANCE_MW = 1e-9
_FORECAST_STRATEGIES = frozenset({
    "powerplant_energy_heuristic_flexable",
    "powerplant_energy_heuristic_block",
    "powerplant_energy_heuristic_linked",
})


class BaseMarket:
    """加载市场参与者，准备电价预测，组织开市报价，并更新交付状态。"""

    def __init__(self, input_path, settings, openings):
        self.input_path = input_path
        self.settings = settings
        self.openings = openings
        self.learning_session = None
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
        self.requires_market_price_forecast = (
            self.requires_price_forecast or bool(self.storages))
        self.demand_profiles = load_hourly_demand_profiles(
            input_path / "demand_df.csv", self.demand_units) # 读取需求侧的负荷预测数据。
        self.availability_profiles = load_hourly_availability_profiles(
            input_path / "availability_df.csv", self.plants) # 读取发电机组的可用性预测数据。

    def prepare(
        self,
        *,
        participant_names=(),
        fuel_price_profiles=None,
        market_price_forecasts=None,
        learning_session=None,
    ):
        """使用场景提供的额外输入，准备价格、开市安排和设备运行状态。"""
        input_path, settings = self.input_path, self.settings
        self.learning_session = learning_session
        fuel_prices = (
            next(iter(fuel_price_profiles.values()))
            if fuel_price_profiles is not None
            else load_fuel_prices(input_path / "fuel_prices_df.csv")
        )
        self.fuel_price_profiles = fuel_price_profiles
        validate_fuel_coverage(self.plants, fuel_prices)

        self.exchange_unit = None
        if settings.exchange_units_file is not None:
            self.exchange_unit = load_exchange_unit(input_path / settings.exchange_units_file)
        all_participant_names = [
            *(plant.name for plant in self.plants),
            *(storage.name for storage in self.storages),
            *(unit.name for unit in self.demand_units),
            *participant_names,
        ]
        if self.exchange_unit is not None:
            all_participant_names.append(self.exchange_unit.name)
        validate_unique_names(all_participant_names, "all market units")

        self.exchange_profiles = {}
        if self.exchange_unit is not None:
            for label, price in (
                ("进口", self.exchange_unit.price_import_eur_per_mwh),
                ("出口", self.exchange_unit.price_export_eur_per_mwh),
            ):
                if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
                    raise InputValidationError(
                        f"电力交换{label}价格（{price:.6f} EUR/MWh）超出配置的报价上下限。"
                    )
            self.exchange_profiles = load_hourly_exchange_profiles(
                input_path / "exchanges_df.csv", self.exchange_unit
            )

        self.marginal_costs = {
            plant.name: plant.marginal_cost(fuel_prices) for plant in self.plants
        } if fuel_price_profiles is None else {}
        self._validate_marginal_costs()
        self._prepare_openings_and_forecasts(market_price_forecasts)
        # 机组初始停机；当前不施加爬坡、CHP 供热承诺及最小连续运行/停机时间约束。
        self.runtime_states = {
            plant.name: False for plant in self.runtime_plants
        }
        self.storage_states = {
            storage.name: StorageRuntimeState(energy_mwh=storage.initial_energy_mwh)
            for storage in self.storages
        }

    def _validate_marginal_costs(self):
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
                            "fuel_prices_df.csv 缺少交付时段 "
                            f"{format_timestamp(cost_time)} 的完整燃料价格数据。"
                        ) from exc
                # 最低价和最高价均允许；越界时报告机组名称、成本，以及分时模式下的交付时间。
                if not self.settings.minimum_bid_price <= marginal_cost <= self.settings.maximum_bid_price:
                    time_detail = "" if cost_time is None else f" 在 {format_timestamp(cost_time)} 时"
                    raise InputValidationError(
                        f"机组 {plant.name!r}{time_detail} 的边际成本"
                        f"（{marginal_cost:.6f} EUR/MWh）超出配置的报价上下限。"
                    )

    def _prepare_openings_and_forecasts(
        self, market_price_forecasts,
    ):
        """筛选可用的开市，并准备机组和储能所需的电价预测。
        检查各开市的交付时段是否越界，以及交付和预测窗口所需的数据是否齐全。
        发现问题时，学习场景抛出 InputValidationError，其余场景警告并跳过该次开市。
        将通过检查的开市保存到 self.valid_openings，保持原有顺序。
        需要电价预测时，优先使用扩展提供的预测，否则根据有效开市生成朴素出清价格预测。
        预测保存为“时段开始时间 -> 电价（EUR/MWh）”的 self.price_forecasts；无需预测时为空字典。
        """
        settings = self.settings
        valid_openings = []
        for opening in self.openings:
            # 检查产品交付范围、刚性负荷、已提供的机组可用率及启用的电力交换数据。
            # 检查时间还包含机组和储能所需的预测窗口；issues 列出发现的问题。
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
                requires_storage_price_forecast=bool(self.storages),
                hard_forecast_horizon=self.learning_session is not None,
            )
            if issues:
                if self.learning_session is not None:
                    raise InputValidationError(
                        f"学习场景在 {format_timestamp(opening.opening_time)} 的开市数据不完整："
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
        self.price_forecasts = {}
        if self.requires_market_price_forecast:
            if market_price_forecasts is not None:
                # 优先直接使用扩展模块提供的预测，不在这里重新计算。
                self.price_forecasts = market_price_forecasts
            else:
                forecast_starts = _forecast_starts(
                    (start for opening in valid_openings for start, _ in opening.products),
                    settings=settings,
                    requires_price_forecast=self.requires_price_forecast,
                    requires_storage_price_forecast=bool(self.storages),
                    hard_forecast_horizon=self.learning_session is not None,
                )
                # 用机组边际成本和可用出力生成朴素卖单，结合刚性负荷及电力交换，
                # 在所需时段按统一出清价机制模拟出清，将出清电价作为预测值。
                self.price_forecasts = _naive_price_forecasts(
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

    def clear_opening(self, opening, scheduled_results, *,
                      additional_demand_bids=()):
        """根据开市时已知的机组运行状态、预计储能电量和预测数据，为本次开市的各产品生成报价。
        汇总需求买单和供给卖单（含外部电力交换、储能及学习报价），检查报价是否符合价格规则，
        additional_demand_bids 接收场景已生成的额外买单。
        再按配置的市场机制逐产品出清或联合出清，返回各产品的市场出清结果元组。
        """
        settings = self.settings
        storage_initial_energies = _project_storage_energies(
            opening=opening,
            storages=self.storages,
            storage_states=self.storage_states,
            scheduled_results=scheduled_results,
        )

        all_demand_bids = []
        for delivery_start, delivery_end in opening.products:
            demand_bids = _demand_bids_for_product(
                # 常规需求单元及其随时间变化的用电需求。
                demand_units=self.demand_units,
                demand_profiles=self.demand_profiles,
                # 外部电力交换单元及各时段的交换计划。
                exchange_unit=self.exchange_unit,
                exchange_profiles=self.exchange_profiles,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                maximum_bid_price=settings.maximum_bid_price,
            ) # 本地刚性负荷买单与弹性负荷买单
            all_demand_bids.extend(demand_bids)

        all_demand_bids.extend(additional_demand_bids)

        offers = _offers_for_opening(
            opening=opening,
            # 发电机组参数及当前已记录的运行状态。
            plants=self.plants,
            runtime_states=self.runtime_states,
            # 机组可用出力、发电成本数据及报价策略所需的电价预测。
            availability_profiles=self.availability_profiles,
            marginal_costs=self.marginal_costs,
            fuel_price_profiles=self.fuel_price_profiles,
            price_forecasts=self.price_forecasts,
            exchange_unit=self.exchange_unit,
            exchange_profiles=self.exchange_profiles,
            learning_session=self.learning_session,
        ) # 机组卖单
        
        clearing_storages = []
        for storage in self.storages:
            # 储能参数及预测的交付初始电量，用于生成充放电报价和构建物理约束。
            initial_energy = storage_initial_energies[storage.name]
            clearing_storages.append((storage, initial_energy))
            storage_bids, storage_offers = storage_heuristic_orders(
                storage, initial_energy, opening.products, self.price_forecasts,
                forecast_start=settings.start,
                forecast_end=settings.end,
            )
            all_demand_bids.extend(storage_bids)
            offers.extend(storage_offers)

        validate_order_prices(all_demand_bids, offers, settings)
        # 按配置的市场机制执行出清。返回的是市场出清结果；
        # 交付开始时需要计入的机组启动成本由 start_delivery 后续补计。
        if settings.market_mechanism == "complex_clearing":
            return clear_complex_opening(
                all_demand_bids,
                offers,
                storages=tuple(clearing_storages),
            )
        if any(offer.bid_type != "SB" for offer in offers):
            raise InputValidationError(
                "EOM 块报价和关联报价策略要求将 market_mechanism 设置为 "
                "complex_clearing。"
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

    def start_delivery(self, result):
        """已成交的产品开始交付时，为需要启动的机组计入启动成本。"""
        accepted_by_plant = defaultdict(float)
        for cleared in result.offers:
            if cleared.offer.offer_type == "power_plant":
                accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
        started_names = {
            name
            for name, is_running in self.runtime_states.items()
            if not is_running and accepted_by_plant[name] > _POWER_TOLERANCE_MW
        }
        if not started_names:
            return result
        plants_by_name = {plant.name: plant for plant in self.plants}
        has_inflexible_offer = {
            cleared.offer.unit_name
            for cleared in result.offers
            if cleared.offer.offer_segment
            in {"inflexible", "block_inflexible", "learning_minimum"}
        }
        applied_names = set()
        cleared_offers = []
        for cleared in result.offers:
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
            cleared_offers.append(replace(cleared, startup_cost_eur=startup_cost))
        return replace(result, offers=tuple(cleared_offers))

    def end_delivery(self, result):
        _record_runtime_states(result, self.runtime_states)
        return _record_storage_states(result, self.storages, self.storage_states)
