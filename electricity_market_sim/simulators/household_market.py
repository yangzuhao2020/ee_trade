"""第三版居民购电计划优化、灵活性评估及采用按报价结算的市场仿真。"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from ..bidding import available_power_mw, naive_offer
from ..clearing_algorithm import clear_pay_as_bid, validate_demand_prices, validate_offer_prices
from ..errors import InputValidationError
from ..household import dispatch_household, plan_household_opening
from ..inputs import (
    load_demand_units,
    load_exact_availability_profiles,
    load_fuel_price_profiles,
    load_household_units,
    load_powerplants,
    load_time_series_profiles,
)
from ..models import (
    DemandBid,
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    SimulationResult,
    SupplyOffer,
)
from ..utils import (
    format_timestamp, hours_between, validate_fuel_coverage, validate_unique_names,)


_POWER_TOLERANCE_MW = 1e-9


def simulate_household_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
) -> SimulationResult:
    """运行 V3 的 15 分钟家庭用户及按报价结算（pay-as-bid）交易流程。"""
    plants = load_powerplants(input_path / "powerplant_units.csv")
    demand_units = load_demand_units(input_path / "demand_units.csv")
    if any(unit.is_elastic for unit in demand_units):
        raise InputValidationError(
            "按报价结算（pay_as_bid）的家庭场景暂不支持弹性需求单元。")
    households = load_household_units(input_path / "residential_dsm_units.csv")
    if not households:
        raise InputValidationError(
            "按报价结算（pay_as_bid）要求 residential_dsm_units.csv 至少包含一个家庭用户。")
    validate_unique_names(
        [
            *(plant.name for plant in plants),
            *(unit.name for unit in demand_units),
            *(household.name for household in households),
        ],
        "发电机组、需求单元和家庭用户",)

    demand_columns = []
    for unit in demand_units:               # 逐个遍历需求单元
        profile_column = unit.profile_column  # 获取当前单元对应的列名
        if profile_column is not None:       # 如果列名不是 None
            demand_columns.append(profile_column)  # 加到列表末尾

    demand_profiles = load_time_series_profiles(
        input_path / "demand_df.csv",
        tuple(demand_columns),
        warn_unused_columns=True,
    )
    forecast_columns = ("price_EOM",) + tuple(
        f"{household.name}_heat_demand" for household in households)
    forecasts = load_time_series_profiles(
        input_path / "forecasts_df.csv", forecast_columns)
    fuel_prices = load_fuel_price_profiles(input_path / "fuel_prices_df.csv")
    availability = load_exact_availability_profiles(
        input_path / "availability_df.csv", plants)
    validate_fuel_coverage(plants, next(iter(fuel_prices.values()), {}))

    household_energy = {household.name: household.initial_energy_mwh for household in households}
    household_heat_pump_power = {household.name: 0.0 for household in households}
    household_charge_power = {household.name: 0.0 for household in households}
    household_discharge_power = {household.name: 0.0 for household in households} 
    # 以上为初始化变量
    
    market_results: list[MarketClearingResult] = []
    household_results = []
    household_flexibility_results = []
    cleared_products: set[tuple[datetime, datetime]] = set()

    for opening in openings:
        if any(product in cleared_products for product in opening.products):
            raise ValueError("同一交付产品不能在两次开市中重复报价。")

        plans_by_household, flexibility_by_household = plan_household_opening(
            households,
            opening.products,
            forecasts["price_EOM"],
            forecasts,
            household_energy,
            household_heat_pump_power,
            household_charge_power,
            household_discharge_power,
        )
        for household in households:
            household_flexibility_results.extend(
                flexibility_by_household[household.name])

        # 已经优化完成的购电计划，这几行负责把结果整理成方便查询的字典
        plan_by_product = {}
        for household_plans in plans_by_household.values():
            for plan in household_plans:
                plan_key = (plan.unit_name, plan.delivery_start)
                plan_by_product[plan_key] = plan

        opening_results: list[MarketClearingResult] = []
        for delivery_start, delivery_end in opening.products:
            duration_hours = hours_between(delivery_start, delivery_end)
            demand_bids: list[DemandBid] = []
            for unit in demand_units:
                assert unit.profile_column is not None
                power = demand_profiles[unit.profile_column][delivery_start]
                demand_bids.append(
                    DemandBid(
                        unit_name=unit.name,
                        operator=unit.operator,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        volume_mwh=power * duration_hours,
                        price_eur_per_mwh=(
                            unit.price_eur_per_mwh
                            if unit.price_eur_per_mwh is not None
                            else settings.maximum_bid_price
                        ),
                        bid_id=f"{unit.name}::load::{delivery_start.isoformat()}",
                        demand_type="inelastic_load",
                    )
                ) # 生成一条买单，普通负荷。

            for household in households:
                plan = plan_by_product[(household.name, delivery_start)]
                volume = plan.planned_grid_power_mw * duration_hours
                if volume <= _POWER_TOLERANCE_MW:
                    continue
                demand_bids.append(
                    DemandBid(
                        unit_name=household.name,
                        operator=household.operator,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        volume_mwh=volume,
                        price_eur_per_mwh=household.bid_price_eur_per_mwh,
                        bid_id=(
                            f"{household.name}::household::{delivery_start.isoformat()}"
                        ),
                        demand_type="household_load",
                    )
                ) # 生成一条买单，家庭负荷。

            try:
                product_fuel_prices = fuel_prices[delivery_start]
            except KeyError as exc:
                raise InputValidationError(
                    f"fuel_prices_df.csv 中缺少 {format_timestamp(delivery_start)} 时刻的燃料价格。") from exc
                
            offers: list[SupplyOffer] = []
            for plant in plants:
                available_power = available_power_mw(plant, delivery_start, availability)
                marginal_cost = plant.marginal_cost(product_fuel_prices)
                offers.append(
                    naive_offer(
                        plant,
                        delivery_start,
                        delivery_end,
                        available_power,
                        marginal_cost,))

            validate_demand_prices(demand_bids, settings)
            validate_offer_prices(offers, settings)
            result = clear_pay_as_bid(demand_bids, offers)
            opening_results.append(replace(result, opening_time=opening.opening_time))

        results_by_start = {result.delivery_start: result for result in opening_results}
        for household in households:
            accepted_grid_energy: dict[datetime, float] = {}
            for delivery_start, _ in opening.products:
                # 获取当前交付时段的出清结果。
                product_result = results_by_start[delivery_start]
                cleared_bid = None
                for cleared in product_result.demand_bids:
                    bid = cleared.bid
                    # 跳过其他参与者的买单。
                    if bid.unit_name != household.name:
                        continue
                    # 只查找居民负荷买单。
                    if bid.demand_type != "household_load":
                        continue
                    # 找到第一条匹配的买单后，结束查找。
                    cleared_bid = cleared
                    break
                if cleared_bid is None:
                    accepted_grid_energy[delivery_start] = 0.0
                else:
                    accepted_grid_energy[delivery_start] = cleared_bid.accepted_energy_mwh
                # 提取当前家庭在每个交付时段实际买到的电量
                
            dispatch = dispatch_household(
                household,
                plans_by_household[household.name], # 购电、充放点计划
                accepted_grid_energy,
                household_energy[household.name], # 取出当前家庭电池已经储存的电量
                initial_heat_pump_power_mw=household_heat_pump_power[household.name],
                initial_battery_charge_power_mw=household_charge_power[household.name],
                initial_battery_discharge_power_mw=household_discharge_power[household.name],
            )
            household_results.extend(dispatch)

            if dispatch: # 保存当前家庭在本批产品调度结束后的实际状态，作为下一次开市优化的初始状态。
                household_energy[household.name] = (
                    dispatch[-1].soc_after * household.battery_capacity_mwh
                )
                household_heat_pump_power[household.name] = dispatch[
                    -1
                ].heat_pump_power_mw
                household_charge_power[household.name] = dispatch[
                    -1
                ].battery_charge_power_mw
                household_discharge_power[household.name] = dispatch[
                    -1
                ].battery_discharge_power_mw

        market_results.extend(opening_results)
        cleared_products.update(opening.products)

    return SimulationResult(
        settings=settings,
        market_results=tuple(
            sorted(market_results, key=lambda result: result.delivery_start)),
        household_results=tuple(
            sorted(household_results,
                key=lambda result: (result.delivery_start, result.unit_name),)),
        household_flexibility_results=tuple(
            sorted(household_flexibility_results,
                key=lambda result: (result.delivery_start, result.unit_name),)),
    )
