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
    HouseholdResult,
    HouseholdDispatchResult,
    HouseholdUnit,
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
    """运行 V3 的 15 分钟家庭用户及按报价结算（pay-as-bid）交易流程。
    随后以相同输入再完整仿真一次不使用电池的基准方案，只用于计算居民的实际节省。
    """
    plants = load_powerplants(input_path / "powerplant_units.csv")
    demand_units = load_demand_units(input_path / "demand_units.csv")
    if any(unit.is_elastic for unit in demand_units):
        raise InputValidationError(
            "按报价结算（pay_as_bid）的家庭场景暂不支持弹性需求单元。")
    households = load_household_units(input_path / "residential_dsm_units.csv")
    validate_unique_names(
        [
            *(plant.name for plant in plants),
            *(unit.name for unit in demand_units),
            *(household.name for household in households),
        ],
        "发电机组、需求单元和家庭用户",)

    demand_profiles = load_time_series_profiles(
        input_path / "demand_df.csv",
        tuple(unit.profile_column for unit in demand_units),
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

    products = [product for opening in openings for product in opening.products]
    if len(products) != len(set(products)):
        raise ValueError("同一交付产品不能在两次开市中重复报价。")
    for delivery_start, _ in products:
        if delivery_start not in fuel_prices:
            raise InputValidationError(
                f"fuel_prices_df.csv 中缺少 {format_timestamp(delivery_start)} 时刻的燃料价格。")

    def run_openings(
        households: tuple[HouseholdUnit, ...],
        evaluate_flexibility: bool,
    ) -> tuple[
        list[MarketClearingResult],
        list[HouseholdDispatchResult],
    ]:
        """按开市顺序完成居民优化、报价、出清及设备调度，返回全部结果。"""
        household_energy = {household.name: household.initial_energy_mwh for household in households}
        household_heat_pump_power = {household.name: 0.0 for household in households}
        household_charge_power = {household.name: 0.0 for household in households}
        household_discharge_power = {household.name: 0.0 for household in households} 
        # 以上为初始化变量
        
        market_results: list[MarketClearingResult] = []
        household_results = []
        for opening in openings:
            plans_by_household, bounds_by_household = plan_household_opening(
                households,
                opening.products,
                forecasts["price_EOM"],
                forecasts,
                household_energy,
                household_charge_power,
                household_discharge_power,
                evaluate_flexibility=evaluate_flexibility,
            )
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

                product_fuel_prices = fuel_prices[delivery_start]
                    
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

            purchases = _household_purchases(opening_results)
            for household in households:
                accepted_grid_energy = {
                    start: purchases.get((household.name, start), (0.0, 0.0))[0]
                    for start, _ in opening.products
                }

                dispatch = dispatch_household(
                    household,
                    plans_by_household[household.name], # 购电、充放点计划
                    accepted_grid_energy,
                    household_energy[household.name], # 取出当前家庭电池已经储存的电量
                    grid_power_bounds_mw=bounds_by_household[household.name],
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
        return market_results, household_results

    market_results, household_results = run_openings(
        households, evaluate_flexibility=True)
    # 基准方案：电池不参与运行，充放电功率上限和自放电率均设为 0，储能量保持初始值；
    # 其余输入和流程完全相同。保留自放电会使储能量低于最低 SOC 而又无法充电，导致求解不可行。
    # 基准结果仅用于计算节省，不进入本次仿真的市场、订单及设备结果。
    baseline_households = tuple(
        replace(
            household,
            battery_max_charge_power_mw=0.0,
            battery_max_discharge_power_mw=0.0,
            battery_loss_rate=0.0,
        )
        for household in households
    )
    baseline_market_results, baseline_household_results = run_openings(
        baseline_households, evaluate_flexibility=False)

    return SimulationResult(
        settings=settings,
        market_results=tuple(
            sorted(market_results, key=lambda result: result.delivery_start)),
        household_results=tuple(sorted(
            _household_results(
                household_results, market_results, baseline_household_results, baseline_market_results,
            ),
            key=lambda result: (result.dispatch.plan.delivery_start, result.dispatch.plan.unit_name),
        )),
    )


def _household_purchases(
    market_results: list[MarketClearingResult],
) -> dict[tuple[str, datetime], tuple[float, float]]:
    """按 (居民名称, 交付开始时间) 汇总居民买单的实际成交电量和实际支付。"""
    purchases: dict[tuple[str, datetime], tuple[float, float]] = {}
    for market in market_results:
        for cleared in market.demand_bids:
            if cleared.bid.demand_type == "household_load":
                purchases[(cleared.bid.unit_name, cleared.bid.delivery_start)] = (
                    cleared.accepted_energy_mwh, cleared.payment_eur)
    return purchases


def _household_results(
    household_results: list[HouseholdDispatchResult],
    market_results: list[MarketClearingResult],
    baseline_household_results: list[HouseholdDispatchResult],
    baseline_market_results: list[MarketClearingResult],
) -> tuple[HouseholdResult, ...]:
    """对比优化方案与基准方案每个产品的实际支付，以及按预测电价的计划节省，
    并保留基准方案的缺供量和 SOC，用于检查两者是否提供了同等服务。

    未提交或未成交的买单按成交电量和支付均为 0 处理。
    计划节省 = 预测电价 × (基准计划购电量 − 优化计划购电量)。
    """
    purchases = _household_purchases(market_results)
    baseline_purchases = _household_purchases(baseline_market_results)
    baseline_dispatch = {
        (result.plan.unit_name, result.plan.delivery_start): result
        for result in baseline_household_results
    }
    results: list[HouseholdResult] = []
    for result in household_results:
        key = (result.plan.unit_name, result.plan.delivery_start)
        energy, payment = purchases.get(key, (0.0, 0.0))
        baseline_energy, baseline_payment = baseline_purchases.get(key, (0.0, 0.0))
        baseline = baseline_dispatch[key]
        results.append(
            HouseholdResult(
                dispatch=result,
                baseline=baseline,
                accepted_grid_energy_mwh=energy,
                payment_eur=payment,
                baseline_grid_energy_mwh=baseline_energy,
                baseline_payment_eur=baseline_payment,
            )
        )
    return tuple(results)
