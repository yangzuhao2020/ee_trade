"""基于边际成本的朴素电价预测、策略所需的预测时间点及开市数据完整性检查。"""

from collections.abc import Iterable
from datetime import datetime, timedelta
from ...bidding import available_power_mw, naive_offer
from ...clearing_algorithm import clear_pay_as_clear
from ...models import ExchangeUnit, MarketOpening, MarketSettings, PowerPlant
from ...utils import format_timestamp
from .demand import _demand_bids_for_product, _exchange_import_offer

_FORECAST_HOURS = 12


def _naive_price_forecasts(
    *,
    settings,
    plants,
    marginal_costs,
    availability_profiles,
    demand_units,
    demand_profiles,
    exchange_unit,
    exchange_profiles,
    forecast_starts,
):
    """基于刚性负荷、可用机组的边际成本报价及进出口计划模拟统一出清，忽略弹性需求、储能及机组启停约束，生成逐时段 EOM 电价预测。TODO: 预测电价算法"""

    prices: dict[datetime, float] = {}
    for delivery_start in sorted(forecast_starts):
        delivery_end = delivery_start + settings.product_duration
        # 刚性负荷按市场最高报价买电；启用电力交换时加入出口需求。
        demand_bids = _demand_bids_for_product(
            demand_units=demand_units,
            demand_profiles=demand_profiles,
            exchange_unit=exchange_unit,
            exchange_profiles=exchange_profiles,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            maximum_bid_price=settings.maximum_bid_price,
            include_elastic=False,
        )
        # 所有机组按可用容量、固定边际成本提交单段卖单。
        offers = []
        for plant in plants:
            available_capacity_mw = available_power_mw(
                plant, delivery_start, availability_profiles
            )
            offers.append(
                naive_offer(
                    plant,
                    delivery_start,
                    delivery_end,
                    available_capacity_mw,
                    marginal_costs[plant.name],
                )
            )
        if exchange_unit is not None:
            # 将计划进口加入供给。
            import_offer = _exchange_import_offer(
                exchange_unit, exchange_profiles[delivery_start], delivery_start, delivery_end)
            offers.append(import_offer)
            
        clearing_result = clear_pay_as_clear(demand_bids, offers)
        prices[delivery_start] = clearing_result.clearing_price_eur_per_mwh
    return prices


def _forecast_starts(
    delivery_starts: Iterable[datetime],
    *,
    settings: MarketSettings,
    requires_price_forecast: bool = False,
    requires_storage_price_forecast: bool = False,
    hard_forecast_horizon: bool = False,
) -> set[datetime]:
    """汇总机组和储能报价所需的参考电价时间点。
    启用相应预测时，机组取交付时刻及之后 12 小时；储能取交付时刻前后
    各 12 小时，用于计算周边均价，判断充放电方向，均按小时取点。
    前后均相对于交付时刻，当前统一使用预测价格，不混用历史成交价。
    储能的时间点限制在仿真起止范围内。
    若 hard_forecast_horizon=True，所有时间点都必须早于仿真结束时间。
    """

    starts: set[datetime] = set()
    for delivery_start in delivery_starts:
        if requires_price_forecast:
            starts.update(
                delivery_start + timedelta(hours=offset)
                for offset in range(_FORECAST_HOURS + 1)
            )
        if requires_storage_price_forecast:
            for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1):
                forecast_start = delivery_start + timedelta(hours=offset)
                if settings.start <= forecast_start <= settings.end:
                    starts.add(forecast_start)
    if hard_forecast_horizon:
        starts = {start for start in starts if start < settings.end}
    return starts


def _opening_coverage_issues(
    *,
    opening,
    settings,
    plants,
    demand_units,
    demand_profiles,
    availability_profiles,
    exchange_unit,
    exchange_profiles,
    requires_price_forecast,
    requires_storage_price_forecast=False,
    hard_forecast_horizon=False,
):
    """检查本次开市的交付是否超出仿真结束时间，以及交付和电价预测所需的
    负荷、已提供的机组可用率及已启用的进出口数据是否齐全。
    返回问题描述列表；无问题时返回空列表。
    """
    issues = []
    delivery_times = {delivery_start for delivery_start, _ in opening.products}
    beyond_end = [
        (delivery_start, delivery_end)
        for delivery_start, delivery_end in opening.products
        if delivery_end > settings.end
    ]
    if beyond_end:
        issues.append("交付时段超出仿真结束时间: " + ", ".join(
                f"{format_timestamp(start)}–{format_timestamp(end)}" for start, end in beyond_end)
        )
    forecast_times = _forecast_starts(
        delivery_times,
        settings=settings,
        requires_price_forecast=requires_price_forecast,
        requires_storage_price_forecast=requires_storage_price_forecast,
        hard_forecast_horizon=hard_forecast_horizon,
    )
    required_times = delivery_times | forecast_times

    def add_missing_issue(source: str, missing: list[datetime]) -> None:
        if not missing:
            return
        kind = ("报价预测数据"
            if any(time in forecast_times for time in missing)
            else "交付数据")
        issues.append(
            f"{kind}缺少 {source}: " + ", ".join(format_timestamp(time) for time in missing))

    for demand_unit in demand_units:
        if demand_unit.is_elastic:
            continue
        missing = sorted(required_times - demand_profiles[demand_unit.name].keys())
        add_missing_issue(f"demand_df.csv/{demand_unit.name}", missing)
    for plant in plants:
        if plant.name not in availability_profiles:
            continue
        missing = sorted(required_times - availability_profiles[plant.name].keys())
        add_missing_issue(f"availability_df.csv/{plant.name}", missing)
    if exchange_unit is not None:
        missing = sorted(required_times - exchange_profiles.keys())
        add_missing_issue("exchanges_df.csv", missing)
    return issues
