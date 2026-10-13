"""基于边际成本的朴素电价预测、策略所需的预测时间点及开市数据完整性检查。"""

from collections.abc import Iterable
from datetime import datetime, timedelta
from ...bidding import available_power_mw, naive_offer
from ...clearing import clear_pay_as_clear
from ...models import ExchangeSchedule, ExchangeUnit, MarketOpening, MarketSettings, PowerPlant
from ...utils import format_timestamp
from .demand import _demand_bids_for_product, _exchange_import_offer

_FORECAST_HOURS = 12


def _naive_price_forecasts(
    *,
    products: list[tuple[datetime, datetime]],
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
    marginal_costs: dict[str, float],
    availability_profiles: dict[str, dict[datetime, float]],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    forecast_starts: set[datetime] | None = None,
) -> dict[datetime, float]:
    """按边际成本报价模拟出清，为机组和储能策略提供朴素 EOM 电价预测。

    使用刚性负荷、机组可用容量及进出口计划，逐时段按 pay_as_clear 出清，
    忽略弹性需求、储能及机组启停约束。返回“时段起点 -> 电价（EUR/MWh）”；
    电价为成交卖单的最高报价，无成交时为 0.0。
    """

    if forecast_starts is None:
        # 默认预测各产品起点及其后 12 小时的逐小时电价，并去重。
        forecast_starts = {
            delivery_start + timedelta(hours=hour)
            for delivery_start, _ in products
            for hour in range(_FORECAST_HOURS + 1)
        }
    prices: dict[datetime, float] = {}
    for delivery_start in sorted(forecast_starts):
        delivery_end = delivery_start + settings.product_duration
        # 刚性负荷按市场最高报价买电；启用电力交换时加入出口需求。
        demand_bids, exchange_schedule = _demand_bids_for_product(
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
        if exchange_unit is not None and exchange_schedule is not None:
            # 将计划进口加入供给。
            import_offer = _exchange_import_offer(
                exchange_unit, exchange_schedule, delivery_start, delivery_end)
            offers.append(import_offer)
            
        clearing_result = clear_pay_as_clear(demand_bids, offers)
        prices[delivery_start] = clearing_result.clearing_price_eur_per_mwh
    return prices


def _forecast_starts(
    delivery_starts: Iterable[datetime],
    *,
    settings: MarketSettings,
    requires_price_forecast: bool = False,
    extra_forecast_offsets: tuple[int, ...] = (),
    requires_storage_price_forecast: bool = False,
) -> set[datetime]:
    """返回各个已启用的策略类别为给定交付时段所需的预测时间点。
    机组需要从交付开始时刻起共 _FORECAST_HOURS + 1 个小时的预测。扩展模块显式
    声明各自所需的时间偏移；储能额外需要以交付时刻为中心的对称窗口，并截取在
    仿真时间范围内。
    """

    starts: set[datetime] = set()
    for delivery_start in delivery_starts:
        if requires_price_forecast:
            starts.update(
                delivery_start + timedelta(hours=offset)
                for offset in range(_FORECAST_HOURS + 1)
            )
        starts.update(
            delivery_start + timedelta(hours=offset)
            for offset in extra_forecast_offsets
        )
        if requires_storage_price_forecast:
            for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1):
                forecast_start = delivery_start + timedelta(hours=offset)
                if settings.start <= forecast_start <= settings.end:
                    starts.add(forecast_start)
    return starts

def _missing_profile_times(
    profile: dict[datetime, object] | None, required_times: set[datetime]
) -> list[datetime]:
    return sorted(
        time for time in required_times if profile is None or time not in profile
    )

def _opening_coverage_issues(
    *,
    opening: MarketOpening,
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    availability_profiles: dict[str, dict[datetime, float]],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, ExchangeSchedule],
    requires_price_forecast: bool,
    extra_forecast_offsets: tuple[int, ...] = (),
    requires_storage_price_forecast: bool = False,
) -> list[str]:
    """返回一次开市无法完整报价的全部已知原因。"""

    issues: list[str] = []
    delivery_times = {delivery_start for delivery_start, _ in opening.products}
    beyond_end = [
        (delivery_start, delivery_end)
        for delivery_start, delivery_end in opening.products
        if delivery_end > settings.end
    ]
    if beyond_end:
        issues.append(
            "交付时段超出仿真结束时间: "
            + ", ".join(
                f"{format_timestamp(start)}–{format_timestamp(end)}" for start, end in beyond_end
            )
        )
    forecast_times = _forecast_starts(
        delivery_times,
        settings=settings,
        requires_price_forecast=requires_price_forecast,
        extra_forecast_offsets=extra_forecast_offsets,
        requires_storage_price_forecast=requires_storage_price_forecast,
    )
    required_times = delivery_times | forecast_times

    def add_missing_issue(source: str, missing: list[datetime]) -> None:
        if not missing:
            return
        kind = (
            "报价预测数据"
            if any(time in forecast_times for time in missing)
            else "交付数据"
        )
        issues.append(
            f"{kind}缺少 {source}: " + ", ".join(format_timestamp(time) for time in missing)
        )

    for demand_unit in demand_units:
        if demand_unit.is_elastic:
            continue
        missing = _missing_profile_times(
            demand_profiles.get(demand_unit.name), required_times
        )
        add_missing_issue(f"demand_df.csv/{demand_unit.name}", missing)
    for plant in plants:
        if plant.name not in availability_profiles:
            continue
        missing = _missing_profile_times(
            availability_profiles[plant.name], required_times
        )
        add_missing_issue(f"availability_df.csv/{plant.name}", missing)
    if exchange_unit is not None:
        missing = _missing_profile_times(exchange_profiles, required_times)
        add_missing_issue("exchanges_df.csv", missing)
    return issues
