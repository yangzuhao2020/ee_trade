"""各出清算法共享的报价限价与订单标识校验、交付时段获取及需求统计。"""

from datetime import datetime
from ..errors import InputValidationError
from ..models import ClearedDemandBid, DemandBid, MarketSettings, SupplyOffer, sum_cleared_demand


_EPSILON = 1e-9


def _validate_price_limits(
    orders: list[SupplyOffer] | list[DemandBid],
    price_field: str,
    label: str,
    settings: MarketSettings,
) -> None:
    for order in orders:
        price = getattr(order, price_field)
        if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
            raise InputValidationError(
                f"{label} for {order.identifier!r} ({price:.6f} EUR/MWh) is outside "
                "the configured bid-price limits."
            )


def validate_offer_prices(
    offers: list[SupplyOffer], settings: MarketSettings
) -> None:
    """供给报价价格超出市场配置的价格上下限时报错。"""

    _validate_price_limits(offers, "bid_price_eur_per_mwh", "Bid price", settings)


def validate_demand_prices(
    demand_bids: list[DemandBid], settings: MarketSettings
) -> None:
    """检查每笔需求买单的报价是否在市场配置的最低报价与最高报价之间（含边界值）。
    遇到越界报价时抛出 InputValidationError，并提示订单标识和报价；全部通过时返回 None。
    """

    _validate_price_limits(
        demand_bids, "price_eur_per_mwh", "Demand bid price", settings
    )


def _validate_order_ids(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
) -> None:
    bid_ids = [bid.identifier for bid in demand_bids]
    offer_ids = [offer.identifier for offer in supply_offers]
    if len(bid_ids) != len(set(bid_ids)):
        raise ValueError("Demand bid identifiers must be unique within a product.")
    if len(offer_ids) != len(set(offer_ids)):
        raise ValueError("Supply offer identifiers must be unique within a product.")


def _single_product(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> tuple[datetime, datetime]:
    """读取交付时段；调用方保证订单非空且属于同一产品。"""
    order = (demand_bids or supply_offers)[0]
    return order.delivery_start, order.delivery_end


def _demand_totals(cleared_demands: tuple[ClearedDemandBid, ...]) -> dict[str, float]:
    """计算各出清算法共用的结果字段：申报需求总量、未满足的刚性负荷和未完成的出口量；成交电量仍由各算法自行汇总。"""
    return {
        "requested_demand_mwh": sum_cleared_demand(cleared_demands, "bid.volume_mwh"),
        "unserved_load_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", demand_type="inelastic_load"
        ),
        "unfulfilled_export_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", bid_type="export"
        ),
    }
