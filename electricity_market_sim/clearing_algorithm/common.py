"""各出清算法共享的报价限价、订单标识校验与交付时段获取。"""

from datetime import datetime
from ..errors import InputValidationError
from ..models import DemandBid, MarketSettings, SupplyOffer


_EPSILON = 1e-9


def validate_order_prices(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    settings: MarketSettings,
) -> None:
    """检查买卖单报价是否在市场限价范围内（含上下限）。"""
    for orders, price_field, label in (
        (demand_bids, "price_eur_per_mwh", "买单"),
        (supply_offers, "bid_price_eur_per_mwh", "卖单"),
    ):
        for order in orders:
            price = getattr(order, price_field)
            if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
                raise InputValidationError(
                    f"{label} {order.identifier!r} 的报价（{price:.6f} 欧元/兆瓦时）"
                    "超出市场配置的报价上下限。"
                )


def _validate_order_ids(demand_bids, supply_offers,):
    bid_ids = [bid.identifier for bid in demand_bids]
    offer_ids = [offer.identifier for offer in supply_offers]
    if len(bid_ids) != len(set(bid_ids)):
        raise ValueError("同一交付产品内的买单标识不得重复。")
    if len(offer_ids) != len(set(offer_ids)):
        raise ValueError("同一交付产品内的卖单标识不得重复。")


def _single_product(demand_bids, supply_offers):
    """读取交付时段；调用方保证订单非空且属于同一产品。"""
    order = (demand_bids or supply_offers)[0]
    return order.delivery_start, order.delivery_end
