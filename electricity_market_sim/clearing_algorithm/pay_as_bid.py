"""按报价排序撮合单个交付产品的买卖订单，并按各卖单报价结算。"""

from collections import defaultdict
from ..models import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    DemandBid,
    MarketClearingResult,
    SupplyOffer,
    Trade,
)
from .common import (
    _EPSILON,
    _single_product,
    _validate_order_ids,
)


def clear_pay_as_bid(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> MarketClearingResult:
    """对单个产品出清，每笔撮合成交都按卖方报价结算。"""

    if not demand_bids and not supply_offers:
        raise ValueError("Cannot clear an empty market.")
    delivery_start, delivery_end = _single_product(demand_bids, supply_offers)
    _validate_order_ids(demand_bids, supply_offers) # 检查买单是否重复、卖单是否重复


    ordered_bids = sorted(
        demand_bids,
        key=lambda bid: (-bid.price_eur_per_mwh, bid.unit_name, bid.identifier),
    ) # 买单价格降序；同价按名称、订单 ID 升序。
    ordered_offers = sorted(
        supply_offers,
        key=lambda offer: (offer.bid_price_eur_per_mwh, offer.unit_name, offer.identifier),
    ) # 卖单价格升序；同价按名称、订单 ID 升序。

    remaining_bids = {bid.identifier: bid.volume_mwh for bid in ordered_bids}
    remaining_offers = {
        offer.identifier: offer.offered_energy_mwh for offer in ordered_offers
    }
    accepted_bids = defaultdict(float)
    accepted_offers = defaultdict(float)
    payments_by_bid = defaultdict(float)
    trades = []

    bid_index = 0
    offer_index = 0
    while bid_index < len(ordered_bids) and offer_index < len(ordered_offers):
        bid = ordered_bids[bid_index]
        offer = ordered_offers[offer_index]
        # 最低卖价超过最高剩余买价，停止撮合。
        if offer.bid_price_eur_per_mwh > bid.price_eur_per_mwh + _EPSILON:
            break
        # 成交量取双方剩余电量的较小值。
        quantity = min(
            remaining_bids[bid.identifier],
            remaining_offers[offer.identifier],
        )
        if quantity > _EPSILON:
            # 每笔成交按卖方报价结算。
            trade = Trade(
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                buyer_bid_id=bid.identifier,
                seller_offer_id=offer.identifier,
                trade_energy_mwh=quantity,
                trade_price_eur_per_mwh=offer.bid_price_eur_per_mwh,
            )
            trades.append(trade)
            # 累计成交量与付款，扣减双方余量。
            accepted_bids[bid.identifier] += quantity
            accepted_offers[offer.identifier] += quantity
            payments_by_bid[bid.identifier] += trade.payment_eur
            remaining_bids[bid.identifier] -= quantity
            remaining_offers[offer.identifier] -= quantity

        # 买单满足或卖单售完后，切换到下一笔。
        if remaining_bids[bid.identifier] <= _EPSILON:
            bid_index += 1
        if remaining_offers[offer.identifier] <= _EPSILON:
            offer_index += 1

    cleared_offers = tuple(
        ClearedSupplyOffer(
            offer=offer,
            accepted_energy_mwh=accepted_offers[offer.identifier],
            clearing_price_eur_per_mwh=(
                offer.bid_price_eur_per_mwh
                if accepted_offers[offer.identifier] > _EPSILON
                else 0.0
            ),
        )
        for offer in supply_offers
    ) # 所有卖单的出清结果，包括未成交的卖单
    cleared_demands = tuple(
        ClearedDemandBid(
            bid=bid,
            accepted_energy_mwh=accepted_bids[bid.identifier],
            clearing_price_eur_per_mwh=(
                payments_by_bid[bid.identifier] / accepted_bids[bid.identifier]
                if accepted_bids[bid.identifier] > _EPSILON
                else 0.0
            ),
        )
        for bid in demand_bids
    ) # 所有买单的出清结果，包括未成交的买单
    return MarketClearingResult(
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        clearing_price_eur_per_mwh=None,
        offers=cleared_offers,
        demand_bids=cleared_demands,
        marginal_unit_name=None,
        pricing_method="pay_as_bid",
        trades=tuple(trades),
    )
