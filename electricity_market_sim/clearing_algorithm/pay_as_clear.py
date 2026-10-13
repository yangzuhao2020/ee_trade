"""采用稳定排序的单节点、单产品统一价格出清。
按报价排序撮合买卖订单，允许订单部分成交。
以已成交卖单的最高报价作为统一结算价；无成交时价格为 0。
返回成交记录、未满足需求和边际参与者等单时段出清结果。"""

from collections import defaultdict
from ..market_models import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    DemandBid,
    MarketClearingResult,
    SupplyOffer,
    Trade,
)
from .common import (
    _EPSILON,
    _demand_totals,
    _single_product,
    _validate_order_ids,
)


def clear_pay_as_clear(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> MarketClearingResult:
    """按稳定的报价优先顺序对单个产品进行出清，并采用统一价格结算。

    供给按 `(price, unit_name)` 排序，需求按 `(-price, unit_name)` 排序。
    这些稳定的排序键有意替代 ASSUME 在同价时的随机排序方式，
    使学习场景的结果可复现。同价时，学习机组的最小出力报价段
    优先于其灵活出力报价段出清，从而使成交电量的报价段归属确定。
    这种同价排序规则并不强制机组达到最小调度出力。
    """

    if not demand_bids and not supply_offers:
        raise ValueError("Cannot clear an empty market.")
    _validate_order_ids(
        demand_bids,
        supply_offers,
        supply_error=(
            "Supply offer identifiers must be unique within a product; "
            "unit names must be unique when no offer_id is supplied."
        ),
    )
    delivery_start, delivery_end = _single_product(demand_bids, supply_offers)

    # 需求按价格从高到低排序；同价时再按名称和原始顺序稳定打破平局。
    ordered_demands = sorted(
        enumerate(demand_bids),
        key=lambda item: (-item[1].price_eur_per_mwh, item[1].unit_name, item[0]),
    )
    ordered_supply = sorted(
        supply_offers,
        key=lambda offer: (
            offer.bid_price_eur_per_mwh,
            offer.unit_name,
            offer.offer_segment != "learning_minimum",
            offer.identifier,
        ),
    )  # 按价格从小到大排。
    accepted_by_offer: dict[str, float] = defaultdict(float)
    supply_index = 0
    remaining_supply = (
        ordered_supply[0].offered_energy_mwh if ordered_supply else 0.0
    )
    marginal_unit_name: str | None = None
    accepted_by_demand_index = [0.0] * len(demand_bids)
    matched_orders: list[tuple[DemandBid, SupplyOffer, float]] = []

    for demand_index, demand in ordered_demands:
        remaining_demand = demand.volume_mwh # 记录这个需求单还剩多少电量没满足。
        while remaining_demand > _EPSILON and supply_index < len(ordered_supply):
            offer = ordered_supply[supply_index] # 当前取一个供给报价
            if offer.bid_price_eur_per_mwh > demand.price_eur_per_mwh:
                # 如果卖方报价比买方最高愿付价格还高，就不成交。
                break
            accepted = min(remaining_demand, remaining_supply)
            # 取需求和供给的最小值作为成交量。
            accepted_by_offer[offer.identifier] += accepted
            accepted_by_demand_index[demand_index] += accepted
            if accepted > _EPSILON:
                matched_orders.append((demand, offer, accepted))
            remaining_demand -= accepted # 这个需求单还剩多少电量没满足。
            remaining_supply -= accepted # 这个供给报价还剩多少电量没成交。
            if accepted > _EPSILON:
                # This offer is the latest accepted offer in the deterministic
                # merit order, so it is the product's marginal unit so far.
                marginal_unit_name = offer.unit_name
            if remaining_supply <= _EPSILON:
                supply_index += 1
                if supply_index < len(ordered_supply):
                    remaining_supply = ordered_supply[supply_index].offered_energy_mwh

    accepted_prices = [
        offer.bid_price_eur_per_mwh
        for offer in ordered_supply
        if accepted_by_offer[offer.identifier] > _EPSILON
    ]
    clearing_price = max(accepted_prices) if accepted_prices else 0.0
    cleared_offers = tuple(
        ClearedSupplyOffer(
            offer=offer,
            accepted_energy_mwh=accepted_by_offer[offer.identifier],
            clearing_price_eur_per_mwh=clearing_price,
        )
        for offer in sorted(
            supply_offers,
            key=lambda offer: (offer.unit_name, offer.offer_segment, offer.identifier),
        )
    )
    cleared_demands = tuple(
        ClearedDemandBid(
            bid=bid,
            accepted_energy_mwh=accepted_by_demand_index[index],
            clearing_price_eur_per_mwh=clearing_price,
        )
        for index, bid in enumerate(demand_bids)
    )
    cleared_energy = sum(item.accepted_energy_mwh for item in cleared_offers)
    trades = tuple(
        Trade(
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            buyer_bid_id=demand.identifier,
            seller_offer_id=offer.identifier,
            trade_energy_mwh=energy,
            trade_price_eur_per_mwh=clearing_price,
        )
        for demand, offer, energy in matched_orders
    )

    return MarketClearingResult(
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        **_demand_totals(cleared_demands),
        cleared_energy_mwh=cleared_energy,
        clearing_price_eur_per_mwh=clearing_price,
        offers=cleared_offers,
        demand_bids=cleared_demands,
        marginal_unit_name=marginal_unit_name,
        trades=trades,
    )
