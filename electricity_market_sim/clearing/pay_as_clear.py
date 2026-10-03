"""Deterministic, single-node pay-as-clear market clearing.

按报价从低到高排序卖单；
按买方愿付价格从高到低匹配需求；
依次成交，允许最后一台机组部分成交；
记录成交电量、未满足负荷、边际机组；
用最后被接受卖单的价格作为统一出清价；
返回该时段的 MarketClearingResult。
"""

from __future__ import annotations

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
    """Clear one product using stable merit order and uniform settlement.

    Supply is ordered by `(price, unit_name)`. Demand is ordered by
    `(-price, unit_name)`. These stable keys deliberately replace ASSUME's random
    tie break so a learning scenario produces reproducible results. At equal
    price, a learning unit's minimum-output segment clears before its flexible
    segment so accepted energy has a deterministic segment attribution. This
    tie break does not impose a minimum dispatched output on the unit.
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
