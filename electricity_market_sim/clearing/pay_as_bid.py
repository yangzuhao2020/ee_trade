"""Single-product merit-order clearing settled at each seller's own bid."""

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


def clear_pay_as_bid(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> MarketClearingResult:
    """Clear one product and settle every match at the seller's offer price."""

    if not demand_bids and not supply_offers:
        raise ValueError("Cannot clear an empty market.")
    delivery_start, delivery_end = _single_product(demand_bids, supply_offers)
    _validate_order_ids(
        demand_bids,
        supply_offers,
        supply_error="Supply offer identifiers must be unique within a product.",
    )

    ordered_bids = sorted(
        demand_bids,
        key=lambda bid: (-bid.price_eur_per_mwh, bid.unit_name, bid.identifier),
    )
    ordered_offers = sorted(
        supply_offers,
        key=lambda offer: (
            offer.bid_price_eur_per_mwh,
            offer.unit_name,
            offer.identifier,
        ),
    )
    remaining_bids = {bid.identifier: bid.volume_mwh for bid in ordered_bids}
    remaining_offers = {
        offer.identifier: offer.offered_energy_mwh for offer in ordered_offers
    }
    accepted_bids: dict[str, float] = defaultdict(float)
    accepted_offers: dict[str, float] = defaultdict(float)
    payments_by_bid: dict[str, float] = defaultdict(float)
    trades: list[Trade] = []

    bid_index = 0
    offer_index = 0
    while bid_index < len(ordered_bids) and offer_index < len(ordered_offers):
        bid = ordered_bids[bid_index]
        offer = ordered_offers[offer_index]
        if offer.bid_price_eur_per_mwh > bid.price_eur_per_mwh + _EPSILON:
            break
        quantity = min(
            remaining_bids[bid.identifier],
            remaining_offers[offer.identifier],
        )
        if quantity > _EPSILON:
            trade = Trade(
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                buyer_bid_id=bid.identifier,
                seller_offer_id=offer.identifier,
                trade_energy_mwh=quantity,
                trade_price_eur_per_mwh=offer.bid_price_eur_per_mwh,
            )
            trades.append(trade)
            accepted_bids[bid.identifier] += quantity
            accepted_offers[offer.identifier] += quantity
            payments_by_bid[bid.identifier] += trade.payment_eur
            remaining_bids[bid.identifier] -= quantity
            remaining_offers[offer.identifier] -= quantity
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
    )
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
    )
    cleared_energy = sum(trade.trade_energy_mwh for trade in trades)
    return MarketClearingResult(
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        **_demand_totals(cleared_demands),
        cleared_energy_mwh=cleared_energy,
        clearing_price_eur_per_mwh=None,
        offers=cleared_offers,
        demand_bids=cleared_demands,
        marginal_unit_name=None,
        pricing_method="pay_as_bid",
        trades=tuple(trades),
    )
