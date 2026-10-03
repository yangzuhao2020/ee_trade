"""Shared validation and result helpers for the clearing algorithms."""

from __future__ import annotations

from datetime import datetime

from ..errors import InputValidationError
from ..market_models import (
    ClearedDemandBid,
    DemandBid,
    SupplyOffer,
    sum_cleared_demand,
)
from ..models import MarketSettings


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
    """Reject supply prices outside the configured market limits."""

    _validate_price_limits(offers, "bid_price_eur_per_mwh", "Bid price", settings)


def validate_demand_prices(
    demand_bids: list[DemandBid], settings: MarketSettings
) -> None:
    """Reject demand prices outside the configured market limits."""

    _validate_price_limits(
        demand_bids, "price_eur_per_mwh", "Demand bid price", settings
    )


def _validate_order_ids(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    *,
    supply_error: str,
) -> None:
    bid_ids = [bid.identifier for bid in demand_bids]
    offer_ids = [offer.identifier for offer in supply_offers]
    if len(bid_ids) != len(set(bid_ids)):
        raise ValueError("Demand bid identifiers must be unique within a product.")
    if len(offer_ids) != len(set(offer_ids)):
        raise ValueError(supply_error)


def _single_product(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> tuple[datetime, datetime]:
    products = {
        (bid.delivery_start, bid.delivery_end) for bid in demand_bids
    } | {(offer.delivery_start, offer.delivery_end) for offer in supply_offers}
    if len(products) != 1:
        raise ValueError("All bids and offers passed to one clearing must share a product.")
    return products.pop()


def _demand_totals(cleared_demands: tuple[ClearedDemandBid, ...]) -> dict[str, float]:
    """Shared result fields; retain each algorithm's own traded-energy sum."""

    return {
        "requested_demand_mwh": sum_cleared_demand(cleared_demands, "bid.volume_mwh"),
        "unserved_load_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", demand_type="inelastic_load"
        ),
        "unfulfilled_export_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", bid_type="export"
        ),
    }
