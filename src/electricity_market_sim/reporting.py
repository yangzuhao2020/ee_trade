"""CSV result writers with stable, documented schemas."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import SimulationResult


def _timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


def _number(value: float) -> str:
    return f"{value:.6f}"


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(output_dir: Path, simulation_result: SimulationResult) -> None:
    """Write the agreed market, unit, and operator result files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(
        simulation_result.market_results,
        key=lambda market_result: market_result.delivery_start,
    )

    market_rows = []
    unit_rows = []
    offer_rows = []
    demand_rows = []
    exchange_rows = []
    operator_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "accepted_energy_mwh": 0.0,
            "revenue_eur": 0.0,
            "variable_cost_eur": 0.0,
            "startup_cost_eur": 0.0,
            "total_cost_eur": 0.0,
            "profit_eur": 0.0,
        }
    )
    for market_result in sorted_results:
        opening_time = market_result.opening_time or market_result.delivery_start
        opening_id = opening_time.isoformat()
        duration_hours = (
            market_result.delivery_end - market_result.delivery_start
        ).total_seconds() / 3600
        market_rows.append(
            {
                "market_id": simulation_result.settings.market_id,
                "opening_id": opening_id,
                "opening_time": _timestamp(opening_time),
                "delivery_start": _timestamp(market_result.delivery_start),
                "delivery_end": _timestamp(market_result.delivery_end),
                "duration_hours": _number(duration_hours),
                "requested_demand_energy_mwh": _number(
                    market_result.requested_demand_mwh
                ),
                "requested_local_demand_energy_mwh": _number(
                    market_result.requested_local_demand_mwh
                ),
                "cleared_local_demand_energy_mwh": _number(
                    market_result.cleared_local_demand_mwh
                ),
                "requested_inelastic_demand_mwh": _number(
                    market_result.requested_inelastic_demand_mwh
                ),
                "cleared_inelastic_demand_mwh": _number(
                    market_result.cleared_inelastic_demand_mwh
                ),
                "requested_elastic_demand_mwh": _number(
                    market_result.requested_elastic_demand_mwh
                ),
                "cleared_elastic_demand_mwh": _number(
                    market_result.cleared_elastic_demand_mwh
                ),
                "unaccepted_elastic_demand_mwh": _number(
                    market_result.unaccepted_elastic_demand_mwh
                ),
                "cleared_energy_mwh": _number(market_result.cleared_energy_mwh),
                "unserved_load_mwh": _number(market_result.unserved_load_mwh),
                "requested_export_mwh": _number(market_result.requested_export_mwh),
                "cleared_export_mwh": _number(market_result.cleared_export_mwh),
                "unfulfilled_export_mwh": _number(
                    market_result.unfulfilled_export_mwh
                ),
                "unserved_total_demand_mwh": _number(
                    market_result.unserved_demand_mwh
                ),
                "offered_import_mwh": _number(market_result.offered_import_mwh),
                "cleared_import_mwh": _number(market_result.cleared_import_mwh),
                "net_exchange_mwh": _number(market_result.net_exchange_mwh),
                "exchange_market_cash_flow_eur": _number(
                    market_result.exchange_cash_flow_eur
                ),
                "clearing_price_eur_per_mwh": _number(
                    market_result.clearing_price_eur_per_mwh
                ),
                "pricing_method": market_result.pricing_method,
                "total_transaction_value_eur": _number(
                    market_result.transaction_value_eur
                ),
            }
        )
        for cleared in market_result.demand_bids:
            demand_rows.append(
                {
                    "opening_id": opening_id,
                    "opening_time": _timestamp(opening_time),
                    "delivery_start": _timestamp(cleared.bid.delivery_start),
                    "delivery_end": _timestamp(cleared.bid.delivery_end),
                    "unit_name": cleared.bid.unit_name,
                    "unit_operator": cleared.bid.operator,
                    "bid_id": cleared.bid.identifier,
                    "bid_type": cleared.bid.bid_type,
                    "demand_type": cleared.bid.demand_type or "",
                    "bid_price_eur_per_mwh": _number(
                        cleared.bid.price_eur_per_mwh
                    ),
                    "requested_energy_mwh": _number(cleared.bid.volume_mwh),
                    "accepted_energy_mwh": _number(cleared.accepted_energy_mwh),
                    "unaccepted_energy_mwh": _number(cleared.unserved_energy_mwh),
                    "clearing_price_eur_per_mwh": _number(
                        cleared.clearing_price_eur_per_mwh
                    ),
                    "payment_eur": _number(cleared.payment_eur),
                }
            )
        powerplant_offers = [
            cleared
            for cleared in market_result.offers
            if cleared.offer.offer_type == "power_plant"
        ]
        for cleared in powerplant_offers:
            offer = cleared.offer
            offer_rows.append(
                {
                    "opening_id": opening_id,
                    "opening_time": _timestamp(opening_time),
                    "delivery_start": _timestamp(offer.delivery_start),
                    "delivery_end": _timestamp(offer.delivery_end),
                    "unit_name": offer.unit_name,
                    "unit_operator": offer.operator,
                    "technology": offer.technology,
                    "offer_id": offer.identifier,
                    "bid_id": offer.complex_identifier,
                    "bid_type": offer.bid_type,
                    "min_acceptance_ratio": _number(offer.min_acceptance_ratio),
                    "parent_bid_id": offer.parent_bid_id or "",
                    "offer_segment": offer.offer_segment,
                    "marginal_cost_eur_per_mwh": _number(
                        offer.marginal_cost_eur_per_mwh
                    ),
                    "bid_price_eur_per_mwh": _number(offer.bid_price_eur_per_mwh),
                    "offered_power_mw": _number(offer.offered_power_mw),
                    "offered_energy_mwh": _number(offer.offered_energy_mwh),
                    "accepted_power_mw": _number(cleared.accepted_power_mw),
                    "accepted_energy_mwh": _number(cleared.accepted_energy_mwh),
                    "clearing_price_eur_per_mwh": _number(
                        cleared.clearing_price_eur_per_mwh
                    ),
                    "accepted_price_eur_per_mwh": _number(
                        cleared.clearing_price_eur_per_mwh
                    ),
                    "revenue_eur": _number(cleared.revenue_eur),
                    "variable_cost_eur": _number(cleared.variable_cost_eur),
                    "startup_cost_eur": _number(cleared.startup_cost_eur),
                    "total_cost_eur": _number(cleared.total_cost_eur),
                    "profit_eur": _number(cleared.profit_eur),
                }
            )

        offers_by_unit: dict[str, list] = defaultdict(list)
        for cleared in powerplant_offers:
            offers_by_unit[cleared.offer.unit_name].append(cleared)
        for unit_name, cleared_offers in sorted(offers_by_unit.items()):
            first_offer = cleared_offers[0].offer
            offered_energy_mwh = sum(
                cleared.offer.offered_energy_mwh for cleared in cleared_offers
            )
            offered_power_mw = sum(
                cleared.offer.offered_power_mw for cleared in cleared_offers
            )
            accepted_energy_mwh = sum(
                cleared.accepted_energy_mwh for cleared in cleared_offers
            )
            accepted_power_mw = accepted_energy_mwh / duration_hours
            weighted_bid_price = (
                sum(
                    cleared.offer.bid_price_eur_per_mwh
                    * cleared.offer.offered_energy_mwh
                    for cleared in cleared_offers
                )
                / offered_energy_mwh
                if offered_energy_mwh > 0
                else first_offer.bid_price_eur_per_mwh
            )
            revenue_eur = sum(cleared.revenue_eur for cleared in cleared_offers)
            variable_cost_eur = sum(
                cleared.variable_cost_eur for cleared in cleared_offers
            )
            startup_cost_eur = sum(
                cleared.startup_cost_eur for cleared in cleared_offers
            )
            total_cost_eur = sum(cleared.total_cost_eur for cleared in cleared_offers)
            profit_eur = sum(cleared.profit_eur for cleared in cleared_offers)
            unit_rows.append(
                {
                    "opening_id": opening_id,
                    "opening_time": _timestamp(opening_time),
                    "delivery_start": _timestamp(market_result.delivery_start),
                    "delivery_end": _timestamp(market_result.delivery_end),
                    "unit_name": unit_name,
                    "unit_operator": first_offer.operator,
                    "technology": first_offer.technology,
                    "marginal_cost_eur_per_mwh": _number(
                        first_offer.marginal_cost_eur_per_mwh
                    ),
                    # For a two-part offer this is the volume-weighted price;
                    # offer_results.csv preserves the individual bid prices.
                    "bid_price_eur_per_mwh": _number(weighted_bid_price),
                    "offered_power_mw": _number(offered_power_mw),
                    "offered_energy_mwh": _number(offered_energy_mwh),
                    "accepted_power_mw": _number(accepted_power_mw),
                    "accepted_energy_mwh": _number(accepted_energy_mwh),
                    "clearing_price_eur_per_mwh": _number(
                        market_result.clearing_price_eur_per_mwh
                    ),
                    "revenue_eur": _number(revenue_eur),
                    "variable_cost_eur": _number(variable_cost_eur),
                    "startup_cost_eur": _number(startup_cost_eur),
                    "total_cost_eur": _number(total_cost_eur),
                    "profit_eur": _number(profit_eur),
                }
            )
            totals = operator_totals[first_offer.operator]
            totals["accepted_energy_mwh"] += accepted_energy_mwh
            totals["revenue_eur"] += revenue_eur
            totals["variable_cost_eur"] += variable_cost_eur
            totals["startup_cost_eur"] += startup_cost_eur
            totals["total_cost_eur"] += total_cost_eur
            totals["profit_eur"] += profit_eur

        import_offer = next(
            (
                cleared
                for cleared in market_result.offers
                if cleared.offer.offer_type == "import"
            ),
            None,
        )
        export_bid = next(
            (
                cleared
                for cleared in market_result.demand_bids
                if cleared.bid.bid_type == "export"
            ),
            None,
        )
        if import_offer is not None or export_bid is not None:
            if import_offer is None or export_bid is None:
                raise ValueError("An Exchange market result must contain both orders.")
            exchange_rows.append(
                {
                    "opening_id": opening_id,
                    "opening_time": _timestamp(opening_time),
                    "delivery_start": _timestamp(market_result.delivery_start),
                    "delivery_end": _timestamp(market_result.delivery_end),
                    "exchange_name": import_offer.offer.unit_name,
                    "exchange_operator": import_offer.offer.operator,
                    "offered_import_power_mw": _number(
                        import_offer.offer.offered_power_mw
                    ),
                    "offered_import_mwh": _number(
                        import_offer.offer.offered_energy_mwh
                    ),
                    "cleared_import_power_mw": _number(
                        import_offer.accepted_power_mw
                    ),
                    "cleared_import_mwh": _number(import_offer.accepted_energy_mwh),
                    "requested_export_power_mw": _number(
                        export_bid.bid.volume_mwh / market_result.duration_hours
                    ),
                    "requested_export_mwh": _number(export_bid.bid.volume_mwh),
                    "cleared_export_power_mw": _number(export_bid.accepted_power_mw),
                    "cleared_export_mwh": _number(export_bid.accepted_energy_mwh),
                    "unfulfilled_export_mwh": _number(
                        export_bid.unserved_energy_mwh
                    ),
                    "net_exchange_mwh": _number(market_result.net_exchange_mwh),
                    "import_revenue_eur": _number(market_result.import_revenue_eur),
                    "export_payment_eur": _number(market_result.export_payment_eur),
                    "exchange_market_cash_flow_eur": _number(
                        market_result.exchange_cash_flow_eur
                    ),
                }
            )

    _write_rows(
        output_dir / "market_results.csv",
        [
            "market_id",
            "opening_id",
            "opening_time",
            "delivery_start",
            "delivery_end",
            "duration_hours",
            "requested_demand_energy_mwh",
            "requested_local_demand_energy_mwh",
            "cleared_local_demand_energy_mwh",
            "requested_inelastic_demand_mwh",
            "cleared_inelastic_demand_mwh",
            "requested_elastic_demand_mwh",
            "cleared_elastic_demand_mwh",
            "unaccepted_elastic_demand_mwh",
            "cleared_energy_mwh",
            "unserved_load_mwh",
            "requested_export_mwh",
            "cleared_export_mwh",
            "unfulfilled_export_mwh",
            "unserved_total_demand_mwh",
            "offered_import_mwh",
            "cleared_import_mwh",
            "net_exchange_mwh",
            "exchange_market_cash_flow_eur",
            "clearing_price_eur_per_mwh",
            "pricing_method",
            "total_transaction_value_eur",
        ],
        market_rows,
    )
    _write_rows(
        output_dir / "demand_results.csv",
        [
            "opening_id",
            "opening_time",
            "delivery_start",
            "delivery_end",
            "unit_name",
            "unit_operator",
            "bid_id",
            "bid_type",
            "demand_type",
            "bid_price_eur_per_mwh",
            "requested_energy_mwh",
            "accepted_energy_mwh",
            "unaccepted_energy_mwh",
            "clearing_price_eur_per_mwh",
            "payment_eur",
        ],
        demand_rows,
    )
    _write_rows(
        output_dir / "unit_results.csv",
        [
            "opening_id",
            "opening_time",
            "delivery_start",
            "delivery_end",
            "unit_name",
            "unit_operator",
            "technology",
            "marginal_cost_eur_per_mwh",
            "bid_price_eur_per_mwh",
            "offered_power_mw",
            "offered_energy_mwh",
            "accepted_power_mw",
            "accepted_energy_mwh",
            "clearing_price_eur_per_mwh",
            "revenue_eur",
            "variable_cost_eur",
            "startup_cost_eur",
            "total_cost_eur",
            "profit_eur",
        ],
        unit_rows,
    )
    _write_rows(
        output_dir / "operator_results.csv",
        [
            "unit_operator",
            "accepted_energy_mwh",
            "revenue_eur",
            "variable_cost_eur",
            "startup_cost_eur",
            "total_cost_eur",
            "profit_eur",
        ],
        (
            {
                "unit_operator": operator,
                **{field: _number(value) for field, value in totals.items()},
            }
            for operator, totals in sorted(operator_totals.items())
        ),
    )
    if exchange_rows:
        _write_rows(
            output_dir / "exchange_results.csv",
            [
                "opening_id",
                "opening_time",
                "delivery_start",
                "delivery_end",
                "exchange_name",
                "exchange_operator",
                "offered_import_power_mw",
                "offered_import_mwh",
                "cleared_import_power_mw",
                "cleared_import_mwh",
                "requested_export_power_mw",
                "requested_export_mwh",
                "cleared_export_power_mw",
                "cleared_export_mwh",
                "unfulfilled_export_mwh",
                "net_exchange_mwh",
                "import_revenue_eur",
                "export_payment_eur",
                "exchange_market_cash_flow_eur",
            ],
            exchange_rows,
        )
    if any(row["offer_segment"] != "single" for row in offer_rows):
        _write_rows(
            output_dir / "offer_results.csv",
            [
                "opening_id",
                "opening_time",
                "delivery_start",
                "delivery_end",
                "unit_name",
                "unit_operator",
                "technology",
                "offer_id",
                "bid_id",
                "bid_type",
                "min_acceptance_ratio",
                "parent_bid_id",
                "offer_segment",
                "marginal_cost_eur_per_mwh",
                "bid_price_eur_per_mwh",
                "offered_power_mw",
                "offered_energy_mwh",
                "accepted_power_mw",
                "accepted_energy_mwh",
                "clearing_price_eur_per_mwh",
                "accepted_price_eur_per_mwh",
                "revenue_eur",
                "variable_cost_eur",
                "startup_cost_eur",
                "total_cost_eur",
                "profit_eur",
            ],
            offer_rows,
        )
