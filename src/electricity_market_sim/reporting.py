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
    operator_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"accepted_energy_mwh": 0.0, "revenue_eur": 0.0, "variable_cost_eur": 0.0, "profit_eur": 0.0}
    )
    for market_result in sorted_results:
        duration_hours = (
            market_result.delivery_end - market_result.delivery_start
        ).total_seconds() / 3600
        market_rows.append(
            {
                "market_id": simulation_result.settings.market_id,
                "delivery_start": _timestamp(market_result.delivery_start),
                "delivery_end": _timestamp(market_result.delivery_end),
                "duration_hours": _number(duration_hours),
                "requested_demand_energy_mwh": _number(
                    market_result.requested_demand_mwh
                ),
                "cleared_energy_mwh": _number(market_result.cleared_energy_mwh),
                "unserved_load_mwh": _number(market_result.unserved_load_mwh),
                "clearing_price_eur_per_mwh": _number(
                    market_result.clearing_price_eur_per_mwh
                ),
                "total_transaction_value_eur": _number(
                    market_result.transaction_value_eur
                ),
            }
        )
        for cleared in market_result.offers:
            offer = cleared.offer
            unit_rows.append(
                {
                    "delivery_start": _timestamp(offer.delivery_start),
                    "delivery_end": _timestamp(offer.delivery_end),
                    "unit_name": offer.unit_name,
                    "unit_operator": offer.operator,
                    "technology": offer.technology,
                    "marginal_cost_eur_per_mwh": _number(offer.marginal_cost_eur_per_mwh),
                    "bid_price_eur_per_mwh": _number(offer.bid_price_eur_per_mwh),
                    "offered_power_mw": _number(offer.offered_power_mw),
                    "offered_energy_mwh": _number(offer.offered_energy_mwh),
                    "accepted_power_mw": _number(cleared.accepted_power_mw),
                    "accepted_energy_mwh": _number(cleared.accepted_energy_mwh),
                    "clearing_price_eur_per_mwh": _number(cleared.clearing_price_eur_per_mwh),
                    "revenue_eur": _number(cleared.revenue_eur),
                    "variable_cost_eur": _number(cleared.variable_cost_eur),
                    "profit_eur": _number(cleared.profit_eur),
                }
            )
            totals = operator_totals[offer.operator]
            totals["accepted_energy_mwh"] += cleared.accepted_energy_mwh
            totals["revenue_eur"] += cleared.revenue_eur
            totals["variable_cost_eur"] += cleared.variable_cost_eur
            totals["profit_eur"] += cleared.profit_eur

    _write_rows(
        output_dir / "market_results.csv",
        [
            "market_id",
            "delivery_start",
            "delivery_end",
            "duration_hours",
            "requested_demand_energy_mwh",
            "cleared_energy_mwh",
            "unserved_load_mwh",
            "clearing_price_eur_per_mwh",
            "total_transaction_value_eur",
        ],
        market_rows,
    )
    _write_rows(
        output_dir / "unit_results.csv",
        [
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
