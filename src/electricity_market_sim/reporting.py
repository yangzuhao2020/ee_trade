"""CSV result writers with stable, documented schemas."""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import SimulationResult


_SETTLEMENT_QUANTUM = Decimal("0.000000000001")


def _timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


def _number(value: float) -> str:
    return f"{value:.6f}"


def _optional_number(value: float | None) -> str:
    return "" if value is None else _number(value)


def _settlement_decimal(value: float) -> Decimal:
    return Decimal(str(value)).quantize(
        _SETTLEMENT_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _settlement_number(value: Decimal) -> str:
    return format(
        value.quantize(_SETTLEMENT_QUANTUM, rounding=ROUND_HALF_UP),
        "f",
    )


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
    trade_rows = []
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
        reported_energy_by_bid: dict[str, Decimal] = defaultdict(Decimal)
        reported_payments_by_bid: dict[str, Decimal] = defaultdict(Decimal)
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
                "clearing_price_eur_per_mwh": _optional_number(
                    market_result.clearing_price_eur_per_mwh
                ),
                "average_trade_price_eur_per_mwh": _optional_number(
                    market_result.average_trade_price_eur_per_mwh
                ),
                "pricing_method": market_result.pricing_method,
                "total_transaction_value_eur": _number(
                    market_result.transaction_value_eur
                ),
            }
        )
        for trade in market_result.trades:
            reported_energy = _settlement_decimal(trade.trade_energy_mwh)
            reported_price = _settlement_decimal(trade.trade_price_eur_per_mwh)
            reported_payment = (reported_energy * reported_price).quantize(
                _SETTLEMENT_QUANTUM,
                rounding=ROUND_HALF_UP,
            )
            reported_energy_by_bid[trade.buyer_bid_id] += reported_energy
            reported_payments_by_bid[trade.buyer_bid_id] += reported_payment
            trade_rows.append(
                {
                    "delivery_start": _timestamp(trade.delivery_start),
                    "delivery_end": _timestamp(trade.delivery_end),
                    "opening_id": opening_id,
                    "buyer_name": trade.buyer_name,
                    "buyer_bid_id": trade.buyer_bid_id,
                    "seller_name": trade.seller_name,
                    "seller_offer_id": trade.seller_offer_id,
                    "trade_energy_mwh": _settlement_number(reported_energy),
                    "trade_price_eur_per_mwh": _settlement_number(reported_price),
                    "payment_eur": _settlement_number(reported_payment),
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
                    "side": cleared.bid.side,
                    "bid_type": cleared.bid.bid_type,
                    "demand_type": cleared.bid.demand_type or "",
                    "bid_price_eur_per_mwh": _number(
                        cleared.bid.price_eur_per_mwh
                    ),
                    "requested_energy_mwh": _number(cleared.bid.volume_mwh),
                    "accepted_energy_mwh": (
                        _settlement_number(
                            reported_energy_by_bid[cleared.bid.identifier]
                        )
                        if market_result.trades
                        else _number(cleared.accepted_energy_mwh)
                    ),
                    "unaccepted_energy_mwh": _number(cleared.unserved_energy_mwh),
                    "clearing_price_eur_per_mwh": _number(
                        cleared.clearing_price_eur_per_mwh
                    ),
                    "payment_eur": (
                        _settlement_number(
                            reported_payments_by_bid[cleared.bid.identifier]
                        )
                        if market_result.trades
                        else _number(cleared.payment_eur)
                    ),
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
                    "side": offer.side,
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
                    "side": first_offer.side,
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
                    "clearing_price_eur_per_mwh": _optional_number(
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

    storage_rows = []
    for storage_result in simulation_result.storage_results:
        duration_hours = (
            storage_result.delivery_end - storage_result.delivery_start
        ).total_seconds() / 3600
        storage_rows.append(
            {
                "opening_id": storage_result.opening_time.isoformat(),
                "opening_time": _timestamp(storage_result.opening_time),
                "delivery_start": _timestamp(storage_result.delivery_start),
                "delivery_end": _timestamp(storage_result.delivery_end),
                "unit_name": storage_result.unit_name,
                "unit_operator": storage_result.operator,
                "technology": storage_result.technology,
                "energy_before_mwh": _number(storage_result.energy_before_mwh),
                "energy_after_mwh": _number(storage_result.energy_after_mwh),
                "soc_before": _number(storage_result.soc_before),
                "soc_after": _number(storage_result.soc_after),
                "offered_charge_power_mw": _number(
                    storage_result.offered_charge_mwh / duration_hours
                ),
                "offered_charge_mwh": _number(storage_result.offered_charge_mwh),
                "accepted_charge_power_mw": _number(
                    storage_result.accepted_charge_mwh / duration_hours
                ),
                "accepted_charge_mwh": _number(storage_result.accepted_charge_mwh),
                "charge_bid_price_eur_per_mwh": (
                    ""
                    if storage_result.charge_bid_price_eur_per_mwh is None
                    else _number(storage_result.charge_bid_price_eur_per_mwh)
                ),
                "offered_discharge_power_mw": _number(
                    storage_result.offered_discharge_mwh / duration_hours
                ),
                "offered_discharge_mwh": _number(
                    storage_result.offered_discharge_mwh
                ),
                "accepted_discharge_power_mw": _number(
                    storage_result.accepted_discharge_mwh / duration_hours
                ),
                "accepted_discharge_mwh": _number(
                    storage_result.accepted_discharge_mwh
                ),
                "discharge_bid_price_eur_per_mwh": (
                    ""
                    if storage_result.discharge_bid_price_eur_per_mwh is None
                    else _number(storage_result.discharge_bid_price_eur_per_mwh)
                ),
                "clearing_price_eur_per_mwh": _number(
                    storage_result.clearing_price_eur_per_mwh
                ),
                "charge_payment_eur": _number(storage_result.charge_payment_eur),
                "discharge_revenue_eur": _number(
                    storage_result.discharge_revenue_eur
                ),
                "additional_charge_cost_eur": _number(
                    storage_result.additional_charge_cost_eur
                ),
                "additional_discharge_cost_eur": _number(
                    storage_result.additional_discharge_cost_eur
                ),
                "net_cash_flow_eur": _number(storage_result.net_cash_flow_eur),
            }
        )

    household_rows = [
        {
            "delivery_start": _timestamp(result.delivery_start),
            "delivery_end": _timestamp(result.delivery_end),
            "unit_name": result.unit_name,
            "forecast_price_eur_per_mwh": _number(
                result.forecast_price_eur_per_mwh
            ),
            "heat_demand_mw_th": _number(result.heat_demand_mw_th),
            "fixed_power_mw": _number(result.fixed_power_mw),
            "planned_grid_power_mw": _number(result.planned_grid_power_mw),
            "heat_pump_power_mw": _number(result.heat_pump_power_mw),
            "battery_charge_power_mw": _number(
                result.battery_charge_power_mw
            ),
            "battery_discharge_power_mw": _number(
                result.battery_discharge_power_mw
            ),
            "soc_before": _number(result.soc_before),
            "soc_after": _number(result.soc_after),
            "unmet_electricity_mwh": _number(result.unmet_electricity_mwh),
            "unmet_heat_mwh_th": _number(result.unmet_heat_mwh_th),
        }
        for result in simulation_result.household_results
    ]
    flexibility_rows = [
        {
            "delivery_start": _timestamp(result.delivery_start),
            "delivery_end": _timestamp(result.delivery_end),
            "unit_name": result.unit_name,
            "minimum_grid_power_mw": _number(result.minimum_grid_power_mw),
            "maximum_grid_power_mw": _number(result.maximum_grid_power_mw),
        }
        for result in simulation_result.household_flexibility_results
    ]
    industry_rows = [
        {
            "datetime": _timestamp(result.delivery_start),
            "window_id": result.window_id,
            "unit_name": result.unit_name,
            "electrolyser_power_mw": _number(result.electrolyser_power_mw),
            "hydrogen_output_mwh": _number(result.hydrogen_output_mwh),
            "dri_power_mw": _number(result.dri_power_mw),
            "dri_output_t": _number(result.dri_output_t),
            "eaf_power_mw": _number(result.eaf_power_mw),
            "planned_steel_output_t": _number(result.planned_steel_output_t),
            "planned_grid_power_mw": _number(result.planned_grid_power_mw),
            "planned_energy_mwh": _number(result.planned_energy_mwh),
            "actual_grid_power_mw": _number(result.actual_grid_power_mw),
            "actual_steel_output_t": _number(result.actual_steel_output_t),
            "forecast_cost_eur": _number(result.forecast_cost_eur),
        }
        for result in simulation_result.industry_results
    ]
    industry_flexibility_rows = [
        {
            "datetime": _timestamp(result.delivery_start),
            "window_id": result.window_id,
            "unit_name": result.unit_name,
            "baseline_power_mw": _number(result.baseline_power_mw),
            "minimum_power_mw": _number(result.minimum_power_mw),
            "maximum_power_mw": _number(result.maximum_power_mw),
            "flex_up_mw": _number(result.flex_up_mw),
            "flex_down_mw": _number(result.flex_down_mw),
        }
        for result in simulation_result.industry_flexibility_results
    ]
    industry_window_rows = [
        {
            "window_id": result.window_id,
            "unit_name": result.unit_name,
            "optimization_start": _timestamp(result.optimization_start),
            "optimization_end": _timestamp(result.optimization_end),
            "commit_start": _timestamp(result.commit_start),
            "commit_end": _timestamp(result.commit_end),
            "baseline_variable_cost_eur": _number(
                result.baseline_variable_cost_eur
            ),
            "maximum_flexible_variable_cost_eur": _number(
                result.maximum_flexible_variable_cost_eur
            ),
        }
        for result in simulation_result.industry_optimization_windows
    ]

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
            "average_trade_price_eur_per_mwh",
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
            "side",
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
    if simulation_result.settings.market_mechanism == "pay_as_bid" or trade_rows:
        _write_rows(
            output_dir / "trade_results.csv",
            [
                "delivery_start",
                "delivery_end",
                "opening_id",
                "buyer_name",
                "buyer_bid_id",
                "seller_name",
                "seller_offer_id",
                "trade_energy_mwh",
                "trade_price_eur_per_mwh",
                "payment_eur",
            ],
            trade_rows,
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
            "side",
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
    if storage_rows:
        _write_rows(
            output_dir / "storage_results.csv",
            [
                "opening_id",
                "opening_time",
                "delivery_start",
                "delivery_end",
                "unit_name",
                "unit_operator",
                "technology",
                "energy_before_mwh",
                "energy_after_mwh",
                "soc_before",
                "soc_after",
                "offered_charge_power_mw",
                "offered_charge_mwh",
                "accepted_charge_power_mw",
                "accepted_charge_mwh",
                "charge_bid_price_eur_per_mwh",
                "offered_discharge_power_mw",
                "offered_discharge_mwh",
                "accepted_discharge_power_mw",
                "accepted_discharge_mwh",
                "discharge_bid_price_eur_per_mwh",
                "clearing_price_eur_per_mwh",
                "charge_payment_eur",
                "discharge_revenue_eur",
                "additional_charge_cost_eur",
                "additional_discharge_cost_eur",
                "net_cash_flow_eur",
            ],
            storage_rows,
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
                "side",
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
    if simulation_result.settings.market_mechanism == "pay_as_bid" or household_rows:
        _write_rows(
            output_dir / "household_results.csv",
            [
                "delivery_start",
                "delivery_end",
                "unit_name",
                "forecast_price_eur_per_mwh",
                "heat_demand_mw_th",
                "fixed_power_mw",
                "planned_grid_power_mw",
                "heat_pump_power_mw",
                "battery_charge_power_mw",
                "battery_discharge_power_mw",
                "soc_before",
                "soc_after",
                "unmet_electricity_mwh",
                "unmet_heat_mwh_th",
            ],
            household_rows,
        )
        _write_rows(
            output_dir / "household_flexibility_results.csv",
            [
                "delivery_start",
                "delivery_end",
                "unit_name",
                "minimum_grid_power_mw",
                "maximum_grid_power_mw",
            ],
            flexibility_rows,
        )
    if industry_rows:
        _write_rows(
            output_dir / "industry_results.csv",
            [
                "datetime",
                "window_id",
                "unit_name",
                "electrolyser_power_mw",
                "hydrogen_output_mwh",
                "dri_power_mw",
                "dri_output_t",
                "eaf_power_mw",
                "planned_steel_output_t",
                "planned_grid_power_mw",
                "planned_energy_mwh",
                "actual_grid_power_mw",
                "actual_steel_output_t",
                "forecast_cost_eur",
            ],
            industry_rows,
        )
        _write_rows(
            output_dir / "industry_flexibility_results.csv",
            [
                "datetime",
                "window_id",
                "unit_name",
                "baseline_power_mw",
                "minimum_power_mw",
                "maximum_power_mw",
                "flex_up_mw",
                "flex_down_mw",
            ],
            industry_flexibility_rows,
        )
        _write_rows(
            output_dir / "industry_optimization_windows.csv",
            [
                "window_id",
                "unit_name",
                "optimization_start",
                "optimization_end",
                "commit_start",
                "commit_end",
                "baseline_variable_cost_eur",
                "maximum_flexible_variable_cost_eur",
            ],
            industry_window_rows,
        )
