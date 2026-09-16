"""Orchestration from scenario files through bids, clearing, and reporting.
读取 YAML、机组、负荷和燃料价格；
计算每台机组的边际成本；
按小时生成需求买单和机组卖单；
调用 clearing.py 对每个时段出清；
汇总为完整的 SimulationResult；
需要写文件时，再调用 reporting.py 输出 CSV。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .clearing import clear_pay_as_clear
from .config import load_market_settings
from .errors import InputValidationError
from .loader import (
    load_demand_units,
    load_exchange_unit,
    load_fuel_prices,
    load_hourly_demand_profiles,
    load_hourly_exchange_profiles,
    load_powerplants,
    validate_fuel_coverage,
)
from .models import (
    DemandBid,
    MarketClearingResult,
    MarketSettings,
    SimulationResult,
    SupplyOffer,
)
from .reporting import write_results


def delivery_products(settings: MarketSettings) -> list[tuple[datetime, datetime]]:
    """Create only products whose delivery end does not exceed `end_date`."""

    products: list[tuple[datetime, datetime]] = []
    opening_time = settings.start
    while True:
        delivery_start = opening_time + settings.first_delivery
        delivery_end = delivery_start + settings.product_duration
        if delivery_end > settings.end:
            return products
        products.append((delivery_start, delivery_end))
        opening_time += settings.opening_frequency


def simulate(
    input_dir: str | Path,
    scenario: str = "base",
) -> SimulationResult:
    """Calculate one supported scenario without writing any output files."""

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    plants = load_powerplants(input_path / "powerplant_units.csv")
    demand_units = load_demand_units(input_path / "demand_units.csv")
    fuel_prices = load_fuel_prices(input_path / "fuel_prices_df.csv")
    validate_fuel_coverage(plants, fuel_prices)
    demand_profiles = load_hourly_demand_profiles(
        input_path / "demand_df.csv", demand_units
    )

    exchange_unit = None
    exchange_profiles = {}
    if settings.exchange_units_file is not None:
        exchange_unit = load_exchange_unit(input_path / settings.exchange_units_file)
        existing_names = {plant.name for plant in plants} | {
            demand_unit.name for demand_unit in demand_units
        }
        if exchange_unit.name in existing_names:
            raise InputValidationError(
                f"Exchange name {exchange_unit.name!r} must not duplicate another unit name."
            )
        for label, price in (
            ("import", exchange_unit.price_import_eur_per_mwh),
            ("export", exchange_unit.price_export_eur_per_mwh),
        ):
            if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
                raise InputValidationError(
                    f"Exchange {label} price ({price:.6f} EUR/MWh) is outside "
                    "the configured bid-price limits."
                )
        exchange_profiles = load_hourly_exchange_profiles(
            input_path / "exchanges_df.csv", exchange_unit
        )

    marginal_costs = {plant.name: plant.marginal_cost(fuel_prices) for plant in plants}
    for plant in plants:
        marginal_cost = marginal_costs[plant.name]
        if not settings.minimum_bid_price <= marginal_cost <= settings.maximum_bid_price:
            raise InputValidationError(
                f"Marginal cost for {plant.name!r} ({marginal_cost:.6f} EUR/MWh) "
                "is outside the configured bid-price limits."
            )

    results: list[MarketClearingResult] = []
    for delivery_start, delivery_end in delivery_products(settings):
        duration_hours = (delivery_end - delivery_start).total_seconds() / 3600
        demand_bids: list[DemandBid] = []
        for demand_unit in demand_units:
            try:
                power_mw = demand_profiles[demand_unit.name][delivery_start]
            except KeyError as exc:
                raise InputValidationError(
                    "demand_df.csv has no complete hourly profile for "
                    f"{delivery_start.isoformat(sep=' ', timespec='minutes')}."
                ) from exc
            demand_bids.append(
                DemandBid(
                    unit_name=demand_unit.name,
                    operator=demand_unit.operator,
                    delivery_start=delivery_start,
                    delivery_end=delivery_end,
                    volume_mwh=power_mw * duration_hours,
                    price_eur_per_mwh=settings.maximum_bid_price,
                )
            )
        if exchange_unit is not None:
            try:
                exchange_schedule = exchange_profiles[delivery_start]
            except KeyError as exc:
                raise InputValidationError(
                    "exchanges_df.csv has no complete hourly profile for "
                    f"{delivery_start.isoformat(sep=' ', timespec='minutes')}."
                ) from exc
            demand_bids.append(
                DemandBid(
                    unit_name=exchange_unit.name,
                    operator=exchange_unit.operator,
                    delivery_start=delivery_start,
                    delivery_end=delivery_end,
                    volume_mwh=exchange_schedule.export_power_mw * duration_hours,
                    price_eur_per_mwh=exchange_unit.price_export_eur_per_mwh,
                    bid_type="export",
                )
            )
        offers = [
            SupplyOffer(
                unit_name=plant.name,
                operator=plant.operator,
                technology=plant.technology,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                offered_power_mw=plant.max_power_mw,
                offered_energy_mwh=plant.max_power_mw * duration_hours,
                bid_price_eur_per_mwh=marginal_costs[plant.name],
                marginal_cost_eur_per_mwh=marginal_costs[plant.name],
            )
            for plant in plants
        ]
        if exchange_unit is not None:
            offers.append(
                SupplyOffer(
                    unit_name=exchange_unit.name,
                    operator=exchange_unit.operator,
                    technology="exchange",
                    delivery_start=delivery_start,
                    delivery_end=delivery_end,
                    offered_power_mw=exchange_schedule.import_power_mw,
                    offered_energy_mwh=(
                        exchange_schedule.import_power_mw * duration_hours
                    ),
                    bid_price_eur_per_mwh=exchange_unit.price_import_eur_per_mwh,
                    # Import price is a bid priority, not an external procurement cost.
                    marginal_cost_eur_per_mwh=0.0,
                    offer_type="import",
                )
            )
        results.append(clear_pay_as_clear(demand_bids, offers))

    return SimulationResult(settings=settings, market_results=tuple(results))


def run_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
) -> SimulationResult:
    """Run a simulation and persist the CSV results before any optional plotting."""

    result = simulate(input_dir=input_dir, scenario=scenario)
    write_results(Path(output_dir), result)
    return result
