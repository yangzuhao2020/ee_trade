"""Public orchestration for all supported electricity-market simulations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import load_market_settings
from .models import MarketOpening, MarketSettings, SimulationResult
from .reporting import write_results
from .simulators.household_market import simulate_household_market
from .simulators.v1_v2 import simulate_v1_v2_market
from .simulators.v4 import simulate_v4_market

__all__ = ["market_openings", "run_simulation", "simulate"]


def market_openings(settings: MarketSettings) -> list[MarketOpening]:
    """Schedule candidate openings without silently shortening their products.

    A candidate is included while its first delivery begins inside the simulation
    horizon. Whether all products and supporting inputs exist is checked later,
    so a V2 day-ahead opening can be skipped whole with a useful warning.
    """

    openings: list[MarketOpening] = []
    if settings.market_mechanism == "pay_as_bid":
        opening_time = settings.start
        seen_products: set[tuple[datetime, datetime]] = set()
        while opening_time + settings.first_delivery < settings.end:
            first_start = opening_time + settings.first_delivery
            products = tuple(
                product
                for index in range(settings.product_count)
                if (
                    product := (
                        first_start + index * settings.product_duration,
                        first_start + (index + 1) * settings.product_duration,
                    )
                )[1]
                <= settings.end
                and product not in seen_products
            )
            if products:
                openings.append(
                    MarketOpening(opening_time=opening_time, products=products)
                )
                seen_products.update(products)
            opening_time += settings.opening_frequency
        return openings

    opening_time = settings.start
    # For a multi-product opening, retain the boundary opening whose first
    # delivery starts exactly at ``end``. Its preflight will deliberately skip
    # the whole opening and emit the required incomplete-product warning. The
    # one-product V1 path retains its established strict boundary.
    includes_end_boundary = settings.product_count > 1
    while opening_time + settings.first_delivery < settings.end or (
        includes_end_boundary and opening_time + settings.first_delivery == settings.end
    ):
        first_start = opening_time + settings.first_delivery
        products = tuple(
            (
                first_start + index * settings.product_duration,
                first_start + (index + 1) * settings.product_duration,
            )
            for index in range(settings.product_count)
        )
        openings.append(MarketOpening(opening_time=opening_time, products=products))
        opening_time += settings.opening_frequency
    return openings


def simulate(input_dir: str | Path, scenario: str = "base") -> SimulationResult:
    """Calculate one supported scenario without writing result files."""

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    openings = tuple(market_openings(settings))
    if settings.industrial_dsm_units_file is not None:
        return simulate_v4_market(input_path, settings, openings)
    if settings.market_mechanism == "pay_as_bid":
        return simulate_household_market(input_path, settings, openings)
    return simulate_v1_v2_market(input_path, settings, openings)


def run_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
) -> SimulationResult:
    """Run a simulation and persist the CSV results before optional plotting."""

    result = simulate(input_dir=input_dir, scenario=scenario)
    write_results(Path(output_dir), result)
    return result
