"""PNG visualisations generated after CSV simulation results are durable."""

from __future__ import annotations

from collections import defaultdict
from math import isclose
from pathlib import Path

from .errors import PlottingError
from .models import MarketClearingResult, SimulationResult


_ENERGY_TOLERANCE_MWH = 1e-7
_UNIT_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
)


def generate_plots(
    result: SimulationResult, output_dir: str | Path
) -> tuple[Path, ...]:
    """Render durable PNG summaries from a complete simulation result.

    This function deliberately owns no simulation or CSV-writing work. Call it
    only after ``write_results`` has completed so a rendering failure cannot
    discard valid market results.
    """

    results = tuple(
        sorted(result.market_results, key=lambda market: market.delivery_start)
    )
    _validate_plot_data(results)
    pyplot, dates = _load_matplotlib()

    plot_directory = Path(output_dir)
    plot_directory.mkdir(parents=True, exist_ok=True)
    unit_colors = _make_unit_colors(results)

    try:
        paths = (
            _plot_market_overview(
                pyplot, dates, results, plot_directory / "market_overview.png"
            ),
            _plot_dispatch_by_unit(
                pyplot,
                dates,
                results,
                unit_colors,
                plot_directory / "dispatch_by_unit.png",
            ),
            _plot_operator_profit(
                pyplot, results, plot_directory / "operator_profit.png"
            ),
            _plot_first_merit_order(
                pyplot,
                results[0],
                unit_colors,
                plot_directory / "merit_order_first_product.png",
            ),
        )
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate PNG plots: {exc}") from exc

    return paths


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.dates as dates
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise PlottingError(
            "PNG generation requires Matplotlib. Install matplotlib or run with --no-plots."
        ) from exc
    return pyplot, dates


def _validate_plot_data(results: tuple[MarketClearingResult, ...]) -> None:
    if not results:
        raise PlottingError("Cannot generate plots because the simulation has no delivery products.")

    for market in results:
        if market.duration_hours <= 0:
            raise PlottingError(
                f"Product at {market.delivery_start!s} has a non-positive duration."
            )
        if not isclose(
            market.requested_demand_mwh,
            market.cleared_energy_mwh + market.unserved_demand_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise PlottingError(
                "Demand energy must equal cleared energy plus unmet demand for "
                f"{market.delivery_start:%Y-%m-%d %H:%M}."
            )


def _make_unit_colors(
    results: tuple[MarketClearingResult, ...]
) -> dict[str, str]:
    unit_names = sorted(
        {cleared.offer.unit_name for market in results for cleared in market.offers}
    )
    return {
        name: _UNIT_COLORS[index % len(_UNIT_COLORS)]
        for index, name in enumerate(unit_names)
    }


def _format_time_axis(axis, dates) -> None:
    axis.xaxis.set_major_locator(dates.DayLocator(interval=3))
    axis.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
    axis.tick_params(axis="x", rotation=30)
    axis.margins(x=0)


def _save_figure(pyplot, figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        pyplot.close(figure)
    return path


def _plot_market_overview(pyplot, dates, results, path: Path) -> Path:
    times = [market.delivery_start for market in results]
    prices = [market.clearing_price_eur_per_mwh for market in results]
    demand = [market.requested_demand_mwh for market in results]
    cleared = [market.cleared_energy_mwh for market in results]
    unserved = [market.unserved_demand_mwh for market in results]

    figure, (price_axis, energy_axis) = pyplot.subplots(
        2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": (1, 1.35)}
    )
    price_axis.step(
        times,
        prices,
        where="mid",
        color="#2F5597",
        linewidth=1.6,
        label="Clearing price",
    )
    price_axis.set_title("Electricity market overview")
    price_axis.set_ylabel("Price (EUR/MWh)")
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper right")

    # The stacked fills make the required identity visually explicit:
    # total demand energy = cleared energy + unmet-demand energy.
    energy_axis.fill_between(
        times,
        0,
        cleared,
        step="mid",
        color="#4C78A8",
        alpha=0.28,
        label="Cleared energy",
    )
    energy_axis.fill_between(
        times,
        cleared,
        demand,
        step="mid",
        color="#E45756",
        alpha=0.45,
        label="Unmet demand",
    )
    energy_axis.step(
        times,
        demand,
        where="mid",
        color="#1F1F1F",
        linewidth=1.1,
        label="Market demand energy",
    )
    energy_axis.step(
        times,
        unserved,
        where="mid",
        color="#E45756",
        linewidth=0.8,
        alpha=0.85,
    )
    energy_axis.set_ylabel("Energy (MWh)")
    energy_axis.set_xlabel("Delivery start")
    energy_axis.grid(axis="y", alpha=0.25)
    energy_axis.legend(loc="upper right", ncol=3)
    _format_time_axis(energy_axis, dates)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_dispatch_by_unit(pyplot, dates, results, unit_colors, path: Path) -> Path:
    times = [market.delivery_start for market in results]
    unit_names = sorted(unit_colors)
    power_by_unit = {name: [] for name in unit_names}
    demand_power: list[float] = []
    marginal_times = []
    marginal_y_positions: list[float] = []

    for market in results:
        accepted_power: dict[str, float] = defaultdict(float)
        for cleared in market.offers:
            accepted_power[cleared.offer.unit_name] += cleared.accepted_power_mw
        cumulative_power = 0.0
        for name in unit_names:
            unit_power = accepted_power.get(name, 0.0)
            power_by_unit[name].append(unit_power)
            if name == market.marginal_unit_name and unit_power > _ENERGY_TOLERANCE_MWH:
                # Place the marker inside the marginal unit's own stacked band.
                marginal_times.append(market.delivery_start)
                marginal_y_positions.append(cumulative_power + unit_power / 2)
            cumulative_power += unit_power
        demand_power.append(market.requested_demand_power_mw)

    figure, axis = pyplot.subplots(figsize=(16, 7.5))
    axis.stackplot(
        times,
        *(power_by_unit[name] for name in unit_names),
        labels=unit_names,
        colors=[unit_colors[name] for name in unit_names],
        alpha=0.78,
    )
    axis.plot(
        times,
        demand_power,
        color="#1F1F1F",
        linewidth=1.2,
        label="Demand power",
        zorder=4,
    )
    if marginal_times:
        axis.scatter(
            marginal_times,
            marginal_y_positions,
            marker="D",
            s=18,
            color="#111111",
            edgecolors="#FFFFFF",
            linewidths=0.35,
            label="Marginal unit",
            zorder=5,
        )
    axis.set_title("Accepted dispatch by unit")
    axis.set_ylabel("Power (MW)")
    axis.set_xlabel("Delivery start")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", ncol=min(3, len(unit_names) + 2))
    _format_time_axis(axis, dates)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_operator_profit(pyplot, results, path: Path) -> Path:
    profits: dict[str, float] = defaultdict(float)
    for market in results:
        for cleared in market.offers:
            if cleared.offer.offer_type != "power_plant":
                continue
            profits[cleared.offer.operator] += cleared.profit_eur

    operators = sorted(profits, key=lambda operator: (profits[operator], operator))
    values = [profits[operator] for operator in operators]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]

    figure, axis = pyplot.subplots(figsize=(10, max(4.5, len(operators) * 0.8 + 1.5)))
    bars = axis.barh(operators, values, color=colors)
    axis.axvline(0, color="#1F1F1F", linewidth=0.8)
    axis.set_title("Cumulative profit by generation operator")
    axis.set_xlabel("Profit (EUR)")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        horizontal_alignment = "left" if value >= 0 else "right"
        offset = max(abs(value) * 0.01, 1.0)
        x_position = value + offset if value >= 0 else value - offset
        axis.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
        )
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_first_merit_order(pyplot, market, unit_colors, path: Path) -> Path:
    figure, axis = pyplot.subplots(figsize=(11, 6.5))
    offers = sorted(
        market.offers,
        key=lambda cleared: (
            cleared.offer.bid_price_eur_per_mwh,
            cleared.offer.unit_name,
            cleared.offer.identifier,
        ),
    )
    cumulative_energy = 0.0
    label_offset = max(market.clearing_price_eur_per_mwh * 0.035, 1.0)

    for cleared in offers:
        offer = cleared.offer
        offered_end = cumulative_energy + offer.offered_energy_mwh
        accepted_end = cumulative_energy + cleared.accepted_energy_mwh
        color = unit_colors[offer.unit_name]
        if cleared.accepted_energy_mwh > _ENERGY_TOLERANCE_MWH:
            axis.hlines(
                offer.bid_price_eur_per_mwh,
                cumulative_energy,
                accepted_end,
                color=color,
                linewidth=7,
            )
        if accepted_end < offered_end:
            axis.hlines(
                offer.bid_price_eur_per_mwh,
                accepted_end,
                offered_end,
                color="#B8B8B8",
                linewidth=4,
            )
        axis.text(
            (cumulative_energy + offered_end) / 2,
            offer.bid_price_eur_per_mwh + label_offset,
            offer.unit_name,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        if (
            offer.unit_name == market.marginal_unit_name
            and cleared.accepted_energy_mwh > _ENERGY_TOLERANCE_MWH
            and isclose(
                offer.bid_price_eur_per_mwh,
                market.clearing_price_eur_per_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            )
        ):
            axis.scatter(
                [accepted_end],
                [offer.bid_price_eur_per_mwh],
                marker="D",
                s=46,
                color="#111111",
                edgecolors="#FFFFFF",
                linewidths=0.7,
                zorder=4,
                label="Marginal unit",
            )
        cumulative_energy = offered_end

    axis.axvline(
        market.requested_demand_mwh,
        color="#1F1F1F",
        linestyle="--",
        linewidth=1.2,
        label="Demand energy",
    )
    axis.axhline(
        market.clearing_price_eur_per_mwh,
        color="#2F5597",
        linestyle=":",
        linewidth=1.3,
        label="Clearing price",
    )
    axis.set_title(
        "Merit order — " + market.delivery_start.strftime("%Y-%m-%d %H:%M")
    )
    axis.set_xlabel("Cumulative offered energy (MWh)")
    axis.set_ylabel("Bid price (EUR/MWh)")
    axis.set_xlim(0, max(cumulative_energy, market.requested_demand_mwh) * 1.03)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)
