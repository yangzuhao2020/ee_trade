"""PNG visualisations generated after CSV simulation results are durable."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
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
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#D4A72C",
)


def generate_plots(
    result: SimulationResult,
    output_dir: str | Path,
    opening_time: datetime | None = None,
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
        paths = [
            _plot_market_overview(
                pyplot, dates, results, plot_directory / "market_overview.png"
            ),
            _plot_market_summary(
                pyplot, dates, results, plot_directory / "market_summary.png"
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
        ]
        if result.settings.market_mechanism == "complex_clearing":
            # A directory may have been rendered by an older version which
            # incorrectly treated complex clearing as a simple merit order.
            (plot_directory / "merit_order_first_product.png").unlink(
                missing_ok=True
            )
            opening_results = _select_opening_results(results, opening_time)
            selected_opening_time = (
                opening_results[0].opening_time or opening_results[0].delivery_start
            )
            opening_directory = (
                plot_directory
                / "openings"
                / selected_opening_time.strftime("%Y-%m-%d_%H-%M")
            )
            opening_directory.mkdir(parents=True, exist_ok=True)
            paths.extend(
                (
                    _plot_opening_dispatch(
                        pyplot,
                        dates,
                        opening_results,
                        unit_colors,
                        opening_directory / "price_and_dispatch.png",
                    ),
                    _plot_offer_acceptance(
                        pyplot,
                        opening_results,
                        opening_directory / "offer_acceptance.png",
                    ),
                )
            )
        else:
            paths.append(
                _plot_first_merit_order(
                    pyplot,
                    results[0],
                    unit_colors,
                    plot_directory / "merit_order_first_product.png",
                )
            )
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate PNG plots: {exc}") from exc

    return tuple(paths)


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
    locator = dates.AutoDateLocator(minticks=4, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
    axis.tick_params(axis="x", rotation=20)
    axis.margins(x=0)


def _select_opening_results(
    results: tuple[MarketClearingResult, ...],
    requested_opening_time: datetime | None,
) -> tuple[MarketClearingResult, ...]:
    """Return all products belonging to the requested or earliest opening."""

    first = results[0]
    opening_time = requested_opening_time or first.opening_time
    if opening_time is None:
        return (first,)
    selected = tuple(
        market for market in results if market.opening_time == opening_time
    )
    if not selected:
        raise PlottingError(
            "No cleared market opening exists at "
            f"{opening_time.isoformat(sep=' ', timespec='minutes')}."
        )
    return selected


def _save_figure(pyplot, figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        pyplot.close(figure)
    return path


def _annotate_no_supply(axis, message: str = "No supply offers submitted") -> None:
    """Mark a valid market state whose supply order book is empty."""

    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F3F4F6",
            "edgecolor": "#D1D5DB",
        },
        zorder=6,
    )


def _plot_market_overview(pyplot, dates, results, path: Path) -> Path:
    times = [market.delivery_start for market in results]
    prices = [market.clearing_price_eur_per_mwh for market in results]
    requested_inelastic = [
        market.requested_inelastic_demand_mwh for market in results
    ]
    cleared_inelastic = [
        market.cleared_inelastic_demand_mwh for market in results
    ]
    cleared_elastic = [market.cleared_elastic_demand_mwh for market in results]
    unserved_load = [market.unserved_load_mwh for market in results]
    unaccepted_elastic = [
        market.unaccepted_elastic_demand_mwh for market in results
    ]
    unfulfilled_export = [market.unfulfilled_export_mwh for market in results]

    figure, (price_axis, demand_axis, unaccepted_axis) = pyplot.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1.45, 0.8)},
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

    demand_axis.fill_between(
        times,
        0,
        cleared_inelastic,
        step="mid",
        color="#4C78A8",
        alpha=0.35,
        label="Served inelastic load",
    )
    cleared_local = [
        inelastic + elastic
        for inelastic, elastic in zip(cleared_inelastic, cleared_elastic)
    ]
    demand_axis.fill_between(
        times,
        cleared_inelastic,
        cleared_local,
        step="mid",
        color="#54A24B",
        alpha=0.35,
        label="Accepted elastic demand",
    )
    demand_axis.step(
        times,
        requested_inelastic,
        where="mid",
        color="#1F1F1F",
        linewidth=1.1,
        label="Requested inelastic load",
    )
    demand_axis.set_ylabel("Energy (MWh)")
    demand_axis.grid(axis="y", alpha=0.25)
    demand_axis.legend(loc="upper right", ncol=3)

    unaccepted_axis.step(
        times,
        unserved_load,
        where="mid",
        color="#E45756",
        linewidth=1.2,
        label="Unserved inelastic load",
    )
    unaccepted_axis.step(
        times,
        unaccepted_elastic,
        where="mid",
        color="#7F7F7F",
        linewidth=1.0,
        label="Unaccepted elastic demand",
    )
    unaccepted_axis.step(
        times,
        unfulfilled_export,
        where="mid",
        color="#F58518",
        linewidth=1.0,
        label="Unfulfilled export",
    )
    unaccepted_axis.set_ylabel("Unaccepted (MWh)")
    unaccepted_axis.set_xlabel("Delivery start")
    unaccepted_axis.grid(axis="y", alpha=0.25)
    unaccepted_axis.legend(loc="upper right", ncol=3)
    _format_time_axis(unaccepted_axis, dates)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_market_summary(pyplot, dates, results, path: Path) -> Path:
    """Plot the cleared EOM volume and uniform price across all products.

    Supply and demand are deliberately plotted separately even though a cleared
    product balances them by definition.  The two series make that balance
    visible and match the market-summary convention used by ASSUME.
    """

    times = [market.delivery_start for market in results]
    cleared_supply_gw = [
        market.cleared_energy_mwh / market.duration_hours / 1_000
        for market in results
    ]
    cleared_demand_gw = [
        sum(demand.accepted_power_mw for demand in market.demand_bids) / 1_000
        for market in results
    ]
    prices = [market.clearing_price_eur_per_mwh for market in results]

    figure = pyplot.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 1, height_ratios=(5, 1.25), hspace=0.12)
    axis = figure.add_subplot(grid[0])
    price_axis = axis.twinx()
    table_axis = figure.add_subplot(grid[1])

    demand_line = axis.step(
        times,
        cleared_demand_gw,
        where="mid",
        color="#54A24B",
        linewidth=1.5,
        linestyle="--",
        label="Cleared demand volume",
        zorder=3,
    )[0]
    supply_line = axis.step(
        times,
        cleared_supply_gw,
        where="mid",
        color="#4C78A8",
        linewidth=1.15,
        label="Cleared supply volume",
        zorder=4,
    )[0]
    price_line = price_axis.step(
        times,
        prices,
        where="mid",
        color="#D4A72C",
        linewidth=1.15,
        label="Clearing price",
        zorder=2,
    )[0]

    axis.set_title("Market summary EOM")
    axis.set_ylabel("Cleared volume (GW)")
    price_axis.set_ylabel("Clearing price (EUR/MWh)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("Delivery start")
    _format_time_axis(axis, dates)
    axis.legend(
        (demand_line, price_line, supply_line),
        ("Cleared demand volume", "Clearing price", "Cleared supply volume"),
        loc="upper left",
        ncol=3,
    )

    table_axis.axis("off")
    table = table_axis.table(
        cellText=[
            _summary_row(cleared_demand_gw, "GW", 2),
            _summary_row(prices, "EUR/MWh", 1),
            _summary_row(cleared_supply_gw, "GW", 2),
        ],
        rowLabels=(
            "Cleared demand volume",
            "Clearing price",
            "Cleared supply volume",
        ),
        colLabels=("Min", "Max", "Mean"),
        cellLoc="right",
        rowLoc="left",
        bbox=(0.12, 0.05, 0.84, 0.9),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_text_props(weight="bold", color="#2F5597")
        if column == -1:
            cell.set_text_props(ha="left")

    return _save_figure(pyplot, figure, path)


def _summary_row(values: list[float], unit: str, decimals: int) -> tuple[str, str, str]:
    """Return display-ready minimum, maximum and mean values for a plot table."""

    mean = sum(values) / len(values)
    return tuple(
        f"{value:.{decimals}f} {unit}" for value in (min(values), max(values), mean)
    )


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
    if unit_names:
        axis.stackplot(
            times,
            *(power_by_unit[name] for name in unit_names),
            labels=unit_names,
            colors=[unit_colors[name] for name in unit_names],
            alpha=0.78,
        )
    else:
        axis.plot(
            times,
            [0.0] * len(times),
            color="#4C78A8",
            linewidth=1.2,
            label="Accepted supply (0 MW)",
            zorder=3,
        )
        _annotate_no_supply(axis)
        axis.set_ylim(bottom=0.0)
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
    if not operators:
        _annotate_no_supply(axis, "No generation offers submitted")
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


def _plot_opening_dispatch(
    pyplot, dates, results, unit_colors, path: Path
) -> Path:
    """Plot prices and accepted unit dispatch for one complete market opening."""

    results = tuple(sorted(results, key=lambda market: market.delivery_start))
    times = [market.delivery_start for market in results]
    prices = [market.clearing_price_eur_per_mwh for market in results]
    unit_names = sorted(
        {
            cleared.offer.unit_name
            for market in results
            for cleared in market.offers
        }
    )
    power_by_unit = {name: [] for name in unit_names}
    total_demand: list[float] = []
    inelastic_demand: list[float] = []

    for market in results:
        accepted_power: dict[str, float] = defaultdict(float)
        for cleared in market.offers:
            accepted_power[cleared.offer.unit_name] += cleared.accepted_power_mw
        for name in unit_names:
            power_by_unit[name].append(accepted_power.get(name, 0.0))
        total_demand.append(market.requested_demand_power_mw)
        inelastic_demand.append(
            market.requested_inelastic_demand_mwh / market.duration_hours
        )

    opening_time = results[0].opening_time or results[0].delivery_start
    figure, (price_axis, dispatch_axis) = pyplot.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1.7)},
    )
    product_word = "product" if len(results) == 1 else "products"
    price_axis.set_title(
        f"Market opening {opening_time:%Y-%m-%d %H:%M} — "
        f"{len(results)} {product_word}"
    )
    price_axis.set_ylabel("Price (EUR/MWh)")
    price_axis.grid(axis="y", alpha=0.25)

    if len(results) == 1:
        _plot_single_product_opening(
            price_axis,
            dispatch_axis,
            results[0],
            prices[0],
            unit_names,
            power_by_unit,
            total_demand[0],
            inelastic_demand[0],
            unit_colors,
        )
    else:
        price_axis.step(
            times,
            prices,
            where="mid",
            color="#2F5597",
            linewidth=1.8,
            label="Hourly uniform price",
        )
        if unit_names:
            dispatch_axis.stackplot(
                times,
                *(power_by_unit[name] for name in unit_names),
                labels=unit_names,
                colors=[unit_colors[name] for name in unit_names],
                alpha=0.78,
            )
        dispatch_axis.plot(
            times,
            total_demand,
            color="#111111",
            linewidth=1.4,
            label="All submitted demand",
            zorder=4,
        )
        dispatch_axis.plot(
            times,
            inelastic_demand,
            color="#E45756",
            linewidth=1.1,
            linestyle="--",
            label="Inelastic load",
            zorder=4,
        )
        interval = max(1, len(times) // 12)
        dispatch_axis.xaxis.set_major_locator(dates.HourLocator(interval=interval))
        dispatch_axis.xaxis.set_major_formatter(dates.DateFormatter("%b %d\n%H:%M"))
        dispatch_axis.margins(x=0)

    if not unit_names:
        if len(results) == 1:
            dispatch_axis.hlines(
                0.0,
                -0.42,
                0.42,
                color="#4C78A8",
                linewidth=1.2,
                label="Accepted supply (0 MW)",
                zorder=3,
            )
        else:
            dispatch_axis.plot(
                times,
                [0.0] * len(times),
                color="#4C78A8",
                linewidth=1.2,
                label="Accepted supply (0 MW)",
                zorder=3,
            )
        _annotate_no_supply(dispatch_axis)
        dispatch_axis.set_ylim(bottom=0.0)
    price_axis.legend(loc="upper right")
    dispatch_axis.set_ylabel("Power (MW)")
    dispatch_axis.set_xlabel("Delivery start")
    dispatch_axis.grid(axis="y", alpha=0.25)
    dispatch_axis.legend(loc="upper left", ncol=min(4, len(unit_names) + 2))
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_single_product_opening(
    price_axis,
    dispatch_axis,
    market,
    price,
    unit_names,
    power_by_unit,
    total_demand,
    inelastic_demand,
    unit_colors,
) -> None:
    """Use bars when a market opening contains only one delivery product."""

    price_bars = price_axis.bar(
        [0],
        [price],
        width=0.55,
        color="#2F5597",
        label="Uniform price",
    )
    price_axis.bar_label(
        price_bars,
        labels=[f"{price:,.2f}"],
        padding=4,
        fontsize=9,
    )

    bottom = 0.0
    for name in unit_names:
        accepted_power = power_by_unit[name][0]
        if accepted_power <= _ENERGY_TOLERANCE_MWH:
            continue
        dispatch_axis.bar(
            [0],
            [accepted_power],
            width=0.55,
            bottom=bottom,
            color=unit_colors[name],
            alpha=0.78,
            label=name,
        )
        bottom += accepted_power

    dispatch_axis.hlines(
        total_demand,
        -0.42,
        0.42,
        color="#111111",
        linewidth=1.5,
        label="All submitted demand",
        zorder=4,
    )
    dispatch_axis.hlines(
        inelastic_demand,
        -0.42,
        0.42,
        color="#E45756",
        linewidth=1.2,
        linestyle="--",
        label="Inelastic load",
        zorder=4,
    )
    dispatch_axis.set_xlim(-0.7, 0.7)
    dispatch_axis.set_xticks([0])
    dispatch_axis.set_xticklabels([market.delivery_start.strftime("%b %d\n%H:%M")])


def _plot_offer_acceptance(pyplot, results, path: Path) -> Path:
    """Render accepted/offered ratios by unit and complex bid segment."""

    results = tuple(sorted(results, key=lambda market: market.delivery_start))
    bid_order = {"BB": 0, "LB": 1, "SB": 2}
    row_keys = sorted(
        {
            (
                cleared.offer.unit_name,
                cleared.offer.bid_type,
                cleared.offer.offer_segment,
            )
            for market in results
            for cleared in market.offers
        },
        key=lambda key: (key[0], bid_order.get(key[1], 99), key[2]),
    )
    opening_time = results[0].opening_time or results[0].delivery_start
    if not row_keys:
        figure, axis = pyplot.subplots(figsize=(9, 5))
        axis.set_title(f"Offer acceptance — opening {opening_time:%Y-%m-%d %H:%M}")
        _annotate_no_supply(axis, "No supply offers submitted for this opening")
        axis.set_axis_off()
        figure.tight_layout()
        return _save_figure(pyplot, figure, path)

    matrix: list[list[float]] = []
    for unit_name, bid_type, segment in row_keys:
        row: list[float] = []
        for market in results:
            matching = [
                cleared
                for cleared in market.offers
                if (
                    cleared.offer.unit_name,
                    cleared.offer.bid_type,
                    cleared.offer.offer_segment,
                )
                == (unit_name, bid_type, segment)
            ]
            offered = sum(item.offer.offered_energy_mwh for item in matching)
            accepted = sum(item.accepted_energy_mwh for item in matching)
            row.append(
                accepted / offered
                if offered > _ENERGY_TOLERANCE_MWH
                else float("nan")
            )
        matrix.append(row)

    figure_width = max(9, len(results) * 0.55)
    figure, axis = pyplot.subplots(
        figsize=(figure_width, max(5, len(row_keys) * 0.48 + 2))
    )
    color_map = pyplot.get_cmap("YlGnBu").copy()
    color_map.set_bad("#E5E7EB")
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=color_map,
    )
    axis.set_title(f"Offer acceptance — opening {opening_time:%Y-%m-%d %H:%M}")
    axis.set_xlabel("Delivery start")
    axis.set_ylabel("Unit · bid type · segment")
    axis.set_xticks(range(len(results)))
    axis.set_xticklabels(
        [market.delivery_start.strftime("%m-%d\n%H:%M") for market in results],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(row_keys)))
    axis.set_yticklabels(
        [f"{unit} · {bid_type} · {segment}" for unit, bid_type, segment in row_keys]
    )
    color_bar = figure.colorbar(image, ax=axis, pad=0.015)
    color_bar.set_label("Accepted / offered energy")
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
