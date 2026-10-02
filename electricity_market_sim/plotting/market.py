"""Market-wide and pay-as-bid visualisations."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from ..errors import PlottingError
from ..market_models import MarketClearingResult
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
)


def _market_price_series(
    results: tuple[MarketClearingResult, ...],
) -> tuple[list[float], str]:
    """Return the price measure that exists for the market's settlement method."""

    uses_pay_as_bid = [market.pricing_method == "pay_as_bid" for market in results]
    if any(uses_pay_as_bid) and not all(uses_pay_as_bid):
        raise PlottingError(
            "Cannot combine pay-as-bid and uniform-price products in one price plot."
        )
    if all(uses_pay_as_bid):
        return (
            [
                (
                    market.average_trade_price_eur_per_mwh
                    if market.average_trade_price_eur_per_mwh is not None
                    else float("nan")
                )
                for market in results
            ],
            "Average trade price",
        )
    return (
        [
            (
                market.clearing_price_eur_per_mwh
                if market.clearing_price_eur_per_mwh is not None
                else float("nan")
            )
            for market in results
        ],
        "Clearing price",
    )


def _plot_market_overview(pyplot, dates, results, path: Path) -> Path:
    times = [market.delivery_start for market in results]
    prices, price_label = _market_price_series(results)
    requested_demand = [market.requested_demand_mwh for market in results]

    def by_type(attribute: str, demand_type: str) -> list[float]:
        return [
            market.demand_total(attribute, demand_type=demand_type) for market in results
        ]

    accepted_layers = (
        (by_type("accepted_energy_mwh", "inelastic_load"), "#4C78A8", "Served inelastic load"),
        (by_type("accepted_energy_mwh", "household_load"), "#72B7B2", "Accepted household load"),
        (by_type("accepted_energy_mwh", "industrial_load"), "#D4A72C", "Accepted industrial load"),
        (by_type("accepted_energy_mwh", "elastic_load"), "#54A24B", "Accepted elastic demand"),
        (by_type("accepted_energy_mwh", "storage_charge"), "#E45756", "Accepted storage charge"),
        ([market.cleared_export_mwh for market in results], "#F58518", "Accepted export"),
    )
    unaccepted_lines = (
        ([market.unserved_load_mwh for market in results], "#E45756", 1.2, "Unserved inelastic load"),
        (by_type("unserved_energy_mwh", "elastic_load"), "#7F7F7F", 1.0, "Unaccepted elastic demand"),
        (by_type("unserved_energy_mwh", "household_load"), "#72B7B2", 1.0, "Unaccepted household load"),
        (by_type("unserved_energy_mwh", "industrial_load"), "#D4A72C", 1.0, "Unaccepted industrial load"),
        (by_type("unserved_energy_mwh", "storage_charge"), "#B279A2", 1.0, "Unaccepted storage charge"),
        ([market.unfulfilled_export_mwh for market in results], "#F58518", 1.0, "Unfulfilled export"),
    )

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
        label=price_label,
    )
    price_axis.set_title("Electricity market overview")
    price_axis.set_ylabel("Price (EUR/MWh)")
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper right")

    lower = None
    for values, color, label in accepted_layers:
        upper = (
            values
            if lower is None
            else [base + value for base, value in zip(lower, values)]
        )
        demand_axis.fill_between(
            times,
            0 if lower is None else lower,
            upper,
            step="mid",
            color=color,
            alpha=0.35,
            label=label,
        )
        lower = upper
    demand_axis.step(
        times,
        requested_demand,
        where="mid",
        color="#1F1F1F",
        linewidth=1.1,
        label="All submitted demand",
    )
    demand_axis.set_ylabel("Energy (MWh)")
    demand_axis.grid(axis="y", alpha=0.25)
    demand_axis.legend(loc="upper right", ncol=3)

    for values, color, linewidth, label in unaccepted_lines:
        unaccepted_axis.step(
            times, values, where="mid", color=color, linewidth=linewidth, label=label
        )
    unaccepted_axis.set_ylabel("Unaccepted (MWh)")
    unaccepted_axis.set_xlabel("Delivery start")
    unaccepted_axis.grid(axis="y", alpha=0.25)
    unaccepted_axis.legend(loc="upper right", ncol=3)
    _format_time_axis(unaccepted_axis, dates)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_market_summary(pyplot, dates, results, path: Path) -> Path:
    """Plot cleared EOM volume and the applicable price across all products.

    Supply and demand are deliberately plotted separately even though a cleared
    product balances them by definition.  The two series make that balance
    visible and match the market-summary convention used by ASSUME.
    """

    times = [market.delivery_start for market in results]
    cleared_supply_gw = [
        market.cleared_energy_mwh / market.duration_hours / 1_000 for market in results
    ]
    cleared_demand_gw = [
        sum(demand.accepted_power_mw for demand in market.demand_bids) / 1_000
        for market in results
    ]
    prices, price_label = _market_price_series(results)

    figure = pyplot.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 1, height_ratios=(5, 1.25), hspace=0.32)
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
        label=price_label,
        zorder=2,
    )[0]

    axis.set_title("Market summary EOM")
    axis.set_ylabel("Cleared volume (GW)")
    price_axis.set_ylabel(f"{price_label} (EUR/MWh)")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    price_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("Delivery start")
    _format_time_axis(axis, dates)
    axis.legend(
        (demand_line, price_line, supply_line),
        ("Cleared demand volume", price_label, "Cleared supply volume"),
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
            price_label,
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

    finite_values = [value for value in values if isfinite(value)]
    if not finite_values:
        return ("N/A", "N/A", "N/A")
    mean = sum(finite_values) / len(finite_values)
    return tuple(
        f"{value:.{decimals}f} {unit}"
        for value in (min(finite_values), max(finite_values), mean)
    )


def _plot_first_pay_as_bid_order_book(pyplot, market, unit_colors, path: Path) -> Path:
    """Plot one pay-as-bid supply book with traded energy and its average price."""

    figure, axis = pyplot.subplots(figsize=(11, 6.5))
    offers = sorted(
        market.offers,
        key=lambda cleared: (
            cleared.offer.bid_price_eur_per_mwh,
            cleared.offer.unit_name,
            cleared.offer.identifier,
        ),
    )
    average_trade_price = market.average_trade_price_eur_per_mwh
    price_scale = max(
        [abs(cleared.offer.bid_price_eur_per_mwh) for cleared in offers]
        + ([abs(average_trade_price)] if average_trade_price is not None else [])
        + [1.0]
    )
    label_offset = price_scale * 0.035
    cumulative_energy = 0.0
    accepted_label_added = False
    unaccepted_label_added = False
    show_unit_labels = len(offers) <= 20

    for cleared in offers:
        offer = cleared.offer
        offered_end = cumulative_energy + offer.offered_energy_mwh
        accepted_end = cumulative_energy + cleared.accepted_energy_mwh
        color = unit_colors[offer.unit_name] if show_unit_labels else "#54A24B"
        if cleared.accepted_energy_mwh > _ENERGY_TOLERANCE_MWH:
            axis.hlines(
                offer.bid_price_eur_per_mwh,
                cumulative_energy,
                accepted_end,
                color=color,
                linewidth=7,
                label=("Traded offer energy" if not accepted_label_added else None),
            )
            accepted_label_added = True
        if accepted_end < offered_end - _ENERGY_TOLERANCE_MWH:
            axis.hlines(
                offer.bid_price_eur_per_mwh,
                accepted_end,
                offered_end,
                color="#B8B8B8",
                linewidth=4,
                label=("Untraded offer energy" if not unaccepted_label_added else None),
            )
            unaccepted_label_added = True
        if show_unit_labels and offer.offered_energy_mwh > _ENERGY_TOLERANCE_MWH:
            axis.text(
                (cumulative_energy + offered_end) / 2,
                offer.bid_price_eur_per_mwh + label_offset,
                offer.unit_name,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        cumulative_energy = offered_end

    if not show_unit_labels:
        axis.text(
            0.99,
            0.02,
            f"{len(offers)} offers; unit labels omitted",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    axis.axvline(
        market.requested_demand_mwh,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.1,
        label="Submitted demand energy",
    )
    axis.axvline(
        market.cleared_energy_mwh,
        color="#1F1F1F",
        linestyle=":",
        linewidth=1.4,
        label="Traded energy",
    )
    if average_trade_price is not None:
        axis.axhline(
            average_trade_price,
            color="#2F5597",
            linestyle="-.",
            linewidth=1.4,
            label="Average trade price",
        )
    if not offers:
        _annotate_no_supply(axis)

    axis.set_title(
        "Pay-as-bid supply offers and trades — "
        + market.delivery_start.strftime("%Y-%m-%d %H:%M")
    )
    axis.set_xlabel("Cumulative offered energy (MWh)")
    axis.set_ylabel("Offer / average trade price (EUR/MWh)")
    maximum_energy = max(
        cumulative_energy,
        market.requested_demand_mwh,
        market.cleared_energy_mwh,
    )
    axis.set_xlim(0, maximum_energy * 1.03 if maximum_energy > 0 else 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)
