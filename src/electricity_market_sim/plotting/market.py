"""Market-wide, merit-order, and pay-as-bid visualisations."""

from __future__ import annotations

from math import isclose, isfinite
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
    cleared_inelastic = [
        market.cleared_inelastic_demand_mwh for market in results
    ]
    cleared_elastic = [market.cleared_elastic_demand_mwh for market in results]
    cleared_household = [
        sum(
            demand.accepted_energy_mwh
            for demand in market.demand_bids
            if demand.bid.demand_type == "household_load"
        )
        for market in results
    ]
    cleared_storage = [
        sum(
            demand.accepted_energy_mwh
            for demand in market.demand_bids
            if demand.bid.demand_type == "storage_charge"
        )
        for market in results
    ]
    cleared_export = [market.cleared_export_mwh for market in results]
    unserved_load = [market.unserved_load_mwh for market in results]
    unaccepted_elastic = [
        market.unaccepted_elastic_demand_mwh for market in results
    ]
    unaccepted_household = [
        sum(
            demand.unserved_energy_mwh
            for demand in market.demand_bids
            if demand.bid.demand_type == "household_load"
        )
        for market in results
    ]
    unaccepted_storage = [
        sum(
            demand.unserved_energy_mwh
            for demand in market.demand_bids
            if demand.bid.demand_type == "storage_charge"
        )
        for market in results
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
        label=price_label,
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
    cleared_with_household = [
        inelastic + household
        for inelastic, household in zip(cleared_inelastic, cleared_household)
    ]
    demand_axis.fill_between(
        times,
        cleared_inelastic,
        cleared_with_household,
        step="mid",
        color="#72B7B2",
        alpha=0.35,
        label="Accepted household load",
    )
    cleared_local = [
        with_household + elastic
        for with_household, elastic in zip(
            cleared_with_household, cleared_elastic
        )
    ]
    demand_axis.fill_between(
        times,
        cleared_with_household,
        cleared_local,
        step="mid",
        color="#54A24B",
        alpha=0.35,
        label="Accepted elastic demand",
    )
    cleared_with_storage = [
        local + storage
        for local, storage in zip(cleared_local, cleared_storage)
    ]
    demand_axis.fill_between(
        times,
        cleared_local,
        cleared_with_storage,
        step="mid",
        color="#E45756",
        alpha=0.35,
        label="Accepted storage charge",
    )
    cleared_total = [
        with_storage + export
        for with_storage, export in zip(cleared_with_storage, cleared_export)
    ]
    demand_axis.fill_between(
        times,
        cleared_with_storage,
        cleared_total,
        step="mid",
        color="#F58518",
        alpha=0.35,
        label="Accepted export",
    )
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
        unaccepted_household,
        where="mid",
        color="#72B7B2",
        linewidth=1.0,
        label="Unaccepted household load",
    )
    unaccepted_axis.step(
        times,
        unaccepted_storage,
        where="mid",
        color="#B279A2",
        linewidth=1.0,
        label="Unaccepted storage charge",
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
    """Plot cleared EOM volume and the applicable price across all products.

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
    prices, price_label = _market_price_series(results)

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


def _plot_first_pay_as_bid_order_book(
    pyplot, market, unit_colors, path: Path
) -> Path:
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
                label=(
                    "Untraded offer energy" if not unaccepted_label_added else None
                ),
            )
            unaccepted_label_added = True
        if offer.offered_energy_mwh > _ENERGY_TOLERANCE_MWH:
            axis.text(
                (cumulative_energy + offered_end) / 2,
                offer.bid_price_eur_per_mwh + label_offset,
                offer.unit_name,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        cumulative_energy = offered_end

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
