"""Accepted dispatch, storage, and operator-result visualisations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..models import StorageDispatchResult
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _UNIT_COLORS,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
)

_MAX_DISPLAYED_DISPATCH_UNITS = 10
_MAX_DISPLAYED_OPERATORS = 20


def _plot_dispatch_by_unit(pyplot, dates, results, path: Path) -> Path:
    times = [market.delivery_start for market in results]
    accepted_energy_by_unit: dict[str, float] = defaultdict(float)
    for market in results:
        for cleared in market.offers:
            accepted_energy_by_unit[cleared.offer.unit_name] += (
                cleared.accepted_energy_mwh
            )
    ranked_units = sorted(
        (
            name
            for name, energy in accepted_energy_by_unit.items()
            if energy > _ENERGY_TOLERANCE_MWH
        ),
        key=lambda name: (-accepted_energy_by_unit[name], name),
    )
    unit_names = ranked_units[:_MAX_DISPLAYED_DISPATCH_UNITS]
    other_unit_names = set(ranked_units[_MAX_DISPLAYED_DISPATCH_UNITS:])
    series_names = [*unit_names]
    if other_unit_names:
        series_names.append("Other accepted supply")
    power_by_unit = {name: [] for name in series_names}
    demand_power: list[float] = []
    marginal_times = []
    marginal_y_positions: list[float] = []

    for market in results:
        accepted_power: dict[str, float] = defaultdict(float)
        for cleared in market.offers:
            accepted_power[cleared.offer.unit_name] += cleared.accepted_power_mw
        if other_unit_names:
            accepted_power["Other accepted supply"] = sum(
                accepted_power.get(name, 0.0) for name in other_unit_names
            )
        cumulative_power = 0.0
        for name in series_names:
            unit_power = accepted_power.get(name, 0.0)
            power_by_unit[name].append(unit_power)
            contains_marginal_unit = name == market.marginal_unit_name or (
                name == "Other accepted supply"
                and market.marginal_unit_name in other_unit_names
            )
            if contains_marginal_unit and unit_power > _ENERGY_TOLERANCE_MWH:
                # Place the marker inside the displayed band containing the
                # marginal unit, including the aggregate band when necessary.
                marginal_times.append(market.delivery_start)
                marginal_y_positions.append(cumulative_power + unit_power / 2)
            cumulative_power += unit_power
        demand_power.append(market.cleared_power_mw)

    figure, axis = pyplot.subplots(figsize=(16, 7.5))
    if series_names:
        series_colors = [
            (
                "#9CA3AF"
                if name == "Other accepted supply"
                else _UNIT_COLORS[index % len(_UNIT_COLORS)]
            )
            for index, name in enumerate(series_names)
        ]
        axis.stackplot(
            times,
            *(power_by_unit[name] for name in series_names),
            labels=series_names,
            colors=series_colors,
            alpha=0.78,
            step="mid",
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
    axis.step(
        times,
        demand_power,
        where="mid",
        color="#1F1F1F",
        linewidth=1.2,
        label="Cleared demand power",
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
    title = "Accepted dispatch by unit"
    if other_unit_names:
        title += (
            f" — {_MAX_DISPLAYED_DISPATCH_UNITS} largest units and "
            f"{len(other_unit_names)} aggregated"
        )
    axis.set_title(title)
    axis.set_ylabel("Power (MW)")
    axis.set_xlabel("Delivery start")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", ncol=3, fontsize=8)
    _format_time_axis(axis, dates)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_operator_profit(
    pyplot,
    results,
    storage_results: tuple[StorageDispatchResult, ...],
    path: Path,
) -> Path:
    profits: dict[str, float] = defaultdict(float)
    for market in results:
        for cleared in market.offers:
            if cleared.offer.offer_type != "power_plant":
                continue
            profits[cleared.offer.operator] += cleared.profit_eur
    for storage in storage_results:
        profits[storage.operator] += storage.net_cash_flow_eur

    displayed_profits = {
        operator: value for operator, value in profits.items() if abs(value) > 1e-6
    }
    ranked_operators = sorted(
        displayed_profits,
        key=lambda operator: (-abs(displayed_profits[operator]), operator),
    )
    omitted_operators = ranked_operators[_MAX_DISPLAYED_OPERATORS:]
    displayed_profits = {
        operator: displayed_profits[operator]
        for operator in ranked_operators[:_MAX_DISPLAYED_OPERATORS]
    }
    omitted_positive = sum(
        profits[operator] for operator in omitted_operators if profits[operator] > 0
    )
    omitted_negative = sum(
        profits[operator] for operator in omitted_operators if profits[operator] < 0
    )
    positive_count = sum(profits[operator] > 0 for operator in omitted_operators)
    negative_count = sum(profits[operator] < 0 for operator in omitted_operators)
    if positive_count:
        displayed_profits[f"Other positive operators ({positive_count})"] = (
            omitted_positive
        )
    if negative_count:
        displayed_profits[f"Other negative operators ({negative_count})"] = (
            omitted_negative
        )
    operators = sorted(
        displayed_profits,
        key=lambda operator: (displayed_profits[operator], operator),
    )
    values = [displayed_profits[operator] for operator in operators]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]

    figure, axis = pyplot.subplots(
        figsize=(10, min(13.5, max(4.5, len(operators) * 0.55 + 1.5)))
    )
    bars = axis.barh(operators, values, color=colors)
    axis.axvline(0, color="#1F1F1F", linewidth=0.8)
    axis.set_title("Cumulative generation and storage result by operator")
    axis.set_xlabel("Profit / net cash flow (EUR)")
    axis.grid(axis="x", alpha=0.25)
    if not operators:
        _annotate_no_supply(axis, "No non-zero generation or storage results")
    maximum_magnitude = max((abs(value) for value in values), default=0.0)
    if operators:
        if maximum_magnitude <= 1e-6:
            axis.set_xlim(-1.0, 1.0)
        else:
            lower = min([0.0, *values]) - maximum_magnitude * 0.08
            upper = max([0.0, *values]) + maximum_magnitude * 0.08
            axis.set_xlim(lower, upper)
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:,.0f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
        )
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_storage_dispatch(
    pyplot,
    dates,
    storage_results: tuple[StorageDispatchResult, ...],
    path: Path,
) -> Path:
    """Plot delivered storage power and the resulting physical SOC trajectory."""

    by_unit: dict[str, list[StorageDispatchResult]] = defaultdict(list)
    for storage in storage_results:
        by_unit[storage.unit_name].append(storage)
    storage_names = sorted(by_unit)
    storage_color_map = pyplot.get_cmap("turbo")
    storage_colors = {
        name: storage_color_map(index / max(1, len(storage_names) - 1))
        for index, name in enumerate(storage_names)
    }

    figure, (soc_axis, power_axis) = pyplot.subplots(
        2,
        1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.1, 1)},
    )
    for unit_name in storage_names:
        rows = sorted(
            by_unit[unit_name],
            key=lambda storage: (storage.delivery_start, storage.opening_time),
        )
        color = storage_colors[unit_name]
        soc_times: list[datetime] = []
        soc_values: list[float] = []
        net_power: list[float] = []
        for row in rows:
            duration_hours = (
                row.delivery_end - row.delivery_start
            ).total_seconds() / 3600
            if not soc_times or soc_times[-1] != row.delivery_start:
                soc_times.append(row.delivery_start)
                soc_values.append(row.soc_before * 100)
            soc_times.append(row.delivery_end)
            soc_values.append(row.soc_after * 100)
            net_power.append(
                (row.accepted_discharge_mwh - row.accepted_charge_mwh) / duration_hours
            )

        soc_axis.plot(
            soc_times,
            soc_values,
            color=color,
            linewidth=1.45,
            label=unit_name,
        )
        contiguous_chunks: list[list[tuple[StorageDispatchResult, float]]] = []
        for row, power in zip(rows, net_power, strict=True):
            if (
                not contiguous_chunks
                or contiguous_chunks[-1][-1][0].delivery_end != row.delivery_start
            ):
                contiguous_chunks.append([])
            contiguous_chunks[-1].append((row, power))
        for index, chunk in enumerate(contiguous_chunks):
            power_times = [chunk[0][0].delivery_start] + [
                row.delivery_end for row, _ in chunk
            ]
            power_values = [power for _, power in chunk] + [chunk[-1][1]]
            power_axis.step(
                power_times,
                power_values,
                where="post",
                color=color,
                linewidth=1.25,
                label=unit_name if index == 0 else None,
            )

    figure.suptitle("Storage dispatch and state of charge")
    soc_axis.set_ylabel("SOC (%)")
    soc_axis.set_ylim(0, 100)
    soc_axis.grid(axis="y", alpha=0.25)

    power_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
    power_axis.set_ylabel("Net power (MW)")
    power_axis.set_xlabel("Delivery start")
    power_axis.grid(axis="y", alpha=0.25)
    power_axis.text(
        0.01,
        0.03,
        "Positive: discharge · Negative: charge",
        transform=power_axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    _format_time_axis(power_axis, dates)
    handles, labels = soc_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(storage_names)),
        fontsize=8,
    )
    legend_rows = (len(storage_names) + 5) // 6
    top = max(0.70, 0.91 - max(0, legend_rows - 1) * 0.035)
    figure.tight_layout(rect=(0, 0, 1, top))
    return _save_figure(pyplot, figure, path)
