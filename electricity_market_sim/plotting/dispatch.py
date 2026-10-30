"""机组成交出力、储能充放电与 SOC，以及运营商利润的 PNG 图表。"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from ..models import StorageDispatchResult
from ..utils import hours_between
from .market import _chinese_font_families
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _UNIT_COLORS,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
)

_MAX_DISPLAYED_DISPATCH_UNITS = 5
_MAX_DISPLAYED_OPERATORS = 5
_MAX_DISPLAYED_STORAGE_UNITS = 2


def _plot_dispatch_by_unit(pyplot, dates, results, path: Path) -> Path:
    """以非堆叠曲线绘制各机组自身的成交出力。"""
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
    power_by_unit = {name: [] for name in series_names}
    marginal_times = []
    marginal_y_positions: list[float] = []

    for market in results:
        accepted_power: dict[str, float] = defaultdict(float)
        for cleared in market.offers:
            accepted_power[cleared.offer.unit_name] += cleared.accepted_power_mw
        for name in series_names:
            unit_power = accepted_power.get(name, 0.0)
            power_by_unit[name].append(unit_power)
            contains_marginal_unit = name == market.marginal_unit_name
            if contains_marginal_unit and unit_power > _ENERGY_TOLERANCE_MWH:
                # 仅标记当前展示机组自身的功率。
                marginal_times.append(market.delivery_start)
                marginal_y_positions.append(unit_power)

    figure, axis = pyplot.subplots(figsize=(16, 7.5)) # 创建画布
    if series_names:
        series_colors = [
            _UNIT_COLORS[index % len(_UNIT_COLORS)]
            for index, name in enumerate(series_names)
        ]
        for name, color in zip(series_names, series_colors):
            axis.step(
                times,
                power_by_unit[name],
                where="mid",
                color=color,
                linewidth=1.4,
                label=name,
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
        ) # Marginal unit 菱形点
    title = f"Accepted dispatch — top {_MAX_DISPLAYED_DISPATCH_UNITS} units by traded energy"
    if other_unit_names:
        title += f" ({len(other_unit_names)} omitted)"
    axis.set_title(title)
    axis.set_ylabel("Power (MW)")
    axis.set_ylim(bottom=0.0)
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
        key=lambda operator: (-displayed_profits[operator], operator),
    )
    operators = ranked_operators[:_MAX_DISPLAYED_OPERATORS][::-1]
    values = [displayed_profits[operator] for operator in operators]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]

    figure, axis = pyplot.subplots(
        figsize=(10, min(13.5, max(4.5, len(operators) * 0.55 + 1.5)))
    )
    bars = axis.barh(operators, values, color=colors)
    axis.axvline(0, color="#1F1F1F", linewidth=0.8)
    axis.set_title(f"Top {_MAX_DISPLAYED_OPERATORS} operators by cumulative profit / net cash flow")
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
    """绘制储能实际交付的功率及由此形成的实际 SOC 轨迹。"""

    by_unit: dict[str, list[StorageDispatchResult]] = defaultdict(list)
    for storage in storage_results:
        by_unit[storage.unit_name].append(storage)
    ranked_storage_names = sorted(
        by_unit,
        key=lambda name: (-sum(row.accepted_charge_mwh + row.accepted_discharge_mwh
                               for row in by_unit[name]), name),
    )
    storage_names = ranked_storage_names[:_MAX_DISPLAYED_STORAGE_UNITS]
    omitted_count = len(ranked_storage_names) - len(storage_names)
    storage_colors = {
        name: _UNIT_COLORS[index]
        for index, name in enumerate(storage_names)
    }
    # A single unit needs no per-unit colors, so net power is colored by direction instead.
    single_unit = len(storage_names) == 1
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
        display_name = f"抽水蓄能电站 {unit_name}" if rows[0].technology == "PSPP" else unit_name
        soc_times: list[datetime] = []
        soc_values: list[float] = []
        net_power: list[float] = []
        for row in rows:
            duration_hours = hours_between(row.delivery_start, row.delivery_end)
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
            color=storage_colors[unit_name],
            linewidth=1.45,
            label=display_name,
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
            if not single_unit:
                power_axis.step(
                    power_times,
                    power_values,
                    where="post",
                    color=storage_colors[unit_name],
                    linewidth=1.25,
                    label=display_name if index == 0 else None,
                )
                continue
            for label, color, clip in (
                ("Discharge", "#54A24B", max),
                ("Charge", "#F58518", min),
            ):
                clipped = [clip(value, 0.0) for value in power_values]
                power_axis.step(
                    power_times,
                    clipped,
                    where="post",
                    color=color,
                    linewidth=1.0,
                    label=label if index == 0 else None,
                )
                power_axis.fill_between(
                    power_times, 0.0, clipped, step="post", color=color, alpha=0.35
                )

    title = "Storage dispatch and state of charge"
    if omitted_count:
        title += f" — top {_MAX_DISPLAYED_STORAGE_UNITS} by charge + discharge energy ({omitted_count} omitted)"
    figure.suptitle(title)
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
    if single_unit:
        power_axis.legend(loc="upper right", ncol=2, fontsize=8)
    _format_time_axis(power_axis, dates)
    handles, labels = soc_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(storage_names)),
        prop={"family": _chinese_font_families(), "size": 8},
    )
    legend_rows = (len(storage_names) + 5) // 6
    top = max(0.70, 0.91 - max(0, legend_rows - 1) * 0.035)
    figure.tight_layout(rect=(0, 0, 1, top))
    return _save_figure(pyplot, figure, path)
