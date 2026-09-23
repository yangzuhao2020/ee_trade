"""Industrial production, execution, and flexibility visualisations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..models import IndustrialDispatchResult, IndustrialFlexibilityResult
from .common import _format_time_axis, _save_figure


def _plot_industry_dispatch(
    pyplot,
    dates,
    results: tuple[IndustrialDispatchResult, ...],
    flexibility_results: tuple[IndustrialFlexibilityResult, ...],
    path: Path,
) -> Path:
    """Plot committed industrial plans, execution, and pointwise power bounds."""

    dispatch_by_unit: dict[str, list[IndustrialDispatchResult]] = defaultdict(list)
    for result in results:
        dispatch_by_unit[result.unit_name].append(result)
    flexibility_by_unit_and_time = {
        (result.unit_name, result.delivery_start): result
        for result in flexibility_results
    }

    figure, (power_axis, steel_axis, cumulative_axis) = pyplot.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1, 1)},
    )
    color_map = pyplot.get_cmap("tab10")

    for index, unit_name in enumerate(sorted(dispatch_by_unit)):
        rows = sorted(
            dispatch_by_unit[unit_name], key=lambda result: result.delivery_start
        )
        color = color_map(index % 10)
        times = [row.delivery_start for row in rows]
        step_times = [*times, rows[-1].delivery_end]
        planned_power = [row.planned_grid_power_mw for row in rows]
        actual_power = [row.actual_grid_power_mw for row in rows]
        planned_steel = [row.planned_steel_output_t for row in rows]
        actual_steel = [row.actual_steel_output_t for row in rows]

        minimum_power: list[float] = []
        maximum_power: list[float] = []
        for row in rows:
            flexibility = flexibility_by_unit_and_time.get(
                (unit_name, row.delivery_start)
            )
            minimum_power.append(
                flexibility.minimum_power_mw
                if flexibility is not None
                else row.planned_grid_power_mw
            )
            maximum_power.append(
                flexibility.maximum_power_mw
                if flexibility is not None
                else row.planned_grid_power_mw
            )

        power_axis.fill_between(
            step_times,
            [*minimum_power, minimum_power[-1]],
            [*maximum_power, maximum_power[-1]],
            step="post",
            color=color,
            alpha=0.16,
            label=f"{unit_name} flexibility range",
        )
        power_axis.step(
            step_times,
            [*planned_power, planned_power[-1]],
            where="post",
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{unit_name} planned power",
        )
        power_axis.step(
            step_times,
            [*actual_power, actual_power[-1]],
            where="post",
            color=color,
            linewidth=1.8,
            label=f"{unit_name} actual power",
        )
        steel_axis.step(
            step_times,
            [*planned_steel, planned_steel[-1]],
            where="post",
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{unit_name} planned steel",
        )
        steel_axis.step(
            step_times,
            [*actual_steel, actual_steel[-1]],
            where="post",
            color=color,
            linewidth=1.8,
            label=f"{unit_name} actual steel",
        )

        cumulative_planned: list[float] = []
        cumulative_actual: list[float] = []
        planned_total = 0.0
        actual_total = 0.0
        for planned, actual in zip(planned_steel, actual_steel, strict=True):
            planned_total += planned
            actual_total += actual
            cumulative_planned.append(planned_total)
            cumulative_actual.append(actual_total)
        cumulative_axis.step(
            [rows[0].delivery_start, *(row.delivery_end for row in rows)],
            [0.0, *cumulative_planned],
            where="post",
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{unit_name} cumulative planned",
        )
        cumulative_axis.step(
            [rows[0].delivery_start, *(row.delivery_end for row in rows)],
            [0.0, *cumulative_actual],
            where="post",
            color=color,
            linewidth=1.8,
            label=f"{unit_name} cumulative actual",
        )

    window_start_by_id = {}
    for result in results:
        existing = window_start_by_id.get(result.window_id)
        if existing is None or result.delivery_start < existing:
            window_start_by_id[result.window_id] = result.delivery_start
    window_starts = sorted(set(window_start_by_id.values()))
    first_delivery = min(result.delivery_start for result in results)
    boundary_label_used = False
    for window_start in window_starts:
        if window_start == first_delivery:
            continue
        for axis in (power_axis, steel_axis, cumulative_axis):
            axis.axvline(
                window_start,
                color="#6B7280",
                linewidth=0.9,
                linestyle=":",
                label=(
                    "Rolling window boundary"
                    if axis is power_axis and not boundary_label_used
                    else None
                ),
            )
        boundary_label_used = True

    figure.suptitle("Industrial plan, execution, and flexibility")
    power_axis.set_ylabel("Grid power (MW)")
    steel_axis.set_ylabel("Steel per product (t)")
    cumulative_axis.set_ylabel("Cumulative steel (t)")
    cumulative_axis.set_xlabel("Delivery start")
    for axis in (power_axis, steel_axis, cumulative_axis):
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper left", ncol=2, fontsize=8)
    _format_time_axis(cumulative_axis, dates)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(pyplot, figure, path)
