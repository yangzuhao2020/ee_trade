"""工业生产计划、实际执行结果及各交付时段购电功率上下界的 PNG 图表。"""

from collections import defaultdict
from pathlib import Path
from ..models import IndustrialResult
from .common import _format_time_axis, _save_figure


_COMPARISON_TOLERANCE = 1e-6


def _plot_industry_dispatch(
    pyplot,
    dates,
    results: tuple[IndustrialResult, ...],
    path: Path,
) -> Path:
    """绘制工业用户的承诺计划、实际执行情况以及逐时段的功率上下界。"""

    dispatch_by_unit: dict[str, list[IndustrialResult]] = defaultdict(list)
    for result in results:
        dispatch_by_unit[result.unit_name].append(result)

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

        minimum_power = [row.minimum_power_mw for row in rows]
        maximum_power = [row.maximum_power_mw for row in rows]

        power_axis.fill_between(
            step_times,
            [*minimum_power, minimum_power[-1]],
            [*maximum_power, maximum_power[-1]],
            step="post",
            color=color,
            alpha=0.16,
            label=f"{unit_name} flexibility range",
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
        cumulative_times = [rows[0].delivery_start, *(row.delivery_end for row in rows)]
        for axis, x, planned, actual, planned_label, actual_label in (
            (power_axis, step_times,
             [*planned_power, planned_power[-1]], [*actual_power, actual_power[-1]],
             "planned power", "actual power"),
            (steel_axis, step_times,
             [*planned_steel, planned_steel[-1]], [*actual_steel, actual_steel[-1]],
             "planned steel", "actual steel"),
            (cumulative_axis, cumulative_times,
             [0.0, *cumulative_planned], [0.0, *cumulative_actual],
             "cumulative planned", "cumulative actual"),
        ):
            matches_plan = all(
                abs(planned_value - actual_value) <= _COMPARISON_TOLERANCE
                for planned_value, actual_value in zip(planned, actual, strict=True)
            )
            if not matches_plan:
                axis.step(
                    x, planned, where="post", color=color, linewidth=1.5,
                    linestyle="--", label=f"{unit_name} {planned_label}",
                )
            label = f"{unit_name} {actual_label}" + (" (matches plan)" if matches_plan else "")
            axis.step(x, actual, where="post", color=color, linewidth=1.8, label=label)

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
