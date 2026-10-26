"""居民建筑实际购电、设备功率和电池 SOC 的 PNG 图表。"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..models import HouseholdDispatchResult, MarketClearingResult
from ..utils import hours_between
from .common import _format_time_axis, _save_figure


def _plot_household_dispatch(
    pyplot,
    dates,
    results: tuple[HouseholdDispatchResult, ...],
    market_results: tuple[MarketClearingResult, ...],
    path: Path,
) -> Path:
    """绘制各居民建筑实际购电、热泵和电池功率及 SOC。"""

    dispatch_by_unit: dict[str, list[HouseholdDispatchResult]] = defaultdict(list)
    for result in results:
        dispatch_by_unit[result.unit_name].append(result)

    accepted_grid_energy: dict[tuple[str, datetime, datetime], float] = defaultdict(float)
    for market in market_results:
        for cleared in market.demand_bids:
            if cleared.bid.demand_type == "household_load":
                key = (
                    cleared.bid.unit_name,
                    cleared.bid.delivery_start,
                    cleared.bid.delivery_end,
                )
                accepted_grid_energy[key] += cleared.accepted_energy_mwh

    household_names = sorted(dispatch_by_unit)
    figure, axes = pyplot.subplots(
        2 * len(household_names),
        1,
        figsize=(16, max(7, 3.5 * len(household_names))),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "grid": "#4C78A8",
        "heat_pump": "#F58518",
        "charge": "#54A24B",
        "discharge": "#E45756",
        "soc": "#2F5597",
    }

    for index, unit_name in enumerate(household_names):
        rows = sorted(
            dispatch_by_unit[unit_name],
            key=lambda result: (result.delivery_start, result.delivery_end),
        )
        soc_axis = axes[2 * index, 0]
        power_axis = axes[2 * index + 1, 0]
        soc_times: list[datetime] = []
        soc_values: list[float] = []
        grid_power: list[float] = []
        heat_pump_power: list[float] = []
        charge_power: list[float] = []
        discharge_power: list[float] = []

        for row in rows:
            if not soc_times or soc_times[-1] != row.delivery_start:
                soc_times.append(row.delivery_start)
                soc_values.append(row.soc_before * 100)
            soc_times.append(row.delivery_end)
            soc_values.append(row.soc_after * 100)
            duration_hours = hours_between(row.delivery_start, row.delivery_end)
            grid_energy = accepted_grid_energy.get(
                (unit_name, row.delivery_start, row.delivery_end), 0.0
            )
            grid_power.append(grid_energy / duration_hours)
            heat_pump_power.append(row.heat_pump_power_mw)
            charge_power.append(row.battery_charge_power_mw)
            discharge_power.append(row.battery_discharge_power_mw)

        soc_axis.plot(
            soc_times,
            soc_values,
            color=colors["soc"],
            linewidth=1.5,
        )
        soc_axis.set_title(unit_name, loc="left", fontsize=10, pad=4)
        soc_axis.set_ylabel("SOC (%)")
        soc_axis.set_ylim(0, 100)
        soc_axis.grid(axis="y", alpha=0.25)

        power_edges = [rows[0].delivery_start] + [row.delivery_end for row in rows]
        for values, label, color in (
            (grid_power, "Grid purchase", colors["grid"]),
            (heat_pump_power, "Heat pump", colors["heat_pump"]),
            (charge_power, "Battery charge", colors["charge"]),
            (discharge_power, "Battery discharge", colors["discharge"]),
        ):
            power_axis.step(
                power_edges,
                [*values, values[-1]],
                where="post",
                color=color,
                linewidth=1.25,
                label=label,
            )
        power_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        power_axis.set_ylabel("Power (MW)")
        power_axis.grid(axis="y", alpha=0.25)
        power_axis.legend(loc="upper right", ncol=4, fontsize=8)

    axes[-1, 0].set_xlabel("Delivery start")
    _format_time_axis(axes[-1, 0], dates)
    figure.suptitle("Household dispatch and state of charge")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return _save_figure(pyplot, figure, path)
