"""居民建筑的电价与电池动作、设备功率、SOC，以及与不使用电池基准方案对比的购电成本 PNG 图表。"""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from ..models import (
    HouseholdCostResult,
    HouseholdDispatchResult,
    HouseholdFlexibilityResult,
    summarize_household_savings,
)
from ..utils import hours_between
from .common import _format_time_axis, _save_figure
from .market import _chinese_font_families


_POWER_TOLERANCE_MW = 1e-9
_SERVICE_TOLERANCE = 1e-6
_COLORS = {
    "forecast": "#F58518",
    "actual_price": "#2F5597",
    "grid": "#4C78A8",
    "baseline": "#1F1F1F",
    "flexibility": "#9ECAE9",
    "charge": "#54A24B",
    "discharge": "#E45756",
    "soc": "#2F5597",
}


def _by_unit(results) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for result in results:
        grouped[result.unit_name].append(result)
    for rows in grouped.values():
        rows.sort(key=lambda result: (result.delivery_start, result.delivery_end))
    return grouped


def _active_spans(
    rows: list[HouseholdDispatchResult], active
) -> list[tuple[datetime, datetime]]:
    """把连续满足条件的交付时段合并为 (开始, 结束) 区间，用于背景着色。"""
    spans: list[tuple[datetime, datetime]] = []
    for row in rows:
        if not active(row):
            continue
        if spans and spans[-1][1] == row.delivery_start:
            spans[-1] = (spans[-1][0], row.delivery_end)
        else:
            spans.append((row.delivery_start, row.delivery_end))
    return spans


def _apply_chinese_fonts(figure) -> None:
    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)


def _plot_household_dispatch(
    pyplot,
    dates,
    results: tuple[HouseholdDispatchResult, ...],
    flexibility_results: tuple[HouseholdFlexibilityResult, ...],
    cost_results: tuple[HouseholdCostResult, ...],
    path: Path,
    *,
    days: int | None = None,
) -> Path:
    """绘制各居民的电价与电池动作、电池功率、SOC 及购电功率。

    days 非空时只绘制首个交付时段起的前 days 天，用于观察日内充放电节奏。
    """

    dispatch_by_unit = _by_unit(results)
    flexibility = {
        (result.unit_name, result.delivery_start): result for result in flexibility_results
    }
    costs = {(result.unit_name, result.delivery_start): result for result in cost_results}
    household_names = sorted(dispatch_by_unit)
    figure, axes = pyplot.subplots(
        4 * len(household_names),
        1,
        figsize=(16, 11 * len(household_names)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.2, 0.9, 0.8, 1.1] * len(household_names)},
    )

    for index, unit_name in enumerate(household_names):
        rows = dispatch_by_unit[unit_name]
        if days is not None:
            limit = rows[0].delivery_start + timedelta(days=days)
            rows = [row for row in rows if row.delivery_start < limit]
        price_axis, battery_axis, soc_axis, grid_axis = axes[4 * index: 4 * index + 4, 0]
        edges = [rows[0].delivery_start] + [row.delivery_end for row in rows]

        def stepped(values: list[float]) -> list[float]:
            return [*values, values[-1]]

        row_costs = [costs.get((unit_name, row.delivery_start)) for row in rows]
        actual_prices = [
            float("nan")
            if cost is None or cost.actual_price_eur_per_mwh is None
            else cost.actual_price_eur_per_mwh
            for cost in row_costs
        ]
        durations = [hours_between(row.delivery_start, row.delivery_end) for row in rows]

        charge_spans = _active_spans(
            rows, lambda row: row.battery_charge_power_mw > _POWER_TOLERANCE_MW)
        discharge_spans = _active_spans(
            rows, lambda row: row.battery_discharge_power_mw > _POWER_TOLERANCE_MW)
        for axis in (price_axis, battery_axis):
            for spans, color in (
                (charge_spans, _COLORS["charge"]),
                (discharge_spans, _COLORS["discharge"]),
            ):
                for start, end in spans:
                    axis.axvspan(start, end, color=color, alpha=0.10, linewidth=0)

        price_axis.step(
            edges, stepped([row.forecast_price_eur_per_mwh for row in rows]),
            where="post", color=_COLORS["forecast"], linewidth=1.3, linestyle="--",
            label="预测电价（优化依据）",
        )
        price_axis.step(
            edges, stepped(actual_prices), where="post",
            color=_COLORS["actual_price"], linewidth=1.3, label="实际成交均价（结算依据）",
        )
        price_axis.set_title(
            f"{unit_name}：电价与电池动作（绿色背景为充电，红色背景为放电）",
            loc="left", fontsize=10,
        )
        price_axis.set_ylabel("电价（欧元/MWh）")
        price_axis.grid(axis="y", alpha=0.25)
        price_axis.legend(loc="upper right", ncol=2, fontsize=8)

        battery_axis.fill_between(
            edges, 0, stepped([row.battery_charge_power_mw * 1000 for row in rows]),
            step="post", color=_COLORS["charge"], alpha=0.8, linewidth=0, label="充电",
        )
        battery_axis.fill_between(
            edges, 0, stepped([-row.battery_discharge_power_mw * 1000 for row in rows]),
            step="post", color=_COLORS["discharge"], alpha=0.8, linewidth=0, label="放电",
        )
        battery_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        battery_axis.set_title("电池充放电功率（充电为正、放电为负）", loc="left", fontsize=10)
        battery_axis.set_ylabel("功率（kW）")
        battery_axis.grid(axis="y", alpha=0.25)
        battery_axis.legend(loc="upper right", ncol=2, fontsize=8)

        soc_times: list[datetime] = []
        soc_values: list[float] = []
        for row in rows:
            if not soc_times or soc_times[-1] != row.delivery_start:
                soc_times.append(row.delivery_start)
                soc_values.append(row.soc_before * 100)
            soc_times.append(row.delivery_end)
            soc_values.append(row.soc_after * 100)
        soc_axis.plot(soc_times, soc_values, color=_COLORS["soc"], linewidth=1.5)
        soc_axis.set_title("电池 SOC", loc="left", fontsize=10)
        soc_axis.set_ylabel("SOC（%）")
        soc_axis.set_ylim(0, min(100.0, max(soc_values) * 1.1 + 1.0))
        soc_axis.grid(axis="y", alpha=0.25)

        flexibility_rows = [flexibility.get((unit_name, row.delivery_start)) for row in rows]
        if all(item is not None for item in flexibility_rows):
            grid_axis.fill_between(
                edges,
                stepped([item.minimum_grid_power_mw for item in flexibility_rows]),
                stepped([item.maximum_grid_power_mw for item in flexibility_rows]),
                step="post", color=_COLORS["flexibility"], alpha=0.45, linewidth=0,
                label="灵活性区间（成本容忍度内）",
            )
        if all(cost is not None for cost in row_costs):
            grid_axis.step(
                edges,
                stepped([
                    cost.baseline_grid_energy_mwh / duration
                    for cost, duration in zip(row_costs, durations)
                ]),
                where="post", color=_COLORS["baseline"], linewidth=1.1, linestyle="--",
                label="基准购电（不使用电池）",
            )
            grid_axis.step(
                edges,
                stepped([
                    cost.accepted_grid_energy_mwh / duration
                    for cost, duration in zip(row_costs, durations)
                ]),
                where="post", color=_COLORS["grid"], linewidth=1.4, label="实际购电（优化后）",
            )
        grid_axis.set_title("电网购电功率", loc="left", fontsize=10)
        grid_axis.set_ylabel("功率（MW）")
        grid_axis.grid(axis="y", alpha=0.25)
        grid_axis.legend(loc="upper right", ncol=3, fontsize=8)

    axes[-1, 0].set_xlabel("交付开始时间")
    _format_time_axis(axes[-1, 0], dates)
    figure.suptitle("居民调度" if days is None else f"居民调度（前 {days} 天）")
    _apply_chinese_fonts(figure)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return _save_figure(pyplot, figure, path)


def _plot_household_savings(
    pyplot,
    dates,
    dispatch_results: tuple[HouseholdDispatchResult, ...],
    cost_results: tuple[HouseholdCostResult, ...],
    path: Path,
) -> Path:
    """绘制各居民相对不使用电池基准方案的累计节省、每日节省及每日购电支付。

    对照不可比（缺供量不同或期末 SOC 低于基准）时标题标红；
    当天缺供量与基准不同的柱子加斜线。
    """

    costs_by_unit = _by_unit(cost_results)
    household_names = sorted(costs_by_unit)
    summaries = {
        summary.unit_name: summary
        for summary in summarize_household_savings(dispatch_results, cost_results)
    }
    dispatch = {(result.unit_name, result.delivery_start): result for result in dispatch_results}
    figure, axes = pyplot.subplots(
        3 * len(household_names),
        1,
        figsize=(16, 10 * len(household_names)),
        sharex=True,
        squeeze=False,
    )

    for index, unit_name in enumerate(household_names):
        rows = costs_by_unit[unit_name]
        cumulative_axis, daily_axis, payment_axis = axes[3 * index: 3 * index + 3, 0]

        summary = summaries[unit_name]
        times = [rows[0].delivery_start] + [row.delivery_end for row in rows]
        actual = [0.0]
        for row in rows:
            actual.append(actual[-1] + row.actual_saving_eur)
        share = summary.actual_saving_share or 0.0
        cumulative_axis.plot(
            times, actual, color=_COLORS["charge"], linewidth=1.8,
            label="实际节省（按实际成交价）",
        )
        cumulative_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        cumulative_axis.text(
            0.01, 0.95,
            f"实际节省 {summary.actual_saving_eur:,.2f} EUR（占基准支付 {share:.2%}）\n"
            f"基准支付 {summary.baseline_payment_eur:,.2f} EUR\n"
            + ("对照可比" if summary.comparable
               else "对照不可比：实际节省不构成同等服务下的节省")
            + "".join(f"\n{line}" for line in summary.difference_lines()),
            transform=cumulative_axis.transAxes, ha="left", va="top", fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4", "facecolor": "white",
                "edgecolor": "#D1D5DB" if summary.comparable else _COLORS["discharge"],
            },
        )
        cumulative_axis.set_title(
            f"{unit_name}：相对不使用电池的累计节省"
            + ("" if summary.comparable else "（对照不可比）"),
            loc="left", fontsize=10,
            color="#1F1F1F" if summary.comparable else _COLORS["discharge"],
        )
        cumulative_axis.set_ylabel("累计节省（EUR）")
        cumulative_axis.grid(axis="y", alpha=0.25)
        cumulative_axis.legend(loc="lower right", fontsize=8)

        daily: dict[datetime, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for row in rows:
            day = datetime.combine(row.delivery_start.date(), datetime.min.time())
            result = dispatch[(unit_name, row.delivery_start)]
            daily[day][0] += row.actual_saving_eur
            daily[day][1] += row.baseline_payment_eur
            daily[day][2] += row.payment_eur
            daily[day][3] += abs(
                result.unmet_electricity_mwh - row.baseline_unmet_electricity_mwh
            ) + abs(result.unmet_heat_mwh_th - row.baseline_unmet_heat_mwh_th)
        days = sorted(daily)
        centers = [day + timedelta(hours=12) for day in days]
        savings = [daily[day][0] for day in days]
        bars = daily_axis.bar(
            centers, savings, width=0.8,
            color=[_COLORS["charge"] if value >= 0 else _COLORS["discharge"] for value in savings],
        )
        for bar, day in zip(bars, days):
            if daily[day][3] > _SERVICE_TOLERANCE:
                bar.set_hatch("//")
                bar.set_edgecolor("#1F1F1F")
        daily_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        daily_axis.set_title(
            "每日实际节省（绿色为节省，红色为多花；斜线表示当天缺供量与基准不同）",
            loc="left", fontsize=10,
        )
        daily_axis.set_ylabel("节省（EUR）")
        daily_axis.grid(axis="y", alpha=0.25)

        offset = timedelta(hours=4.8)
        payment_axis.bar(
            [center - offset for center in centers], [daily[day][1] for day in days],
            width=0.4, color="#BAB0AC", label="基准支付（不使用电池）",
        )
        payment_axis.bar(
            [center + offset for center in centers], [daily[day][2] for day in days],
            width=0.4, color=_COLORS["grid"], label="实际支付（优化后）",
        )
        payment_axis.set_title("每日购电支付", loc="left", fontsize=10)
        payment_axis.set_ylabel("支付（EUR）")
        payment_axis.grid(axis="y", alpha=0.25)
        payment_axis.legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1, 0].set_xlabel("交付日期")
    _format_time_axis(axes[-1, 0], dates)
    figure.suptitle("居民购电成本与节省（对比：不使用电池的基准方案）")
    _apply_chinese_fonts(figure)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return _save_figure(pyplot, figure, path)
