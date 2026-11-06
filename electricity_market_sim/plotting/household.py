"""居民建筑的电价与电池动作、设备功率、SOC，以及与不使用电池基准方案对比的购电成本 PNG 图表。"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

from ..models import HouseholdResult
from ..utils import hours_between
from .common import _format_time_axis, _save_figure
from .market import _chinese_font_families


_POWER_TOLERANCE_MW = 1e-9
COMPARISON_TOLERANCE = 1e-6
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


def _summarize_household_savings(
    *,
    baseline_payment_eur: float,
    actual_saving_eur: float,
    heat_differences: Iterable[float],
    final_soc_difference: float,
    electricity_differences: Iterable[float] = (),
) -> dict[str, float | bool | None]:
    """供 PNG 和交互图共用的节省比例与对照检查；输入为单个居民的时序差值。

    缺供净差仅供参考；逐时段绝对差之和避免不同时段的差异相互抵消。
    同等服务要求缺供量相同且期末 SOC 不低于基准，剩余电量不折算为金额。
    """
    heat = list(heat_differences)
    electricity = list(electricity_differences)
    heat_mismatch = sum(abs(value) for value in heat)
    electricity_mismatch = sum(abs(value) for value in electricity)
    return {
        "actual_saving_share": actual_saving_eur / baseline_payment_eur if baseline_payment_eur else None,
        "unmet_heat_difference_mwh_th": sum(heat),
        "unmet_heat_mismatch_mwh_th": heat_mismatch,
        "unmet_electricity_difference_mwh": sum(electricity),
        "unmet_electricity_mismatch_mwh": electricity_mismatch,
        "comparable": (
            heat_mismatch <= COMPARISON_TOLERANCE
            and electricity_mismatch <= COMPARISON_TOLERANCE
            and final_soc_difference >= -COMPARISON_TOLERANCE
        ),
    }


def _by_unit(results) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for result in results:
        grouped[result.dispatch.plan.unit_name].append(result)
    for rows in grouped.values():
        rows.sort(key=lambda result: (result.dispatch.plan.delivery_start, result.dispatch.plan.delivery_end))
    return grouped


def _active_spans(
    rows: list[HouseholdResult], active
) -> list[tuple[datetime, datetime]]:
    """把连续满足条件的交付时段合并为 (开始, 结束) 区间，用于背景着色。"""
    spans: list[tuple[datetime, datetime]] = []
    for row in rows:
        if not active(row):
            continue
        plan = row.dispatch.plan
        if spans and spans[-1][1] == plan.delivery_start:
            spans[-1] = (spans[-1][0], plan.delivery_end)
        else:
            spans.append((plan.delivery_start, plan.delivery_end))
    return spans


def _apply_chinese_fonts(figure) -> None:
    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)


def _plot_household_dispatch(
    pyplot,
    dates,
    results: tuple[HouseholdResult, ...],
    path: Path,
    *,
    days: int | None = None,
) -> Path:
    """绘制各居民的电价与电池动作、电池功率、SOC 及购电功率。

    days 非空时只绘制首个交付时段起的前 days 天，用于观察日内充放电节奏。
    """

    dispatch_by_unit = _by_unit(results)
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
            limit = rows[0].dispatch.plan.delivery_start + timedelta(days=days)
            rows = [row for row in rows if row.dispatch.plan.delivery_start < limit]
        dispatches = [row.dispatch for row in rows]
        plans = [dispatch.plan for dispatch in dispatches]
        price_axis, battery_axis, soc_axis, grid_axis = axes[4 * index: 4 * index + 4, 0]
        edges = [plans[0].delivery_start] + [plan.delivery_end for plan in plans]

        def stepped(values: list[float]) -> list[float]:
            return [*values, values[-1]]

        actual_prices = [
            float("nan")
            if cost.actual_price_eur_per_mwh is None
            else cost.actual_price_eur_per_mwh
            for cost in rows
        ]
        durations = [hours_between(plan.delivery_start, plan.delivery_end) for plan in plans]

        charge_spans = _active_spans(
            rows, lambda row: row.dispatch.battery_charge_power_mw > _POWER_TOLERANCE_MW)
        discharge_spans = _active_spans(
            rows, lambda row: row.dispatch.battery_discharge_power_mw > _POWER_TOLERANCE_MW)
        for axis in (price_axis, battery_axis):
            for spans, color in (
                (charge_spans, _COLORS["charge"]),
                (discharge_spans, _COLORS["discharge"]),
            ):
                for start, end in spans:
                    axis.axvspan(start, end, color=color, alpha=0.10, linewidth=0)

        price_axis.step(
            edges, stepped([plan.forecast_price_eur_per_mwh for plan in plans]),
            where="post", color=_COLORS["forecast"], linewidth=1.3, linestyle="--",
            label=f"{unit_name} 预测购电价（优化依据）",
        )
        price_axis.step(
            edges, stepped(actual_prices), where="post",
            color=_COLORS["actual_price"], linewidth=1.3, label=f"{unit_name} 实际购电均价",
        )
        price_axis.set_title(
            f"{unit_name}：居民购电价格与电池动作（绿色背景为充电，红色背景为放电）",
            loc="left", fontsize=10,
        )
        price_axis.set_ylabel("电价（欧元/MWh）")
        price_axis.grid(axis="y", alpha=0.25)
        price_axis.legend(loc="upper right", ncol=2, fontsize=8)

        battery_axis.fill_between(
            edges, 0, stepped([dispatch.battery_charge_power_mw * 1000 for dispatch in dispatches]),
            step="post", color=_COLORS["charge"], alpha=0.8, linewidth=0, label="充电",
        )
        battery_axis.fill_between(
            edges, 0, stepped([-dispatch.battery_discharge_power_mw * 1000 for dispatch in dispatches]),
            step="post", color=_COLORS["discharge"], alpha=0.8, linewidth=0, label="放电",
        )
        battery_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        battery_axis.set_title("电池充放电功率（充电为正、放电为负）", loc="left", fontsize=10)
        battery_axis.set_ylabel("功率（kW）")
        battery_axis.grid(axis="y", alpha=0.25)
        battery_axis.legend(loc="upper right", ncol=2, fontsize=8)

        soc_times: list[datetime] = []
        soc_values: list[float] = []
        for dispatch in dispatches:
            plan = dispatch.plan
            if not soc_times or soc_times[-1] != plan.delivery_start:
                soc_times.append(plan.delivery_start)
                soc_values.append(dispatch.soc_before * 100)
            soc_times.append(plan.delivery_end)
            soc_values.append(dispatch.soc_after * 100)
        soc_axis.plot(soc_times, soc_values, color=_COLORS["soc"], linewidth=1.5)
        soc_axis.set_title("电池 SOC", loc="left", fontsize=10)
        soc_axis.set_ylabel("SOC（%）")
        soc_axis.set_ylim(0, min(100.0, max(soc_values) * 1.1 + 1.0))
        soc_axis.grid(axis="y", alpha=0.25)

        grid_axis.fill_between(
            edges,
            stepped([dispatch.grid_power_bounds_mw[0] for dispatch in dispatches]),
            stepped([dispatch.grid_power_bounds_mw[1] for dispatch in dispatches]),
            step="post", color=_COLORS["flexibility"], alpha=0.45, linewidth=0,
            label="灵活性区间（成本容忍度内）",
        )
        grid_axis.step(
            edges,
            stepped([
                cost.baseline_grid_energy_mwh / duration
                for cost, duration in zip(rows, durations)
            ]),
            where="post", color=_COLORS["baseline"], linewidth=1.1, linestyle="--",
            label="基准购电（不使用电池）",
        )
        grid_axis.step(
            edges,
            stepped([
                cost.accepted_grid_energy_mwh / duration
                for cost, duration in zip(rows, durations)
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
    results: tuple[HouseholdResult, ...],
    path: Path,
) -> Path:
    """绘制各居民相对不使用电池基准方案的累计节省。

    对照不可比（缺供量不同或期末 SOC 低于基准）时标题标红。
    """

    costs_by_unit = _by_unit(results)
    household_names = sorted(costs_by_unit)
    figure, axes = pyplot.subplots(
        len(household_names),
        1,
        figsize=(16, 4.5 * len(household_names)),
        sharex=True,
        squeeze=False,
    )

    for index, unit_name in enumerate(household_names):
        rows = costs_by_unit[unit_name]
        cumulative_axis = axes[index, 0]

        times = [rows[0].dispatch.plan.delivery_start] + [row.dispatch.plan.delivery_end for row in rows]
        actual = [0.0]
        for row in rows:
            actual.append(actual[-1] + row.actual_saving_eur)
        baseline = sum(row.baseline_payment_eur for row in rows)
        saving = baseline - sum(row.payment_eur for row in rows)
        soc_gap = rows[-1].dispatch.soc_after - rows[-1].baseline.soc_after
        summary = _summarize_household_savings(
            baseline_payment_eur=baseline,
            actual_saving_eur=saving,
            heat_differences=(row.dispatch.unmet_heat_mwh_th - row.baseline.unmet_heat_mwh_th for row in rows),
            final_soc_difference=soc_gap,
        )
        comparable = summary["comparable"]
        share = summary["actual_saving_share"] or 0.0
        cumulative_axis.plot(
            times, actual, color=_COLORS["charge"], linewidth=1.8,
            label="实际节省（按实际成交价）",
        )
        cumulative_axis.axhline(0, color="#1F1F1F", linewidth=0.8)
        cumulative_axis.text(
            0.01, 0.95,
            f"实际节省 {saving:,.2f} EUR（占基准支付 {share:.2%}）\n"
            f"基准支付 {baseline:,.2f} EUR\n"
            + ("对照可比" if comparable
               else "对照不可比：实际节省不构成同等服务下的节省")
            + f"\n缺热量逐时段不一致 {summary['unmet_heat_mismatch_mwh_th']:.6f} MWh_th"
            + f"（净差 {summary['unmet_heat_difference_mwh_th']:+.6f}）"
            + f"\n期末 SOC 差 {soc_gap * 100:+.2f} 个百分点",
            transform=cumulative_axis.transAxes, ha="left", va="top", fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4", "facecolor": "white",
                "edgecolor": "#D1D5DB" if comparable else _COLORS["discharge"],
            },
        )
        cumulative_axis.set_title(
            f"{unit_name}：相对不使用电池的累计节省"
            + ("" if comparable else "（对照不可比）"),
            loc="left", fontsize=10,
            color="#1F1F1F" if comparable else _COLORS["discharge"],
        )
        cumulative_axis.set_ylabel("累计节省（EUR）")
        cumulative_axis.grid(axis="y", alpha=0.25)
        cumulative_axis.legend(loc="lower right", fontsize=8)

    axes[-1, 0].set_xlabel("交付时间")
    _format_time_axis(axes[-1, 0], dates)
    figure.suptitle("居民购电成本与节省（对比：不使用电池的基准方案）")
    _apply_chinese_fonts(figure)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return _save_figure(pyplot, figure, path)
