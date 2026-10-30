"""学习机组报价、出力与收益、学习曲线，以及学习策略与边际成本基准的对比图表。"""

import csv
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any
from ..errors import PlottingError
from ..deep_learning.learning_metrics import LEARNING_METRIC_FIELDS
from ..models import LearningStepResult, MarketClearingResult
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _UNIT_COLORS,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
)

_LEARNING_SEGMENTS = ("learning_minimum", "learning_flexible")
_CURVE_FIELDS = (
    ("mean_reward_per_hour", "每小时平均奖励"),
    ("total_profit_eur", "总利润（欧元，受时段长度影响）"),
)
_COMPARISON_FIELDS = (
    ("total_profit_eur", "总利润（欧元）"),
    ("accepted_energy_mwh", "成交电量（MWh）"),
    ("minimum_segment_average_bid_eur_per_mwh", "最低出力段平均报价（欧元/MWh）"),
    ("flexible_segment_average_bid_eur_per_mwh", "灵活出力段平均报价（欧元/MWh）"),
)


def _chinese_font_families() -> list[str]:
    from matplotlib.font_manager import fontManager

    available_fonts = {font.name for font in fontManager.ttflist}
    return [
        name for name in (
            "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
        ) if name in available_fonts
    ] + ["DejaVu Sans"]


def comparison_periods(rows: list[dict[str, Any]]) -> list[str]:
    """按出现顺序返回指标中含 Actor/基线对比的数据时段。"""

    return list(dict.fromkeys(
        row["phase"].rsplit("_", 1)[0]
        for row in rows
        if row["phase"].endswith(("_actor", "_baseline"))
    ))


def _plot_learning_unit(
    pyplot,
    dates,
    results: tuple[MarketClearingResult, ...],
    learning_steps: tuple[LearningStepResult, ...],
    path: Path,
    episode_label: str,
) -> Path:
    """按完整交付时段绘制两段报价、总成交功率和利润。"""

    markets = {market.delivery_start: market for market in results}
    segments_by_start = {
        start: {
            cleared.offer.offer_segment: cleared
            for cleared in market.offers
            if cleared.offer.offer_segment in _LEARNING_SEGMENTS
        }
        for start, market in markets.items()
    }
    unit_names = sorted(
        {
            cleared.offer.unit_name
            for segments in segments_by_start.values()
            for cleared in segments.values()
        }
    )
    steps = sorted(learning_steps, key=lambda step: step.delivery_start)
    times = [step.delivery_start for step in steps]

    def segment_series(
        segment: str, value, missing: float = float("nan")
    ) -> list[float]:
        return [
            value(segments_by_start[time][segment])
            if segment in segments_by_start.get(time, {})
            else missing
            for time in times
        ]

    accepted = {
        segment: segment_series(segment, lambda item: item.accepted_power_mw, 0.0)
        for segment in _LEARNING_SEGMENTS
    }

    figure, (price_axis, power_axis, profit_axis) = pyplot.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1, 1)},
    )
    for segment, label, color in (
        ("learning_minimum", "最低出力段报价", "#4C78A8"),
        ("learning_flexible", "灵活出力段报价", "#F58518"),
    ):
        price_axis.step(
            times,
            segment_series(segment, lambda item: item.offer.bid_price_eur_per_mwh),
            where="mid",
            color=color,
            linewidth=1.1,
            label=label,
        )
    price_axis.set_ylabel("报价（欧元/MWh）")
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper left", ncol=2, fontsize=8)

    available_power = [step.available_power_mw for step in steps]
    accepted_power = [
        minimum + flexible
        for minimum, flexible in zip(
            accepted["learning_minimum"], accepted["learning_flexible"]
        )
    ]
    power_axis.fill_between(
        times, 0, accepted_power, step="mid", color="#F58518", alpha=0.45,
        linewidth=0, label="总成交功率",
    )
    power_axis.step(
        times,
        available_power,
        where="mid",
        color="#1F1F1F",
        linestyle="--",
        linewidth=1.1,
        label="可用功率",
    )
    power_top = max([*available_power, *accepted_power, 0.0])
    power_axis.set_ylim(0.0, power_top * 1.05 if power_top > 0 else 1.0)
    if max(accepted_power, default=0.0) <= _ENERGY_TOLERANCE_MWH:
        _annotate_no_supply(power_axis, "无成交出力")
    power_axis.set_ylabel("功率（MW）")
    power_axis.grid(axis="y", alpha=0.25)
    power_axis.legend(loc="upper left", ncol=2, fontsize=8)

    profit_axis.step(
        times,
        [step.profit_eur for step in steps],
        where="mid",
        color="#54A24B",
        linewidth=1.1,
        label="利润",
    )
    profit_axis.axhline(0, color="#1F1F1F", linewidth=0.6)
    profit_axis.set_ylabel("每时段利润（欧元）")
    profit_axis.set_xlabel("交付开始时间")
    profit_axis.grid(axis="y", alpha=0.25)
    profit_axis.legend(loc="upper left", fontsize=8)
    _format_time_axis(profit_axis, dates)
    profit_axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(
        profit_axis.xaxis.get_major_locator(),
        formats=["%Y", "%m", "%d", "%H:%M", "%H:%M", "%S"],
        zero_formats=["", "%Y", "%m-%d", "%m-%d", "%H:%M", "%H:%M"],
        offset_formats=["", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d %H:%M"],
    ))

    unit_label = "、".join(unit_names) or "学习机组"
    figure.suptitle(f"学习机组报价与收益——{unit_label}（{episode_label}）")
    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _read_learning_metrics(path: Path) -> list[dict[str, Any]]:
    """读取第五版训练器或评估器写出的指标 CSV。"""

    if not path.is_file():
        raise PlottingError(f"Learning metrics file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(
            {"run", "phase", "episode", *LEARNING_METRIC_FIELDS}
            - set(reader.fieldnames or ())
        )
        if missing:
            raise PlottingError(
                f"{path.name} is missing learning metric columns: {', '.join(missing)}."
            )
        try:
            return [
                {
                    "run": int(row["run"]),
                    "phase": row["phase"],
                    "episode": int(row["episode"]),
                    **{name: float(row[name]) for name in LEARNING_METRIC_FIELDS},
                }
                for row in reader
            ]
        except (TypeError, ValueError) as exc:
            raise PlottingError(
                f"{path.name} contains a missing or non-numeric learning metric."
            ) from exc


def _plot_learning_curves(pyplot, rows: list[dict[str, Any]], path: Path) -> Path:
    """绘制每次独立运行中训练回合和验证回合的指标。"""

    runs = sorted({row["run"] for row in rows})
    figure, axes = pyplot.subplots(len(_CURVE_FIELDS), 1, figsize=(12, 7), sharex=True)
    initial_episodes = [
        row["episode"] for row in rows if row["phase"] == "initial_experience"
    ]
    if initial_episodes:
        for axis in axes:
            axis.axvspan(
                min(initial_episodes) - 0.5,
                max(initial_episodes) + 0.5,
                color="#E5E7EB",
                alpha=0.7,
                zorder=0,
                label="初始经验收集（不更新策略网络）",
            )
    for index, run in enumerate(runs):
        color = _UNIT_COLORS[index % len(_UNIT_COLORS)]
        training_color = _UNIT_COLORS[(index + 1) % len(_UNIT_COLORS)]
        label_prefix = f"第 {run} 次运行：" if len(runs) > 1 else ""
        run_rows = sorted(
            (row for row in rows if row["run"] == run), key=lambda row: row["episode"]
        )
        initial = [row for row in run_rows if row["phase"] == "initial_experience"]
        training = [row for row in run_rows if row["phase"] == "train"]
        validation = [row for row in run_rows if row["phase"] == "validation"]
        best = (
            max(validation, key=lambda row: row["total_reward"])
            if validation
            else None
        )
        for axis, (field, _) in zip(axes, _CURVE_FIELDS):
            if initial:
                axis.plot(
                    [row["episode"] for row in initial],
                    [row[field] for row in initial],
                    color=color,
                    alpha=0.45,
                    linewidth=1.0,
                    linestyle="--",
                )
            if training:
                connected = [*initial[-1:], *training]
                axis.plot(
                    [row["episode"] for row in connected],
                    [row[field] for row in connected],
                    color=training_color,
                    alpha=0.85,
                    linewidth=1.2,
                    label=f"{label_prefix}训练",
                )
            if validation:
                axis.plot(
                    [row["episode"] for row in validation],
                    [row[field] for row in validation],
                    color=color,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    label=f"{label_prefix}验证",
                )
            if best is not None:
                axis.scatter(
                    [best["episode"]],
                    [best[field]],
                    marker="*",
                    s=150,
                    color=color,
                    edgecolors="#1F1F1F",
                    linewidths=0.6,
                    zorder=4,
                    label=f"{label_prefix}最佳验证模型",
                )
    for axis, (_, label) in zip(axes, _CURVE_FIELDS):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    axes[-1].set_xlabel("训练轮次")
    axes[-1].xaxis.set_major_locator(pyplot.MaxNLocator(integer=True))
    figure.suptitle("学习曲线")
    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _format_metric(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1_000 else f"{value:.3g}"


def _plot_evaluation_comparison(
    pyplot, rows: list[dict[str, Any]], path: Path, period: str
) -> Path:
    """对比确定性 Actor 与边际成本基准策略在指定数据时段上的结果。"""

    font_families = _chinese_font_families()
    phases = (
        (f"{period}_actor", "学习策略", "#4C78A8"),
        (f"{period}_baseline", "边际成本基准", "#9CA3AF"),
    )
    run_count = max(
        len([row for row in rows if row["phase"] == phase])
        for phase, _, _ in phases
    )
    figure, axes = pyplot.subplots(2, 2, figsize=(12, 7.5))
    for axis, (field, title) in zip(axes.flat, _COMPARISON_FIELDS):
        for position, (phase, _, color) in enumerate(phases):
            values = [row[field] for row in rows if row["phase"] == phase]
            if not values:
                continue
            mean = fmean(values)
            spread = pstdev(values)
            axis.bar(
                position,
                mean,
                width=0.6,
                color=color,
                yerr=spread if len(values) > 1 else None,
                capsize=4,
            )
            is_negative = mean < 0
            axis.annotate(
                _format_metric(mean),
                (
                    position,
                    min(mean - spread, *values)
                    if is_negative
                    else max(mean + spread, *values),
                ),
                xytext=(0, -3 if is_negative else 3),
                textcoords="offset points",
                ha="center",
                va="top" if is_negative else "bottom",
                fontsize=8,
            )
            if len(values) > 1:
                axis.scatter(
                    [position] * len(values),
                    values,
                    s=12,
                    color="#1F1F1F",
                    zorder=3,
                )
        axis.axhline(0, color="#1F1F1F", linewidth=0.6)
        axis.set_xticks(
            range(len(phases)),
            [label for _, label, _ in phases],
            fontfamily=font_families,
        )
        axis.set_xlim(-0.6, len(phases) - 0.4)
        axis.margins(y=0.15)
        axis.set_title(title, fontsize=10, fontfamily=font_families)
        axis.grid(axis="y", alpha=0.25)
    period_label = {"train": "训练期", "validation": "验证期", "test": "测试期"}.get(period, period)
    spread_note = (
        f"{run_count} 次独立运行；柱形为均值，误差线为总体标准差，圆点为各次结果"
        if run_count > 1 else "1 次独立运行"
    )
    figure.suptitle(
        f"学习策略与边际成本基准对比（无探索噪声）——{period_label}，{spread_note}",
        fontfamily=font_families,
    )
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)
