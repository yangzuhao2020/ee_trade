"""?????????????? PNG ???"""

from pathlib import Path
from ..errors import PlottingError
from ..models import MarketClearingResult
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
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


def _market_price_series(
    results: tuple[MarketClearingResult, ...],
) -> tuple[list[float], str]:
    """根据市场的结算方式，返回该方式下存在的价格指标。"""

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


def _plot_market_overview(
    pyplot, dates, results, path: Path, forecast_prices: dict | None = None
) -> Path:
    """forecast_prices 非空时（居民场景），在价格子图叠加居民优化所用的预测电价。"""
    times = [market.delivery_start for market in results]
    prices, price_label = _market_price_series(results)
    price_label = {"Clearing price": "出清电价", "Average trade price": "全市场成交量加权均价"}[price_label]
    requested_demand = [
        market.requested_demand_mwh - market.requested_storage_charge_mwh
        for market in results
    ]

    def by_type(attribute: str, demand_type: str) -> list[float]:
        return [
            market.demand_total(attribute, demand_type=demand_type) for market in results
        ]

    accepted_layers = (
        (by_type("accepted_energy_mwh", "inelastic_load"), "#4C78A8", "已满足的刚性负荷"),
        (by_type("accepted_energy_mwh", "household_load"), "#72B7B2", "已成交的居民用电需求"),
        (by_type("accepted_energy_mwh", "industrial_load"), "#D4A72C", "已成交的工业用电需求"),
        (by_type("accepted_energy_mwh", "elastic_load"), "#54A24B", "已成交的弹性需求"),
        ([market.cleared_export_mwh for market in results], "#F58518", "已成交的外送需求"),
    )
    demand_types = {demand.bid.demand_type for market in results for demand in market.demand_bids}
    unaccepted_lines = (
        ("inelastic_load", [market.unserved_load_mwh for market in results], "#E45756", "未满足的刚性负荷"),
        ("elastic_load", by_type("unserved_energy_mwh", "elastic_load"), "#7F7F7F", "未成交的弹性需求"),
        ("household_load", by_type("unserved_energy_mwh", "household_load"), "#72B7B2", "未成交的居民用电需求"),
        ("industrial_load", by_type("unserved_energy_mwh", "industrial_load"), "#D4A72C", "未成交的工业用电需求"),
        ("export", [market.unfulfilled_export_mwh for market in results], "#F58518", "未满足的外送需求"),
    )
    unaccepted_panels = [panel for panel in unaccepted_lines if panel[0] in demand_types]

    figure, axes = pyplot.subplots(
        2 + len(unaccepted_panels),
        1,
        figsize=(16, 7 + 2.5 * len(unaccepted_panels)),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.45, *([0.8] * len(unaccepted_panels))]},
    )
    price_axis, demand_axis = axes[:2]
    figure.suptitle("电力市场概览")
    price_axis.step(
        times,
        prices,
        where="mid",
        color="#2F5597",
        linewidth=1.6,
        label=price_label,
    )
    if forecast_prices:
        price_axis.step(
            times,
            [forecast_prices.get(time, float("nan")) for time in times],
            where="mid",
            color="#F58518",
            linewidth=1.2,
            linestyle="--",
            label="居民预测购电价（优化依据）",
        )
    price_axis.set_title(price_label, loc="left", fontsize=10)
    price_axis.set_ylabel("电价（欧元/MWh）")
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper right")

    lower = None
    for values, color, label in accepted_layers:
        if not any(abs(value) > _ENERGY_TOLERANCE_MWH for value in values):
            continue
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
        label="总申报需求（不含储能充电）",
    )
    demand_axis.set_title("成交需求与总申报需求", loc="left", fontsize=10)
    demand_axis.set_ylabel("电量（MWh）")
    demand_axis.grid(axis="y", alpha=0.25)
    demand_axis.legend(loc="upper right", ncol=3)

    for axis, (_, values, color, label) in zip(axes[2:], unaccepted_panels):
        axis.step(times, values, where="mid", color=color, linewidth=1.2)
        axis.set_title(label, loc="left", fontsize=10)
        axis.set_ylabel("未成交电量（MWh）")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        if all(abs(value) <= _ENERGY_TOLERANCE_MWH for value in values):
            axis.set_ylim(0, 1)
            axis.set_yticks([0])
            axis.text(0.98, 0.75, "全时段为 0", transform=axis.transAxes, ha="right")
    axes[-1].set_xlabel("交付开始时间")
    _format_time_axis(axes[-1], dates)
    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _plot_market_summary(pyplot, dates, results, path: Path) -> Path:
    """绘制所有产品的 EOM 供需两侧成交功率，电价由市场概览图展示。

    尽管按定义每个已出清产品的供需相等，这里仍刻意分别绘制供给和需求。
    两条曲线能直观展示这种平衡，也与 ASSUME 市场汇总图的惯例一致。
    """

    times = [market.delivery_start for market in results]
    cleared_supply_gw = [
        market.cleared_energy_mwh / market.duration_hours / 1_000 for market in results
    ]
    cleared_demand_gw = [
        sum(demand.accepted_power_mw for demand in market.demand_bids) / 1_000
        for market in results
    ]
    figure, axis = pyplot.subplots(figsize=(16, 7))

    demand_line = axis.step(
        times,
        cleared_demand_gw,
        where="mid",
        color="#54A24B",
        linewidth=1.5,
        linestyle="--",
        label="需求侧成交功率",
        zorder=3,
    )[0]
    supply_line = axis.step(
        times,
        cleared_supply_gw,
        where="mid",
        color="#4C78A8",
        linewidth=1.15,
        label="供给侧成交功率",
        zorder=4,
    )[0]
    axis.set_title("电能量市场出清汇总")
    axis.set_ylabel("成交功率（GW）")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("交付开始时间")
    _format_time_axis(axis, dates)
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(
        axis.xaxis.get_major_locator(),
        formats=["%Y", "%m", "%d", "%H:%M", "%H:%M", "%S"],
        zero_formats=["", "%Y", "%m-%d", "%m-%d", "%H:%M", "%H:%M"],
        offset_formats=["", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d %H:%M"],
    ))
    axis.legend(
        (demand_line, supply_line),
        ("需求侧成交功率", "供给侧成交功率"),
        loc="upper left",
        ncol=2,
    )

    from matplotlib.text import Text

    font_families = _chinese_font_families()
    for text in figure.findobj(Text):
        text.set_fontfamily(font_families)
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)
