"""从仿真结果 CSV 文件构建交互式 Plotly 图表。
展示市场供需、成交价格与出力、运营商利润，以及可用的储能和工业数据。"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative
from plotly.subplots import make_subplots

from electricity_market_sim.plotting.household import (
    COMPARISON_TOLERANCE,
    _summarize_household_savings,
)

ENERGY_TOLERANCE_MWH = 1e-7
MAX_DISPATCH_UNITS = 5
MAX_OPERATORS = 5
MAX_STORAGE_UNITS = 2
# Above this many traces a unified hover box would cover the whole chart.
MAX_UNIFIED_HOVER_TRACES = 12
UNIT_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#D4A72C",
)
TABLES = {
    "market": "market_results.csv",
    "demand": "demand_results.csv",
    "unit": "unit_results.csv",
    "operator": "operator_results.csv",
    "exchange": "exchange_results.csv",
    "storage": "storage_results.csv",
    "industry": "industry_results.csv",
    "household": "household_results.csv",
}
# write_results() always writes these; the others exist only when the run has such data.
REQUIRED_TABLES = ("market", "demand", "unit", "operator")
TIME_COLUMNS = ("opening_time", "delivery_start", "delivery_end", "datetime")
ACCEPTED_LAYERS = (
    ("inelastic_load", "#4C78A8", "已满足的刚性负荷"),
    ("household_load", "#72B7B2", "已成交的居民用电需求"),
    ("industrial_load", "#D4A72C", "已成交的工业用电需求"),
    ("elastic_load", "#54A24B", "已成交的弹性需求"),
    ("export", "#F58518", "已成交的外送需求"),
)
UNACCEPTED_LINES = (
    ("inelastic_load", "#E45756", "未满足的刚性负荷"),
    ("elastic_load", "#7F7F7F", "未成交的弹性需求"),
    ("household_load", "#72B7B2", "未成交的居民用电需求"),
    ("industrial_load", "#D4A72C", "未成交的工业用电需求"),
    ("export", "#F58518", "未满足的外送需求"),
)
DEMAND_TYPES = [*(demand_type for demand_type, _, _ in ACCEPTED_LAYERS), "storage_charge"]
# 较早版本的 household_results.csv 缺少部分或全部成本对照列，相应内容不绘制。
HOUSEHOLD_SAVING_COLUMNS = (
    "baseline_payment_eur", "actual_saving_eur",
    "unmet_electricity_mwh", "baseline_unmet_electricity_mwh",
    "unmet_heat_mwh_th", "baseline_unmet_heat_mwh_th", "soc_after", "baseline_soc_after",
)


def has_results(run_dir: Path) -> bool:
    return all((run_dir / TABLES[key]).is_file() for key in REQUIRED_TABLES)


def load_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    """读取一次运行目录下的各个 CSV 表，并解析时间列。

    缺失或仅含表头的可选表会被略过，依赖这些表的图表也随之跳过。
    必需表缺失或 CSV 没有表头时直接抛错。
    """

    tables = {}
    for key, filename in TABLES.items():
        path = run_dir / filename
        if key not in REQUIRED_TABLES and not path.is_file():
            continue
        table = pd.read_csv(path)
        if table.empty and key not in REQUIRED_TABLES:
            continue
        for column in TIME_COLUMNS:
            if column in table:
                table[column] = pd.to_datetime(table[column])
        tables[key] = table
    return tables


def build_charts(tables: dict[str, pd.DataFrame]) -> list[go.Figure]:
    """按显示优先级顺序，返回适用于本次运行的所有图表。"""

    if tables["market"].empty:
        return []
    figures = [
        market_overview(tables),
        market_summary(tables),
        dispatch_by_unit(tables),
    ]
    if "household" not in tables:
        figures.append(operator_profit(tables))
    else:
        figures.append(household_dispatch(tables))
        figures.append(household_savings(tables))
    if "storage" in tables:
        figures.append(storage_dispatch(tables))
    if "industry" in tables:
        figures.append(industry_dispatch(tables))
    return figures


def market_overview(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    price, price_label = _price(market)
    requested = _demand_by_type(tables, "requested_energy_mwh")
    accepted = _demand_by_type(tables, "accepted_energy_mwh")
    unaccepted = _demand_by_type(tables, "unaccepted_energy_mwh")
    demand_types = set(tables["demand"]["demand_type"])
    household_mode = "household_load" in demand_types
    demand_titles = (
        ["居民申报需求", "已成交的居民用电需求"]
        if household_mode else ["成交需求与总申报需求"]
    )
    unaccepted_panels = [
        (demand_type, color, label)
        for demand_type, color, label in UNACCEPTED_LINES
        if demand_type in demand_types and (not household_mode or demand_type != "inelastic_load")
    ]
    demand_row_count = 1 + len(demand_titles)
    row_count = demand_row_count + len(unaccepted_panels)

    figure = make_subplots(
        rows=row_count, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[1, *([1.2, 1.2] if household_mode else [1.45]), *([0.8] * len(unaccepted_panels))],
        subplot_titles=[
            price_label, *demand_titles,
            *[label for _, _, label in unaccepted_panels],
        ],
    )
    figure.add_trace(_line(times, price, price_label, "#2F5597"), row=1, col=1)
    if household_mode:
        requested_values = requested["household_load"]
        requested_label = "居民申报需求"
    else:
        for demand_type, color, label in ACCEPTED_LAYERS:
            if accepted[demand_type].abs().max() > ENERGY_TOLERANCE_MWH:
                figure.add_trace(
                    _line(times, accepted[demand_type], label, color, width=0.5, stackgroup="accepted"),
                    row=2,
                    col=1,
                )
        requested_values = market["requested_demand_energy_mwh"] - requested["storage_charge"].to_numpy()
        requested_label = "总申报需求（不含储能充电）"
    figure.add_trace(
        _line(
            times,
            requested_values,
            requested_label,
            "#1F1F1F",
            width=1.1,
        ),
        row=2,
        col=1,
    )
    if household_mode:
        figure.add_trace(
            _line(times, accepted["household_load"], "已成交的居民用电需求", "#72B7B2"),
            row=3,
            col=1,
        )
    for row, (demand_type, color, label) in enumerate(unaccepted_panels, start=demand_row_count + 1):
        values = unaccepted[demand_type]
        figure.add_trace(
            _line(times, values, label, color, width=1.2, showlegend=False), row=row, col=1
        )
        figure.update_yaxes(title_text="未成交电量（MWh）", rangemode="tozero", row=row, col=1)
        if values.abs().max() <= ENERGY_TOLERANCE_MWH:
            figure.update_yaxes(range=[0, 1], tickvals=[0], row=row, col=1)
            figure.add_annotation(
                x=0.98, y=0.75, xref=f"x{row} domain", yref=f"y{row} domain",
                text="全时段为 0", showarrow=False, xanchor="right",
            )
    figure.update_yaxes(title_text="电价（欧元/MWh）", row=1, col=1)
    figure.update_yaxes(title_text="电量（MWh）", row=2, col=1)
    if household_mode:
        figure.update_yaxes(rangemode="tozero", row=2, col=1)
        figure.update_yaxes(title_text="电量（MWh）", rangemode="tozero", row=3, col=1)
    figure.update_xaxes(title_text="交付开始时间", row=row_count, col=1)
    title = "电力市场概览"
    if household_mode:
        title += f"<br><sup>未满足的刚性负荷合计：{unaccepted['inelastic_load'].sum():,.6f} MWh</sup>"
    figure = _finish(figure, title, (860 if household_mode else 600) + 220 * len(unaccepted_panels))
    if household_mode:
        figure.update_layout(margin_t=110)
    return figure


def market_summary(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    duration = market["duration_hours"]
    cleared_demand_gw = _demand_by_type(tables, "accepted_energy_mwh").sum(axis=1) / duration / 1_000
    cleared_supply_gw = market["cleared_energy_mwh"] / duration / 1_000

    figure = go.Figure()
    figure.add_trace(
        _line(times, cleared_demand_gw, "需求侧成交功率", "#54A24B", dash="dash")
    )
    figure.add_trace(
        _line(times, cleared_supply_gw, "供给侧成交功率", "#4C78A8", width=1.15)
    )
    figure.update_yaxes(title_text="成交功率（GW）")
    figure.update_xaxes(title_text="交付开始时间", tickformat="%m-%d\n%H:%M")
    return _finish(figure, "电能量市场出清汇总", 540)


def dispatch_by_unit(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    supply = _accepted_supply(tables)
    energy = supply.groupby("unit_name")["energy_mwh"].sum()
    ranked = energy[energy > ENERGY_TOLERANCE_MWH].sort_values(ascending=False, kind="stable").index
    power = (
        supply.pivot_table(index="delivery_start", columns="unit_name", values="power_mw", aggfunc="sum")
        .reindex(times)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    selected = ranked[:MAX_DISPATCH_UNITS]
    omitted_count = len(ranked) - len(selected)

    figure = go.Figure()
    for index, name in enumerate(selected):
        color = UNIT_COLORS[index % len(UNIT_COLORS)]
        figure.add_trace(_line(times, power[name], name, color, width=1.25))
    if selected.empty:
        _annotate_empty(figure, "暂无供给报价")
    title = f"成交电量最多的 {MAX_DISPATCH_UNITS} 个供给单元"
    if omitted_count:
        title += f"（其余 {omitted_count} 个未展示）"
    figure.update_yaxes(title_text="功率（MW）", rangemode="tozero")
    figure.update_xaxes(title_text="交付开始时间")
    return _finish(figure, title, 560)


def operator_profit(tables: dict[str, pd.DataFrame]) -> go.Figure:
    metric_label = "利润／净现金流" if "storage" in tables else "利润"
    title = f"累计{metric_label}最高的 {MAX_OPERATORS} 家运营商"
    profits = tables["operator"].groupby("unit_operator")["profit_eur"].sum()
    profits = profits[profits.abs() > 1e-6]
    ranked = sorted(profits.index, key=lambda operator: (-profits[operator], operator))
    operators = ranked[:MAX_OPERATORS][::-1]
    values = [profits[operator] for operator in operators]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=operators,
            orientation="h",
            marker_color=["#54A24B" if value >= 0 else "#E45756" for value in values],
            text=[f"{value:,.0f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{metric_label}：%{{x:,.0f}} 欧元<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#1F1F1F", line_width=0.8)
    if not operators:
        _annotate_empty(figure, f"暂无非零的{metric_label}")
    figure.update_xaxes(title_text=f"{metric_label}（欧元）")
    return _finish(
        figure,
        title,
        360,
        hovermode="closest",
    )


def storage_dispatch(tables: dict[str, pd.DataFrame]) -> go.Figure:
    storage = tables["storage"].sort_values(["unit_name", "delivery_start", "opening_time"], kind="stable")
    energy = storage.groupby("unit_name")[["accepted_charge_mwh", "accepted_discharge_mwh"]].sum().sum(axis=1)
    ranked = energy.sort_values(ascending=False, kind="stable").index
    names = ranked[:MAX_STORAGE_UNITS].tolist()
    omitted_count = len(ranked) - len(names)
    # A single unit needs no per-unit colors, so net power is colored by direction instead.
    single_unit = len(names) == 1
    colors = UNIT_COLORS[:len(names)]

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=(1.1, 1))
    for name, color in zip(names, colors):
        rows = storage[storage["unit_name"] == name]
        display_name = f"抽水蓄能电站 {name}" if rows["technology"].iloc[0] == "PSPP" else name
        soc_x, soc_y, power_x, power_y = [], [], [], []
        previous_end = previous_power = None
        for row in rows.itertuples(index=False):
            if row.delivery_start != previous_end:
                # A delivery gap breaks both curves instead of bridging the missing period.
                if previous_end is not None:
                    soc_x.append(None)
                    soc_y.append(None)
                    power_x += [previous_end, None]
                    power_y += [previous_power, None]
                soc_x.append(row.delivery_start)
                soc_y.append(row.soc_before * 100)
            previous_end = row.delivery_end
            previous_power = (row.accepted_discharge_mwh - row.accepted_charge_mwh) / row.duration_hours
            soc_x.append(row.delivery_end)
            soc_y.append(row.soc_after * 100)
            power_x.append(row.delivery_start)
            power_y.append(previous_power)
        power_x.append(previous_end)
        power_y.append(previous_power)
        figure.add_trace(_line(soc_x, soc_y, display_name, color, shape="linear", legendgroup=name), row=1, col=1)
        if not single_unit:
            figure.add_trace(
                _line(power_x, power_y, display_name, color, shape="hv", width=1.25, legendgroup=name, showlegend=False),
                row=2,
                col=1,
            )
            continue
        for label, power_color, clip in (("放电", "#54A24B", max), ("充电", "#F58518", min)):
            red, green, blue = hex_to_rgb(power_color)
            figure.add_trace(
                _line(
                    power_x,
                    [None if value is None else clip(value, 0.0) for value in power_y],
                    label,
                    power_color,
                    shape="hv",
                    width=1.0,
                    fill="tozeroy",
                    fillcolor=f"rgba({red}, {green}, {blue}, 0.35)",
                ),
                row=2,
                col=1,
            )
    figure.add_hline(y=0, line_color="#1F1F1F", line_width=0.8, row=2, col=1)
    figure.update_yaxes(title_text="荷电状态（%）", range=[0, 100], row=1, col=1)
    figure.update_yaxes(title_text="净功率（MW）<br>正值放电／负值充电", row=2, col=1)
    figure.update_xaxes(title_text="交付开始时间", row=2, col=1)
    figure.update_layout(hoverlabel={"namelength": -1})
    title = "储能调度与荷电状态"
    if omitted_count:
        title += f"——充放电量前 {MAX_STORAGE_UNITS} 个，其余 {omitted_count} 个未展示"
    return _finish(figure, title, 720)


def industry_dispatch(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    industry = tables["industry"].rename(columns={"datetime": "delivery_start"})
    industry["delivery_end"] = industry["delivery_start"].map(market.set_index("delivery_start")["delivery_end"])

    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=(1.35, 1, 1)
    )
    for index, (name, rows) in enumerate(industry.sort_values("delivery_start").groupby("unit_name")):
        color = qualitative.D3[index % len(qualitative.D3)]
        red, green, blue = hex_to_rgb(color)
        step_x = [*rows["delivery_start"], rows["delivery_end"].iloc[-1]]
        cumulative_x = [rows["delivery_start"].iloc[0], *rows["delivery_end"]]
        band = {"shape": "hv", "width": 0.5, "legendgroup": f"{name} 灵活性"}
        figure.add_trace(
            _line(step_x, _repeat_last(rows["minimum_power_mw"]), f"{name} 最小功率", color, **band),
            row=1,
            col=1,
        )
        figure.add_trace(
            _line(
                step_x,
                _repeat_last(rows["maximum_power_mw"]),
                f"{name} 最大功率",
                color,
                fill="tonexty",
                fillcolor=f"rgba({red}, {green}, {blue}, 0.16)",
                **band,
            ),
            row=1,
            col=1,
        )
        for row, x, planned, actual, planned_label, actual_label in (
            (1, step_x,
             _repeat_last(rows["planned_grid_power_mw"]),
             _repeat_last(rows["actual_grid_power_mw"]), "计划功率", "实际功率"),
            (2, step_x,
             _repeat_last(rows["planned_steel_output_t"]),
             _repeat_last(rows["actual_steel_output_t"]), "计划钢产量", "实际钢产量"),
            (3, cumulative_x,
             [0.0, *rows["planned_steel_output_t"].cumsum()],
             [0.0, *rows["actual_steel_output_t"].cumsum()], "累计计划钢产量", "累计实际钢产量"),
        ):
            matches_plan = all(
                abs(planned_value - actual_value) <= COMPARISON_TOLERANCE
                for planned_value, actual_value in zip(planned, actual, strict=True)
            )
            if not matches_plan:
                figure.add_trace(
                    _line(x, planned, f"{name} {planned_label}", color, shape="hv", dash="dash"),
                    row=row,
                    col=1,
                )
            label = f"{name} {actual_label}" + ("（与计划一致）" if matches_plan else "")
            figure.add_trace(
                _line(x, actual, label, color, shape="hv", width=1.8),
                row=row,
                col=1,
            )

    window_starts = set(industry.groupby("window_id")["delivery_start"].min())
    boundaries = sorted(window_starts - {industry["delivery_start"].min()})
    for boundary in boundaries:
        figure.add_vline(x=boundary, line_dash="dot", line_color="#6B7280", line_width=0.9)
    title = "工业生产计划、执行与灵活性"
    if boundaries:
        title += "<br><sup>竖向点线表示滚动窗口边界</sup>"
    figure.update_yaxes(title_text="电网取电功率（MW）", row=1, col=1)
    figure.update_yaxes(title_text="每交付时段钢产量（吨）", row=2, col=1)
    figure.update_yaxes(title_text="累计钢产量（吨）", row=3, col=1)
    figure.update_xaxes(title_text="交付开始时间", row=3, col=1)
    return _finish(figure, title, 820)


def household_dispatch(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """各居民的电价、电池充放电功率及 SOC。"""

    household = tables["household"].sort_values("delivery_start", kind="stable")
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=(1.2, 0.9, 0.8),
        subplot_titles=["居民购电价格", "电池充放电功率（充电为正、放电为负）", "电池 SOC"],
    )
    grouped = household.groupby("unit_name", sort=True)
    for name, rows in grouped:
        prefix = f"{name} " if len(grouped) > 1 else ""
        step_x = [*rows["delivery_start"], rows["delivery_end"].iloc[-1]]
        step = {"shape": "hv", "legendgroup": name}
        figure.add_trace(
            _line(step_x, _repeat_last(rows["forecast_price_eur_per_mwh"]), f"{name} 预测购电价（优化依据）",
                  "#F58518", dash="dash", width=1.3, **step),
            row=1, col=1,
        )
        if "actual_price_eur_per_mwh" in rows:
            figure.add_trace(
                _line(step_x, _repeat_last(rows["actual_price_eur_per_mwh"]), f"{name} 实际购电均价",
                      "#2F5597", width=1.3, **step),
                row=1, col=1,
            )
        for label, values, color in (
            ("充电", rows["battery_charge_power_mw"] * 1000, "#54A24B"),
            ("放电", -rows["battery_discharge_power_mw"] * 1000, "#E45756"),
        ):
            red, green, blue = hex_to_rgb(color)
            figure.add_trace(
                _line(step_x, _repeat_last(values), f"{prefix}{label}", color, width=0.8, fill="tozeroy",
                      fillcolor=f"rgba({red}, {green}, {blue}, 0.45)", **step),
                row=2, col=1,
            )
        figure.add_trace(
            _line([rows["delivery_start"].iloc[0], *rows["delivery_end"]],
                  [rows["soc_before"].iloc[0] * 100, *(rows["soc_after"] * 100)],
                  f"{prefix}SOC", "#2F5597", shape="linear", legendgroup=name),
            row=3, col=1,
        )
    figure.add_hline(y=0, line_color="#1F1F1F", line_width=0.8, row=2, col=1)
    figure.update_yaxes(title_text="电价（欧元/MWh）", row=1, col=1)
    figure.update_yaxes(title_text="功率（kW）", row=2, col=1)
    figure.update_yaxes(title_text="SOC（%）", rangemode="tozero", row=3, col=1)
    figure.update_xaxes(title_text="交付开始时间", row=3, col=1)
    return _finish(figure, "居民调度", 740)


def household_savings(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """各居民相对不使用电池基准方案的累计节省，并标明对照是否可比。

    只有每个时段的缺电量、缺热量都与基准相同且期末 SOC 不低于基准时，支付差额才是同等服务下的节省；
    结果缺少对照列时只显示提示。
    """

    title = "居民购电成本与节省（对比：不使用电池的基准方案）"
    missing = [column for column in HOUSEHOLD_SAVING_COLUMNS if column not in tables["household"]]
    if missing:
        figure = go.Figure()
        _annotate_empty(figure, "结果缺少对照数据（" + "、".join(missing) + "），请重新运行仿真生成")
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return _finish(figure, title, 320)

    household = tables["household"].sort_values("delivery_start", kind="stable")
    figure = go.Figure()
    grouped = household.groupby("unit_name", sort=True)
    notes = []
    all_comparable = True
    for name, rows in grouped:
        prefix = f"{name} " if len(grouped) > 1 else ""
        cumulative_x = [rows["delivery_start"].iloc[0], *rows["delivery_end"]]
        figure.add_trace(
            _line(cumulative_x, [0.0, *rows["actual_saving_eur"].cumsum()], f"{prefix}实际节省（按实际成交价）",
                  "#54A24B", shape="linear", width=1.8, legendgroup=name),
        )

        baseline = rows["baseline_payment_eur"].sum()
        saving = rows["actual_saving_eur"].sum()
        soc_gap = rows["soc_after"].iloc[-1] - rows["baseline_soc_after"].iloc[-1]
        summary = _summarize_household_savings(
            baseline_payment_eur=baseline,
            actual_saving_eur=saving,
            heat_differences=rows["unmet_heat_mwh_th"] - rows["baseline_unmet_heat_mwh_th"],
            final_soc_difference=soc_gap,
            electricity_differences=rows["unmet_electricity_mwh"] - rows["baseline_unmet_electricity_mwh"],
        )
        comparable = summary["comparable"]
        all_comparable &= comparable
        share = f"（{summary['actual_saving_share']:.2%}）" if baseline else ""
        verdict = "对照可比" if comparable else "<b>对照不可比：实际节省不构成同等服务下的节省</b>"
        notes.append(
            f"{name}：实际节省 {saving:,.2f} 欧元{share}；"
            f"{verdict}<br>缺电量逐时段不一致 {summary['unmet_electricity_mismatch_mwh']:.6f} MWh"
            f"（净差 {summary['unmet_electricity_difference_mwh']:+.6f}），"
            f"缺热量逐时段不一致 {summary['unmet_heat_mismatch_mwh_th']:.6f} MWh_th"
            f"（净差 {summary['unmet_heat_difference_mwh_th']:+.6f}），"
            f"期末 SOC 差 {soc_gap * 100:+.2f} 个百分点"
        )
    figure.add_hline(y=0, line_color="#1F1F1F", line_width=0.8)
    figure.add_annotation(
        text="<br>".join(notes), x=0.01, y=0.99, xref="x domain", yref="y domain", xanchor="left",
        yanchor="top", showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="#D1D5DB" if all_comparable else "#E45756",
    )
    figure.update_yaxes(title_text="累计节省（欧元）")
    figure.update_xaxes(title_text="交付时间")
    if not all_comparable:
        title += "<br><sup><span style='color:#E45756'>对照不可比</span></sup>"
    return _finish(figure, title, 540)


def _price(market: pd.DataFrame) -> tuple[pd.Series, str]:
    """根据市场的结算方式，返回该方式下存在的价格指标。"""

    if (market["pricing_method"] == "pay_as_bid").all():
        return market["average_trade_price_eur_per_mwh"], "全市场成交量加权均价"
    return market["clearing_price_eur_per_mwh"], "出清电价"


def _demand_by_type(tables: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """按市场产品（行）和需求类型，汇总 demand_results.csv 中的某一列。"""

    return (
        tables["demand"]
        .pivot_table(index="delivery_start", columns="demand_type", values=column, aggfunc="sum")
        .reindex(index=tables["market"]["delivery_start"], columns=DEMAND_TYPES)
        .fillna(0.0)
        .reset_index(drop=True)
    )


def _accepted_supply(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """跨所有供给报价类型，统计每个单元在每个产品上被接受的功率和电量。"""

    columns = ["delivery_start", "unit_name", "power_mw", "energy_mwh"]
    unit = tables["unit"]
    parts = [unit.rename(columns={
        "accepted_power_mw": "power_mw", "accepted_energy_mwh": "energy_mwh",
    })[columns]]
    if "storage" in tables:
        storage = tables["storage"]
        parts.append(
            storage.assign(
                power_mw=storage["accepted_discharge_mwh"] / storage["duration_hours"],
                energy_mwh=storage["accepted_discharge_mwh"],
            )[columns]
        )
    if "exchange" in tables:
        exchange = tables["exchange"]
        parts.append(
            exchange.assign(
                unit_name=exchange["exchange_name"],
                power_mw=exchange["cleared_import_power_mw"],
                energy_mwh=exchange["cleared_import_power_mw"] * exchange["duration_hours"],
            )[columns]
        )
    return pd.concat(parts, ignore_index=True)


def _line(x, y, name: str, color: str, *, shape: str = "hvh", dash: str | None = None, width: float = 1.5, **trace) -> go.Scatter:
    """创建一条折线轨迹；默认的 "hvh" 形状在中点处换阶，效果类似 matplotlib 的 where="mid"。"""

    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": color, "shape": shape, "dash": dash, "width": width},
        **trace,
    )


def _repeat_last(values: pd.Series) -> list[float]:
    """重复最后一个值，使阶梯线也能覆盖最后一个交付产品。"""

    values = values.tolist()
    return [*values, values[-1]]


def _annotate_empty(figure: go.Figure, message: str) -> None:
    figure.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, bgcolor="#F3F4F6"
    )


def _finish(figure: go.Figure, title: str, height: int, hovermode: str | None = None) -> go.Figure:
    if hovermode is None:
        hovermode = "x unified" if len(figure.data) <= MAX_UNIFIED_HOVER_TRACES else "closest"
    figure.update_layout(
        title=title, height=height, hovermode=hovermode, margin={"t": 80, "b": 40},
        font={"family": "Microsoft YaHei, SimHei, Noto Sans CJK SC, sans-serif"},
    )
    return figure
