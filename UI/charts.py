"""从仿真结果 CSV 文件构建交互式 Plotly 图表。
展示市场供需、成交价格与出力、运营商利润，以及可用的储能、工业和按报价结算订单簿数据。"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative, sample_colorscale
from plotly.subplots import make_subplots

ENERGY_TOLERANCE_MWH = 1e-7
MAX_DISPATCH_UNITS = 10
MAX_OPERATORS = 20
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
    "offer": "offer_results.csv",
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
    ("storage_charge", "#E45756", "已成交的储能充电需求"),
    ("export", "#F58518", "已成交的外送需求"),
)
UNACCEPTED_LINES = (
    ("inelastic_load", "#E45756", "未满足的刚性负荷"),
    ("elastic_load", "#7F7F7F", "未成交的弹性需求"),
    ("household_load", "#72B7B2", "未成交的居民用电需求"),
    ("industrial_load", "#D4A72C", "未成交的工业用电需求"),
    ("storage_charge", "#B279A2", "未成交的储能充电需求"),
    ("export", "#F58518", "未满足的外送需求"),
)
DEMAND_TYPES = [demand_type for demand_type, _, _ in ACCEPTED_LAYERS]


def has_results(run_dir: Path) -> bool:
    return all((run_dir / TABLES[key]).is_file() for key in REQUIRED_TABLES)


def load_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    """读取一次运行目录下的各个 CSV 表，并解析时间列。

    缺失或没有数据行的可选表会被略过，依赖这些表的图表也随之跳过。
    """

    tables = {}
    for key, filename in TABLES.items():
        path = run_dir / filename
        if not path.is_file():
            continue
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
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
    if "storage" in tables:
        figures.append(storage_dispatch(tables))
    if "industry" in tables:
        figures.append(industry_dispatch(tables))
    if (tables["market"]["pricing_method"] == "pay_as_bid").all():
        figures.append(pay_as_bid_order_book(tables))
    return figures


def market_overview(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    price, price_label = _price(market)
    accepted = _demand_by_type(tables, "accepted_energy_mwh")
    unaccepted = _demand_by_type(tables, "unaccepted_energy_mwh")
    demand_types = set(tables["demand"]["demand_type"])
    unaccepted_panels = [
        (demand_type, color, label)
        for demand_type, color, label in UNACCEPTED_LINES
        if demand_type in demand_types
    ]
    row_count = 2 + len(unaccepted_panels)

    figure = make_subplots(
        rows=row_count, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.04, 0.25 / (row_count - 1)),
        row_heights=[1, 1.45, *([0.8] * len(unaccepted_panels))],
        subplot_titles=[price_label, "成交需求与总申报需求", *[label for _, _, label in unaccepted_panels]],
    )
    figure.add_trace(_line(times, price, price_label, "#2F5597"), row=1, col=1)
    for demand_type, color, label in ACCEPTED_LAYERS:
        if accepted[demand_type].abs().max() > ENERGY_TOLERANCE_MWH:
            figure.add_trace(
                _line(times, accepted[demand_type], label, color, width=0.5, stackgroup="accepted"),
                row=2,
                col=1,
            )
    figure.add_trace(
        _line(times, market["requested_demand_energy_mwh"], "总申报需求", "#1F1F1F", width=1.1),
        row=2,
        col=1,
    )
    for row, (demand_type, color, label) in enumerate(unaccepted_panels, start=3):
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
    figure.update_xaxes(title_text="交付开始时间", row=row_count, col=1)
    figure.update_layout(font={"family": "Microsoft YaHei, SimHei, Noto Sans CJK SC, sans-serif"})
    return _finish(figure, "电力市场概览", 600 + 220 * len(unaccepted_panels))


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
    figure.update_layout(font={"family": "Microsoft YaHei, SimHei, Noto Sans CJK SC, sans-serif"})
    return _finish(figure, "电能量市场出清汇总", 540)


def dispatch_by_unit(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    supply = _accepted_supply(tables)
    energy = supply.groupby("unit_name")["energy_mwh"].sum().sort_index()
    ranked = energy[energy > ENERGY_TOLERANCE_MWH].sort_values(ascending=False, kind="stable").index
    power = (
        supply.pivot_table(index="delivery_start", columns="unit_name", values="power_mw", aggfunc="sum")
        .reindex(times)
        .fillna(0.0)
        .reset_index(drop=True)
    )
    series = {name: power[name] for name in ranked[:MAX_DISPATCH_UNITS]}
    others = ranked[MAX_DISPATCH_UNITS:]
    if len(others):
        series["其他成交供给"] = power[others].sum(axis=1)

    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        color = "#9CA3AF" if name == "其他成交供给" else UNIT_COLORS[index % len(UNIT_COLORS)]
        figure.add_trace(_line(times, values, name, color, width=0.5, stackgroup="supply"))
    figure.add_trace(
        _line(times, market["cleared_energy_mwh"] / market["duration_hours"], "需求侧成交功率", "#1F1F1F", width=1.2)
    )
    if not series:
        _annotate_empty(figure, "暂无供给报价")
    title = "各机组成交出力"
    if len(others):
        title += f"——展示成交电量最多的 {MAX_DISPATCH_UNITS} 个单元，其余 {len(others)} 个合并"
    figure.update_yaxes(title_text="功率（MW）")
    figure.update_xaxes(title_text="交付开始时间")
    return _finish(figure, title, 560)


def operator_profit(tables: dict[str, pd.DataFrame]) -> go.Figure:
    metric_label = "利润／净现金流" if "storage" in tables else "利润"
    title = "各运营商累计发电利润与储能净现金流" if "storage" in tables else "各运营商累计利润"
    profits = tables["operator"].groupby("unit_operator")["profit_eur"].sum()
    if "storage" in tables:
        storage_cash_flow = tables["storage"].groupby("unit_operator")["net_cash_flow_eur"].sum()
        profits = profits.add(storage_cash_flow, fill_value=0.0)
    profits = profits[profits.abs() > 1e-6]
    ranked = sorted(profits.index, key=lambda operator: (-abs(profits[operator]), operator))
    shown = profits[ranked[:MAX_OPERATORS]].to_dict()
    omitted = profits[ranked[MAX_OPERATORS:]]
    for sign, values in (("盈利", omitted[omitted > 0]), ("亏损", omitted[omitted < 0])):
        if len(values):
            shown[f"其他{sign}运营商（{len(values)} 家）"] = values.sum()
    operators = sorted(shown, key=lambda operator: (shown[operator], operator))
    values = [shown[operator] for operator in operators]

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
        max(360, 30 * len(operators) + 140),
        hovermode="closest",
    )


def storage_dispatch(tables: dict[str, pd.DataFrame]) -> go.Figure:
    storage = tables["storage"].sort_values(["unit_name", "delivery_start", "opening_time"], kind="stable")
    names = sorted(storage["unit_name"].unique())
    colors = sample_colorscale("Turbo", [index / max(1, len(names) - 1) for index in range(len(names))])

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=(1.1, 1))
    for name, color in zip(names, colors):
        soc_x, soc_y, power_x, power_y = [], [], [], []
        previous_end = previous_power = None
        for row in storage[storage["unit_name"] == name].itertuples(index=False):
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
        figure.add_trace(_line(soc_x, soc_y, name, color, shape="linear", legendgroup=name), row=1, col=1)
        figure.add_trace(
            _line(power_x, power_y, name, color, shape="hv", width=1.25, legendgroup=name, showlegend=False),
            row=2,
            col=1,
        )
    figure.add_hline(y=0, line_color="#1F1F1F", line_width=0.8, row=2, col=1)
    figure.update_yaxes(title_text="荷电状态（%）", range=[0, 100], row=1, col=1)
    figure.update_yaxes(title_text="净功率（MW）<br>正值放电／负值充电", row=2, col=1)
    figure.update_xaxes(title_text="交付开始时间", row=2, col=1)
    return _finish(figure, "储能调度与荷电状态", 720)


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
        for kind, dash, width in (("planned", "dash", 1.5), ("actual", None, 1.8)):
            kind_label = {"planned": "计划", "actual": "实际"}[kind]
            line = {"shape": "hv", "dash": dash, "width": width}
            figure.add_trace(
                _line(step_x, _repeat_last(rows[f"{kind}_grid_power_mw"]), f"{name} {kind_label}功率", color, **line),
                row=1,
                col=1,
            )
            figure.add_trace(
                _line(step_x, _repeat_last(rows[f"{kind}_steel_output_t"]), f"{name} {kind_label}钢产量", color, **line),
                row=2,
                col=1,
            )
            figure.add_trace(
                _line(
                    cumulative_x,
                    [0.0, *rows[f"{kind}_steel_output_t"].cumsum()],
                    f"{name} 累计{kind_label}钢产量",
                    color,
                    **line,
                ),
                row=3,
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


def pay_as_bid_order_book(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """第一个产品的供给订单簿；按报价结算市场中只有发电机组的报价。"""

    market = tables["market"].iloc[0]
    start = market["delivery_start"]
    columns = ["unit_name", "bid_price_eur_per_mwh", "offered_energy_mwh", "accepted_energy_mwh"]
    units = tables["unit"][tables["unit"]["delivery_start"] == start]
    # unit_results.csv merges split offers at a volume-weighted price; use the
    # individual orders from offer_results.csv wherever they were written.
    offers = tables.get("offer", units.iloc[0:0])
    detailed = offers[offers["delivery_start"] == start]
    book = pd.concat(
        [detailed[columns], units[~units["unit_name"].isin(detailed["unit_name"])][columns]]
    ).sort_values(["bid_price_eur_per_mwh", "unit_name"], kind="stable")
    offered_end = book["offered_energy_mwh"].cumsum()
    offered_start = offered_end - book["offered_energy_mwh"]
    accepted_end = offered_start + book["accepted_energy_mwh"]
    traded = book["accepted_energy_mwh"] > ENERGY_TOLERANCE_MWH
    untraded = accepted_end < offered_end - ENERGY_TOLERANCE_MWH

    average_price = market["average_trade_price_eur_per_mwh"]
    prices = [*book["bid_price_eur_per_mwh"], *([average_price] if pd.notna(average_price) else [])]
    padding = max([abs(price) for price in prices] + [1.0]) * 0.08
    price_range = [min(prices, default=0.0) - padding, max(prices, default=0.0) + padding]
    energy_end = max(offered_end.max() if len(book) else 0.0, market["requested_demand_energy_mwh"], market["cleared_energy_mwh"])

    figure = go.Figure()
    figure.add_trace(_segments(offered_start[traded], accepted_end[traded], book[traded], "已成交供给电量", "#54A24B", 7))
    figure.add_trace(_segments(accepted_end[untraded], offered_end[untraded], book[untraded], "未成交供给电量", "#B8B8B8", 4))
    for energy, label, color, dash in (
        (market["requested_demand_energy_mwh"], "申报需求电量", "#7F7F7F", "dash"),
        (market["cleared_energy_mwh"], "成交电量", "#1F1F1F", "dot"),
    ):
        figure.add_trace(_line([energy, energy], price_range, label, color, shape="linear", dash=dash, width=1.2))
    if pd.notna(average_price):
        figure.add_trace(
            _line([0.0, energy_end], [average_price] * 2, "平均成交电价", "#2F5597", shape="linear", dash="dashdot", width=1.4)
        )
    if book.empty:
        _annotate_empty(figure, "暂无供给报价")
    figure.update_xaxes(title_text="累计申报供给电量（MWh）", range=[0, energy_end * 1.03 or 1.0])
    figure.update_yaxes(title_text="报价／平均成交电价（欧元/MWh）")
    return _finish(
        figure,
        f"按报价结算的供给订单与成交——{start:%Y-%m-%d %H:%M}",
        520,
        hovermode="closest",
    )


def _price(market: pd.DataFrame) -> tuple[pd.Series, str]:
    """根据市场的结算方式，返回该方式下存在的价格指标。"""

    if (market["pricing_method"] == "pay_as_bid").all():
        return market["average_trade_price_eur_per_mwh"], "平均成交电价"
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
    parts = [unit.assign(power_mw=unit["accepted_power_mw"], energy_mwh=unit["accepted_energy_mwh"])[columns]]
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


def _segments(starts, ends, book: pd.DataFrame, name: str, color: str, width: float) -> go.Scatter:
    """绘制订单簿的水平线段，并为每段加入中点，使悬停在线段任意位置都能显示该报价。"""

    x, y, text = [], [], []
    for start, end, price, unit in zip(starts, ends, book["bid_price_eur_per_mwh"], book["unit_name"]):
        x += [start, (start + end) / 2, end, None]
        y += [price, price, price, None]
        text += [unit, unit, unit, None]
    return go.Scatter(
        x=x,
        y=y,
        text=text,
        name=name,
        mode="lines",
        line={"color": color, "width": width},
        hovertemplate="%{text}<br>报价：%{y:.2f} 欧元/MWh<br>累计电量：%{x:,.1f} MWh<extra></extra>",
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
