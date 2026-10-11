"""Interactive Plotly charts rebuilt from the CSV files written by write_results().

Each chart mirrors the PNG renderer of the same name in
electricity_market_sim/plotting, but reads outputs/<dataset>/<scenario>/*.csv
instead of an in-memory SimulationResult.
"""

from __future__ import annotations

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
}
# write_results() always writes these; the others exist only when the run has such data.
REQUIRED_TABLES = ("market", "demand", "unit", "operator")
TIME_COLUMNS = ("opening_time", "delivery_start", "delivery_end", "datetime")
ACCEPTED_LAYERS = (
    ("inelastic_load", "#4C78A8", "Served inelastic load"),
    ("household_load", "#72B7B2", "Accepted household load"),
    ("industrial_load", "#D4A72C", "Accepted industrial load"),
    ("elastic_load", "#54A24B", "Accepted elastic demand"),
    ("storage_charge", "#E45756", "Accepted storage charge"),
    ("export", "#F58518", "Accepted export"),
)
UNACCEPTED_LINES = (
    ("inelastic_load", "#E45756", "Unserved inelastic load"),
    ("elastic_load", "#7F7F7F", "Unaccepted elastic demand"),
    ("household_load", "#72B7B2", "Unaccepted household load"),
    ("industrial_load", "#D4A72C", "Unaccepted industrial load"),
    ("storage_charge", "#B279A2", "Unaccepted storage charge"),
    ("export", "#F58518", "Unfulfilled export"),
)
DEMAND_TYPES = [demand_type for demand_type, _, _ in ACCEPTED_LAYERS]


def has_results(run_dir: Path) -> bool:
    return all((run_dir / TABLES[key]).is_file() for key in REQUIRED_TABLES)


def load_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the CSV tables of one run directory, parsing timestamp columns.

    Optional tables that are missing or have no rows are left out, so the charts
    depending on them are skipped.
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
    """Return every chart that applies to the run, in display priority order."""

    if tables["market"].empty:
        return []
    figures = [
        market_overview(tables),
        market_summary(tables),
        dispatch_by_unit(tables),
        operator_profit(tables),
    ]
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

    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=(1, 1.45, 0.8)
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
        _line(times, market["requested_demand_energy_mwh"], "All submitted demand", "#1F1F1F", width=1.1),
        row=2,
        col=1,
    )
    for demand_type, color, label in UNACCEPTED_LINES:
        # Unserved inelastic load is always shown; other rejected demand only if it occurs.
        if demand_type == "inelastic_load" or unaccepted[demand_type].abs().max() > ENERGY_TOLERANCE_MWH:
            figure.add_trace(_line(times, unaccepted[demand_type], label, color, width=1.2), row=3, col=1)
    figure.update_yaxes(title_text="Price (EUR/MWh)", row=1, col=1)
    figure.update_yaxes(title_text="Energy (MWh)", row=2, col=1)
    figure.update_yaxes(title_text="Unaccepted (MWh)", row=3, col=1)
    figure.update_xaxes(title_text="Delivery start", row=3, col=1)
    return _finish(figure, "Electricity market overview", 820)


def market_summary(tables: dict[str, pd.DataFrame]) -> go.Figure:
    market = tables["market"]
    times = market["delivery_start"]
    duration = market["duration_hours"]
    price, price_label = _price(market)
    cleared_demand_gw = _demand_by_type(tables, "accepted_energy_mwh").sum(axis=1) / duration / 1_000
    cleared_supply_gw = market["cleared_energy_mwh"] / duration / 1_000

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        _line(times, cleared_demand_gw, "Cleared demand volume", "#54A24B", dash="dash"), secondary_y=False
    )
    figure.add_trace(_line(times, price, price_label, "#D4A72C", width=1.15), secondary_y=True)
    figure.add_trace(
        _line(times, cleared_supply_gw, "Cleared supply volume", "#4C78A8", width=1.15), secondary_y=False
    )
    figure.update_yaxes(title_text="Cleared volume (GW)", secondary_y=False)
    figure.update_yaxes(title_text=f"{price_label} (EUR/MWh)", secondary_y=True, showgrid=False)
    figure.update_xaxes(title_text="Delivery start")
    statistics = " · ".join(
        f"{label}: {_min_max_mean(values, unit, decimals)}"
        for label, values, unit, decimals in (
            ("Cleared demand", cleared_demand_gw, "GW", 2),
            (price_label, price, "EUR/MWh", 1),
            ("Cleared supply", cleared_supply_gw, "GW", 2),
        )
    )
    return _finish(figure, f"Market summary EOM<br><sup>Min / Max / Mean — {statistics}</sup>", 540)


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
        series["Other accepted supply"] = power[others].sum(axis=1)

    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        color = "#9CA3AF" if name == "Other accepted supply" else UNIT_COLORS[index % len(UNIT_COLORS)]
        figure.add_trace(_line(times, values, name, color, width=0.5, stackgroup="supply"))
    figure.add_trace(
        _line(times, market["cleared_energy_mwh"] / market["duration_hours"], "Cleared demand power", "#1F1F1F", width=1.2)
    )
    if not series:
        _annotate_empty(figure, "No supply offers submitted")
    title = "Accepted dispatch by unit"
    if len(others):
        title += f" — {MAX_DISPATCH_UNITS} largest units and {len(others)} aggregated"
    figure.update_yaxes(title_text="Power (MW)")
    figure.update_xaxes(title_text="Delivery start")
    return _finish(figure, title, 560)


def operator_profit(tables: dict[str, pd.DataFrame]) -> go.Figure:
    profits = tables["operator"].groupby("unit_operator")["profit_eur"].sum()
    if "storage" in tables:
        storage_cash_flow = tables["storage"].groupby("unit_operator")["net_cash_flow_eur"].sum()
        profits = profits.add(storage_cash_flow, fill_value=0.0)
    profits = profits[profits.abs() > 1e-6]
    ranked = sorted(profits.index, key=lambda operator: (-abs(profits[operator]), operator))
    shown = profits[ranked[:MAX_OPERATORS]].to_dict()
    omitted = profits[ranked[MAX_OPERATORS:]]
    for sign, values in (("positive", omitted[omitted > 0]), ("negative", omitted[omitted < 0])):
        if len(values):
            shown[f"Other {sign} operators ({len(values)})"] = values.sum()
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
            hovertemplate="%{y}<br>%{x:,.0f} EUR<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#1F1F1F", line_width=0.8)
    if not operators:
        _annotate_empty(figure, "No non-zero generation or storage results")
    figure.update_xaxes(title_text="Profit / net cash flow (EUR)")
    return _finish(
        figure,
        "Cumulative generation and storage result by operator",
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
    figure.update_yaxes(title_text="SOC (%)", range=[0, 100], row=1, col=1)
    figure.update_yaxes(title_text="Net power (MW)<br>+ discharge / − charge", row=2, col=1)
    figure.update_xaxes(title_text="Delivery start", row=2, col=1)
    return _finish(figure, "Storage dispatch and state of charge", 720)


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
        band = {"shape": "hv", "width": 0.5, "legendgroup": f"{name} flexibility"}
        figure.add_trace(
            _line(step_x, _repeat_last(rows["minimum_power_mw"]), f"{name} minimum power", color, **band),
            row=1,
            col=1,
        )
        figure.add_trace(
            _line(
                step_x,
                _repeat_last(rows["maximum_power_mw"]),
                f"{name} maximum power",
                color,
                fill="tonexty",
                fillcolor=f"rgba({red}, {green}, {blue}, 0.16)",
                **band,
            ),
            row=1,
            col=1,
        )
        for kind, dash, width in (("planned", "dash", 1.5), ("actual", None, 1.8)):
            line = {"shape": "hv", "dash": dash, "width": width}
            figure.add_trace(
                _line(step_x, _repeat_last(rows[f"{kind}_grid_power_mw"]), f"{name} {kind} power", color, **line),
                row=1,
                col=1,
            )
            figure.add_trace(
                _line(step_x, _repeat_last(rows[f"{kind}_steel_output_t"]), f"{name} {kind} steel", color, **line),
                row=2,
                col=1,
            )
            figure.add_trace(
                _line(
                    cumulative_x,
                    [0.0, *rows[f"{kind}_steel_output_t"].cumsum()],
                    f"{name} cumulative {kind} steel",
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
    title = "Industrial plan, execution, and flexibility"
    if boundaries:
        title += "<br><sup>Dotted vertical lines: rolling window boundaries</sup>"
    figure.update_yaxes(title_text="Grid power (MW)", row=1, col=1)
    figure.update_yaxes(title_text="Steel per product (t)", row=2, col=1)
    figure.update_yaxes(title_text="Cumulative steel (t)", row=3, col=1)
    figure.update_xaxes(title_text="Delivery start", row=3, col=1)
    return _finish(figure, title, 820)


def pay_as_bid_order_book(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """Supply order book of the first product; pay-as-bid markets only have plant offers."""

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
    figure.add_trace(_segments(offered_start[traded], accepted_end[traded], book[traded], "Traded offer energy", "#54A24B", 7))
    figure.add_trace(_segments(accepted_end[untraded], offered_end[untraded], book[untraded], "Untraded offer energy", "#B8B8B8", 4))
    for energy, label, color, dash in (
        (market["requested_demand_energy_mwh"], "Submitted demand energy", "#7F7F7F", "dash"),
        (market["cleared_energy_mwh"], "Traded energy", "#1F1F1F", "dot"),
    ):
        figure.add_trace(_line([energy, energy], price_range, label, color, shape="linear", dash=dash, width=1.2))
    if pd.notna(average_price):
        figure.add_trace(
            _line([0.0, energy_end], [average_price] * 2, "Average trade price", "#2F5597", shape="linear", dash="dashdot", width=1.4)
        )
    if book.empty:
        _annotate_empty(figure, "No supply offers submitted")
    figure.update_xaxes(title_text="Cumulative offered energy (MWh)", range=[0, energy_end * 1.03 or 1.0])
    figure.update_yaxes(title_text="Offer / average trade price (EUR/MWh)")
    return _finish(
        figure,
        f"Pay-as-bid supply offers and trades — {start:%Y-%m-%d %H:%M}",
        520,
        hovermode="closest",
    )


def _price(market: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return the price measure that exists for the market's settlement method."""

    if (market["pricing_method"] == "pay_as_bid").all():
        return market["average_trade_price_eur_per_mwh"], "Average trade price"
    return market["clearing_price_eur_per_mwh"], "Clearing price"


def _demand_by_type(tables: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """Sum one demand_results.csv column per market product (row) and demand type."""

    return (
        tables["demand"]
        .pivot_table(index="delivery_start", columns="demand_type", values=column, aggfunc="sum")
        .reindex(index=tables["market"]["delivery_start"], columns=DEMAND_TYPES)
        .fillna(0.0)
        .reset_index(drop=True)
    )


def _accepted_supply(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Accepted power and energy per unit and product across all supply offer types."""

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
    """A line trace; the default "hvh" shape steps at midpoints like matplotlib's where="mid"."""

    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": color, "shape": shape, "dash": dash, "width": width},
        **trace,
    )


def _segments(starts, ends, book: pd.DataFrame, name: str, color: str, width: float) -> go.Scatter:
    """Horizontal order-book segments, with a midpoint so hovering anywhere shows the offer."""

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
        hovertemplate="%{text}<br>%{y:.2f} EUR/MWh<br>%{x:,.1f} MWh<extra></extra>",
    )


def _repeat_last(values: pd.Series) -> list[float]:
    """Repeat the final value so a step line also covers the last delivery product."""

    values = values.tolist()
    return [*values, values[-1]]


def _min_max_mean(values: pd.Series, unit: str, decimals: int) -> str:
    finite = values.dropna()
    if finite.empty:
        return "N/A"
    return " / ".join(f"{value:.{decimals}f}" for value in (finite.min(), finite.max(), finite.mean())) + f" {unit}"


def _annotate_empty(figure: go.Figure, message: str) -> None:
    figure.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, bgcolor="#F3F4F6"
    )


def _finish(figure: go.Figure, title: str, height: int, hovermode: str | None = None) -> go.Figure:
    if hovermode is None:
        hovermode = "x unified" if len(figure.data) <= MAX_UNIFIED_HOVER_TRACES else "closest"
    figure.update_layout(title=title, height=height, hovermode=hovermode, margin={"t": 80, "b": 40})
    return figure
