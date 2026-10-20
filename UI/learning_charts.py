"""根据学习结果构建交互图，并兼容旧输出中的 PNG 图。"""

from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATA_FILES = (
    "learning_results.csv",
    "learning_metrics.csv",
    "learning_test_metrics.csv",
    "learning_evaluation_metrics.csv",
)
CHART_NAMES = (
    "learning_unit",
    "learning_curves",
    "learning_train_comparison",
    "learning_validation_comparison",
    "learning_test_comparison",
)
PERIODS = ("train", "validation", "test")
SEGMENTS = ("learning_minimum", "learning_flexible")
ACCEPTED_TOLERANCE_MW = 1e-7
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")
CURVE_FIELDS = (
    ("mean_reward_per_hour", "Mean reward per hour"),
    ("total_reward", "Total reward"),
    ("total_profit_eur", "Total profit (EUR)"),
)
COMPARISON_FIELDS = (
    ("total_profit_eur", "Total profit (EUR)"),
    ("total_reward", "Total reward"),
    ("mean_reward_per_hour", "Mean reward per hour"),
    ("accepted_energy_mwh", "Accepted energy (MWh)"),
    ("minimum_segment_average_bid_eur_per_mwh", "Minimum-segment avg bid (EUR/MWh)"),
    ("flexible_segment_average_bid_eur_per_mwh", "Flexible-segment avg bid (EUR/MWh)"),
    ("minimum_segment_acceptance_ratio", "Minimum-segment acceptance ratio"),
    ("flexible_segment_acceptance_ratio", "Flexible-segment acceptance ratio"),
)
METRIC_FIELDS = {name for name, _ in (*CURVE_FIELDS, *COMPARISON_FIELDS)}
UNIT_FIELDS = {
    "delivery_start", "available_power_mw", "clearing_price_eur_per_mwh",
    "profit_eur", "regret_eur", "reward",
}
OFFER_FIELDS = {
    "delivery_start", "offer_segment", "bid_price_eur_per_mwh",
    "marginal_cost_eur_per_mwh", "accepted_power_mw",
}


class LearningChartError(ValueError):
    """学习图数据校验失败时展示给用户的错误。"""


def _error_text(exc: Exception) -> str:
    if isinstance(exc, LearningChartError):
        return str(exc)
    return "读取或绘制失败，请检查结果 CSV 的格式和数值。"


def _read_csv(path: Path, fields: set[str], *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise LearningChartError(f"{path.name} 文件为空或没有表头。") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise LearningChartError(f"无法读取 {path.name}，请检查 CSV 文件格式。") from exc
    missing = fields - set(table.columns)
    if missing:
        raise LearningChartError(f"{path.name} 缺少列：{', '.join(sorted(missing))}。")
    if table.empty:
        raise LearningChartError(f"{path.name} 没有数据行。")
    for field in dates:
        try:
            table[field] = pd.to_datetime(table[field], errors="raise")
        except (TypeError, ValueError) as exc:
            raise LearningChartError(f"{path.name} 的 {field} 列包含无效时间。") from exc
    for field in fields - set(dates) - {"phase", "offer_segment"}:
        try:
            table[field] = pd.to_numeric(table[field], errors="raise")
        except (TypeError, ValueError) as exc:
            raise LearningChartError(f"{path.name} 的 {field} 列包含无效数字。") from exc
    return table


def _step(x, y, name: str, color: str, *, dash: str | None = None, **kwargs) -> go.Scatter:
    line = {"color": color, "shape": "hvh", "width": 1.2}
    if dash is not None:
        line["dash"] = dash
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=line, **kwargs,
    )


def learning_unit(results: pd.DataFrame, offers: pd.DataFrame) -> go.Figure:
    """根据交付结果和报价记录绘制学习机组图。"""

    results = results.sort_values("delivery_start")
    if results["delivery_start"].duplicated().any():
        raise LearningChartError("learning_results.csv 的 delivery_start 列存在重复时间。")
    offers = offers[offers["offer_segment"].isin(SEGMENTS)]
    if offers.duplicated(["delivery_start", "offer_segment"]).any():
        raise LearningChartError("offer_results.csv 存在重复的学习报价分段记录。")
    segments = {
        name: offers[offers["offer_segment"] == name].set_index("delivery_start")
        .reindex(results["delivery_start"])
        for name in SEGMENTS
    }
    times = results["delivery_start"]
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=(1.25, 1, 1), specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    figure.add_trace(_step(times, results["clearing_price_eur_per_mwh"], "Clearing price", "#1F1F1F"), row=1, col=1)
    for name, label, color in (
        (SEGMENTS[0], "Minimum-segment bid", COLORS[0]),
        (SEGMENTS[1], "Flexible-segment bid", COLORS[1]),
    ):
        figure.add_trace(_step(times, segments[name]["bid_price_eur_per_mwh"], label, color), row=1, col=1)
    marginal_cost = segments[SEGMENTS[0]]["marginal_cost_eur_per_mwh"].combine_first(
        segments[SEGMENTS[1]]["marginal_cost_eur_per_mwh"]
    )
    figure.add_trace(_step(times, marginal_cost, "Marginal cost", COLORS[3], dash="dash"), row=1, col=1)
    accepted = {
        name: segments[name]["accepted_power_mw"].fillna(0.0) for name in SEGMENTS
    }
    for name, label, color in (
        (SEGMENTS[0], "Accepted minimum segment", COLORS[0]),
        (SEGMENTS[1], "Accepted flexible segment", COLORS[1]),
    ):
        figure.add_trace(
            _step(times, accepted[name], label, color,
                  stackgroup="accepted"), row=2, col=1,
        )
    figure.add_trace(_step(times, results["available_power_mw"], "Available power", "#1F1F1F"), row=2, col=1)
    if (accepted[SEGMENTS[0]] + accepted[SEGMENTS[1]]).max() <= ACCEPTED_TOLERANCE_MW:
        figure.add_annotation(
            x=0.5, y=0.5, xref="x2 domain", yref="y2 domain",
            text="No accepted output", showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.85)",
        )
    figure.add_trace(_step(times, results["profit_eur"], "Profit", COLORS[2]), row=3, col=1)
    figure.add_trace(_step(times, -results["regret_eur"], "Regret penalty (negative)", COLORS[5]), row=3, col=1)
    figure.add_trace(_step(times, results["reward"], "Reward", "#1F1F1F"), row=3, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Price (EUR/MWh)", row=1, col=1)
    figure.update_yaxes(title_text="Power (MW)", rangemode="tozero", row=2, col=1)
    figure.update_yaxes(title_text="EUR per delivery", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Normalised reward", row=3, col=1, secondary_y=True)
    figure.update_xaxes(title_text="Delivery start", row=3, col=1)
    start, end = times.iloc[0], times.iloc[-1]
    total_reward = results["reward"].sum()
    total_profit = results["profit_eur"].sum()
    figure.update_layout(
        title=(
            "Learning unit: bids, dispatch and reward"
            f"<br><sup>{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"
            f" · total reward {total_reward:.3f}"
            f" · total profit {total_profit:,.0f} EUR</sup>"
        ),
        height=900, hovermode="x unified", margin={"t": 95, "b": 40},
    )
    return figure


def learning_curves(metrics: pd.DataFrame) -> go.Figure:
    """绘制初始经验、训练和验证曲线，并标出最佳验证回合。"""

    metrics = metrics[metrics["phase"].isin(("initial_experience", "train", "validation"))]
    if metrics.empty:
        raise LearningChartError("learning_metrics.csv 没有训练或验证记录。")
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    initial_episodes = metrics.loc[metrics["phase"] == "initial_experience", "episode"]
    if not initial_episodes.empty:
        for row in range(1, 4):
            figure.add_vrect(
                x0=initial_episodes.min() - 0.5, x1=initial_episodes.max() + 0.5,
                fillcolor="#E5E7EB", opacity=0.7, line_width=0, row=row, col=1,
            )
    for index, (run, run_rows) in enumerate(metrics.groupby("run", sort=True)):
        color = COLORS[index % len(COLORS)]
        run_rows = run_rows.sort_values("episode")
        initial = run_rows[run_rows["phase"] == "initial_experience"]
        training = run_rows[run_rows["phase"] == "train"]
        validation = run_rows[run_rows["phase"] == "validation"]
        best = validation.loc[validation["total_reward"].idxmax()] if not validation.empty else None
        for row, (field, label) in enumerate(CURVE_FIELDS, start=1):
            for phase_rows, phase, dash, mode in (
                (initial, "initial experience", "dash", "lines"),
                (pd.concat([initial.tail(1), training]) if not training.empty else training,
                 "training", "solid", "lines"),
                (validation, "validation", "solid", "lines+markers"),
            ):
                if phase_rows.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=phase_rows["episode"], y=phase_rows[field],
                        name=f"Run {run} {phase}", mode=mode,
                        line={"color": color, "dash": dash, "width": 1.5},
                        marker={"size": 5}, legendgroup=f"{run}-{phase}",
                        showlegend=row == 1,
                    ), row=row, col=1,
                )
            if best is not None:
                figure.add_trace(
                    go.Scatter(
                        x=[best["episode"]], y=[best[field]], mode="markers",
                        name=f"Run {run} best validation", marker={"symbol": "star", "size": 13, "color": color,
                        "line": {"color": "#1F1F1F", "width": 1}},
                        legendgroup=f"{run}-best", showlegend=row == 1,
                    ), row=row, col=1,
                )
            figure.update_yaxes(title_text=label, row=row, col=1)
    figure.update_xaxes(title_text="Episode", tickformat="d", row=3, col=1)
    figure.update_layout(
        title="Learning curves — dashed: initial experience; star: best validation reward",
        height=850, hovermode="x unified", margin={"t": 85, "b": 40},
    )
    return figure


def learning_comparison(metrics: pd.DataFrame, period: str) -> go.Figure:
    """沿用 PNG 图的八项指标，对比 Actor 和基准策略。"""

    phases = ((f"{period}_actor", "Actor", COLORS[0]),
              (f"{period}_baseline", "Baseline", "#9CA3AF"))
    if any(metrics.loc[metrics["phase"] == phase].empty for phase, _, _ in phases):
        period_label = {"train": "训练期", "validation": "验证期", "test": "测试期"}.get(period, period)
        raise LearningChartError(f"{period_label}对比图缺少 Actor 或基准策略记录。")
    figure = make_subplots(
        rows=2, cols=4, subplot_titles=[label for _, label in COMPARISON_FIELDS],
        horizontal_spacing=0.07, vertical_spacing=0.16,
    )
    for index, (field, _) in enumerate(COMPARISON_FIELDS):
        row, col = divmod(index, 4)
        for phase, label, color in phases:
            rows = metrics[metrics["phase"] == phase]
            values = [float(value) for value in rows[field]]
            mean = fmean(values)
            spread = pstdev(values)
            figure.add_trace(
                go.Bar(
                    x=[label], y=[mean], name=label, marker_color=color,
                    text=[f"{mean:,.0f}" if abs(mean) >= 1_000 else f"{mean:.3g}"],
                    textposition="outside", cliponaxis=False,
                    error_y={"type": "data", "array": [spread], "visible": True} if len(values) > 1 else None,
                    customdata=[[len(values), spread]],
                    hovertemplate="%{x}<br>Mean: %{y:,.4g}<br>Runs: %{customdata[0]}<br>Std: %{customdata[1]:,.4g}<extra></extra>",
                    legendgroup=label, showlegend=index == 0,
                ), row=row + 1, col=col + 1,
            )
            if len(values) > 1:
                figure.add_trace(
                    go.Scatter(
                        x=[label] * len(values), y=values, mode="markers",
                        text=[f"Run {run}" for run in rows["run"]],
                        marker={"size": 6, "color": "#1F1F1F"},
                        hovertemplate="%{text}<br>%{y:,.4g}<extra></extra>",
                        showlegend=False,
                    ), row=row + 1, col=col + 1,
                )
        figure.add_hline(y=0, line_color="#1F1F1F", line_width=0.6, row=row + 1, col=col + 1)
    figure.update_layout(
        title=f"Deterministic Actor vs marginal-cost baseline — {period} period",
        height=780, barmode="group", hovermode="closest", margin={"t": 90, "b": 40},
    )
    return figure


def build_learning_charts(run_dir: Path) -> tuple[dict[str, go.Figure], dict[str, str]]:
    """分别构建学习图，使单张图的错误不影响其他图。"""

    figures: dict[str, go.Figure] = {}
    errors: dict[str, str] = {}
    results_path = run_dir / "learning_results.csv"
    if not results_path.is_file():
        return figures, errors
    offer_path = run_dir / "offer_results.csv"
    if offer_path.is_file():
        try:
            results = _read_csv(results_path, UNIT_FIELDS, dates=("delivery_start",))
            offers = _read_csv(offer_path, OFFER_FIELDS, dates=("delivery_start",))
            figures["learning_unit"] = learning_unit(results, offers)
        except Exception as exc:
            errors["learning_unit"] = _error_text(exc)

    metric_tables: dict[str, pd.DataFrame] = {}
    metric_errors: dict[str, str] = {}
    for filename in DATA_FILES[1:]:
        path = run_dir / filename
        if not path.is_file():
            continue
        try:
            metric_tables[filename] = _read_csv(
                path, {"run", "phase", "episode", *METRIC_FIELDS}
            )
        except Exception as exc:
            metric_errors[filename] = _error_text(exc)

    training = metric_tables.get("learning_metrics.csv")
    if training is not None:
        if training["phase"].isin(("initial_experience", "train", "validation")).any():
            try:
                figures["learning_curves"] = learning_curves(training)
            except Exception as exc:
                errors["learning_curves"] = _error_text(exc)
    elif "learning_metrics.csv" in metric_errors:
        errors["learning_curves"] = metric_errors["learning_metrics.csv"]
    if "learning_evaluation_metrics.csv" in metric_errors:
        # 评估文件损坏时无法判断其对应训练期还是验证期，因此单独报告文件错误。
        errors["learning_evaluation_metrics.csv"] = metric_errors["learning_evaluation_metrics.csv"]

    for period in PERIODS:
        name = f"learning_{period}_comparison"
        candidates = [
            (run_dir / filename, table)
            for filename, table in metric_tables.items()
            if {f"{period}_actor", f"{period}_baseline"}.issubset(set(table["phase"]))
        ]
        if candidates:
            path, table = max(candidates, key=lambda item: (item[0].stat().st_mtime_ns, item[0].name))
            try:
                figure = learning_comparison(table, period)
                figure.update_layout(title=f"{figure.layout.title.text}<br><sup>Source: {path.name}</sup>")
                figures[name] = figure
            except Exception as exc:
                errors[name] = _error_text(exc)
        else:
            likely_file = {
                "train": "learning_evaluation_metrics.csv",
                "validation": "learning_metrics.csv",
                "test": "learning_test_metrics.csv",
            }[period]
            if likely_file in metric_errors:
                errors[name] = metric_errors[likely_file]
    return figures, errors


def learning_items(
    run_dir: Path, figures: dict[str, go.Figure], errors: dict[str, str]
) -> list[tuple[str, str, go.Figure | Path | str]]:
    """为每个学习主题选择交互图、错误提示或当前 PNG。"""

    current = (run_dir / "learning_results.csv").is_file()
    newest_result = max(
        (path.stat().st_mtime_ns for path in
         (run_dir / name for name in ("market_results.csv", "demand_results.csv", "unit_results.csv", "operator_results.csv"))
         if path.is_file()),
        default=0,
    )
    items = []
    for name in CHART_NAMES:
        png = run_dir / "plots" / f"{name}.png"
        if current and name in figures:
            items.append(("figure", name, figures[name]))
        elif current and name in errors:
            items.append(("error", name, errors[name]))
        elif png.is_file() and (current or png.stat().st_mtime_ns >= newest_result):
            items.append(("image", name, png))
    return items
