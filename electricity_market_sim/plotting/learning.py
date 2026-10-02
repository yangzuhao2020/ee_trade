"""Version 5 learning-unit, training-curve, and Actor/baseline visualisations."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from ..errors import PlottingError
from ..learning_metrics import LEARNING_METRIC_FIELDS
from ..market_models import MarketClearingResult
from ..models import LearningStepResult
from .common import (
    _ENERGY_TOLERANCE_MWH,
    _UNIT_COLORS,
    _annotate_no_supply,
    _format_time_axis,
    _save_figure,
)

_LEARNING_SEGMENTS = ("learning_minimum", "learning_flexible")
_CURVE_FIELDS = (
    ("total_reward", "Total reward"),
    ("discounted_reward", "Discounted reward"),
    ("total_profit_eur", "Total profit (EUR)"),
)
_COMPARISON_FIELDS = (
    ("total_profit_eur", "Total profit (EUR)"),
    ("total_reward", "Total reward"),
    ("discounted_reward", "Discounted reward"),
    ("accepted_energy_mwh", "Accepted energy (MWh)"),
    ("minimum_segment_average_bid_eur_per_mwh", "Minimum-segment avg bid (EUR/MWh)"),
    ("flexible_segment_average_bid_eur_per_mwh", "Flexible-segment avg bid (EUR/MWh)"),
    ("minimum_segment_acceptance_ratio", "Minimum-segment acceptance ratio"),
    ("flexible_segment_acceptance_ratio", "Flexible-segment acceptance ratio"),
)
_COMPARISON_PHASES = (
    ("evaluation", "Actor", "#4C78A8"),
    ("baseline", "Baseline", "#9CA3AF"),
)


def _plot_learning_unit(
    pyplot,
    dates,
    results: tuple[MarketClearingResult, ...],
    learning_steps: tuple[LearningStepResult, ...],
    path: Path,
    episode_label: str,
) -> Path:
    """Plot the learning unit's two bids, dispatch, and reward per delivery."""

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

    clearing_prices = [
        (
            markets[time].clearing_price_eur_per_mwh
            if time in markets and markets[time].clearing_price_eur_per_mwh is not None
            else float("nan")
        )
        for time in times
    ]
    marginal_costs = [
        next(
            (
                cleared.offer.marginal_cost_eur_per_mwh
                for cleared in segments_by_start.get(time, {}).values()
            ),
            float("nan"),
        )
        for time in times
    ]
    accepted = {
        segment: segment_series(segment, lambda item: item.accepted_power_mw, 0.0)
        for segment in _LEARNING_SEGMENTS
    }

    figure, (price_axis, power_axis, reward_axis) = pyplot.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1, 1)},
    )
    price_axis.step(
        times,
        clearing_prices,
        where="mid",
        color="#1F1F1F",
        linewidth=1.3,
        label="Clearing price",
    )
    for segment, label, color in (
        ("learning_minimum", "Minimum-segment bid", "#4C78A8"),
        ("learning_flexible", "Flexible-segment bid", "#F58518"),
    ):
        price_axis.step(
            times,
            segment_series(segment, lambda item: item.offer.bid_price_eur_per_mwh),
            where="mid",
            color=color,
            linewidth=1.1,
            label=label,
        )
    price_axis.step(
        times,
        marginal_costs,
        where="mid",
        color="#E45756",
        linestyle="--",
        linewidth=1.0,
        label="Marginal cost",
    )
    price_axis.set_ylabel("Price (EUR/MWh)")
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper left", ncol=4, fontsize=8)

    power_axis.stackplot(
        times,
        accepted["learning_minimum"],
        accepted["learning_flexible"],
        labels=("Accepted minimum segment", "Accepted flexible segment"),
        colors=("#4C78A8", "#F58518"),
        alpha=0.6,
        step="mid",
    )
    available_power = [step.available_power_mw for step in steps]
    accepted_power = [
        minimum + flexible
        for minimum, flexible in zip(
            accepted["learning_minimum"], accepted["learning_flexible"]
        )
    ]
    power_axis.step(
        times,
        available_power,
        where="mid",
        color="#1F1F1F",
        linewidth=1.1,
        label="Available power",
    )
    power_top = max([*available_power, *accepted_power, 0.0])
    power_axis.set_ylim(0.0, power_top * 1.05 if power_top > 0 else 1.0)
    if max(accepted_power, default=0.0) <= _ENERGY_TOLERANCE_MWH:
        _annotate_no_supply(power_axis, "No accepted output")
    power_axis.set_ylabel("Power (MW)")
    power_axis.grid(axis="y", alpha=0.25)
    power_axis.legend(loc="upper left", ncol=3, fontsize=8)

    reward_twin = reward_axis.twinx()
    reward_axis.step(
        times,
        [step.profit_eur for step in steps],
        where="mid",
        color="#54A24B",
        linewidth=1.1,
        label="Profit",
    )
    reward_axis.step(
        times,
        [-step.regret_eur for step in steps],
        where="mid",
        color="#B279A2",
        linewidth=1.1,
        label="Regret penalty (negative)",
    )
    reward_twin.step(
        times,
        [step.reward for step in steps],
        where="mid",
        color="#1F1F1F",
        linewidth=0.9,
        alpha=0.7,
        label="Reward",
    )
    reward_axis.axhline(0, color="#1F1F1F", linewidth=0.6)
    reward_axis.set_ylabel("EUR per delivery")
    reward_twin.set_ylabel("Normalised reward")
    reward_axis.set_xlabel("Delivery start")
    reward_axis.grid(axis="y", alpha=0.25)
    handles, labels = reward_axis.get_legend_handles_labels()
    twin_handles, twin_labels = reward_twin.get_legend_handles_labels()
    reward_axis.legend(
        [*handles, *twin_handles],
        [*labels, *twin_labels],
        loc="upper left",
        ncol=3,
        fontsize=8,
    )
    _format_time_axis(reward_axis, dates)

    unit_label = ", ".join(unit_names) or "learning unit"
    figure.suptitle(
        f"{episode_label} — {unit_label}: bids, dispatch and reward "
        f"(total reward {sum(step.reward for step in steps):.3f}, "
        f"total profit {sum(step.profit_eur for step in steps):,.0f} EUR)"
    )
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _read_learning_metrics(path: Path) -> list[dict[str, Any]]:
    """Read a metrics CSV written by the Version 5 trainer or evaluator."""

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
    """Plot training and validation episode metrics for every independent run."""

    runs = sorted({row["run"] for row in rows})
    figure, axes = pyplot.subplots(len(_CURVE_FIELDS), 1, figsize=(12, 10), sharex=True)
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
                label="Initial experience (no Actor updates)",
            )
    for index, run in enumerate(runs):
        color = _UNIT_COLORS[index % len(_UNIT_COLORS)]
        run_rows = sorted(
            (row for row in rows if row["run"] == run), key=lambda row: row["episode"]
        )
        initial = [row for row in run_rows if row["phase"] == "initial_experience"]
        training = [row for row in run_rows if row["phase"] == "train"]
        validation = [row for row in run_rows if row["phase"] == "validation"]
        best = (
            max(validation, key=lambda row: row["discounted_reward"])
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
                    color=color,
                    alpha=0.45,
                    linewidth=1.0,
                    label=f"Run {run} training",
                )
            if validation:
                axis.plot(
                    [row["episode"] for row in validation],
                    [row[field] for row in validation],
                    color=color,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    label=f"Run {run} validation",
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
                    label=f"Run {run} best (discounted)",
                )
    for axis, (_, label) in zip(axes, _CURVE_FIELDS):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Episode")
    axes[-1].xaxis.set_major_locator(pyplot.MaxNLocator(integer=True))
    figure.suptitle(
        f"Learning curves — {len(runs)} independent run(s); dashed: initial "
        "experience; star: validation episode with the highest discounted reward"
    )
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)


def _format_metric(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1_000 else f"{value:.3g}"


def _plot_evaluation_comparison(pyplot, rows: list[dict[str, Any]], path: Path) -> Path:
    """Compare deterministic Actor and marginal-cost baseline evaluations."""

    run_count = max(
        len([row for row in rows if row["phase"] == phase])
        for phase, _, _ in _COMPARISON_PHASES
    )
    figure, axes = pyplot.subplots(2, 4, figsize=(16, 7.5))
    for axis, (field, title) in zip(axes.flat, _COMPARISON_FIELDS):
        for position, (phase, _, color) in enumerate(_COMPARISON_PHASES):
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
            range(len(_COMPARISON_PHASES)),
            [label for _, label, _ in _COMPARISON_PHASES],
        )
        axis.set_xlim(-0.6, len(_COMPARISON_PHASES) - 0.4)
        axis.margins(y=0.15)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    spread_note = "mean ± population std, dots are runs" if run_count > 1 else "1 run"
    figure.suptitle(f"Deterministic Actor vs marginal-cost baseline — {spread_note}")
    figure.tight_layout()
    return _save_figure(pyplot, figure, path)
