"""第五版训练、评估、测试和绘图共享的回合指标、CSV 写出及跨运行统计。"""

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from ..models import SimulationResult

LEARNING_METRIC_DECIMALS = {
    "total_reward": 12,
    "mean_reward_per_hour": 12,
    "discounted_reward": 12,
    "total_profit_eur": 6,
    "accepted_energy_mwh": 6,
    "minimum_segment_average_bid_eur_per_mwh": 6,
    "flexible_segment_average_bid_eur_per_mwh": 6,
    "minimum_segment_acceptance_ratio": 12,
    "flexible_segment_acceptance_ratio": 12,
}
LEARNING_METRIC_FIELDS = tuple(LEARNING_METRIC_DECIMALS)


@dataclass(frozen=True)
class _EpisodeMetrics:
    run: int
    phase: str
    episode: int
    total_reward: float
    mean_reward_per_hour: float
    discounted_reward: float
    total_profit_eur: float
    accepted_energy_mwh: float
    minimum_segment_average_bid_eur_per_mwh: float
    flexible_segment_average_bid_eur_per_mwh: float
    minimum_segment_acceptance_ratio: float
    flexible_segment_acceptance_ratio: float


def _metrics(
    result: SimulationResult, *, run: int, phase: str, episode: int, gamma: float
) -> _EpisodeMetrics:
    total_reward = 0.0
    discounted_reward = 0.0
    total_profit = 0.0
    accepted_energy = 0.0
    total_hours = 0.0
    segment_totals = {
        "learning_minimum": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
        "learning_flexible": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
    }
    durations = {
        market.delivery_start: market.duration_hours for market in result.market_results
    }
    for index, step in enumerate(result.learning_steps):
        total_reward += step.reward
        discounted_reward += gamma**index * step.reward
        total_profit += step.profit_eur
        accepted_energy += step.accepted_power_mw * durations[step.delivery_start]
        total_hours += durations[step.delivery_start]
    for market in result.market_results:
        for cleared in market.offers:
            segment = cleared.offer.offer_segment
            if segment not in segment_totals:
                continue
            totals = segment_totals[segment]
            offered = cleared.offer.offered_energy_mwh
            totals["offered"] += offered
            totals["accepted"] += cleared.accepted_energy_mwh
            totals["price"] += cleared.offer.bid_price_eur_per_mwh * offered

    def average_price(segment: str) -> float:
        totals = segment_totals[segment]
        return totals["price"] / totals["offered"] if totals["offered"] else 0.0

    def acceptance_ratio(segment: str) -> float:
        totals = segment_totals[segment]
        return totals["accepted"] / totals["offered"] if totals["offered"] else 0.0

    return _EpisodeMetrics(
        run=run,
        phase=phase,
        episode=episode,
        total_reward=total_reward,
        mean_reward_per_hour=total_reward / total_hours if total_hours else 0.0,
        discounted_reward=discounted_reward,
        total_profit_eur=total_profit,
        accepted_energy_mwh=accepted_energy,
        minimum_segment_average_bid_eur_per_mwh=average_price("learning_minimum"),
        flexible_segment_average_bid_eur_per_mwh=average_price("learning_flexible"),
        minimum_segment_acceptance_ratio=acceptance_ratio("learning_minimum"),
        flexible_segment_acceptance_ratio=acceptance_ratio("learning_flexible"),
    )


def _write_metrics(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["run", "phase", "episode", *LEARNING_METRIC_FIELDS]
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "run": item.run,
                    "phase": item.phase,
                    "episode": item.episode,
                    **{
                        name: f"{getattr(item, name):.{decimals}f}"
                        for name, decimals in LEARNING_METRIC_DECIMALS.items()
                    },
                }
            )


def _comparison_phases(period: str) -> tuple[str, str]:
    """返回某个数据时段上 Actor 与边际成本基线对比的阶段名。"""

    return f"{period}_actor", f"{period}_baseline"


def _write_run_summary(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    """写出 Actor 与基线对比指标在各独立运行间的均值和总体标准差。"""
    phases = list(dict.fromkeys(
        item.phase for item in metrics
        if item.phase.endswith(("_actor", "_baseline"))
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "run_count"]
    for name in LEARNING_METRIC_FIELDS:
        fieldnames.extend((f"{name}_mean", f"{name}_std"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for phase in phases:
            phase_metrics = [item for item in metrics if item.phase == phase]
            row: dict[str, str | int] = {
                "phase": phase,
                "run_count": len(phase_metrics),
            }
            for name in LEARNING_METRIC_FIELDS:
                values = [float(getattr(item, name)) for item in phase_metrics]
                row[f"{name}_mean"] = f"{fmean(values):.12f}"
                row[f"{name}_std"] = f"{pstdev(values):.12f}"
            writer.writerow(row)
