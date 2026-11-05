"""第五版训练、评估、测试和绘图共享的回合指标、CSV 写出及跨运行统计。"""

import csv
from dataclasses import asdict, dataclass
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
    """统计学习入口已限定为每个产品 1 小时的回合指标。"""
    total_reward = 0.0
    discounted_reward = 0.0
    total_profit = 0.0
    accepted_energy = 0.0
    segment_totals = {
        "learning_minimum": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
        "learning_flexible": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
    }
    for index, step in enumerate(result.learning_steps):
        total_reward += step.reward
        discounted_reward += gamma**index * step.reward
        total_profit += step.profit_eur
        accepted_energy += step.accepted_power_mw
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

    def segment_ratio(segment: str, field: str) -> float:
        totals = segment_totals[segment]
        return totals[field] / totals["offered"] if totals["offered"] else 0.0

    return _EpisodeMetrics(
        run=run,
        phase=phase,
        episode=episode,
        total_reward=total_reward,
        mean_reward_per_hour=(
            total_reward / len(result.learning_steps) if result.learning_steps else 0.0
        ),
        discounted_reward=discounted_reward,
        total_profit_eur=total_profit,
        accepted_energy_mwh=accepted_energy,
        minimum_segment_average_bid_eur_per_mwh=segment_ratio("learning_minimum", "price"),
        flexible_segment_average_bid_eur_per_mwh=segment_ratio("learning_flexible", "price"),
        minimum_segment_acceptance_ratio=segment_ratio("learning_minimum", "accepted"),
        flexible_segment_acceptance_ratio=segment_ratio("learning_flexible", "accepted"),
    )


def _write_metrics(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["run", "phase", "episode", *LEARNING_METRIC_FIELDS]
        )
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            row.update({
                name: f"{row[name]:.{decimals}f}"
                for name, decimals in LEARNING_METRIC_DECIMALS.items()
            })
            writer.writerow(row)


def _write_run_summary(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    """写出 Actor 与基线对比指标在各独立运行间的均值和总体标准差。"""
    phase_groups: dict[str, list[_EpisodeMetrics]] = {}
    for item in metrics:
        if item.phase.endswith(("_actor", "_baseline")):
            phase_groups.setdefault(item.phase, []).append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "run_count"]
    for name in LEARNING_METRIC_FIELDS:
        fieldnames.extend((f"{name}_mean", f"{name}_std"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for phase, phase_metrics in phase_groups.items():
            row: dict[str, str | int] = {
                "phase": phase,
                "run_count": len(phase_metrics),
            }
            for name in LEARNING_METRIC_FIELDS:
                values = [getattr(item, name) for item in phase_metrics]
                row[f"{name}_mean"] = f"{fmean(values):.12f}"
                row[f"{name}_std"] = f"{pstdev(values):.12f}"
            writer.writerow(row)
