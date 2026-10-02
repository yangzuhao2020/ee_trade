"""Version 5 episode-metric columns shared by the trainer and learning plots."""

from __future__ import annotations

LEARNING_METRIC_DECIMALS = {
    "total_reward": 12,
    "discounted_reward": 12,
    "total_profit_eur": 6,
    "accepted_energy_mwh": 6,
    "minimum_segment_average_bid_eur_per_mwh": 6,
    "flexible_segment_average_bid_eur_per_mwh": 6,
    "minimum_segment_acceptance_ratio": 12,
    "flexible_segment_acceptance_ratio": 12,
}
LEARNING_METRIC_FIELDS = tuple(LEARNING_METRIC_DECIMALS)
