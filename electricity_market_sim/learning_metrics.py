"""第五版训练与绘图共享的回合指标字段及输出精度定义。"""

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
