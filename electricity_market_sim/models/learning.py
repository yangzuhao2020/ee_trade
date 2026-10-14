"""第五版学习配置，以及学习过程中逐时段的动作、奖励与经验回放记录。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LearningConfig:
    """经校验的 Version 5 单智能体 MATD3 学习配置。"""
    learning_mode: bool
    algorithm: str
    learning_rate: float
    training_episodes: int
    initial_experience_episodes: int
    replay_buffer_size: int
    batch_size: int
    gamma: float
    train_frequency_steps: int
    gradient_steps: int
    validation_interval: int
    exploration_noise_std: float
    noise_sigma: float
    noise_scale: float
    noise_dt: float
    action_noise_schedule: str
    tau: float
    policy_delay: int
    target_policy_noise: float
    target_noise_clip: float
    actor_architecture: str = "mlp"
    device: str = "cpu"
    continue_learning: bool = False
    trained_policies_save_path: str | None = None
    trained_policies_load_path: str | None = None
    max_bid_price: float = 100.0


@dataclass(frozen=True)
class LearningStepResult:
    """单个学习交付时段可供审计的动作、调度结果与奖励。"""
    delivery_start: datetime
    action: tuple[float, float]
    available_power_mw: float
    accepted_power_mw: float
    clearing_price_eur_per_mwh: float
    profit_eur: float
    regret_eur: float
    reward: float


@dataclass(frozen=True)
class LearningTransition:
    """一条经验回放缓冲区记录，使用未排序的实际执行动作。"""
    delivery_start: datetime
    state: tuple[float, ...]
    action: tuple[float, float]
    reward: float
    next_state: tuple[float, ...]
    done: bool
