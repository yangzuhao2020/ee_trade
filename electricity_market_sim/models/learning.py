"""第五版学习配置，以及学习过程中逐时段的动作、奖励与经验回放记录。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

LEARNING_PERIOD_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class LearningConfig:
    """MATD3 / GRPO 的公共设置及各自训练参数。"""
    learning_mode: bool
    algorithm: str
    learning_rate: float
    gamma: float
    validation_interval: int
    training_episodes: int = 30
    initial_experience_episodes: int = 5
    replay_buffer_size: int = 100000
    batch_size: int = 128
    train_frequency_steps: int = 100
    gradient_steps: int = 10
    exploration_noise_std: float = 0.2
    noise_sigma: float = 0.1
    noise_scale: float = 1.0
    noise_dt: float = 1.0
    tau: float = 0.005
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    device: str = "cpu"
    trained_policies_save_path: str | None = None
    trained_policies_load_path: str | None = None
    max_bid_price: float = 100.0
    # 学习模式必填，按名称保存交付时段 [start, end)。
    periods: dict[str, tuple[datetime, datetime]] | None = None
    training_steps: int = 0
    group_size: int = 12
    sampling_workers: int = 4
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    target_kl: float = 0.02
    initial_std: float = 0.2
    min_std: float = 0.02
    max_std: float = 1.0
    advantage_threshold: float = 1e-6
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    min_lr_ratio: float = 0.25

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
