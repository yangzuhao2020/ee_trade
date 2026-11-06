"""第五版学习机组的状态构造、报价生成、奖励计算及回合记录。
通过策略回调接入报价决策，不依赖 PyTorch；网络和参数更新由 matd3_agent.py / grpo_agent.py 负责，训练编排由 training.py 负责。"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import cos, isfinite, pi, sin
import numpy as np
from ..bidding import segment_supply_offers
from ..errors import InputValidationError
from ..models import (
    LearningStepResult,
    LearningTransition,
    MarketClearingResult,
    PowerPlant,
    SupplyOffer,
)

OBSERVATION_SIZE = 16
ACTION_SIZE = 2
PRICE_HISTORY_HOURS = 12
REWARD_SCALE_EUR_PER_MWH = 100.0
_POWER_TOLERANCE_MW = 1e-9

ActionProvider = Callable[[tuple[float, ...]], Sequence[float]]
TransitionConsumer = Callable[[LearningTransition], None]


@dataclass
class _PendingLearningStep:
    delivery_start: datetime
    state: tuple[float, ...]
    action: tuple[float, float]
    available_power_mw: float
    result: LearningStepResult | None = None


class LearningEpisodeSession:
    """管理学习策略与市场仿真的逐小时交互。
    首先是构建策略输入的 16 维观测，调用策略生成两段报价动作，转换为市场报价；
    出清后计算奖励，并记录状态、动作、奖励、下一状态及终止标记。
    """
    def __init__(
        self,
        *,
        plant: PowerPlant,
        action_provider: ActionProvider,
        transition_consumer: TransitionConsumer | None = None,
        enforce_action_bounds: bool = True,
        price_scale_eur_per_mwh: float,
    ) -> None:
        if not isfinite(plant.max_power_mw) or plant.max_power_mw <= 0:
            raise ValueError("Rated power must be positive and finite.")
        if not isfinite(price_scale_eur_per_mwh) or price_scale_eur_per_mwh <= 0:
            raise ValueError("Price scale must be positive and finite.")
        self.plant = plant
        self.action_provider = action_provider
        self.transition_consumer = transition_consumer
        self.enforce_action_bounds = enforce_action_bounds
        self.price_scale_eur_per_mwh = price_scale_eur_per_mwh
        self._history: deque[float] = deque([0.0] * PRICE_HISTORY_HOURS, maxlen=PRICE_HISTORY_HOURS)
        self._pending: _PendingLearningStep | None = None
        self._steps: list[LearningStepResult] = []
        self._transitions: list[LearningTransition] = []

    def _observation(
        self, delivery_start: datetime, marginal_cost_eur_per_mwh: float
    ) -> tuple[float, ...]:
        """构建 16 维观测：历史出清价、交付小时编码、上期出力和自身成本。
        价格和成本使用与动作映射相同的 max_bid_price 缩放；成本始终放在最后一维。
        不读取负荷或预测电价；小时编码使用场景中的交付时间。
        历史价格窗口在回合开始时补零，缩放参数在会话入口校验。
        """

        previous = self._pending.result if self._pending is not None else None
        hour = (
            delivery_start.hour + delivery_start.minute / 60
            + delivery_start.second / 3600 + delivery_start.microsecond / 3_600_000_000
        )
        hour_angle = 2 * pi * hour / 24
        return (
            *(value / self.price_scale_eur_per_mwh for value in self._history),
            sin(hour_angle),
            cos(hour_angle),
            (previous.accepted_power_mw if previous is not None else 0.0) / self.plant.max_power_mw,
            marginal_cost_eur_per_mwh / self.price_scale_eur_per_mwh,
        )

    def _offers(
        self,
        delivery_start: datetime,
        delivery_end: datetime,
        available_power_mw: float,
        marginal_cost_eur_per_mwh: float,
        action: tuple[float, float],
    ) -> tuple[SupplyOffer, ...]:
        """将实际执行的动作映射为可各自独立出清的最低出力段报价和灵活出力段报价。
        调用方必须先校验这两个值都是有限数，并施加所需的边界限制。
        经验数据中保留动作的原始顺序，只对报价价格进行排序。
        """
        if available_power_mw < self.plant.min_power_mw - _POWER_TOLERANCE_MW:
            return ()
        low_price, high_price = sorted(
            value * self.price_scale_eur_per_mwh for value in action
        )
        segment_specs = (
            ("learning_minimum", min(self.plant.min_power_mw, available_power_mw), low_price),
            (
                "learning_flexible",
                max(0.0, available_power_mw - self.plant.min_power_mw),
                high_price,
            ),
        )
        return tuple(
            segment_supply_offers(
                self.plant,
                delivery_start,
                delivery_end,
                segment_specs,
                marginal_cost_eur_per_mwh,
            )
        )

    def _step_result(self, market_result: MarketClearingResult) -> LearningStepResult:
        """出清后计算利润、遗憾（regret）塑形项和归一化奖励。"""
        pending = self._pending
        # 整个市场无成交时，不使用出清价计算奖励。
        has_trades = any(
            item.accepted_power_mw > _POWER_TOLERANCE_MW for item in market_result.offers
        )
        if has_trades:
            price = market_result.clearing_price_eur_per_mwh
        else:
            price = None
        if price is None:
            accepted_power_mw = 0.0
            profit_eur = 0.0
            regret_eur = 0.0
            reward = 0.0
        else:
            # 筛选本机组的报价记录，包括未成交的报价。
            cleared = []
            for item in market_result.offers:
                if item.offer.unit_name == self.plant.name:
                    cleared.append(item)
                    
            # 汇总各报价段的成交电量和实际利润。
            accepted_energy_mwh = sum(item.accepted_energy_mwh for item in cleared)
            accepted_power_mw = accepted_energy_mwh / market_result.duration_hours
            profit_eur = sum(item.profit_eur for item in cleared)
            # 成交功率超过最小出力时，减轻遗憾惩罚。
            regret_factor = 0.1 if accepted_power_mw > self.plant.min_power_mw else 0.5
            
            # 按未成交可用功率计算错失利润，仅惩罚正值。
            if cleared:
                unused_power_mw = pending.available_power_mw - accepted_power_mw
                margin_eur_per_mwh = price - cleared[0].offer.marginal_cost_eur_per_mwh
                missed_profit_eur = unused_power_mw * margin_eur_per_mwh * market_result.duration_hours
                regret_eur = regret_factor * max(0.0, missed_profit_eur)
            else:
                regret_eur = 0.0
            # 奖励归一化分母：价格尺度 × 额定最大功率 × 时长。
            denominator = (
                REWARD_SCALE_EUR_PER_MWH * self.plant.max_power_mw * market_result.duration_hours
            )
            reward = (profit_eur - regret_eur) / denominator # TODO: 奖励的计算
        return LearningStepResult(
            delivery_start=market_result.delivery_start,
            action=pending.action,
            available_power_mw=pending.available_power_mw,
            accepted_power_mw=accepted_power_mw,
            clearing_price_eur_per_mwh=0.0 if price is None else price,
            profit_eur=profit_eur,
            regret_eur=regret_eur,
            reward=reward,
        )

    def _emit_transition(self, *, next_state, done,):
        """保存本小时状态、报价动作、奖励及下一小时状态，每条训练记录只保存一次。"""
        pending = self._pending
        if pending is None:
            return
        if pending.result is None:
            raise RuntimeError(
                f"Learning result was not recorded for {pending.delivery_start!s}."
            )
        transition = LearningTransition(
            delivery_start=pending.delivery_start,
            state=pending.state,
            action=pending.action,
            reward=pending.result.reward,
            next_state=next_state,
            done=done,
        )
        self._steps.append(pending.result)
        self._transitions.append(transition)
        self._pending = None
        if self.transition_consumer is not None:
            self.transition_consumer(transition)

    def offers_for_product(
        self,
        *,
        delivery_start,
        delivery_end,
        available_power_mw,
        marginal_cost_eur_per_mwh,
    ):
        previous_start = delivery_start - timedelta(hours=1)
        if self._pending is not None and self._pending.delivery_start != previous_start:
            raise InputValidationError("学习回合的交付时段必须按小时连续排列。")
        state = self._observation(delivery_start, marginal_cost_eur_per_mwh)
        self._emit_transition(next_state=state, done=False)
        values = np.asarray(self.action_provider(state), dtype=np.float64)
        
        if values.shape != (ACTION_SIZE,):
            raise ValueError(f"学习策略必须返回包含 {ACTION_SIZE} 个数值的一维报价动作。")
        if not np.isfinite(values).all():
            raise ValueError("学习策略的报价动作不能包含 NaN 或无穷大。")
        if self.enforce_action_bounds:
            values = np.clip(values, -1.0, 1.0)
            
        action = float(values[0]), float(values[1])
        self._pending = _PendingLearningStep(
            delivery_start=delivery_start,
            state=state,
            action=action,
            available_power_mw=available_power_mw,
        )
        return self._offers(
            delivery_start, delivery_end, 
            available_power_mw, 
            marginal_cost_eur_per_mwh, action
        )

    def record_result(self, market_result: MarketClearingResult):
        pending = self._pending
        if pending is None or pending.delivery_start != market_result.delivery_start:
            return
        pending.result = self._step_result(market_result)
        self._history.append(pending.result.clearing_price_eur_per_mwh or 0.0)

