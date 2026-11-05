"""第五版学习机组的状态构造、报价生成、奖励计算及回合记录。
通过策略回调接入报价决策，不依赖 PyTorch；网络和参数更新由 agent.py 负责，训练编排由 training.py 负责。"""

from collections import deque
from collections.abc import Callable, Iterable, Sequence
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
PRICE_SCALE_EUR_PER_MWH = 100.0
REWARD_SCALE_EUR_PER_MWH = 100.0
_POWER_TOLERANCE_MW = 1e-9

ActionProvider = Callable[[tuple[float, ...]], Sequence[float]]
TransitionConsumer = Callable[[LearningTransition], None]


def build_learning_observation(
    *,
    delivery_start: datetime,
    historical_clearing_prices_eur_per_mwh: Iterable[float],
    previous_power_mw: float,
    rated_power_mw: float,
    marginal_cost_eur_per_mwh: float,
    price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
) -> tuple[float, ...]:
    """构建 16 维观测：历史出清价、交付小时编码、上期出力和自身成本。

    价格和成本使用与动作映射相同的 max_bid_price 缩放；成本始终放在最后一维。
    不读取负荷或预测电价；小时编码使用场景中的交付时间。
    调用方提供已补齐的 12 小时价格窗口，并在回合入口校验固定缩放参数。
    """

    hour = (
        delivery_start.hour + delivery_start.minute / 60
        + delivery_start.second / 3600 + delivery_start.microsecond / 3_600_000_000
    )
    hour_angle = 2 * pi * hour / 24
    return (
        *(value / price_scale_eur_per_mwh for value in historical_clearing_prices_eur_per_mwh),
        sin(hour_angle),
        cos(hour_angle),
        previous_power_mw / rated_power_mw,
        marginal_cost_eur_per_mwh / price_scale_eur_per_mwh,
    )


def learning_offers(
    *,
    plant: PowerPlant,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power_mw: float,
    marginal_cost_eur_per_mwh: float,
    action: tuple[float, float],
    price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
) -> tuple[SupplyOffer, ...]:
    """将实际执行的动作映射为可各自独立出清的最低出力段报价和灵活出力段报价。

    调用方必须先校验这两个值都是有限数，并施加所需的边界限制。
    经验数据中保留动作的原始顺序，只对报价价格进行排序。
    """
    if available_power_mw < plant.min_power_mw - _POWER_TOLERANCE_MW:
        return ()
    low_price, high_price = sorted(
        value * price_scale_eur_per_mwh for value in action
    )
    segment_specs = (
        ("learning_minimum", min(plant.min_power_mw, available_power_mw), low_price),
        (
            "learning_flexible",
            max(0.0, available_power_mw - plant.min_power_mw),
            high_price,
        ),
    )
    return tuple(
        segment_supply_offers(
            plant,
            delivery_start,
            delivery_end,
            segment_specs,
            marginal_cost_eur_per_mwh,
        )
    )


def _valid_clearing_price(market_result: MarketClearingResult) -> float | None:
    """返回 EOM 出清价格；没有被接受的供给报价确定该价格时返回 None。"""

    if any(item.accepted_power_mw > _POWER_TOLERANCE_MW for item in market_result.offers):
        return market_result.clearing_price_eur_per_mwh
    return None


def learning_step_result(
    *,
    market_result: MarketClearingResult,
    plant: PowerPlant,
    action: tuple[float, float],
    available_power_mw: float,
) -> LearningStepResult:
    """出清后计算利润、遗憾（regret）塑形项和归一化奖励。"""

    price = _valid_clearing_price(market_result)
    if price is None or available_power_mw < plant.min_power_mw - _POWER_TOLERANCE_MW:
        return LearningStepResult(
            delivery_start=market_result.delivery_start,
            action=action,
            available_power_mw=available_power_mw,
            accepted_power_mw=0.0,
            clearing_price_eur_per_mwh=0.0 if price is None else price,
            profit_eur=0.0,
            regret_eur=0.0,
            reward=0.0,
        )
    cleared = tuple(
        item for item in market_result.offers if item.offer.unit_name == plant.name
    )
    accepted_energy_mwh = sum(item.accepted_energy_mwh for item in cleared)
    accepted_power_mw = accepted_energy_mwh / market_result.duration_hours
    profit_eur = sum(item.profit_eur for item in cleared)
    regret_factor = 0.1 if accepted_power_mw > plant.min_power_mw else 0.5
    regret_eur = (
        regret_factor
        * max(
            0.0,
            (available_power_mw - accepted_power_mw)
            * (price - cleared[0].offer.marginal_cost_eur_per_mwh)
            * market_result.duration_hours,
        )
        if cleared
        else 0.0
    )
    denominator = (
        REWARD_SCALE_EUR_PER_MWH * plant.max_power_mw * market_result.duration_hours
    )
    return LearningStepResult(
        delivery_start=market_result.delivery_start,
        action=action,
        available_power_mw=available_power_mw,
        accepted_power_mw=accepted_power_mw,
        clearing_price_eur_per_mwh=price,
        profit_eur=profit_eur,
        regret_eur=regret_eur,
        reward=(profit_eur - regret_eur) / denominator,
    )


@dataclass
class _PendingLearningStep:
    delivery_start: datetime
    state: tuple[float, ...]
    action: tuple[float, float]
    available_power_mw: float
    result: LearningStepResult | None = None


class LearningEpisodeSession:
    """Connect a policy to sequential market openings and collect transitions."""

    def __init__(
        self,
        *,
        plant: PowerPlant,
        action_provider: ActionProvider,
        transition_consumer: TransitionConsumer | None = None,
        enforce_action_bounds: bool = True,
        price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
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

    def _emit_transition(
        self,
        *,
        next_state: tuple[float, ...],
        done: bool,
    ) -> None:
        """发布一个已完成的状态转移，每个转移只发布一次。"""

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
        delivery_start: datetime,
        delivery_end: datetime,
        available_power_mw: float,
        marginal_cost_eur_per_mwh: float,
    ) -> tuple[SupplyOffer, ...]:
        previous_start = delivery_start - timedelta(hours=1)
        if self._pending is not None and self._pending.delivery_start != previous_start:
            raise InputValidationError(
                "Learning delivery products must form a continuous hourly episode."
            )
        previous = self._pending.result if self._pending is not None else None
        state = build_learning_observation(
            delivery_start=delivery_start,
            historical_clearing_prices_eur_per_mwh=self._history,
            previous_power_mw=previous.accepted_power_mw if previous is not None else 0.0,
            rated_power_mw=self.plant.max_power_mw,
            marginal_cost_eur_per_mwh=marginal_cost_eur_per_mwh,
            price_scale_eur_per_mwh=self.price_scale_eur_per_mwh,
        )
        self._emit_transition(next_state=state, done=False)
        values = np.asarray(self.action_provider(state), dtype=np.float64)
        if values.shape != (ACTION_SIZE,):
            raise ValueError(f"Learning action must have shape ({ACTION_SIZE},).")
        if not np.isfinite(values).all():
            raise ValueError("Learning action must contain only finite values.")
        if self.enforce_action_bounds:
            values = np.clip(values, -1.0, 1.0)
        action = float(values[0]), float(values[1])
        self._pending = _PendingLearningStep(
            delivery_start=delivery_start,
            state=state,
            action=action,
            available_power_mw=available_power_mw,
        )
        return learning_offers(
            plant=self.plant,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            available_power_mw=available_power_mw,
            marginal_cost_eur_per_mwh=marginal_cost_eur_per_mwh,
            action=action,
            price_scale_eur_per_mwh=self.price_scale_eur_per_mwh,
        )

    def record_result(self, market_result: MarketClearingResult) -> None:
        pending = self._pending
        if pending is None or pending.delivery_start != market_result.delivery_start:
            return
        pending.result = learning_step_result(
            market_result=market_result,
            plant=self.plant,
            action=pending.action,
            available_power_mw=pending.available_power_mw,
        )
        self._history.append(pending.result.clearing_price_eur_per_mwh or 0.0)

    def finalize(
        self,
    ) -> tuple[tuple[LearningStepResult, ...], tuple[LearningTransition, ...]]:
        self._emit_transition(
            next_state=(0.0,) * OBSERVATION_SIZE,
            done=True,
        )
        return tuple(self._steps), tuple(self._transitions)
