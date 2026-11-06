"""第五版市场入口：校验学习机组，创建学习会话并汇总回合结果。"""

from dataclasses import replace
from pathlib import Path
from ..errors import InputValidationError
from ..deep_learning.learning import (
    OBSERVATION_SIZE, ActionProvider, LearningEpisodeSession, TransitionConsumer,
)
from ..models import (
    MarketOpening,
    MarketSettings,
    PowerPlant,
    SimulationResult,
)
from .base_market import BaseMarket
from .eom import simulate_eom_market


def _learning_plant(settings, plants):
    """校验并返回第五版唯一的学习型参与者。"""
    learning_plants = tuple(
        plant
        for plant in plants
        if plant.bidding_strategy == "powerplant_energy_learning"
    )
    if len(learning_plants) != 1:
        raise InputValidationError(
            "第五版要求恰好配置一台使用 powerplant_energy_learning 策略的机组。"
        )
    if settings.product_count != 1:
        raise InputValidationError(
            "第五版要求每次 EOM 开市仅提供一个产品。"
        )
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "第五版要求 EOM 的 market_mechanism 设置为 pay_as_clear。"
        )
    max_bid_price = settings.learning_config.max_bid_price
    if (
        settings.minimum_bid_price > -max_bid_price
        or settings.maximum_bid_price < max_bid_price
    ):
        raise InputValidationError(
            "EOM 报价上下限必须覆盖学习动作对应的报价范围 "
            f"[-{max_bid_price:g}, {max_bid_price:g}]。"
        )
    return learning_plants[0]


def simulate_v5_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
    learning_action_provider: ActionProvider,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_enforce_action_bounds: bool = True,
) -> SimulationResult:
    """准备一个使用确定性策略或训练策略回调的学习市场。"""

    market = BaseMarket(input_path, settings, openings)
    session = LearningEpisodeSession(
        plant=_learning_plant(settings, market.plants),
        action_provider=learning_action_provider,
        transition_consumer=learning_transition_consumer,
        enforce_action_bounds=learning_enforce_action_bounds,
        price_scale_eur_per_mwh=settings.learning_config.max_bid_price,
    )
    market.prepare(learning_session=session)
    result = simulate_eom_market(market)
    session._emit_transition(next_state=(0.0,) * OBSERVATION_SIZE, done=True)
    steps = tuple(session._steps)
    transitions = tuple(session._transitions)
    return replace(result, learning_steps=steps, learning_transitions=transitions)
