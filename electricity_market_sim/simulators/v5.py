"""第五版学习机组的市场扩展：生成学习报价并记录奖励与状态转移。"""

from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from ..errors import InputValidationError
from ..deep_learning.learning import (
    ActionProvider, LearningEpisodeSession, PRICE_HISTORY_HOURS, TransitionConsumer,
)
from ..models import (
    DemandBid,
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    PowerPlant,
    SimulationResult,
    SupplyOffer,
)
from .base_market import BaseMarket
from .eom import EomExtensionInputs, ScheduledProductKey, simulate_eom_market


def _learning_plant(
    settings: MarketSettings, plants: tuple[PowerPlant, ...]
) -> PowerPlant:
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


class _LearningMarketExtension:
    """Replace one plant's bids and collect rewards before the next opening."""

    settle_before_opening = True

    def __init__(self, action_provider: ActionProvider,
                 transition_consumer: TransitionConsumer | None = None,
                 enforce_action_bounds: bool = True) -> None:
        self.action_provider = action_provider
        self.transition_consumer = transition_consumer
        self.enforce_action_bounds = enforce_action_bounds
        self._history: deque[MarketClearingResult] = deque(maxlen=PRICE_HISTORY_HOURS)

    def prepare(self, market: BaseMarket) -> EomExtensionInputs:
        self.plant = _learning_plant(market.settings, market.plants)
        return EomExtensionInputs(
            incomplete_opening_error="Learning opening",
            hard_forecast_horizon=True,
        )

    def initialize(self, market: BaseMarket) -> None:
        self.session = LearningEpisodeSession(
            plant=self.plant,
            action_provider=self.action_provider,
            transition_consumer=self.transition_consumer,
            enforce_action_bounds=self.enforce_action_bounds,
            price_scale_eur_per_mwh=market.settings.learning_config.max_bid_price,
        )

    def bids_for_products(
        self, products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[DemandBid, ...]:
        return ()

    def offers_for_plant(
        self, plant: PowerPlant,
        products: tuple[tuple[datetime, datetime], ...],
        available_powers: dict[datetime, float],
        marginal_cost_at: Callable[[datetime], float],
        scheduled_results: dict[ScheduledProductKey, MarketClearingResult],
    ) -> tuple[SupplyOffer, ...] | None:
        if plant.name != self.plant.name:
            return None
        return tuple(
            offer
            for start, end in products
            for offer in self.session.offers_for_product(
                delivery_start=start,
                delivery_end=end,
                available_power_mw=available_powers[start],
                marginal_cost_eur_per_mwh=marginal_cost_at(start),
                scheduled_results={result.delivery_start: result for result in self._history},
            )
        )

    def record_delivery_start(self, result: MarketClearingResult) -> None:
        self.session.record_result(result)
        self._history.append(result)

    def finalize(self, result: SimulationResult) -> SimulationResult:
        steps, transitions = self.session.finalize()
        return replace(
            result, learning_steps=steps, learning_transitions=transitions,
        )


def simulate_v5_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
    learning_action_provider: ActionProvider,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_enforce_action_bounds: bool = True,
) -> SimulationResult:
    """准备一个挂接了确定性策略或训练策略钩子的常规市场。"""

    market = BaseMarket(input_path, settings, openings)
    market.prepare(_LearningMarketExtension(
        action_provider=learning_action_provider,
        transition_consumer=learning_transition_consumer,
        enforce_action_bounds=learning_enforce_action_bounds,
    ))
    return simulate_eom_market(market)
