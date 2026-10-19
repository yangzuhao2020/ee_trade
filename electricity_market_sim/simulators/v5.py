"""第五版学习机组的市场扩展：生成学习报价并记录奖励与状态转移。"""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from ..errors import InputValidationError
from ..deep_learning.learning import (
    ActionProvider, LearningEpisodeSession, TransitionConsumer,
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
) -> PowerPlant | None:
    """校验并返回第五版唯一的学习型参与者。"""

    learning_plants = tuple(
        plant
        for plant in plants
        if plant.bidding_strategy == "powerplant_energy_learning"
    )
    if not learning_plants:
        if (
            settings.learning_config is not None
            and settings.learning_config.learning_mode
        ):
            raise InputValidationError(
                "learning_mode requires exactly one powerplant_energy_learning unit."
            )
        return None
    if len(learning_plants) != 1:
        raise InputValidationError(
            "Version 5 requires exactly one powerplant_energy_learning unit."
        )
    if settings.learning_config is None or not settings.learning_config.learning_mode:
        raise InputValidationError(
            "powerplant_energy_learning requires learning_config.learning_mode: true."
        )
    if settings.product_count != 1 or settings.product_duration != timedelta(hours=1):
        raise InputValidationError(
            "Version 5 requires one 1-hour product per EOM opening."
        )
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError(
            "Version 5 requires EOM market_mechanism: pay_as_clear."
        )
    max_bid_price = settings.learning_config.max_bid_price
    if (
        settings.minimum_bid_price > -max_bid_price
        or settings.maximum_bid_price < max_bid_price
    ):
        raise InputValidationError(
            "The EOM price limits must contain the learning action range "
            f"[-{max_bid_price:g}, {max_bid_price:g}]."
        )
    return learning_plants[0]


class _LearningMarketExtension:
    """Replace one plant's bids and collect rewards before the next opening."""

    settle_before_opening = True

    def __init__(self, action_provider: ActionProvider | None,
                 transition_consumer: TransitionConsumer | None = None,
                 enforce_action_bounds: bool = True) -> None:
        self.action_provider = action_provider
        self.transition_consumer = transition_consumer
        self.enforce_action_bounds = enforce_action_bounds
        self.session: LearningEpisodeSession | None = None

    def prepare(self, market: BaseMarket) -> EomExtensionInputs:
        self.plant = _learning_plant(market.settings, market.plants)
        if self.plant is not None and self.action_provider is None:
            raise InputValidationError(
                "A learning_action_provider is required for a Version 5 market episode."
            )
        return EomExtensionInputs(
            incomplete_opening_error="Learning opening",
            hard_forecast_horizon=True,
        )

    def initialize(self, market: BaseMarket) -> None:
        assert self.action_provider is not None
        assert self.plant is not None
        assert market.settings.learning_config is not None
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
        assert self.session is not None
        return tuple(
            offer
            for start, end in products
            for offer in self.session.offers_for_product(
                delivery_start=start,
                delivery_end=end,
                available_power_mw=available_powers[start],
                marginal_cost_eur_per_mwh=marginal_cost_at(start),
                scheduled_results=scheduled_results,
            )
        )

    def record_delivery_start(self, result: MarketClearingResult) -> None:
        assert self.session is not None
        self.session.record_result(result)

    def record_delivery(self, result: MarketClearingResult) -> None:
        pass

    def finalize(self, result: SimulationResult) -> SimulationResult:
        assert self.session is not None
        steps, transitions = self.session.finalize()
        return replace(
            result, learning_steps=steps, learning_transitions=transitions,
        )


def simulate_v5_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
    learning_action_provider: ActionProvider | None = None,
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
