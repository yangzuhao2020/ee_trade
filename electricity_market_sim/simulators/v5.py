"""V5 learning participant extension over the conventional EOM market."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

from ..errors import InputValidationError
from ..learning import (
    ActionProvider, LearningEpisodeSession, TransitionConsumer, calculate_load_base,
)
from ..market_models import DemandBid, MarketClearingResult, SupplyOffer
from ..models import MarketOpening, MarketSettings, PowerPlant, SimulationResult
from .base_market import BaseMarket
from .base_market.demand import _profile_power
from .base_market.forecasts import _forecast_starts
from .base_market.plants import _available_power
from .eom import EomExtensionInputs, ScheduledProductKey, simulate_eom_market

_FORECAST_HOURS = 12


def _learning_plant(
    settings: MarketSettings, plants: tuple[PowerPlant, ...]
) -> PowerPlant | None:
    """Validate and return Version 5's sole learning participant."""

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

def _residual_load_forecasts(
    *,
    forecast_times: set[datetime],
    demand_units,
    demand_profiles: dict[str, dict[datetime, float]],
    plants: tuple[PowerPlant, ...],
    availability_profiles: dict[str, dict[datetime, float]],
) -> dict[datetime, float]:
    """Return demand minus forecast wind/solar production for each hour."""

    variable_renewables = tuple(
        plant
        for plant in plants
        if any(label in plant.technology.lower() for label in ("wind", "solar"))
    )
    residual: dict[datetime, float] = {}
    for forecast_time in sorted(forecast_times):
        demand_mw = sum(
            _profile_power(demand_profiles, unit.name, forecast_time, "demand_df.csv")
            for unit in demand_units
            if not unit.is_elastic
        )
        renewable_mw = sum(
            _available_power(plant, forecast_time, availability_profiles)
            for plant in variable_renewables
        )
        residual[forecast_time] = demand_mw - renewable_mw
    return residual

class _LearningMarketExtension:
    """Replace one plant's bids and collect rewards before the next opening."""

    settle_before_opening = True

    def __init__(self, action_provider: ActionProvider | None,
                 transition_consumer: TransitionConsumer | None = None,
                 load_base_mw: float | None = None,
                 enforce_action_bounds: bool = True) -> None:
        self.action_provider = action_provider
        self.transition_consumer = transition_consumer
        self.load_base_mw = load_base_mw
        self.enforce_action_bounds = enforce_action_bounds
        self.session: LearningEpisodeSession | None = None

    def prepare(self, market: BaseMarket) -> EomExtensionInputs:
        self.plant = _learning_plant(market.settings, market.plants)
        if self.plant is not None and self.action_provider is None:
            raise InputValidationError(
                "A learning_action_provider is required for a Version 5 market episode."
            )
        if self.load_base_mw is not None:
            try:
                self.load_base_mw = float(self.load_base_mw)
            except (TypeError, ValueError) as exc:
                raise InputValidationError("The saved learning load base must be numeric.") from exc
            if not isfinite(self.load_base_mw) or self.load_base_mw <= 0:
                raise InputValidationError(
                    "The saved learning load base must be a positive finite value."
                )
            if self.plant is None:
                raise InputValidationError("A saved learning load base requires a learning plant.")
        return EomExtensionInputs(
            forecast_offsets=tuple(range(_FORECAST_HOURS)),
            incomplete_opening_error="Learning opening",
        )

    def initialize(self, market: BaseMarket) -> None:
        training_times = sorted(
            {start for opening in market.valid_openings for start, _ in opening.products}
        )
        forecast_times = _forecast_starts(
            training_times, settings=market.settings,
            extra_forecast_offsets=tuple(range(_FORECAST_HOURS)),
        )
        residual_forecasts = _residual_load_forecasts(
            forecast_times=forecast_times,
            demand_units=market.demand_units,
            demand_profiles=market.demand_profiles,
            plants=market.plants,
            availability_profiles=market.availability_profiles,
        )
        load_base = (
            calculate_load_base([residual_forecasts[start] for start in training_times])
            if self.load_base_mw is None else self.load_base_mw
        )
        assert self.action_provider is not None
        assert self.plant is not None
        assert market.settings.learning_config is not None
        self.session = LearningEpisodeSession(
            plant=self.plant,
            action_provider=self.action_provider,
            residual_load_forecasts_mw=residual_forecasts,
            price_forecasts_eur_per_mwh=market.price_forecasts,
            load_base_mw=load_base,
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
            learning_load_base_mw=self.session.load_base_mw,
        )


def simulate_v5_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
    learning_action_provider: ActionProvider | None = None,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_load_base_mw: float | None = None,
    learning_enforce_action_bounds: bool = True,
) -> SimulationResult:
    """Prepare a conventional market with deterministic or training policy hooks."""

    market = BaseMarket(input_path, settings, openings)
    market.prepare(_LearningMarketExtension(
        learning_action_provider, learning_transition_consumer,
        learning_load_base_mw, learning_enforce_action_bounds,
    ))
    return simulate_eom_market(market)
