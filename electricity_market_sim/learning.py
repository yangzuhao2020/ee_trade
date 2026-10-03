"""Version 5 observation, bidding, reward, and episode bookkeeping.

This module deliberately has no PyTorch dependency.  It is the market-facing
boundary of the learning system and can therefore be tested with a fixed policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import numpy as np

from .bidding import segment_supply_offers
from .errors import InputValidationError
from .market_models import MarketClearingResult, SupplyOffer
from .models import LearningStepResult, LearningTransition, PowerPlant
from .time_utils import hours_between

OBSERVATION_SIZE = 38
ACTION_SIZE = 2
FORECAST_HOURS = 12
PRICE_SCALE_EUR_PER_MWH = 100.0
REWARD_SCALE_EUR_PER_MWH = 100.0
_POWER_TOLERANCE_MW = 1e-9

ActionProvider = Callable[[tuple[float, ...]], Sequence[float]]
TransitionConsumer = Callable[[LearningTransition], None]


def calculate_load_base(residual_loads_mw: Sequence[float]) -> float:
    """Return the fixed signed-load normalization base for the training split."""

    values = tuple(float(value) for value in residual_loads_mw)
    if not values or any(not isfinite(value) for value in values):
        raise InputValidationError(
            "Training residual-load forecasts must be a non-empty finite sequence."
        )
    return max(1.0, max(abs(value) for value in values))


def build_learning_observation(
    *,
    delivery_start: datetime,
    residual_load_forecasts_mw: Mapping[datetime, float],
    price_forecasts_eur_per_mwh: Mapping[datetime, float],
    historical_clearing_prices_eur_per_mwh: Sequence[float],
    previous_power_mw: float,
    rated_power_mw: float,
    marginal_cost_eur_per_mwh: float,
    load_base_mw: float,
    price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
) -> tuple[float, ...]:
    """Build the documented 38-vector, rejecting incomplete future windows."""

    if rated_power_mw <= 0 or load_base_mw <= 0:
        raise ValueError("Rated power and load base must be positive.")
    future_times = tuple(
        delivery_start + timedelta(hours=offset) for offset in range(FORECAST_HOURS)
    )
    try:
        residual = [float(residual_load_forecasts_mw[time]) for time in future_times]
    except KeyError as exc:
        raise InputValidationError(
            "Residual-load forecast is missing future hour "
            f"{exc.args[0].isoformat(sep=' ')}."
        ) from exc
    try:
        prices = [float(price_forecasts_eur_per_mwh[time]) for time in future_times]
    except KeyError as exc:
        raise InputValidationError(
            f"Price forecast is missing future hour {exc.args[0].isoformat(sep=' ')}."
        ) from exc
    history = [float(value) for value in historical_clearing_prices_eur_per_mwh]
    if len(history) > FORECAST_HOURS:
        history = history[-FORECAST_HOURS:]
    history = [0.0] * (FORECAST_HOURS - len(history)) + history
    raw_values = [
        *residual,
        *prices,
        *history,
        float(previous_power_mw),
        float(marginal_cost_eur_per_mwh),
    ]
    if any(not isfinite(value) for value in raw_values):
        raise InputValidationError("Learning observation contains a non-finite value.")
    observation = (
        *(value / load_base_mw for value in residual),
        *(value / price_scale_eur_per_mwh for value in prices),
        *(value / price_scale_eur_per_mwh for value in history),
        previous_power_mw / rated_power_mw,
        marginal_cost_eur_per_mwh / price_scale_eur_per_mwh,
    )
    assert len(observation) == OBSERVATION_SIZE
    return tuple(float(value) for value in observation)


def _validated_learning_action(action: Sequence[float]) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (ACTION_SIZE,):
        raise ValueError(f"Learning action must have shape ({ACTION_SIZE},).")
    if not np.isfinite(values).all():
        raise ValueError("Learning action must contain only finite values.")
    return values


def clip_learning_action(action: Sequence[float]) -> tuple[float, float]:
    """Validate and clip the executed two-dimensional action."""

    values = _validated_learning_action(action)
    clipped = np.clip(values, -1.0, 1.0)
    return float(clipped[0]), float(clipped[1])


def learning_offers(
    *,
    plant: PowerPlant,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power_mw: float,
    marginal_cost_eur_per_mwh: float,
    action: Sequence[float],
    enforce_action_bounds: bool = True,
    price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
) -> tuple[SupplyOffer, ...]:
    """Map an unsorted action to independently clearable minimum/flexible offers."""

    if enforce_action_bounds:
        executed_action = clip_learning_action(action)
    else:
        values = _validated_learning_action(action)
        executed_action = float(values[0]), float(values[1])
    if available_power_mw < plant.min_power_mw - _POWER_TOLERANCE_MW:
        return ()
    duration_hours = hours_between(delivery_start, delivery_end)
    if duration_hours <= 0:
        raise ValueError("Learning offer duration must be positive.")
    low_price, high_price = sorted(
        value * price_scale_eur_per_mwh for value in executed_action
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
    """Return the EOM price, or None when no accepted supply order set it."""

    price = market_result.clearing_price_eur_per_mwh
    if (
        price is None
        or not isfinite(price)
        or not any(
            item.accepted_power_mw > _POWER_TOLERANCE_MW
            for item in market_result.offers
        )
    ):
        return None
    return price


def learning_step_result(
    *,
    market_result: MarketClearingResult,
    plant: PowerPlant,
    action: tuple[float, float],
    available_power_mw: float,
) -> LearningStepResult:
    """Calculate profit, regret shaping, and normalized reward after clearing."""

    price = _valid_clearing_price(market_result)
    if price is None:
        return LearningStepResult(
            delivery_start=market_result.delivery_start,
            action=action,
            available_power_mw=available_power_mw,
            accepted_power_mw=0.0,
            clearing_price_eur_per_mwh=0.0,
            profit_eur=0.0,
            regret_eur=0.0,
            reward=0.0,
        )
    if available_power_mw < plant.min_power_mw - _POWER_TOLERANCE_MW:
        return LearningStepResult(
            delivery_start=market_result.delivery_start,
            action=action,
            available_power_mw=available_power_mw,
            accepted_power_mw=0.0,
            clearing_price_eur_per_mwh=price,
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
        residual_load_forecasts_mw: Mapping[datetime, float],
        price_forecasts_eur_per_mwh: Mapping[datetime, float],
        load_base_mw: float,
        transition_consumer: TransitionConsumer | None = None,
        enforce_action_bounds: bool = True,
        price_scale_eur_per_mwh: float = PRICE_SCALE_EUR_PER_MWH,
    ) -> None:
        self.plant = plant
        self.action_provider = action_provider
        self.residual_load_forecasts_mw = residual_load_forecasts_mw
        self.price_forecasts_eur_per_mwh = price_forecasts_eur_per_mwh
        self.load_base_mw = load_base_mw
        self.transition_consumer = transition_consumer
        self.enforce_action_bounds = enforce_action_bounds
        self.price_scale_eur_per_mwh = price_scale_eur_per_mwh
        self._pending: dict[datetime, _PendingLearningStep] = {}
        self._transitions: list[LearningTransition] = []
        self._emitted_starts: set[datetime] = set()

    def _emit_transition(
        self,
        *,
        delivery_start: datetime,
        next_state: tuple[float, ...],
        done: bool,
    ) -> None:
        """Publish one completed transition exactly once."""

        if delivery_start in self._emitted_starts:
            return
        pending = self._pending[delivery_start]
        if pending.result is None:
            raise RuntimeError(
                f"Learning reward was not recorded before the next action for "
                f"{delivery_start!s}."
            )
        transition = LearningTransition(
            delivery_start=delivery_start,
            state=pending.state,
            action=pending.action,
            reward=pending.result.reward,
            next_state=next_state,
            done=done,
        )
        self._transitions.append(transition)
        self._emitted_starts.add(delivery_start)
        if self.transition_consumer is not None:
            self.transition_consumer(transition)

    def _emit_previous_transition(
        self, delivery_start: datetime, state: tuple[float, ...]
    ) -> None:
        """Complete the preceding step before requesting the current action."""

        pending_starts = sorted(self._pending.keys() - self._emitted_starts)
        if not pending_starts:
            return
        previous_start = delivery_start - timedelta(hours=1)
        if pending_starts != [previous_start]:
            raise InputValidationError(
                "Learning delivery products must form a continuous hourly episode."
            )
        self._emit_transition(
            delivery_start=previous_start,
            next_state=state,
            done=False,
        )

    def offers_for_product(
        self,
        *,
        delivery_start: datetime,
        delivery_end: datetime,
        available_power_mw: float,
        marginal_cost_eur_per_mwh: float,
        scheduled_results: Mapping[object, MarketClearingResult],
    ) -> tuple[SupplyOffer, ...]:
        earlier = sorted(
            (
                result
                for result in scheduled_results.values()
                if result.delivery_start < delivery_start
            ),
            key=lambda result: result.delivery_start,
        )
        history = [
            float(_valid_clearing_price(result) or 0.0)
            for result in earlier[-FORECAST_HOURS:]
        ]
        previous_start = delivery_start - timedelta(hours=1)
        previous = next(
            (
                result
                for result in reversed(earlier)
                if result.delivery_start == previous_start
            ),
            None,
        )
        previous_power_mw = 0.0
        if previous is not None:
            previous_power_mw = sum(
                item.accepted_power_mw
                for item in previous.offers
                if item.offer.unit_name == self.plant.name
            )
        state = build_learning_observation(
            delivery_start=delivery_start,
            residual_load_forecasts_mw=self.residual_load_forecasts_mw,
            price_forecasts_eur_per_mwh=self.price_forecasts_eur_per_mwh,
            historical_clearing_prices_eur_per_mwh=history,
            previous_power_mw=previous_power_mw,
            rated_power_mw=self.plant.max_power_mw,
            marginal_cost_eur_per_mwh=marginal_cost_eur_per_mwh,
            load_base_mw=self.load_base_mw,
            price_scale_eur_per_mwh=self.price_scale_eur_per_mwh,
        )
        self._emit_previous_transition(delivery_start, state)
        provided_action = self.action_provider(state)
        if self.enforce_action_bounds:
            action = clip_learning_action(provided_action)
        else:
            values = _validated_learning_action(provided_action)
            action = float(values[0]), float(values[1])
        self._pending[delivery_start] = _PendingLearningStep(
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
            enforce_action_bounds=self.enforce_action_bounds,
            price_scale_eur_per_mwh=self.price_scale_eur_per_mwh,
        )

    def record_result(self, market_result: MarketClearingResult) -> None:
        pending = self._pending.get(market_result.delivery_start)
        if pending is None:
            return
        pending.result = learning_step_result(
            market_result=market_result,
            plant=self.plant,
            action=pending.action,
            available_power_mw=pending.available_power_mw,
        )

    def finalize(
        self,
    ) -> tuple[tuple[LearningStepResult, ...], tuple[LearningTransition, ...]]:
        starts = sorted(self._pending)
        for start in starts:
            pending = self._pending[start]
            if pending.result is None:
                raise RuntimeError(f"Learning result was not recorded for {start!s}.")
        remaining_starts = sorted(self._pending.keys() - self._emitted_starts)
        if remaining_starts:
            if len(remaining_starts) != 1 or remaining_starts[0] != starts[-1]:
                raise InputValidationError(
                    "Learning delivery products must form a continuous hourly episode."
                )
            self._emit_transition(
                delivery_start=remaining_starts[0],
                next_state=(0.0,) * OBSERVATION_SIZE,
                done=True,
            )
        steps = tuple(self._pending[start].result for start in starts)
        assert all(step is not None for step in steps)
        return tuple(step for step in steps if step is not None), tuple(
            self._transitions
        )
