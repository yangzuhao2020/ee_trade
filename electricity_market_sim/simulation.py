"""Public orchestration for all supported electricity-market simulations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from .config import load_market_settings
from .errors import InputValidationError
from .learning import ActionProvider, TransitionConsumer
from .models import MarketOpening, MarketSettings, SimulationResult
from .reporting import write_results
from .simulators.v5 import simulate_v5_market
from .simulators.household_market import simulate_household_market
from .simulators.v1_v2 import simulate_v1_v2_market
from .simulators.v4 import simulate_v4_market

__all__ = [
    "LearningEpisodeRunner",
    "market_openings",
    "run_simulation",
    "simulate",
    "simulate_learning_episode",
]


class LearningEpisodeRunner:
    """Run multiple episodes with one immutable observation normalization base."""

    def __init__(
        self,
        input_dir: str | Path,
        scenario: str = "base",
        *,
        load_base_mw: float | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.scenario = scenario
        self._load_base_mw = load_base_mw

    @property
    def load_base_mw(self) -> float | None:
        """Return the base to persist in or restore from a checkpoint."""

        return self._load_base_mw

    def run_episode(
        self,
        action_provider: ActionProvider,
        transition_consumer: TransitionConsumer | None = None,
        *,
        enforce_action_bounds: bool = True,
    ) -> SimulationResult:
        """Run an episode, calculating the shared base only on the first call."""

        result = simulate_learning_episode(
            self.input_dir,
            action_provider,
            scenario=self.scenario,
            transition_consumer=transition_consumer,
            load_base_mw=self._load_base_mw,
            enforce_action_bounds=enforce_action_bounds,
        )
        result_base = result.learning_load_base_mw
        if result_base is None:
            raise RuntimeError("The learning episode did not produce a load base.")
        if self._load_base_mw is None:
            self._load_base_mw = result_base
        elif result_base != self._load_base_mw:
            raise RuntimeError("The learning load base changed between episodes.")
        return result


def market_openings(settings: MarketSettings) -> list[MarketOpening]:
    """Schedule candidate openings without silently shortening their products.

    A candidate is included while its first delivery begins inside the simulation
    horizon. Whether all products and supporting inputs exist is checked later,
    so a V2 day-ahead opening can be skipped whole with a useful warning.
    """

    openings: list[MarketOpening] = []
    if settings.market_mechanism == "pay_as_bid":
        opening_time = settings.start
        seen_products: set[tuple[datetime, datetime]] = set()
        
        while opening_time + settings.first_delivery < settings.end:
            first_start = opening_time + settings.first_delivery
            products = tuple(product for index in range(settings.product_count)
                if (product := (first_start + index * settings.product_duration,
                                first_start + (index + 1) * settings.product_duration,
                    ))[1]<= settings.end and product not in seen_products)
            
            if products:
                openings.append(MarketOpening(opening_time=opening_time, products=products))
                seen_products.update(products)
            
            opening_time += settings.opening_frequency
        return openings

    opening_time = settings.start
    # For a multi-product opening, retain the boundary opening whose first
    # delivery starts exactly at ``end``. Its preflight will deliberately skip
    # the whole opening and emit the required incomplete-product warning. The
    # one-product V1 path retains its established strict boundary.
    includes_end_boundary = settings.product_count > 1
    while opening_time + settings.first_delivery < settings.end or (
        includes_end_boundary and opening_time + settings.first_delivery == settings.end
    ):
        first_start = opening_time + settings.first_delivery
        products = tuple(
            (
                first_start + index * settings.product_duration,
                first_start + (index + 1) * settings.product_duration,
            )
            for index in range(settings.product_count)
        )
        openings.append(MarketOpening(opening_time=opening_time, products=products))
        opening_time += settings.opening_frequency
    return openings


def simulate(
    input_dir: str | Path,
    scenario: str = "base",
    learning_action_provider: ActionProvider | None = None,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_load_base_mw: float | None = None,
    learning_enforce_action_bounds: bool = True,
) -> SimulationResult:
    """Calculate one supported scenario without writing result files."""

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    openings = tuple(market_openings(settings))
    is_learning = (settings.learning_config is not None and settings.learning_config.learning_mode)
    
    if is_learning: # 强化学习部分
        if learning_action_provider is None:
            raise InputValidationError("Version 5 simulate() requires a learning_action_provider.")
        
        return simulate_v5_market(
            input_path,
            settings,
            openings,
            learning_action_provider=learning_action_provider,
            learning_transition_consumer=learning_transition_consumer,
            learning_load_base_mw=learning_load_base_mw,
            learning_enforce_action_bounds=learning_enforce_action_bounds,
        ) # 第五版

    if settings.industrial_dsm_units_file is not None:
        return simulate_v4_market(input_path, settings, openings) # 第四版本
    if settings.market_mechanism == "pay_as_bid":
        return simulate_household_market(input_path, settings, openings) # 第三版本
    
    return simulate_v1_v2_market(input_path, settings, openings) # 第一、二版本


def simulate_learning_episode(
    input_dir: str | Path,
    action_provider: ActionProvider,
    scenario: str = "base",
    transition_consumer: TransitionConsumer | None = None,
    load_base_mw: float | None = None,
    enforce_action_bounds: bool = True,
) -> SimulationResult:
    """Run one Version 5 episode with injected policy and replay callbacks."""

    return simulate(
        input_dir,
        scenario=scenario,
        learning_action_provider=action_provider,
        learning_transition_consumer=transition_consumer,
        learning_load_base_mw=load_base_mw,
        learning_enforce_action_bounds=enforce_action_bounds,
    )


def run_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    learning_action_provider: ActionProvider | None = None,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_load_base_mw: float | None = None,
) -> SimulationResult:
    """Run a simulation and persist the CSV results before optional plotting."""

    result = simulate(
        input_dir=input_dir,
        scenario=scenario,
        learning_action_provider=learning_action_provider,
        learning_transition_consumer=learning_transition_consumer,
        learning_load_base_mw=learning_load_base_mw,
    )
    write_results(Path(output_dir), result)
    return result
