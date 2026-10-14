"""一次仿真运行的完整结果汇总。"""

from __future__ import annotations
from dataclasses import dataclass
from .clearing import MarketClearingResult
from .household import HouseholdDispatchResult, HouseholdFlexibilityResult
from .industry import (
    IndustrialDispatchResult,
    IndustrialFlexibilityResult,
    IndustrialOptimizationWindowResult,
)
from .learning import LearningStepResult, LearningTransition
from .market import MarketSettings
from .storage import StorageDispatchResult


@dataclass(frozen=True)
class SimulationResult:
    """一次仿真运行的完整内存结果。"""
    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
    storage_results: tuple[StorageDispatchResult, ...] = ()
    household_results: tuple[HouseholdDispatchResult, ...] = ()
    household_flexibility_results: tuple[HouseholdFlexibilityResult, ...] = ()
    industry_results: tuple[IndustrialDispatchResult, ...] = ()
    industry_flexibility_results: tuple[IndustrialFlexibilityResult, ...] = ()
    industry_optimization_windows: tuple[IndustrialOptimizationWindowResult, ...] = ()
    learning_steps: tuple[LearningStepResult, ...] = ()
    learning_transitions: tuple[LearningTransition, ...] = ()
    learning_load_base_mw: float | None = None
