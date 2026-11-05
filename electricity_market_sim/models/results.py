"""一次仿真运行的完整结果汇总。"""

from __future__ import annotations
from dataclasses import dataclass
from .clearing import MarketClearingResult
from .household import (
    HouseholdResult,
)
from .industry import (
    IndustrialDispatchResult,
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
    household_results: tuple[HouseholdResult, ...] = ()
    industry_results: tuple[IndustrialDispatchResult, ...] = ()
    industry_optimization_windows: tuple[IndustrialOptimizationWindowResult, ...] = ()
    learning_steps: tuple[LearningStepResult, ...] = ()
    learning_transitions: tuple[LearningTransition, ...] = ()
