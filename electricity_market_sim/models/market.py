"""能量市场配置、市场开市安排与完整仿真结果。"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .clearing import MarketClearingResult
from .household import HouseholdResult
from .industry import IndustrialResult, Industrial_Optimization_Window_Result
from .learning import LearningConfig, LearningStepResult, LearningTransition
from .storage import StorageDispatchResult


@dataclass(frozen=True)
class MarketSettings:
    """将 EOM（能量市场）参数集中封装为一个对象，便于跨模块传递和共享。
    """
    start: datetime
    end: datetime
    time_step: timedelta
    opening_frequency: timedelta
    product_duration: timedelta
    product_count: int
    first_delivery: timedelta
    maximum_bid_price: float
    minimum_bid_price: float
    market_mechanism: str
    market_id: str = "EOM"
    exchange_units_file: str | None = None
    industrial_dsm_units_file: str | None = None
    additional_fields: frozenset[str] = field(default_factory=frozenset)
    learning_config: LearningConfig | None = None
    seed: int | None = None


@dataclass(frozen=True)
class MarketOpening:
    """一次市场开市时同时提供的所有交付产品。"""
    opening_time: datetime # 本次市场开盘/开市的时刻，时间点。
    products: tuple[tuple[datetime, datetime], ...] # 本次开市可交易的各交付时段，比如 10:00 到 11:00


@dataclass(frozen=True)
class SimulationResult:
    """一次仿真运行的完整内存结果。"""
    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
    storage_results: tuple[StorageDispatchResult, ...] = ()
    household_results: tuple[HouseholdResult, ...] = ()
    industry_results: tuple[IndustrialResult, ...] = ()
    industry_optimization_windows: tuple[Industrial_Optimization_Window_Result, ...] = ()
    learning_steps: tuple[LearningStepResult, ...] = ()
    learning_transitions: tuple[LearningTransition, ...] = ()
