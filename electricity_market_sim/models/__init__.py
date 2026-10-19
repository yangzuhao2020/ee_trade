"""电力市场仿真的配置、参与者参数及运行结果数据结构。
按领域划分为以下子模块：
- market：市场配置 MarketSettings 与开市安排 MarketOpening
- learning：学习配置 LearningConfig 及学习步结果、经验回放记录
- plants：可调度发电机组 PowerPlant
- demand：需求 DemandUnit 与进出口交换 ExchangeUnit、ExchangeSchedule
- storage：储能参数、联合出清上下文及调度结果
- household：居民参与者参数、运行计划、调度及灵活性结果
- industry：工业参与者参数、生产计划、调度、灵活性及滚动优化窗口结果
- orders：出清前的市场买单 DemandBid 与卖单 SupplyOffer
- clearing：出清后的订单结果、成交记录 Trade 及单时段出清结果 MarketClearingResult
- results：完整仿真结果 SimulationResult
还提供边际成本、储能量与 SOC 换算、净现金流、工业灵活性幅度、成交收支与利润等派生计算，
并校验成交电量平衡与结算一致性。
"""

from .clearing import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    MarketClearingResult,
    Trade,
    sum_cleared_demand,
)
from .demand import DemandUnit, ExchangeSchedule, ExchangeUnit
from .household import (
    HouseholdDispatchResult,
    HouseholdFlexibilityResult,
    HouseholdPlan,
    HouseholdUnit,
)
from .industry import (
    IndustrialDevice,
    IndustrialDispatchResult,
    IndustrialFlexibilityResult,
    IndustrialOptimizationWindowResult,
    IndustrialPlan,
    IndustrialUnit,
)
from .learning import (
    LEARNING_PERIOD_NAMES,
    LearningConfig,
    LearningPeriod,
    LearningStepResult,
    LearningTransition,
)
from .market import MarketOpening, MarketSettings
from .orders import DemandBid, SupplyOffer
from .plants import PowerPlant
from .results import SimulationResult
from .storage import StorageClearingContext, StorageDispatchResult, StorageUnit
