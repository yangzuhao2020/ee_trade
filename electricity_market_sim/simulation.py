"""统一协调所有受支持的电力市场仿真。"""

from datetime import datetime
from pathlib import Path
from .config import load_market_settings
from .errors import InputValidationError
from .learning import ActionProvider, TransitionConsumer
from .models import MarketOpening, MarketSettings, SimulationResult
from .reporting import write_results
from .simulators.v5 import simulate_v5_market
from .simulators.household_market import simulate_household_market
from .simulators.eom_scenarios import (
    simulate_v1_v2_market,
    simulate_v4_market,
)


class LearningEpisodeRunner:
    """使用同一个固定的观测归一化基准运行多轮仿真。"""
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
        """返回用于保存到检查点或从检查点恢复的归一化基准。"""
        return self._load_base_mw

    def run_episode(
        self,
        action_provider: ActionProvider,
        transition_consumer: TransitionConsumer | None = None,
        *,
        enforce_action_bounds: bool = True,
    ) -> SimulationResult:
        """运行一轮仿真，仅在首次调用时计算各轮共用的归一化基准。"""

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
    """生成候选开市安排，保留完整的交易产品集合。
    首个产品的交付开始时间位于仿真范围内时，纳入候选开市。
    所有产品及其所需输入是否齐备，会在后续检查；因此第二版的日前开市
    可能被整体跳过，并给出说明原因的警告。
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
    # 多产品开市保留首个产品的交付开始时间恰好等于 end 的边界开市。
    # 预检查会整体跳过该次开市，并按要求发出产品不完整的警告。
    # 第一版的单产品流程继续使用原有的严格边界。
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
    """计算一个受支持场景的仿真结果，不写入结果文件。"""

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    openings = tuple(market_openings(settings)) # 开市报价的时间。
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
    """使用传入的策略和经验回放回调，运行第五版的一轮仿真。"""
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
    """运行仿真并保存 CSV 结果，随后可按需绘图。"""

    result = simulate(
        input_dir=input_dir,
        scenario=scenario,
        learning_action_provider=learning_action_provider,
        learning_transition_consumer=learning_transition_consumer,
        learning_load_base_mw=learning_load_base_mw,
    )
    write_results(Path(output_dir), result)
    return result
