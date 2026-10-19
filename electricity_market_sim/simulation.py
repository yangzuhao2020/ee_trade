"""统一协调所有受支持的电力市场仿真。"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from .config import load_market_settings
from .errors import InputValidationError
from .deep_learning.learning import ActionProvider, TransitionConsumer
from .models import LEARNING_PERIOD_NAMES, MarketOpening, MarketSettings, SimulationResult
from .reporting import write_results
from .simulators.v5 import simulate_v5_market
from .simulators.household_market import simulate_household_market
from .simulators.eom import simulate_v1_v2_market
from .simulators.industry_market import simulate_v4_market


def learning_period_settings(settings: MarketSettings, period: str) -> MarketSettings:
    """把按交付时间配置的学习时段换算为市场的开市起止时间。
    第一次开市提前 first_delivery，使首个交付时段恰好从 period.start 开始；
    最后一个交付时段在 period.end 结束。顶层 start_date 和 EOM.start_date 不再生效。
    """
    if period not in LEARNING_PERIOD_NAMES:
        raise InputValidationError(
            "第五版仿真必须指定 learning period：train、validation 或 test。"
        )
    if settings.learning_config is None:
        raise InputValidationError("learning period 需要 learning_config。")
    try:
        delivery = settings.learning_config.period(period)
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
    return replace(
        settings, start=delivery.start - settings.first_delivery, end=delivery.end
    )


class LearningEpisodeRunner:
    """用同一输入目录、场景和数据时段重复运行回合。"""
    def __init__(
        self,
        input_dir: str | Path,
        scenario: str = "base",
        *,
        period: str,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.scenario = scenario
        self.period = period

    def run_episode(
        self,
        action_provider: ActionProvider,
        transition_consumer: TransitionConsumer | None = None,
        *,
        enforce_action_bounds: bool = True,
    ) -> SimulationResult:
        """重新初始化市场，使用传入的策略运行一轮仿真。"""

        return simulate_learning_episode(
            self.input_dir,
            action_provider,
            scenario=self.scenario,
            transition_consumer=transition_consumer,
            enforce_action_bounds=enforce_action_bounds,
            period=self.period,
        )


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
    learning_enforce_action_bounds: bool = True,
    learning_period: str | None = None,
) -> SimulationResult:
    """计算一个受支持场景的仿真结果，不写入结果文件。
    第五版必须通过 learning_period 指定 train、validation 或 test 时段。
    """

    input_path = Path(input_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    is_learning = (settings.learning_config is not None and settings.learning_config.learning_mode)
    
    if is_learning: # 强化学习部分
        if learning_action_provider is None:
            raise InputValidationError("Version 5 simulate() requires a learning_action_provider.")
        period_settings = learning_period_settings(settings, learning_period)
        return simulate_v5_market(
            input_path,
            period_settings,
            tuple(market_openings(period_settings)),
            learning_action_provider=learning_action_provider,
            learning_transition_consumer=learning_transition_consumer,
            learning_enforce_action_bounds=learning_enforce_action_bounds,
        ) # 第五版

    openings = tuple(market_openings(settings)) # 开市报价的时间。
    if settings.industrial_dsm_units_file is not None:
        return simulate_v4_market(input_path, settings, openings) # 第四版本
    elif settings.market_mechanism == "pay_as_bid":
        return simulate_household_market(input_path, settings, openings) # 第三版本
    else:
        return simulate_v1_v2_market(input_path, settings, openings) # 第一、二版本


def simulate_learning_episode(
    input_dir: str | Path,
    action_provider: ActionProvider,
    scenario: str = "base",
    transition_consumer: TransitionConsumer | None = None,
    enforce_action_bounds: bool = True,
    *,
    period: str,
) -> SimulationResult:
    """使用传入的策略和经验回放回调，在指定数据时段运行第五版的一轮仿真。"""
    return simulate(
        input_dir,
        scenario=scenario,
        learning_action_provider=action_provider,
        learning_transition_consumer=transition_consumer,
        learning_enforce_action_bounds=enforce_action_bounds,
        learning_period=period,
    )


def run_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    learning_action_provider: ActionProvider | None = None,
    learning_transition_consumer: TransitionConsumer | None = None,
    learning_period: str | None = None,
) -> SimulationResult:
    """运行仿真并保存 CSV 结果，随后可按需绘图。"""

    result = simulate(
        input_dir=input_dir,
        scenario=scenario,
        learning_action_provider=learning_action_provider,
        learning_transition_consumer=learning_transition_consumer,
        learning_period=learning_period,
    )
    write_results(Path(output_dir), result)
    return result
