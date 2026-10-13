"""电力市场仿真的配置、参与者参数及运行结果数据结构。
包括：
- 市场与学习配置：MarketSettings、LearningConfig
- 市场开市安排：MarketOpening
- 发电机组、需求、储能、居民及工业参与者的参数
- 储能、居民、工业及学习过程的运行结果
- 完整仿真结果：SimulationResult
还提供边际成本、储能量与 SOC 换算、净现金流及工业灵活性幅度等派生计算。
市场订单、成交记录和单时段出清结果的数据结构定义在 market_models.py 中。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .market_models import MarketClearingResult


@dataclass(frozen=True)
class LearningConfig:
    """经校验的 Version 5 单智能体 MATD3 学习配置。"""
    learning_mode: bool
    algorithm: str
    learning_rate: float
    training_episodes: int
    initial_experience_episodes: int
    replay_buffer_size: int
    batch_size: int
    gamma: float
    train_frequency_steps: int
    gradient_steps: int
    validation_interval: int
    exploration_noise_std: float
    noise_sigma: float
    noise_scale: float
    noise_dt: float
    action_noise_schedule: str
    tau: float
    policy_delay: int
    target_policy_noise: float
    target_noise_clip: float
    actor_architecture: str = "mlp"
    device: str = "cpu"
    continue_learning: bool = False
    trained_policies_save_path: str | None = None
    trained_policies_load_path: str | None = None
    max_bid_price: float = 100.0


@dataclass(frozen=True)
class MarketSettings:
    """简单与复杂 V2 市场开市共用的 EOM（能量市场）配置。"""
    start: datetime
    end: datetime
    time_step: timedelta
    opening_frequency: timedelta
    # 加载时必填并校验；当前事件调度尚未使用该字段。
    opening_duration: timedelta
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
class DemandUnit:
    """EOM 需求侧参与者，可选择根据需求弹性输入参数生成。"""
    name: str
    operator: str
    bidding_strategy: str = "demand_energy_naive"
    profile_column: str | None = None
    max_power_mw: float | None = None
    elasticity: float | None = None
    elasticity_model: str | None = None
    max_price_eur_per_mwh: float | None = None
    num_bids: int | None = None
    price_eur_per_mwh: float | None = None

    @property
    def is_elastic(self) -> bool:
        return self.bidding_strategy == "demand_energy_heuristic_elastic"


@dataclass(frozen=True)
class ExchangeUnit:
    """代表计划进口与出口的唯一虚拟参与者。"""
    name: str
    operator: str
    price_import_eur_per_mwh: float
    price_export_eur_per_mwh: float


@dataclass(frozen=True)
class ExchangeSchedule:
    """单个交付小时的计划进口与出口功率（均为正值）。"""
    import_power_mw: float
    export_power_mw: float


@dataclass(frozen=True)
class PowerPlant:
    """可调度发电机组及其报价策略所用的参数。"""
    name: str
    operator: str
    technology: str
    bidding_strategy: str
    fuel_type: str
    emission_factor: float
    max_power_mw: float
    min_power_mw: float
    efficiency: float
    additional_cost_eur_per_mwh: float
    # 当 ``powerplant_units.csv`` 中缺少对应列时，
    # 以下可选 V2 参数沿用 V2 示例的默认值。
    start_cost_eur: float = 0.0
    min_operating_time_hours: float = 1.0
    # 为保持兼容性而加载并校验；最小停机时间约束并未实际执行。
    min_down_time_hours: float = 1.0

    def marginal_cost(self, fuel_prices: dict[str, float]) -> float:
        """返回与 ASSUME 一致的可变边际成本。
        输入的排放因子按单位燃料投入计，因此燃料成本和 CO2 成本都要除以发电效率。
        """
        # 所提供的 V2 示例中，波动性可再生能源没有燃料价格列；
        # 按惯例其燃料成本分量为零。
        fuel_price = fuel_prices.get(self.fuel_type, 0.0)
        co2_price = fuel_prices["co2"]
        co2_cost = co2_price * self.emission_factor
        fuel_and_co2_cost = (fuel_price + co2_cost) / self.efficiency
        return fuel_and_co2_cost + self.additional_cost_eur_per_mwh


@dataclass(frozen=True)
class StorageUnit:
    """储能参与者及其 EOM 策略所用的物理约束参数。"""
    name: str
    operator: str
    technology: str
    bidding_strategy: str
    max_power_charge_mw: float
    max_power_discharge_mw: float
    efficiency_charge: float
    efficiency_discharge: float
    min_soc: float
    max_soc: float
    capacity_mwh: float
    initial_soc: float
    additional_cost_charge_eur_per_mwh: float
    additional_cost_discharge_eur_per_mwh: float
    natural_inflow_mw: float = 0.0

    @property
    def min_energy_mwh(self) -> float:
        return self.min_soc * self.capacity_mwh

    @property
    def max_energy_mwh(self) -> float:
        return self.max_soc * self.capacity_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.capacity_mwh


@dataclass(frozen=True)
class HouseholdUnit:
    """配备热泵和内部电池的单个建筑参与者。"""
    name: str
    operator: str
    bidding_strategy: str
    objective: str
    flexibility_measure: str
    cost_tolerance_percent: float
    is_prosumer: bool
    fixed_power_mw: float
    heat_pump_max_power_mw: float
    heat_pump_min_power_mw: float
    heat_pump_ramp_up_mw: float
    heat_pump_ramp_down_mw: float
    cop: float
    battery_capacity_mwh: float
    battery_min_soc: float
    battery_max_soc: float
    battery_initial_soc: float
    battery_efficiency_charge: float
    battery_efficiency_discharge: float
    battery_max_charge_power_mw: float
    battery_max_discharge_power_mw: float
    battery_ramp_up_mw: float
    battery_ramp_down_mw: float
    battery_loss_rate: float = 0.0
    bid_price_eur_per_mwh: float = 3000.0

    @property
    def initial_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_initial_soc

    @property
    def min_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_min_soc

    @property
    def max_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_max_soc


@dataclass(frozen=True)
class HouseholdPlan:
    """居民用户基于预测、针对单个交付产品制定的运行计划。"""
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    planned_heat_pump_power_mw: float
    planned_battery_discharge_power_mw: float


@dataclass(frozen=True)
class HouseholdDispatchResult:
    """对应买单出清后建筑的实际运行结果。"""
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    heat_pump_power_mw: float
    battery_charge_power_mw: float
    battery_discharge_power_mw: float
    soc_before: float
    soc_after: float
    unmet_electricity_mwh: float
    unmet_heat_mwh_th: float


@dataclass(frozen=True)
class HouseholdFlexibilityResult:
    """单个已计划居民产品在成本容忍度内的电网功率上下界。"""
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    minimum_grid_power_mw: float
    maximum_grid_power_mw: float


@dataclass(frozen=True)
class IndustrialDevice:
    """工业钢铁厂内部的单个物理设备。"""
    technology: str
    fuel_type: str
    max_power_mw: float
    min_power_mw: float
    ramp_up_mw: float
    ramp_down_mw: float
    efficiency: float = 0.0
    specific_dri_demand: float = 0.0
    specific_electricity_consumption: float = 0.0
    specific_hydrogen_consumption: float = 0.0
    specific_iron_ore_consumption: float = 0.0
    specific_lime_demand: float = 0.0
    lime_co2_factor: float = 0.0
    min_operating_time_hours: float = 0.0
    min_down_time_hours: float = 0.0


@dataclass(frozen=True)
class IndustrialUnit:
    """由三行设备数据组装而成的单个钢铁厂参与者。"""
    name: str
    operator: str
    bidding_strategy: str
    objective: str
    flexibility_measure: str
    cost_tolerance_percent: float
    demand_t: float
    load_profile_deviation: float
    horizon_mode: str
    look_ahead: timedelta
    commit_horizon: timedelta
    rolling_step: timedelta
    forecast_price_column: str
    electrolyser: IndustrialDevice
    dri_plant: IndustrialDevice
    eaf: IndustrialDevice


@dataclass(frozen=True)
class IndustrialPlan:
    """工业用户基于预测、针对单个交付产品制定的生产计划。"""
    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    electrolyser_power_mw: float
    hydrogen_output_mwh: float
    dri_power_mw: float
    dri_output_t: float
    eaf_power_mw: float
    planned_steel_output_t: float
    planned_grid_power_mw: float
    planned_energy_mwh: float
    forecast_cost_eur: float


@dataclass(frozen=True)
class IndustrialDispatchResult:
    """工业报价出清后的计划与实际钢铁生产结果。"""
    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    electrolyser_power_mw: float
    hydrogen_output_mwh: float
    dri_power_mw: float
    dri_output_t: float
    eaf_power_mw: float
    planned_steel_output_t: float
    planned_grid_power_mw: float
    planned_energy_mwh: float
    actual_grid_power_mw: float
    actual_steel_output_t: float
    forecast_cost_eur: float


@dataclass(frozen=True)
class IndustrialFlexibilityResult:
    """单个已承诺产品在成本容忍度内的逐点功率上下界。"""
    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    baseline_power_mw: float
    minimum_power_mw: float
    maximum_power_mw: float

    @property
    def flex_up_mw(self) -> float:
        return self.maximum_power_mw - self.baseline_power_mw

    @property
    def flex_down_mw(self) -> float:
        return self.baseline_power_mw - self.minimum_power_mw


@dataclass(frozen=True)
class IndustrialOptimizationWindowResult:
    """单个滚动优化窗口的成本合计（不重复计算）。"""
    window_id: str
    unit_name: str
    optimization_start: datetime
    optimization_end: datetime
    commit_start: datetime
    commit_end: datetime
    baseline_variable_cost_eur: float
    maximum_flexible_variable_cost_eur: float


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


@dataclass(frozen=True)
class LearningStepResult:
    """单个学习交付时段可供审计的动作、调度结果与奖励。"""
    delivery_start: datetime
    action: tuple[float, float]
    available_power_mw: float
    accepted_power_mw: float
    clearing_price_eur_per_mwh: float
    profit_eur: float
    regret_eur: float
    reward: float


@dataclass(frozen=True)
class LearningTransition:
    """一条经验回放缓冲区记录，使用未排序的实际执行动作。"""
    delivery_start: datetime
    state: tuple[float, ...]
    action: tuple[float, float]
    reward: float
    next_state: tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class StorageClearingContext:
    """单次联合复杂出清开市所需的储能物理数据。"""
    unit_name: str
    initial_energy_mwh: float
    min_energy_mwh: float
    max_energy_mwh: float
    efficiency_charge: float
    efficiency_discharge: float


@dataclass(frozen=True)
class StorageDispatchResult:
    """单个储能单元在单个产品上的成交调度结果及 SOC 变化。"""
    opening_time: datetime
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    operator: str
    technology: str
    energy_before_mwh: float
    energy_after_mwh: float
    capacity_mwh: float
    offered_charge_mwh: float
    accepted_charge_mwh: float
    charge_bid_price_eur_per_mwh: float | None
    offered_discharge_mwh: float
    accepted_discharge_mwh: float
    discharge_bid_price_eur_per_mwh: float | None
    clearing_price_eur_per_mwh: float
    charge_payment_eur: float
    discharge_revenue_eur: float
    additional_charge_cost_eur: float
    additional_discharge_cost_eur: float

    @property
    def soc_before(self) -> float:
        return self.energy_before_mwh / self.capacity_mwh

    @property
    def soc_after(self) -> float:
        return self.energy_after_mwh / self.capacity_mwh

    @property
    def net_cash_flow_eur(self) -> float:
        return (
            self.discharge_revenue_eur
            - self.charge_payment_eur
            - self.additional_charge_cost_eur
            - self.additional_discharge_cost_eur
        )
