"""工业钢铁厂参与者参数，以及其生产计划、调度结果、灵活性与滚动优化窗口结果。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta


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
