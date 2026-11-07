"""工业钢铁厂参与者参数，以及其生产计划、调度结果、灵活性与滚动优化窗口结果。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..utils import hours_between


@dataclass(frozen=True)
class IndustrialDevice:
    """工业钢铁厂内部的单个物理设备。"""
    technology: str
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


@dataclass(frozen=True)
class IndustrialUnit:
    """钢铁厂参与者参数，包含电解槽、直接还原装置和电弧炉。"""
    name: str
    operator: str
    cost_tolerance_percent: float
    demand_t: float
    load_profile_deviation: float
    look_ahead: timedelta
    commit_horizon: timedelta
    forecast_price_column: str
    electrolyser: IndustrialDevice
    dri_plant: IndustrialDevice
    eaf: IndustrialDevice


@dataclass(frozen=True)
class IndustrialResult:
    """单个交付时段的工业承诺计划与执行结果；未交付时实际值为 None。"""
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
    forecast_cost_eur: float
    minimum_power_mw: float
    maximum_power_mw: float
    actual_grid_power_mw: float | None = None
    actual_steel_output_t: float | None = None

    @property
    def planned_grid_power_mw(self) -> float:
        return self.electrolyser_power_mw + self.dri_power_mw + self.eaf_power_mw

    @property
    def planned_energy_mwh(self) -> float:
        return self.planned_grid_power_mw * hours_between(self.delivery_start, self.delivery_end)


@dataclass(frozen=True)
class Industrial_Optimization_Window_Result:
    """单个滚动优化窗口的基准可变成本及灵活性成本上限，不含负荷偏差惩罚。"""
    window_id: str
    unit_name: str
    optimization_start: datetime
    optimization_end: datetime
    commit_start: datetime
    commit_end: datetime
    baseline_variable_cost_eur: float
    maximum_flexible_variable_cost_eur: float
