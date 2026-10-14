"""居民建筑参与者参数，以及其运行计划、调度结果与灵活性上下界。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


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
