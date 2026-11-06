"""居民建筑参与者参数，以及其运行计划、调度结果与灵活性上下界。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from ..utils import hours_between


@dataclass(frozen=True)
class HouseholdUnit:
    """配备热泵和内部电池的单个建筑参与者。"""
    name: str
    operator: str
    cost_tolerance_percent: float
    heat_pump_max_power_mw: float
    heat_pump_ramp_up_mw: float
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
    planned_grid_power_mw: float
    planned_heat_pump_power_mw: float
    planned_battery_discharge_power_mw: float


@dataclass(frozen=True)
class HouseholdDispatchResult:
    """对应买单出清后建筑的实际运行结果。"""
    plan: HouseholdPlan
    heat_pump_power_mw: float
    battery_charge_power_mw: float
    battery_discharge_power_mw: float
    soc_before: float
    soc_after: float
    unmet_heat_mwh_th: float
    # 基准仿真不评估灵活性，用 None 明确表示没有上下界结果。
    grid_power_bounds_mw: tuple[float, float] | None


@dataclass(frozen=True)
class HouseholdResult:
    """两次仿真完成后的居民结果，引用实际与无电池基准调度，并保存各自结算。"""
    dispatch: HouseholdDispatchResult
    baseline: HouseholdDispatchResult
    accepted_grid_energy_mwh: float
    payment_eur: float
    baseline_grid_energy_mwh: float
    baseline_payment_eur: float

    @property
    def forecast_saving_eur(self) -> float:
        plan = self.dispatch.plan
        shift = self.baseline.plan.planned_grid_power_mw - plan.planned_grid_power_mw
        shifted_energy = shift * hours_between(plan.delivery_start, plan.delivery_end)
        return plan.forecast_price_eur_per_mwh * shifted_energy

    @property
    def actual_price_eur_per_mwh(self) -> float | None:
        """实际支付 ÷ 实际成交电量；未成交时没有成交均价。"""
        if self.accepted_grid_energy_mwh <= 0:
            return None
        return self.payment_eur / self.accepted_grid_energy_mwh

    @property
    def actual_saving_eur(self) -> float:
        """基准方案实际支付 − 优化方案实际支付。"""
        return self.baseline_payment_eur - self.payment_eur
