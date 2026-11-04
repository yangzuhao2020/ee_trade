"""居民建筑参与者参数，以及其运行计划、调度结果与灵活性上下界。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HouseholdUnit:
    """配备热泵和内部电池的单个建筑参与者。"""
    name: str
    operator: str
    cost_tolerance_percent: float
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
class HouseholdCostResult:
    """单个居民产品的实际购电支付，并与不使用电池的基准方案对比。

    基准方案以相同输入重新完整仿真，电池不参与运行（充放电功率和自放电率均为 0）；
    两者都按各自买单的实际成交电量和逐笔成交价结算。基准方案的缺供量和 SOC
    用于检查两者是否提供了同等服务，见 ``summarize_household_savings``。
    """
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    accepted_grid_energy_mwh: float
    payment_eur: float
    baseline_grid_energy_mwh: float
    baseline_payment_eur: float
    forecast_saving_eur: float
    baseline_unmet_electricity_mwh: float
    baseline_unmet_heat_mwh_th: float
    baseline_soc_after: float

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


_COMPARISON_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HouseholdSavingSummary:
    """单个居民在整个仿真期内与基准方案的支付对比及对照有效性。

    净差为“优化方案 − 基准方案”的合计，仅供参考；不一致量为各时段
    |优化方案 − 基准方案| 之和，不同时段的差异不会相互抵消。只有每个时段的
    缺电量、缺热量都与基准相同，且优化方案期末 SOC 不低于基准方案时，支付差额
    才是同等服务下的节省（comparable=True）。剩余电量差异不折算为金额。
    """
    unit_name: str
    baseline_payment_eur: float
    payment_eur: float
    unmet_electricity_difference_mwh: float
    unmet_heat_difference_mwh_th: float
    unmet_electricity_mismatch_mwh: float
    unmet_heat_mismatch_mwh_th: float
    final_soc_difference: float

    @property
    def actual_saving_eur(self) -> float:
        return self.baseline_payment_eur - self.payment_eur

    @property
    def actual_saving_share(self) -> float | None:
        if not self.baseline_payment_eur:
            return None
        return self.actual_saving_eur / self.baseline_payment_eur

    @property
    def comparable(self) -> bool:
        return (
            self.unmet_electricity_mismatch_mwh <= _COMPARISON_TOLERANCE
            and self.unmet_heat_mismatch_mwh_th <= _COMPARISON_TOLERANCE
            and self.final_soc_difference >= -_COMPARISON_TOLERANCE
        )

    def difference_lines(self) -> tuple[str, ...]:
        """三项对照差异的中文说明，供图表使用。"""
        return (
            f"缺电量逐时段不一致 {self.unmet_electricity_mismatch_mwh:.6f} MWh"
            f"（净差 {self.unmet_electricity_difference_mwh:+.6f}）",
            f"缺热量逐时段不一致 {self.unmet_heat_mismatch_mwh_th:.6f} MWh_th"
            f"（净差 {self.unmet_heat_difference_mwh_th:+.6f}）",
            f"期末 SOC 差 {self.final_soc_difference * 100:+.2f} 个百分点",
        )


def summarize_household_savings(
    dispatch_results: tuple[HouseholdDispatchResult, ...],
    cost_results: tuple[HouseholdCostResult, ...],
) -> tuple[HouseholdSavingSummary, ...]:
    """按居民汇总支付，以及缺供量和期末 SOC 与基准方案的差异。"""
    dispatch = {
        (result.unit_name, result.delivery_start): result for result in dispatch_results
    }
    grouped: dict[str, list[HouseholdCostResult]] = {}
    for cost in cost_results:
        grouped.setdefault(cost.unit_name, []).append(cost)
    summaries = []
    for unit_name in sorted(grouped):
        costs = sorted(grouped[unit_name], key=lambda cost: cost.delivery_end)
        rows = [dispatch[(unit_name, cost.delivery_start)] for cost in costs]
        electricity = [
            row.unmet_electricity_mwh - cost.baseline_unmet_electricity_mwh
            for row, cost in zip(rows, costs)
        ]
        heat = [
            row.unmet_heat_mwh_th - cost.baseline_unmet_heat_mwh_th
            for row, cost in zip(rows, costs)
        ]
        summaries.append(
            HouseholdSavingSummary(
                unit_name=unit_name,
                baseline_payment_eur=sum(cost.baseline_payment_eur for cost in costs),
                payment_eur=sum(cost.payment_eur for cost in costs),
                unmet_electricity_difference_mwh=sum(electricity),
                unmet_heat_difference_mwh_th=sum(heat),
                unmet_electricity_mismatch_mwh=sum(abs(value) for value in electricity),
                unmet_heat_mismatch_mwh_th=sum(abs(value) for value in heat),
                final_soc_difference=rows[-1].soc_after - costs[-1].baseline_soc_after,
            )
        )
    return tuple(summaries)


@dataclass(frozen=True)
class HouseholdFlexibilityResult:
    """单个已计划居民产品在成本容忍度内的电网功率上下界。"""
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    minimum_grid_power_mw: float
    maximum_grid_power_mw: float
