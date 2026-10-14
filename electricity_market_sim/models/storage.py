"""储能参与者参数、联合出清所需的储能物理数据及储能调度结果。"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


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
