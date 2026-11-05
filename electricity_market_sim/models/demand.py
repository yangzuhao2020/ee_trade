"""需求参与者，以及计划进口与出口的虚拟交换参与者。"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DemandUnit:
    """
    DemandUnit 对象
        ↓ 结合负荷数据、交付时段和报价策略
    DemandBid 买单
    """
    name: str
    operator: str
    bidding_strategy: str = "demand_energy_naive"
    profile_column: str | None = None
    max_power_mw: float | None = None
    elasticity: float | None = None
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
