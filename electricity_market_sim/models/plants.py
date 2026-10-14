"""可调度发电机组参数及其边际成本计算。"""

from __future__ import annotations
from dataclasses import dataclass


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
        """边际成本的计算。
        """
        # 所提供的 V2 示例中，波动性可再生能源没有燃料价格列；
        # 按惯例其燃料成本分量为零。
        fuel_price = fuel_prices.get(self.fuel_type, 0.0)
        co2_price = fuel_prices["co2"]
        co2_cost = co2_price * self.emission_factor
        fuel_and_co2_cost = (fuel_price + co2_cost) / self.efficiency
        return fuel_and_co2_cost + self.additional_cost_eur_per_mwh
