"""市场买卖订单及其派生标识与功率。"""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from ..utils import hours_between


@dataclass(frozen=True)
class DemandBid:
    """把“一条买单”定义成统一的数据对象。字段更清楚、便于传递和检查。"""
    unit_name: str
    operator: str
    delivery_start: datetime
    delivery_end: datetime
    volume_mwh: float
    price_eur_per_mwh: float
    bid_id: str | None = None # 订单标识
    demand_type: str = "inelastic_load"

    def __post_init__(self) -> None:
        if self.delivery_end <= self.delivery_start:
            raise ValueError("买单的交付时长必须大于 0。")
        if not isfinite(self.volume_mwh) or self.volume_mwh < 0:
            raise ValueError("买单电量必须是有限的非负数。")
        if not isfinite(self.price_eur_per_mwh):
            raise ValueError("买单价格必须是有限值。")
        if self.demand_type not in {
            "inelastic_load", # 刚性负荷
            "elastic_load", # 弹性负荷
            "household_load",
            "industrial_load",
            "export",
            "storage_charge",
        }:
            raise ValueError(
                "demand_type 必须是 inelastic_load、elastic_load、household_load、"
                "industrial_load、export 或 storage_charge。"
            )

    @property
    def bid_type(self) -> str:
        """由需求分类推导报告使用的市场侧订单角色。"""
        if self.demand_type in {"export", "storage_charge"}:
            return self.demand_type
        return "local_load"

    @property
    def identifier(self) -> str:
        """返回稳定的订单标识，供需求明细结果报告使用。"""
        return self.bid_id or self.unit_name


@dataclass(frozen=True)
class SupplyOffer:
    """出清前的卖单；保存申报电量，报告使用的平均功率由电量和时长推导。"""
    unit_name: str
    operator: str
    technology: str
    delivery_start: datetime
    delivery_end: datetime
    offered_energy_mwh: float
    bid_price_eur_per_mwh: float
    marginal_cost_eur_per_mwh: float
    offer_type: str = "power_plant"
    # V2 中同一机组可以提交多条卖单。``unit_name`` 仍表示物理机组，
    # ``offer_id`` 用于标识一条市场订单。
    offer_id: str | None = None
    offer_segment: str = "single"
    # 复杂出清的附加字段：``offer_id`` 标识订单在单个交付产品上的分段，
    # ``bid_id`` 标识一条可能跨多个交付产品的市场订单。
    bid_id: str | None = None
    bid_type: str = "SB"
    parent_bid_id: str | None = None

    def __post_init__(self) -> None:
        """校验交付时长、电量、价格、边际成本和报价类型，以及 V2 的最小接受比例约定。"""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("卖单的交付时长必须大于 0。")
        if not isfinite(self.offered_energy_mwh) or self.offered_energy_mwh < 0:
            raise ValueError("卖单申报电量必须是有限的非负数。")
        if not isfinite(self.bid_price_eur_per_mwh):
            raise ValueError("卖单价格必须是有限值。")
        if not isfinite(self.marginal_cost_eur_per_mwh):
            raise ValueError("卖单边际成本必须是有限值。")
        if self.bid_type not in {"SB", "BB", "LB"}:
            raise ValueError(f"不支持的 bid_type：{self.bid_type!r}。")

    @property
    def min_acceptance_ratio(self) -> float:
        return 1.0 if self.bid_type == "BB" else 0.0

    @property
    def offered_power_mw(self) -> float:
        return self.offered_energy_mwh / hours_between(self.delivery_start, self.delivery_end)

    @property
    def identifier(self) -> str:
        """返回唯一的订单键"""
        return self.offer_id or self.unit_name

    @property
    def complex_identifier(self) -> str:
        """返回块报价各分段共用的订单标识。"""
        return self.bid_id or self.identifier
