"""??????????????????????????????????"""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


@dataclass(frozen=True)
class DemandBid:
    """把“一条买单”定义成统一的数据对象。字段更清楚、便于传递和检查。"""
    unit_name: str
    operator: str
    delivery_start: datetime
    delivery_end: datetime
    volume_mwh: float
    price_eur_per_mwh: float
    bid_type: str = "local_load"
    bid_id: str | None = None # 订单标识
    # ``bid_type`` 保留 V1 电力交换逻辑使用的市场侧订单角色。
    # ``demand_type`` 区分刚性负荷与价格响应负荷，
    # 避免将未成交的弹性买单报告为非自愿限电。
    demand_type: str | None = None

    def __post_init__(self) -> None:
        if self.delivery_end <= self.delivery_start:
            raise ValueError("买单的交付时长必须大于 0。")
        if not isfinite(self.volume_mwh) or self.volume_mwh < 0:
            raise ValueError("买单电量必须是有限的非负数。")
        if not isfinite(self.price_eur_per_mwh):
            raise ValueError("买单价格必须是有限值。")
        if self.bid_type not in {"local_load", "export", "storage_charge"}:
            raise ValueError(
                "买单的 bid_type 必须是 local_load、export 或 storage_charge。"
            )
        demand_type = self.demand_type
        if demand_type is None:
            if self.bid_type == "export":
                demand_type = "export"
            elif self.bid_type == "storage_charge":
                demand_type = "storage_charge"
            else:
                demand_type = "inelastic_load"
            object.__setattr__(self, "demand_type", demand_type)
        if demand_type not in {
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
        if (self.bid_type == "export") != (demand_type == "export"):
            raise ValueError("出口买单必须使用 demand_type='export'。")
        if (self.bid_type == "storage_charge") != (
            demand_type == "storage_charge"
        ):
            raise ValueError(
                "储能充电买单必须使用 demand_type='storage_charge'。")

    @property
    def identifier(self) -> str:
        """返回稳定的订单标识，供需求明细结果报告使用。"""
        return self.bid_id or self.unit_name

    @property
    def side(self) -> str:
        return "buy"


@dataclass(frozen=True)
class SupplyOffer:
    """出清之前，一条卖单的数据结构，申报电量用于市场出清，申报功率保留用于结果报告。"""
    unit_name: str
    operator: str
    technology: str
    delivery_start: datetime
    delivery_end: datetime
    offered_power_mw: float
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
    min_acceptance_ratio: float = 0.0
    parent_bid_id: str | None = None

    def __post_init__(self) -> None:
        """校验交付时长、报价功率、电量、价格、边际成本和报价类型，并强制执行 V2 有意收窄的最小接受比例约定。"""

        if self.delivery_end <= self.delivery_start:
            raise ValueError("卖单的交付时长必须大于 0。")
        if not isfinite(self.offered_power_mw) or self.offered_power_mw < 0:
            raise ValueError("卖单申报功率必须是有限的非负数。")
        if not isfinite(self.offered_energy_mwh) or self.offered_energy_mwh < 0:
            raise ValueError("卖单申报电量必须是有限的非负数。")
        if not isfinite(self.bid_price_eur_per_mwh):
            raise ValueError("卖单价格必须是有限值。")
        if not isfinite(self.marginal_cost_eur_per_mwh):
            raise ValueError("卖单边际成本必须是有限值。")
        if self.bid_type not in {"SB", "BB", "LB"}:
            raise ValueError(f"不支持的 bid_type：{self.bid_type!r}。")
        # SB 普通报价 BB 块报价 LB 关联子报价
        if not isfinite(self.min_acceptance_ratio):
            raise ValueError("min_acceptance_ratio 必须是有限值。")
        if self.bid_type == "BB" and self.min_acceptance_ratio != 1.0:
            raise ValueError("第二版仅支持 MAR=1 的 BB 订单。")
        if self.bid_type in {"SB", "LB"} and self.min_acceptance_ratio != 0.0:
            raise ValueError("第二版仅支持 MAR=0 的 SB 和 LB 订单。")

    @property
    def identifier(self) -> str:
        """返回唯一的订单键"""
        return self.offer_id or self.unit_name

    @property
    def complex_identifier(self) -> str:
        """返回块报价各分段共用的订单标识。"""
        return self.bid_id or self.identifier

    @property
    def side(self) -> str:
        return "sell"
