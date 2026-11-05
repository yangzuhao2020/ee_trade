"""订单成交结果、逐笔交易和单产品出清汇总。"""

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from operator import attrgetter
from ..utils import hours_between
from .orders import DemandBid, SupplyOffer


_ENERGY_TOLERANCE_MWH = 1e-7


@dataclass(frozen=True)
class ClearedSupplyOffer:
    """出清后，保存单条卖单的原始报价和出清结果，包括成交电量为零的情况。"""
    offer: SupplyOffer
    accepted_energy_mwh: float
    clearing_price_eur_per_mwh: float
    startup_cost_eur: float = 0.0

    @property
    def duration_hours(self) -> float:
        return hours_between(self.offer.delivery_start, self.offer.delivery_end)

    @property
    def accepted_power_mw(self) -> float:
        return self.accepted_energy_mwh / self.duration_hours

    @property
    def revenue_eur(self) -> float:
        # 售电收入 = 实际成交电量（MWh）× 结算价格（EUR/MWh）
        return self.accepted_energy_mwh * self.clearing_price_eur_per_mwh

    @property
    def variable_cost_eur(self) -> float:
        # 可变成本 = 成交电量（MWh）× 单位电量边际成本（EUR/MWh）
        return self.accepted_energy_mwh * self.offer.marginal_cost_eur_per_mwh

    @property
    def total_cost_eur(self) -> float:
        """可变成本加上可能产生的整台机组启动成本。"""
        return self.variable_cost_eur + self.startup_cost_eur

    @property
    def profit_eur(self) -> float:
        # 利润
        return self.revenue_eur - self.total_cost_eur


@dataclass(frozen=True)
class ClearedDemandBid:
    """出清后，保存单条买单的原始报价、成交电量及结算信息，包括成交电量为零的情况。"""
    bid: DemandBid
    accepted_energy_mwh: float
    clearing_price_eur_per_mwh: float

    @property
    def duration_hours(self) -> float:
        return hours_between(self.bid.delivery_start, self.bid.delivery_end)

    @property
    def accepted_power_mw(self) -> float:
        return self.accepted_energy_mwh / self.duration_hours

    @property
    def unserved_energy_mwh(self) -> float:
        # 未成交电量 = 申报购买电量 − 实际成交电量
        return self.bid.volume_mwh - self.accepted_energy_mwh

    @property
    def payment_eur(self) -> float:
        # 付款金额 = 实际成交电量（MWh）× 结算价格（EUR/MWh）
        return self.accepted_energy_mwh * self.clearing_price_eur_per_mwh


@dataclass(frozen=True)
class Trade:
    """出清后，保存一条买单与一条卖单之间的单笔成交记录。
    包含交付时段、订单标识、成交电量和结算价格。
    同一订单可对应多笔成交。
    | 买单 | 卖单 | 成交电量 | 成交价格 | 付款金额 |
    | B1   | S1  | 6 MWh | 50 EUR/MWh | 300 EUR |
    | B1   | S2  | 4 MWh | 60 EUR/MWh | 240 EUR |
    """
    delivery_start: datetime  # 本笔成交所属产品的交付开始时间。
    delivery_end: datetime  # 本笔成交所属产品的交付结束时间。
    buyer_bid_id: str  # 买单标识，对应 DemandBid.identifier。
    seller_offer_id: str  # 卖单标识，对应 SupplyOffer.identifier。
    trade_energy_mwh: float  # 本笔实际成交电量（MWh），一条订单的总成交量可由多笔记录累加得到。
    trade_price_eur_per_mwh: float  # 本笔成交的实际结算价格（EUR/MWh）。

    @property
    def payment_eur(self) -> float:
        """付款金额 = 成交电量（MWh）× 成交价格（EUR/MWh）。
        若成交价格为负，付款金额也为负，表示本笔成交由卖方向买方支付。
        """
        return self.trade_energy_mwh * self.trade_price_eur_per_mwh


@dataclass(frozen=True)
class MarketClearingResult:
    """单个交付产品的市场出清结果。"""
    delivery_start: datetime
    delivery_end: datetime
    clearing_price_eur_per_mwh: float | None
    offers: tuple[ClearedSupplyOffer, ...]
    demand_bids: tuple[ClearedDemandBid, ...]
    marginal_unit_name: str | None
    pricing_method: str = "merit_order"
    opening_time: datetime | None = None
    trades: tuple[Trade, ...] = ()

    def __post_init__(self) -> None:
        """校验结果报告和绘图所用的物理量与财务量。"""
        if self.delivery_end <= self.delivery_start:
            raise ValueError("市场出清结果的交付时长必须大于 0。")
        if self.opening_time is not None and self.opening_time > self.delivery_start:
            raise ValueError("开市时间不能晚于交付开始时间。")
        if self.pricing_method == "pay_as_bid":
            if self.clearing_price_eur_per_mwh is not None:
                raise ValueError("pay_as_bid 市场没有统一出清价。")
        elif self.clearing_price_eur_per_mwh is None or not isfinite(
            self.clearing_price_eur_per_mwh
        ):
            raise ValueError("出清价必须是有限值。")
        if any(
            value < -_ENERGY_TOLERANCE_MWH
            for value in (
                self.cleared_energy_mwh,
                self.unserved_load_mwh,
                self.unfulfilled_export_mwh,
            )
        ):
            raise ValueError("市场电量不能为负数。")
        if any(
            cleared.offer.delivery_start != self.delivery_start
            or cleared.offer.delivery_end != self.delivery_end
            for cleared in self.offers
        ) or any(
            cleared.bid.delivery_start != self.delivery_start
            or cleared.bid.delivery_end != self.delivery_end
            for cleared in self.demand_bids
        ):
            raise ValueError("所有已出清的买单和卖单必须属于同一市场产品。")
        if any(
            cleared.accepted_energy_mwh < -_ENERGY_TOLERANCE_MWH
            or cleared.accepted_energy_mwh
            > cleared.offer.offered_energy_mwh + _ENERGY_TOLERANCE_MWH
            for cleared in self.offers
        ) or any(
            cleared.accepted_energy_mwh < -_ENERGY_TOLERANCE_MWH
            or cleared.accepted_energy_mwh
            > cleared.bid.volume_mwh + _ENERGY_TOLERANCE_MWH
            for cleared in self.demand_bids
        ):
            raise ValueError("成交电量必须介于 0 和申报电量之间。")
        if any(
            cleared.offer.offer_type
            not in {"power_plant", "import", "storage_discharge"}
            for cleared in self.offers
        ):
            raise ValueError(
                "卖单的 offer_type 必须是 power_plant、import 或 storage_discharge。")

        accepted_demand_energy = self.demand_total("accepted_energy_mwh")
        if not isclose(
            accepted_demand_energy,
            sum(offer.accepted_energy_mwh for offer in self.offers),
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("买单成交电量之和必须等于出清电量。")

    @property
    def requested_demand_mwh(self) -> float:
        return self.demand_total("bid.volume_mwh")

    @property
    def cleared_energy_mwh(self) -> float:
        return sum(offer.accepted_energy_mwh for offer in self.offers)

    @property
    def unserved_load_mwh(self) -> float:
        return self.demand_total("unserved_energy_mwh", demand_type="inelastic_load")

    @property
    def unfulfilled_export_mwh(self) -> float:
        return self.demand_total("unserved_energy_mwh", bid_type="export")

    @property
    def duration_hours(self) -> float:
        """交付时长（小时），用于将电量换算为功率。"""
        return hours_between(self.delivery_start, self.delivery_end)

    def demand_total(
        self,
        attribute: str,
        demand_type: str | None = None,
        bid_type: str | None = None,
    ) -> float:
        """按需求类别或订单角色累加属性，支持嵌套属性路径。"""
        value_of = attrgetter(attribute)
        return sum(
            value_of(demand)
            for demand in self.demand_bids
            if (demand_type is None or demand.bid.demand_type == demand_type)
            and (bid_type is None or demand.bid.bid_type == bid_type)
        )

    @property
    def unserved_demand_mwh(self) -> float:
        """除储能充电外全部未被接受的需求，包括对价格敏感的弹性需求报价。
        储能未成交充电是未执行的充电计划，不属于未满足需求，由 unaccepted_storage_charge_mwh 单独统计。
        """
        return self.demand_total("unserved_energy_mwh") - self.unaccepted_storage_charge_mwh

    @property
    def requested_storage_charge_mwh(self) -> float:
        return self.demand_total("bid.volume_mwh", demand_type="storage_charge")

    @property
    def unaccepted_storage_charge_mwh(self) -> float:
        return self.demand_total("unserved_energy_mwh", demand_type="storage_charge")

    @property
    def unaccepted_elastic_demand_mwh(self) -> float:
        return self.demand_total("unserved_energy_mwh", demand_type="elastic_load")

    @property
    def requested_inelastic_demand_mwh(self) -> float:
        return self.demand_total("bid.volume_mwh", demand_type="inelastic_load")

    @property
    def cleared_inelastic_demand_mwh(self) -> float:
        return self.demand_total("accepted_energy_mwh", demand_type="inelastic_load")

    @property
    def requested_elastic_demand_mwh(self) -> float:
        return self.demand_total("bid.volume_mwh", demand_type="elastic_load")

    @property
    def cleared_elastic_demand_mwh(self) -> float:
        return self.demand_total("accepted_energy_mwh", demand_type="elastic_load")

    @property
    def requested_local_demand_mwh(self) -> float:
        return self.demand_total("bid.volume_mwh", bid_type="local_load")

    @property
    def cleared_local_demand_mwh(self) -> float:
        return self.demand_total("accepted_energy_mwh", bid_type="local_load")

    @property
    def cleared_export_mwh(self) -> float:
        return self.demand_total("accepted_energy_mwh", bid_type="export")

    @property
    def cleared_import_mwh(self) -> float:
        return sum(
            offer.accepted_energy_mwh
            for offer in self.offers
            if offer.offer.offer_type == "import"
        )

    @property
    def net_exchange_mwh(self) -> float:
        """正值表示向本地市场的净进口。"""
        return self.cleared_import_mwh - self.cleared_export_mwh

    @property
    def import_revenue_eur(self) -> float:
        import_revenues = (
            cleared_offer.revenue_eur
            for cleared_offer in self.offers
            if cleared_offer.offer.offer_type == "import"
        )
        return sum(import_revenues)

    @property
    def export_payment_eur(self) -> float:
        return self.demand_total("payment_eur", bid_type="export")

    @property
    def transaction_value_eur(self) -> float:
        if self.pricing_method == "pay_as_bid":
            return sum(trade.payment_eur for trade in self.trades)
        return self.cleared_energy_mwh * self.clearing_price_eur_per_mwh

    @property
    def average_trade_price_eur_per_mwh(self) -> float | None:
        if not self.trades:
            return None
        total_payment_eur = sum(trade.payment_eur for trade in self.trades)
        total_trade_energy_mwh = sum(
            trade.trade_energy_mwh for trade in self.trades
        )
        return total_payment_eur / total_trade_energy_mwh
