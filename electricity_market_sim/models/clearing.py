""""""

from collections import defaultdict
from collections.abc import Iterable
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


def sum_cleared_demand(
    demand_bids: Iterable[ClearedDemandBid],
    attribute: str,
    demand_type: str | None = None,
    bid_type: str | None = None,
) -> float:
    """按顺序累加需求出清结果的某个字段，组装市场结果之前或之后都可使用。"""
    value_of = attrgetter(attribute)
    matched_values = []
    for demand in demand_bids:
        matches_demand_type = (
            demand_type is None or demand.bid.demand_type == demand_type
        )
        if not matches_demand_type:
            continue

        matches_bid_type = bid_type is None or demand.bid.bid_type == bid_type
        if not matches_bid_type:
            continue

        demand_value = value_of(demand)
        matched_values.append(demand_value)

    return sum(matched_values)


@dataclass(frozen=True)
class Trade:
    """出清后，保存一条买单与一条卖单之间的单笔成交记录。
    包含交付时段、订单标识、成交电量和结算价格。
    同一订单可对应多笔成交，供 MarketClearingResult 汇总核对。
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

    def __post_init__(self) -> None:
        """创建成交记录后，检查交付时段、成交电量和价格是否符合基本要求。"""
        # 交付结束时间必须晚于开始时间，以保证产品具有正的交付时长。
        if self.delivery_end <= self.delivery_start:
            raise ValueError("成交记录的交付时长必须大于 0。")
        # 只记录实际发生的成交；电量为零的订单由对应的订单出清结果保存。
        if self.trade_energy_mwh <= 0:
            raise ValueError("成交电量必须为正数。")
        # 价格允许为负，但必须是有限值，排除 NaN 和正负无穷。
        if not isfinite(self.trade_price_eur_per_mwh):
            raise ValueError("成交价格必须是有限值。")

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
    requested_demand_mwh: float
    cleared_energy_mwh: float
    unserved_load_mwh: float
    unfulfilled_export_mwh: float
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
                self.requested_demand_mwh,
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
        if self.pricing_method != "pay_as_bid" and any(
            not isclose(
                cleared.clearing_price_eur_per_mwh,
                self.clearing_price_eur_per_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            )
            for cleared in (*self.offers, *self.demand_bids)
        ):
            raise ValueError("所有已出清订单必须使用市场出清价。")
        offer_ids = [cleared.offer.identifier for cleared in self.offers]
        if len(offer_ids) != len(set(offer_ids)):
            raise ValueError(
                "同一产品内的卖单标识必须唯一；" "未提供 offer_id 时，机组名称必须唯一。")
        if any(
            cleared.offer.offer_type
            not in {"power_plant", "import", "storage_discharge"}
            for cleared in self.offers
        ):
            raise ValueError(
                "卖单的 offer_type 必须是 power_plant、import 或 storage_discharge。")

        requested_demand = self.demand_total("bid.volume_mwh")
        if not isclose(
            self.requested_demand_mwh,
            requested_demand,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("需求申报电量必须等于各买单电量之和。")
        
        accepted_energy = sum(offer.accepted_energy_mwh for offer in self.offers)
        if not isclose(
            accepted_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("卖单成交电量之和必须等于出清电量。")
        
        accepted_demand_energy = self.demand_total("accepted_energy_mwh")
        if not isclose(
            accepted_demand_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("买单成交电量之和必须等于出清电量。")
        
        expected_unserved_load = self.demand_total(
            "unserved_energy_mwh", demand_type="inelastic_load")
        if not isclose(
            self.unserved_load_mwh,
            expected_unserved_load,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("未满足负荷必须等于本地负荷买单中未满足的电量。")
        expected_unfulfilled_export = self.demand_total(
            "unserved_energy_mwh", bid_type="export")
        if not isclose(
            self.unfulfilled_export_mwh,
            expected_unfulfilled_export,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("未完成出口电量必须等于出口买单中未满足的电量。")

        if self.pricing_method not in {"merit_order", "dual", "pay_as_bid"}:
            raise ValueError("pricing_method 必须是 merit_order、dual 或 pay_as_bid。")

        if self.pricing_method == "pay_as_bid" or self.trades:
            # 只要采用 pay_as_bid，或者 trades 非空，就核对成交明细与订单、市场汇总结果是否一致。
            self._validate_trades()
            
        if self.pricing_method == "merit_order" and self.cleared_energy_mwh > _ENERGY_TOLERANCE_MWH:
            marginal_offers = []
            for cleared_offer in self.offers:
                is_marginal_unit = (cleared_offer.offer.unit_name == self.marginal_unit_name)
                if not is_marginal_unit:
                    continue

                has_accepted_energy = (cleared_offer.accepted_energy_mwh > _ENERGY_TOLERANCE_MWH) 
                # 检查是否有接受的电量
                if not has_accepted_energy:
                    continue

                matches_clearing_price = isclose(
                    cleared_offer.offer.bid_price_eur_per_mwh,
                    self.clearing_price_eur_per_mwh,
                    rel_tol=0.0,
                    abs_tol=_ENERGY_TOLERANCE_MWH,
                )
                if not matches_clearing_price:
                    continue

                marginal_offers.append(cleared_offer)
                
            if not marginal_offers:
                raise ValueError("边际机组必须有一条以出清价成交的卖单。")
            
        elif self.pricing_method == "merit_order" and self.marginal_unit_name is not None:
            raise ValueError("未出清的市场不能有边际机组。")

    @property
    def duration_hours(self) -> float:
        """交付时长（小时），用于将电量换算为功率。"""
        return hours_between(self.delivery_start, self.delivery_end)

    @property
    def cleared_power_mw(self) -> float:
        return self.cleared_energy_mwh / self.duration_hours

    def demand_total(
        self,
        attribute: str,
        demand_type: str | None = None,
        bid_type: str | None = None,
    ) -> float:
        return sum_cleared_demand(
            self.demand_bids,
            attribute,
            demand_type=demand_type,
            bid_type=bid_type,
        )

    @property
    def unserved_demand_mwh(self) -> float:
        """全部未被接受的需求，包括对价格敏感的弹性需求报价。"""
        return self.demand_total("unserved_energy_mwh")

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
        assert self.clearing_price_eur_per_mwh is not None
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

    def _validate_trades(self) -> None:
        """交付时段一致：所有成交记录都属于当前市场产品。
        总成交电量一致：各笔成交电量之和等于 cleared_energy_mwh。
        订单存在：每笔成交中的买单、卖单编号都能找到对应订单。
        成交价格正确：pay_as_bid 按卖单报价结算，其他机制按统一出清价结算；成交价还不能超过买方最高愿付价。
        每条订单的成交量一致：按订单编号累加逐笔成交电量，必须等于对应 ClearedSupplyOffer、ClearedDemandBid 中的成交量。
        买方付款一致：按买单累加逐笔付款，必须等于该买单汇总的付款金额。
        """

        if any(
            trade.delivery_start != self.delivery_start
            or trade.delivery_end != self.delivery_end
            for trade in self.trades
        ):
            raise ValueError("所有成交记录必须属于同一市场产品。")
        trade_energy = sum(trade.trade_energy_mwh for trade in self.trades)
        if not isclose(
            trade_energy,
            self.cleared_energy_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise ValueError("成交记录的总电量必须等于市场出清电量。")

        offers = {cleared.offer.identifier: cleared for cleared in self.offers}
        bids = {cleared.bid.identifier: cleared for cleared in self.demand_bids}
        traded_by_offer: dict[str, float] = defaultdict(float)
        traded_by_bid: dict[str, float] = defaultdict(float)
        payment_by_bid: dict[str, float] = defaultdict(float)
        
        for trade in self.trades:
            if trade.seller_offer_id not in offers:
                raise ValueError(f"未知的卖单：{trade.seller_offer_id!r}。")
            if trade.buyer_bid_id not in bids:
                raise ValueError(f"未知的买单：{trade.buyer_bid_id!r}。")
            
            offer = offers[trade.seller_offer_id].offer
            bid = bids[trade.buyer_bid_id].bid
            expected_trade_price = (
                offer.bid_price_eur_per_mwh
                if self.pricing_method == "pay_as_bid"
                else self.clearing_price_eur_per_mwh
            )
            assert expected_trade_price is not None
            if not isclose(
                trade.trade_price_eur_per_mwh,
                expected_trade_price,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("成交价格必须符合市场结算规则。")
            if trade.trade_price_eur_per_mwh > bid.price_eur_per_mwh + _ENERGY_TOLERANCE_MWH:
                raise ValueError("成交价格不能高于买单报价。")
            traded_by_offer[trade.seller_offer_id] += trade.trade_energy_mwh
            traded_by_bid[trade.buyer_bid_id] += trade.trade_energy_mwh
            payment_by_bid[trade.buyer_bid_id] += trade.payment_eur

        for identifier, cleared in offers.items():
            accepted = traded_by_offer[identifier]
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("成交记录的电量必须与卖单成交电量一致。")
            if self.pricing_method == "pay_as_bid":
                expected_price = (
                    cleared.offer.bid_price_eur_per_mwh
                    if accepted > _ENERGY_TOLERANCE_MWH
                    else 0.0
                )
                if not isclose(
                    cleared.clearing_price_eur_per_mwh,
                    expected_price,
                    rel_tol=0.0,
                    abs_tol=_ENERGY_TOLERANCE_MWH,
                ):
                    raise ValueError("pay_as_bid 卖方必须按自身报价结算。")
        for identifier, cleared in bids.items():
            accepted = traded_by_bid[identifier]
            if not isclose(
                accepted,
                cleared.accepted_energy_mwh,
                rel_tol=0.0,
                abs_tol=_ENERGY_TOLERANCE_MWH,
            ):
                raise ValueError("成交记录的电量必须与买单成交电量一致。")
            if not isclose(
                payment_by_bid[identifier],
                cleared.payment_eur,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("成交记录的支付金额必须与买单结算支付一致。")
