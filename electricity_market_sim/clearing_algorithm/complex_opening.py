"""考虑块报价、关联报价及跨时段储能 SOC 约束的多产品联合出清。"""

from collections import defaultdict
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from ..models import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    MarketClearingResult,
)
from .common import _EPSILON


def clear_complex_opening(demand_bids, supply_offers, storages=()):
    """TODO:复杂出清主函数，考虑块报价、关联报价和储能电量约束，联合出清本次开市的所有产品。"""

    products = tuple(
        sorted({(bid.delivery_start, bid.delivery_end) for bid in demand_bids}|{(offer.delivery_start, offer.delivery_end) for offer in supply_offers}))
    if not products:
        raise ValueError("本次开市没有买卖订单，无法进行复杂出清。")
    product_rows = {product: index for index, product in enumerate(products)}
    _validate_complex_offers(supply_offers)
    storage_order_names = {
        bid.unit_name
        for bid in demand_bids
        if bid.demand_type == "storage_charge"
    } | {
        offer.unit_name
        for offer in supply_offers
        if offer.offer_type == "storage_discharge"
    }
    context_names = set()
    for storage, initial_energy in storages:
        if storage.name in context_names:
            raise ValueError(
                f"储能 {storage.name!r} 的出清上下文重复。"
            )
        if not storage.min_energy_mwh <= initial_energy <= storage.max_energy_mwh:
            raise ValueError(
                f"储能 {storage.name!r} 的初始电量超出 SOC 允许范围。"
            )
        if storage.efficiency_charge <= 0 or storage.efficiency_discharge <= 0:
            raise ValueError(
                f"储能 {storage.name!r} 的充电和放电效率必须大于 0。"
            )
        context_names.add(storage.name)
    missing_contexts = sorted(storage_order_names - context_names)
    if missing_contexts:
        raise ValueError(
            "以下储能订单缺少物理出清参数："
            + ", ".join(missing_contexts)
            + "。"
        )

    disabled_parent_bids = set()
    while True:
        active_offers = [
            offer
            for offer in supply_offers
            if offer.complex_identifier not in disabled_parent_bids
            and offer.parent_bid_id not in disabled_parent_bids
        ]
        solution = _solve_complex_order_book(
            demand_bids,
            active_offers,
            products,
            product_rows,
            storages,
        )
        accepted_supply, accepted_demand, prices = solution
        negative_parent = _lowest_negative_parent_bid(
            active_offers,
            accepted_supply,
            prices,
        )
        if negative_parent is None:
            final_accepted_supply = {
                offer.identifier: accepted_supply.get(offer.identifier, 0.0)
                for offer in supply_offers
            }
            return _complex_results(
                demand_bids,
                supply_offers,
                final_accepted_supply,
                accepted_demand,
                prices,
                products,
            )
        disabled_parent_bids.add(negative_parent)


def _validate_complex_offers(supply_offers):
    """检查同一交付产品内的卖单标识是否唯一，以及关联报价（LB）引用的父块报价（BB）是否存在。"""
    offer_ids_by_product = defaultdict(set)
    parent_ids = {
        offer.complex_identifier
        for offer in supply_offers
        if offer.bid_type == "BB"
    }
    for offer in supply_offers:
        product = (offer.delivery_start, offer.delivery_end)
        if offer.identifier in offer_ids_by_product[product]:
            raise ValueError(f"Duplicate supply offer identifier {offer.identifier!r}.")
        offer_ids_by_product[product].add(offer.identifier)
        if offer.bid_type == "LB" and offer.parent_bid_id not in parent_ids:
            raise ValueError(
                f"Linked bid {offer.identifier!r} has no BB parent "
                f"{offer.parent_bid_id!r}."
            )


def _solve_complex_order_book(
    demand_bids, supply_offers, products, product_rows, storages=()
):
    """先求解 MILP，固定 BB 决策后求解 LP，返回成交电量和对偶价格。"""
    variables, objective, bounds, constraints, integrality = _build_complex_matrices(
        demand_bids, supply_offers, products, product_rows, storages
    )
    mip_result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )  # TODO: 调用求解器，求解。
    if not mip_result.success:
        raise ValueError(f"复杂出清的混合整数线性规划求解失败：{mip_result.message}")

    lp_bounds = list(zip(bounds.lb, bounds.ub))
    for variable_index, variable in enumerate(variables):
        if variable["bid_type"] == "BB":
            fixed = float(round(float(mip_result.x[variable_index])))
            lp_bounds[variable_index] = (fixed, fixed)

    equality, *inequalities = constraints
    lp_result = linprog(
        c=objective,
        A_ub=inequalities[0].A if inequalities else None,
        b_ub=inequalities[0].ub if inequalities else None,
        A_eq=equality.A,
        b_eq=equality.lb,
        bounds=lp_bounds,
        method="highs",
    )
    if not lp_result.success:
        raise ValueError(f"复杂出清的线性规划定价失败：{lp_result.message}")

    return _decode_complex_solution(
        demand_bids, supply_offers, products, product_rows, variables, lp_result
    )


def _build_complex_variables(demand_bids, supply_offers, products, storages):
    """整理优化模型中“需要求解哪些变量”，并确定它们的排列顺序，供后续构建矩阵使用"""
    variables = []
    # 每笔买单对应一个接受比例变量。
    for demand_index, bid in enumerate(demand_bids):
        variables.append(
            {
                "kind": "demand",
                "indices": (demand_index,),
                "bid_type": "DB",
                "bid_id": bid.identifier,
                "parent_bid_id": None,
            }
        )

    # 同一块报价的各时段卖单共用一个接受变量。
    block_legs = defaultdict(list)
    simple_legs = []
    for offer_index, offer in enumerate(supply_offers):
        if offer.bid_type == "BB":
            block_legs[offer.complex_identifier].append(offer_index)
        else:
            simple_legs.append(offer_index)
    for bid_id in sorted(block_legs):
        variables.append(
            {
                "kind": "supply",
                "indices": tuple(block_legs[bid_id]),
                "bid_type": "BB",
                "bid_id": bid_id,
                "parent_bid_id": None,
            }
        )
    # 普通报价和关联报价各自对应一个接受比例变量。
    for offer_index in sorted(simple_legs, key=lambda index: supply_offers[index].identifier):
        offer = supply_offers[offer_index]
        variables.append(
            {
                "kind": "supply",
                "indices": (offer_index,),
                "bid_type": offer.bid_type,
                "bid_id": offer.complex_identifier,
                "parent_bid_id": offer.parent_bid_id,
            }
        )

    # 每台储能、每个交付时段设置一个期末电量变量。
    storage_state_variables = {}
    for storage, _ in storages:
        for product in products:
            # 保存变量索引，供储能电量约束引用。
            storage_state_variables[(storage.name, product)] = len(variables)
            variables.append(
                {
                    "kind": "storage_energy",
                    "indices": (),
                    "bid_type": "STATE",
                    "bid_id": f"{storage.name}::{product[0].isoformat()}",
                    "parent_bid_id": None,
                    "storage_name": storage.name,
                }
            )
    return variables, storage_state_variables


def _build_complex_matrices(demand_bids, supply_offers, 
                            products, product_rows, storages):
    """构建目标函数、变量边界、供需平衡、储能电量和父子报价约束。
    储能变量表示各产品结束时的电量，按成交充放电量及效率更新。
    TODO: 将约束变为矩阵。
    """
    variables, storage_state_variables = _build_complex_variables(
        demand_bids, supply_offers, products, storages)
    storages_by_name = {storage.name: storage for storage, _ in storages}
    number_variables = len(variables)
    objective = np.zeros(number_variables)
    balance = np.zeros((len(products), number_variables))
    lower_bounds = np.zeros(number_variables)
    upper_bounds = np.ones(number_variables)
    parent_variable = {}
    demand_variable = {}
    supply_variable = {}
    
    for variable_index, variable in enumerate(variables):
        if variable["kind"] == "demand":
            demand_index = variable["indices"][0]
            demand_variable[demand_index] = variable_index
            bid = demand_bids[demand_index]
            # 买方申报价值取负，平衡式中的需求取正。
            objective[variable_index] = -bid.price_eur_per_mwh * bid.volume_mwh
            balance[
                product_rows[(bid.delivery_start, bid.delivery_end)], variable_index
            ] = bid.volume_mwh

        elif variable["kind"] == "supply":
            offers = [supply_offers[index] for index in variable["indices"]]
            # 合计卖单各段的报价金额，作为目标系数。
            objective[variable_index] = sum(
                offer.bid_price_eur_per_mwh * offer.offered_energy_mwh
                for offer in offers
            )
            # 平衡式中的供给取负。
            for offer_index, offer in zip(variable["indices"], offers):
                supply_variable[offer_index] = variable_index
                balance[
                    product_rows[(offer.delivery_start, offer.delivery_end)],
                    variable_index,
                ] -= offer.offered_energy_mwh
            if variable["bid_type"] == "BB":
                # 保存父块变量索引，供关联报价约束使用。
                parent_variable[variable["bid_id"]] = variable_index

        else:
            # 储能变量表示电量，设置其上下限。
            storage = storages_by_name[variable["storage_name"]]
            lower_bounds[variable_index] = storage.min_energy_mwh
            upper_bounds[variable_index] = storage.max_energy_mwh

    storage_balance_rows = []
    storage_balance_rhs = []
    
    # 电量平衡：期末 = 期初 + 充电量×效率 - 放电量/效率。
    for storage, initial_energy in storages:
        previous_state_index = None
        for product in products:
            row = np.zeros(number_variables)
            state_index = storage_state_variables[(storage.name, product)]
            row[state_index] = 1.0
            # 首时段使用初始电量，后续衔接上一时段。
            if previous_state_index is None:
                right_hand_side = initial_energy
            else:
                row[previous_state_index] = -1.0
                right_hand_side = 0.0

            # 成交充电量乘以效率，得到入库电量。
            for demand_index, bid in enumerate(demand_bids):
                if (
                    bid.unit_name == storage.name
                    and bid.demand_type == "storage_charge"
                    and (bid.delivery_start, bid.delivery_end) == product
                ):
                    row[demand_variable[demand_index]] -= (
                        bid.volume_mwh * storage.efficiency_charge
                    )
            # 成交放电量除以效率，得到出库电量。
            for offer_index, offer in enumerate(supply_offers):
                if (
                    offer.unit_name == storage.name
                    and offer.offer_type == "storage_discharge"
                    and (offer.delivery_start, offer.delivery_end) == product
                ):
                    row[supply_variable[offer_index]] += (
                        offer.offered_energy_mwh / storage.efficiency_discharge
                    )
            storage_balance_rows.append(row)
            storage_balance_rhs.append(right_hand_side)
            previous_state_index = state_index

    child_constraints = []
    for variable_index, variable in enumerate(variables):
        if variable["bid_type"] != "LB":
            continue
        parent_index = parent_variable[variable["parent_bid_id"]]
        row = np.zeros(number_variables)
        row[variable_index] = 1.0
        row[parent_index] = -1.0
        child_constraints.append(row)

    equality = balance
    equality_rhs = np.zeros(len(products))
    if storage_balance_rows:
        equality = np.vstack([balance, *storage_balance_rows])
        equality_rhs = np.concatenate(
            [np.zeros(len(products)), np.asarray(storage_balance_rhs)])

    constraints = [LinearConstraint(equality, equality_rhs, equality_rhs)]
    if child_constraints:
        constraints.append(
            LinearConstraint(np.vstack(child_constraints), -np.inf, 0.0)
        )
    integrality = np.array(
        [1 if variable["bid_type"] == "BB" else 0 for variable in variables],
        dtype=int,
    )
    return variables, objective, Bounds(lower_bounds, upper_bounds), constraints, integrality


def _decode_complex_solution(
    demand_bids, supply_offers, 
    products, product_rows, 
    variables, lp_result
):
    """将 LP 接受比例和供需平衡对偶值还原为成交电量及各产品出清价。"""
    accepted_supply = {}
    accepted_demand = [0.0] * len(demand_bids)
    for variable_index, variable in enumerate(variables):
        acceptance = max(0.0, min(1.0, float(lp_result.x[variable_index])))
        if variable["kind"] == "demand":
            bid = demand_bids[variable["indices"][0]]
            accepted_demand[variable["indices"][0]] = bid.volume_mwh * acceptance
        elif variable["kind"] == "supply":
            for offer_index in variable["indices"]:
                offer = supply_offers[offer_index]
                accepted_supply[offer.identifier] = offer.offered_energy_mwh * acceptance
    prices = {
        product: -float(lp_result.eqlin.marginals[product_rows[product]])
        for product in products
    }
    return accepted_supply, accepted_demand, prices


def _lowest_negative_parent_bid(supply_offers, accepted_supply, prices):
    """返回已接受报价组中合计价差收益最负的 BB 标识；计入关联 LB，无负收益组时返回 None。
    剔除就是让这组报价在本次开市中不成交,也就是意味着不成交是吗"""
    blocks = defaultdict(list)
    children = defaultdict(list)
    for offer in supply_offers:
        if offer.bid_type == "BB":
            blocks[offer.complex_identifier].append(offer)
        elif offer.bid_type == "LB":
            children[offer.parent_bid_id].append(offer)

    negative = []
    for parent_id, legs in blocks.items():
        if sum(accepted_supply[leg.identifier] for leg in legs) <= _EPSILON:
            continue
        surplus = sum(
            accepted_supply[leg.identifier]
            * (
                prices[(leg.delivery_start, leg.delivery_end)]
                - leg.bid_price_eur_per_mwh
            )
            for leg in legs
        )
        surplus += sum(
            accepted_supply[child.identifier]
            * (
                prices[(child.delivery_start, child.delivery_end)]
                - child.bid_price_eur_per_mwh
            )
            for child in children[parent_id]
        )
        if surplus < -_EPSILON:
            negative.append((surplus, parent_id))
    return min(negative, default=(0.0, None), key=lambda item: (item[0], item[1]))[1]


def _complex_results(
    demand_bids, supply_offers, accepted_supply, accepted_demand, prices, products
):
    """将联合求解的开市结果拆分回每个产品各自的出清结果。"""
    results = []
    for delivery_start, delivery_end in products:
        product = (delivery_start, delivery_end)
        offers = [
            offer
            for offer in supply_offers
            if (offer.delivery_start, offer.delivery_end) == product
        ]
        indexed_demands = [
            (index, bid)
            for index, bid in enumerate(demand_bids)
            if (bid.delivery_start, bid.delivery_end) == product
        ]
        cleared_offers = tuple(
            ClearedSupplyOffer(
                offer=offer,
                accepted_energy_mwh=accepted_supply[offer.identifier],
                clearing_price_eur_per_mwh=prices[product],
            )
            for offer in sorted(
                offers,
                key=lambda offer: (offer.unit_name, offer.offer_segment, offer.identifier),
            )
        )
        cleared_demands = tuple(
            ClearedDemandBid(
                bid=bid,
                accepted_energy_mwh=accepted_demand[index],
                clearing_price_eur_per_mwh=prices[product],
            )
            for index, bid in indexed_demands
        )
        results.append(
            MarketClearingResult(
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                clearing_price_eur_per_mwh=prices[product],
                offers=cleared_offers,
                demand_bids=cleared_demands,
                pricing_method="dual",
            )
        )
    return tuple(results)
