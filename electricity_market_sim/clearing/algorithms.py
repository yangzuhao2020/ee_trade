"""Deterministic, single-node pay-as-clear market clearing.
clearing.py 是市场出清的核心文件。

按报价从低到高排序卖单；
按买方愿付价格从高到低匹配需求；
依次成交，允许最后一台机组部分成交；
记录成交电量、未满足负荷、边际机组；
用最后被接受卖单的价格作为统一出清价；
返回该时段的 MarketClearingResult。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
    Bounds = None
    LinearConstraint = None
    linprog = None
    milp = None
    _SCIPY_IMPORT_ERROR: Exception | None = exc
else:
    _SCIPY_IMPORT_ERROR = None

from ..errors import InputValidationError
from ..market_models import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    DemandBid,
    MarketClearingResult,
    SupplyOffer,
    Trade,
    sum_cleared_demand,
)
from ..models import MarketSettings, StorageClearingContext


_EPSILON = 1e-9


def _require_complex_optimizer() -> None:
    if any(tool is None for tool in (Bounds, LinearConstraint, linprog, milp)):
        detail = f": {_SCIPY_IMPORT_ERROR}" if _SCIPY_IMPORT_ERROR else ""
        raise InputValidationError(
            "Complex market clearing requires SciPy optimization support "
            f"(milp and linprog){detail}."
        )


def _validate_price_limits(
    orders: list[SupplyOffer] | list[DemandBid],
    price_field: str,
    label: str,
    settings: MarketSettings,
) -> None:
    for order in orders:
        price = getattr(order, price_field)
        if not settings.minimum_bid_price <= price <= settings.maximum_bid_price:
            raise InputValidationError(
                f"{label} for {order.identifier!r} ({price:.6f} EUR/MWh) is outside "
                "the configured bid-price limits."
            )


def validate_offer_prices(
    offers: list[SupplyOffer], settings: MarketSettings
) -> None:
    """Reject supply prices outside the configured market limits."""

    _validate_price_limits(offers, "bid_price_eur_per_mwh", "Bid price", settings)


def validate_demand_prices(
    demand_bids: list[DemandBid], settings: MarketSettings
) -> None:
    """Reject demand prices outside the configured market limits."""

    _validate_price_limits(
        demand_bids, "price_eur_per_mwh", "Demand bid price", settings
    )


def _validate_order_ids(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    *,
    supply_error: str,
) -> None:
    bid_ids = [bid.identifier for bid in demand_bids]
    offer_ids = [offer.identifier for offer in supply_offers]
    if len(bid_ids) != len(set(bid_ids)):
        raise ValueError("Demand bid identifiers must be unique within a product.")
    if len(offer_ids) != len(set(offer_ids)):
        raise ValueError(supply_error)


def _single_product(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> tuple[datetime, datetime]:
    products = {
        (bid.delivery_start, bid.delivery_end) for bid in demand_bids
    } | {(offer.delivery_start, offer.delivery_end) for offer in supply_offers}
    if len(products) != 1:
        raise ValueError("All bids and offers passed to one clearing must share a product.")
    return products.pop()


def _demand_totals(cleared_demands: tuple[ClearedDemandBid, ...]) -> dict[str, float]:
    """Shared result fields; retain each algorithm's own traded-energy sum."""

    return {
        "requested_demand_mwh": sum_cleared_demand(cleared_demands, "bid.volume_mwh"),
        "unserved_load_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", demand_type="inelastic_load"
        ),
        "unfulfilled_export_mwh": sum_cleared_demand(
            cleared_demands, "unserved_energy_mwh", bid_type="export"
        ),
    }


def clear_pay_as_clear(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> MarketClearingResult:
    """Clear one product using stable merit order and uniform settlement.

    Supply is ordered by `(price, unit_name)`. Demand is ordered by
    `(-price, unit_name)`. These stable keys deliberately replace ASSUME's random
    tie break so a learning scenario produces reproducible results. At equal
    price, a learning unit's minimum-output segment clears before its flexible
    segment so accepted energy has a deterministic segment attribution. This
    tie break does not impose a minimum dispatched output on the unit.
    """

    if not demand_bids and not supply_offers:
        raise ValueError("Cannot clear an empty market.")
    _validate_order_ids(
        demand_bids,
        supply_offers,
        supply_error=(
            "Supply offer identifiers must be unique within a product; "
            "unit names must be unique when no offer_id is supplied."
        ),
    )
    delivery_start, delivery_end = _single_product(demand_bids, supply_offers)

    # 需求按价格从高到低排序；同价时再按名称和原始顺序稳定打破平局。
    ordered_demands = sorted(
        enumerate(demand_bids),
        key=lambda item: (-item[1].price_eur_per_mwh, item[1].unit_name, item[0]),
    )
    ordered_supply = sorted(
        supply_offers,
        key=lambda offer: (
            offer.bid_price_eur_per_mwh,
            offer.unit_name,
            offer.offer_segment != "learning_minimum",
            offer.identifier,
        ),
    )  # 按价格从小到大排。
    accepted_by_offer: dict[str, float] = defaultdict(float)
    supply_index = 0
    remaining_supply = (
        ordered_supply[0].offered_energy_mwh if ordered_supply else 0.0
    )
    marginal_unit_name: str | None = None
    accepted_by_demand_index = [0.0] * len(demand_bids)
    matched_orders: list[tuple[DemandBid, SupplyOffer, float]] = []

    for demand_index, demand in ordered_demands:
        remaining_demand = demand.volume_mwh # 记录这个需求单还剩多少电量没满足。
        while remaining_demand > _EPSILON and supply_index < len(ordered_supply):
            offer = ordered_supply[supply_index] # 当前取一个供给报价
            if offer.bid_price_eur_per_mwh > demand.price_eur_per_mwh:
                # 如果卖方报价比买方最高愿付价格还高，就不成交。
                break
            accepted = min(remaining_demand, remaining_supply)
            # 取需求和供给的最小值作为成交量。
            accepted_by_offer[offer.identifier] += accepted
            accepted_by_demand_index[demand_index] += accepted
            if accepted > _EPSILON:
                matched_orders.append((demand, offer, accepted))
            remaining_demand -= accepted # 这个需求单还剩多少电量没满足。
            remaining_supply -= accepted # 这个供给报价还剩多少电量没成交。
            if accepted > _EPSILON:
                # This offer is the latest accepted offer in the deterministic
                # merit order, so it is the product's marginal unit so far.
                marginal_unit_name = offer.unit_name
            if remaining_supply <= _EPSILON:
                supply_index += 1
                if supply_index < len(ordered_supply):
                    remaining_supply = ordered_supply[supply_index].offered_energy_mwh

    accepted_prices = [
        offer.bid_price_eur_per_mwh
        for offer in ordered_supply
        if accepted_by_offer[offer.identifier] > _EPSILON
    ]
    clearing_price = max(accepted_prices) if accepted_prices else 0.0
    cleared_offers = tuple(
        ClearedSupplyOffer(
            offer=offer,
            accepted_energy_mwh=accepted_by_offer[offer.identifier],
            clearing_price_eur_per_mwh=clearing_price,
        )
        for offer in sorted(
            supply_offers,
            key=lambda offer: (offer.unit_name, offer.offer_segment, offer.identifier),
        )
    )
    cleared_demands = tuple(
        ClearedDemandBid(
            bid=bid,
            accepted_energy_mwh=accepted_by_demand_index[index],
            clearing_price_eur_per_mwh=clearing_price,
        )
        for index, bid in enumerate(demand_bids)
    )
    cleared_energy = sum(item.accepted_energy_mwh for item in cleared_offers)
    trades = tuple(
        Trade(
            delivery_start=delivery_start,
            delivery_end=delivery_end,
            buyer_bid_id=demand.identifier,
            seller_offer_id=offer.identifier,
            trade_energy_mwh=energy,
            trade_price_eur_per_mwh=clearing_price,
        )
        for demand, offer, energy in matched_orders
    )

    return MarketClearingResult(
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        **_demand_totals(cleared_demands),
        cleared_energy_mwh=cleared_energy,
        clearing_price_eur_per_mwh=clearing_price,
        offers=cleared_offers,
        demand_bids=cleared_demands,
        marginal_unit_name=marginal_unit_name,
        trades=trades,
    )


def clear_pay_as_bid(
    demand_bids: list[DemandBid], supply_offers: list[SupplyOffer]
) -> MarketClearingResult:
    """Clear one product and settle every match at the seller's offer price."""

    if not demand_bids and not supply_offers:
        raise ValueError("Cannot clear an empty market.")
    delivery_start, delivery_end = _single_product(demand_bids, supply_offers)
    _validate_order_ids(
        demand_bids,
        supply_offers,
        supply_error="Supply offer identifiers must be unique within a product.",
    )

    ordered_bids = sorted(
        demand_bids,
        key=lambda bid: (-bid.price_eur_per_mwh, bid.unit_name, bid.identifier),
    )
    ordered_offers = sorted(
        supply_offers,
        key=lambda offer: (
            offer.bid_price_eur_per_mwh,
            offer.unit_name,
            offer.identifier,
        ),
    )
    remaining_bids = {bid.identifier: bid.volume_mwh for bid in ordered_bids}
    remaining_offers = {
        offer.identifier: offer.offered_energy_mwh for offer in ordered_offers
    }
    accepted_bids: dict[str, float] = defaultdict(float)
    accepted_offers: dict[str, float] = defaultdict(float)
    payments_by_bid: dict[str, float] = defaultdict(float)
    trades: list[Trade] = []

    bid_index = 0
    offer_index = 0
    while bid_index < len(ordered_bids) and offer_index < len(ordered_offers):
        bid = ordered_bids[bid_index]
        offer = ordered_offers[offer_index]
        if offer.bid_price_eur_per_mwh > bid.price_eur_per_mwh + _EPSILON:
            break
        quantity = min(
            remaining_bids[bid.identifier],
            remaining_offers[offer.identifier],
        )
        if quantity > _EPSILON:
            trade = Trade(
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                buyer_bid_id=bid.identifier,
                seller_offer_id=offer.identifier,
                trade_energy_mwh=quantity,
                trade_price_eur_per_mwh=offer.bid_price_eur_per_mwh,
            )
            trades.append(trade)
            accepted_bids[bid.identifier] += quantity
            accepted_offers[offer.identifier] += quantity
            payments_by_bid[bid.identifier] += trade.payment_eur
            remaining_bids[bid.identifier] -= quantity
            remaining_offers[offer.identifier] -= quantity
        if remaining_bids[bid.identifier] <= _EPSILON:
            bid_index += 1
        if remaining_offers[offer.identifier] <= _EPSILON:
            offer_index += 1

    cleared_offers = tuple(
        ClearedSupplyOffer(
            offer=offer,
            accepted_energy_mwh=accepted_offers[offer.identifier],
            clearing_price_eur_per_mwh=(
                offer.bid_price_eur_per_mwh
                if accepted_offers[offer.identifier] > _EPSILON
                else 0.0
            ),
        )
        for offer in supply_offers
    )
    cleared_demands = tuple(
        ClearedDemandBid(
            bid=bid,
            accepted_energy_mwh=accepted_bids[bid.identifier],
            clearing_price_eur_per_mwh=(
                payments_by_bid[bid.identifier] / accepted_bids[bid.identifier]
                if accepted_bids[bid.identifier] > _EPSILON
                else 0.0
            ),
        )
        for bid in demand_bids
    )
    cleared_energy = sum(trade.trade_energy_mwh for trade in trades)
    return MarketClearingResult(
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        **_demand_totals(cleared_demands),
        cleared_energy_mwh=cleared_energy,
        clearing_price_eur_per_mwh=None,
        offers=cleared_offers,
        demand_bids=cleared_demands,
        marginal_unit_name=None,
        pricing_method="pay_as_bid",
        trades=tuple(trades),
    )


def clear_complex_opening(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    storage_contexts: tuple[StorageClearingContext, ...] = (),
) -> tuple[MarketClearingResult, ...]:
    """Jointly clear all products of one complex EOM market opening.

    The solver follows the V2 ASSUME-compatible process: solve the order-book
    MILP, fix block accept/reject decisions, recover hourly prices from the
    balance-constraint duals of the resulting LP, then remove exactly one
    lowest-surplus accepted BB or linked family and repeat when necessary.
    """

    _require_complex_optimizer()

    products = tuple(
        sorted(
            {
                (bid.delivery_start, bid.delivery_end) for bid in demand_bids
            }
            | {
                (offer.delivery_start, offer.delivery_end) for offer in supply_offers
            }
        )
    )
    if not products:
        raise ValueError("Cannot clear an empty complex market opening.")
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
    context_names = {context.unit_name for context in storage_contexts}
    missing_contexts = sorted(storage_order_names - context_names)
    if missing_contexts:
        raise ValueError(
            "Storage orders require physical clearing contexts for: "
            + ", ".join(missing_contexts)
            + "."
        )

    disabled_parent_bids: set[str] = set()
    max_iterations = 1 + len(
        {offer.complex_identifier for offer in supply_offers if offer.bid_type == "BB"}
    )
    for _ in range(max_iterations):
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
            storage_contexts,
        )
        accepted_supply, accepted_demand, prices = solution
        negative_parent = _lowest_negative_parent_bid(
            active_offers,
            accepted_supply,
            prices,
        )
        if negative_parent is None:
            # ``active_offers`` is only the order book used for the final
            # re-clearing round.  Preserve the originally submitted order book
            # in the result so a BB/LB family removed for negative surplus is
            # reported as rejected (accepted energy 0), rather than appearing
            # never to have been submitted.
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

    raise ValueError("Complex clearing did not converge while removing block bids.")


def _validate_complex_offers(supply_offers: list[SupplyOffer]) -> None:
    """Validate relationships which only exist in a complex opening."""

    offer_ids_by_product: dict[tuple, set[str]] = defaultdict(set)
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
        if offer.bid_type == "BB" and offer.min_acceptance_ratio != 1.0:
            raise ValueError("Version two supports BB orders only with MAR=1.")
        if offer.bid_type in {"SB", "LB"} and offer.min_acceptance_ratio != 0.0:
            raise ValueError("Version two supports SB and LB orders only with MAR=0.")
        if offer.bid_type == "LB" and offer.parent_bid_id not in parent_ids:
            raise ValueError(
                f"Linked bid {offer.identifier!r} has no BB parent "
                f"{offer.parent_bid_id!r}."
            )


def _solve_complex_order_book(
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    products: tuple[tuple, ...],
    product_rows: dict[tuple, int],
    storage_contexts: tuple[StorageClearingContext, ...] = (),
) -> tuple[dict[str, float], list[float], dict[tuple, float]]:
    """Solve MILP then LP and return energy acceptances and dual prices.

    Storage energy variables describe the end of every product.  Their balance
    equations use accepted grid-side charge and discharge energy, so rejected
    bids never change SOC and a multi-product opening cannot over-charge or
    over-discharge a unit.
    """

    variables: list[dict] = []
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

    block_legs: dict[str, list[int]] = defaultdict(list)
    simple_legs: list[int] = []
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

    contexts_by_name: dict[str, StorageClearingContext] = {}
    storage_state_variables: dict[tuple[str, tuple], int] = {}
    for context in storage_contexts:
        if context.unit_name in contexts_by_name:
            raise ValueError(
                f"Duplicate storage clearing context for {context.unit_name!r}."
            )
        if not context.min_energy_mwh <= context.initial_energy_mwh <= context.max_energy_mwh:
            raise ValueError(
                f"Initial energy for storage {context.unit_name!r} is outside "
                "its SOC bounds."
            )
        if context.efficiency_charge <= 0 or context.efficiency_discharge <= 0:
            raise ValueError(
                f"Storage {context.unit_name!r} clearing efficiencies must be positive."
            )
        contexts_by_name[context.unit_name] = context
        for product in products:
            storage_state_variables[(context.unit_name, product)] = len(variables)
            variables.append(
                {
                    "kind": "storage_energy",
                    "indices": (),
                    "bid_type": "STATE",
                    "bid_id": f"{context.unit_name}::{product[0].isoformat()}",
                    "parent_bid_id": None,
                    "storage_name": context.unit_name,
                    "product": product,
                }
            )

    number_variables = len(variables)
    objective = np.zeros(number_variables)
    balance = np.zeros((len(products), number_variables))
    lower_bounds = np.zeros(number_variables)
    upper_bounds = np.ones(number_variables)
    parent_variable: dict[str, int] = {}
    demand_variable: dict[int, int] = {}
    supply_variable: dict[int, int] = {}
    for variable_index, variable in enumerate(variables):
        if variable["kind"] == "demand":
            demand_index = variable["indices"][0]
            demand_variable[demand_index] = variable_index
            bid = demand_bids[demand_index]
            objective[variable_index] = -bid.price_eur_per_mwh * bid.volume_mwh
            balance[
                product_rows[(bid.delivery_start, bid.delivery_end)], variable_index
            ] = bid.volume_mwh
        elif variable["kind"] == "supply":
            offers = [supply_offers[index] for index in variable["indices"]]
            objective[variable_index] = sum(
                offer.bid_price_eur_per_mwh * offer.offered_energy_mwh
                for offer in offers
            )
            for offer_index, offer in zip(variable["indices"], offers, strict=True):
                supply_variable[offer_index] = variable_index
                balance[
                    product_rows[(offer.delivery_start, offer.delivery_end)],
                    variable_index,
                ] -= offer.offered_energy_mwh
            if variable["bid_type"] == "BB":
                parent_variable[variable["bid_id"]] = variable_index
        else:
            context = contexts_by_name[variable["storage_name"]]
            lower_bounds[variable_index] = context.min_energy_mwh
            upper_bounds[variable_index] = context.max_energy_mwh

    storage_balance_rows: list[np.ndarray] = []
    storage_balance_rhs: list[float] = []
    for context in storage_contexts:
        previous_state_index: int | None = None
        for product in products:
            row = np.zeros(number_variables)
            state_index = storage_state_variables[(context.unit_name, product)]
            row[state_index] = 1.0
            if previous_state_index is None:
                right_hand_side = context.initial_energy_mwh
            else:
                row[previous_state_index] = -1.0
                right_hand_side = 0.0

            for demand_index, bid in enumerate(demand_bids):
                if (
                    bid.unit_name == context.unit_name
                    and bid.demand_type == "storage_charge"
                    and (bid.delivery_start, bid.delivery_end) == product
                ):
                    row[demand_variable[demand_index]] -= (
                        bid.volume_mwh * context.efficiency_charge
                    )
            for offer_index, offer in enumerate(supply_offers):
                if (
                    offer.unit_name == context.unit_name
                    and offer.offer_type == "storage_discharge"
                    and (offer.delivery_start, offer.delivery_end) == product
                ):
                    row[supply_variable[offer_index]] += (
                        offer.offered_energy_mwh / context.efficiency_discharge
                    )
            storage_balance_rows.append(row)
            storage_balance_rhs.append(right_hand_side)
            previous_state_index = state_index

    child_constraints: list[np.ndarray] = []
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
            [np.zeros(len(products)), np.asarray(storage_balance_rhs)]
        )

    constraints = [LinearConstraint(equality, equality_rhs, equality_rhs)]
    if child_constraints:
        constraints.append(
            LinearConstraint(np.vstack(child_constraints), -np.inf, 0.0)
        )
    integrality = np.array(
        [1 if variable["bid_type"] == "BB" else 0 for variable in variables],
        dtype=int,
    )
    mip_result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"disp": False},
    )
    if not mip_result.success or mip_result.x is None:
        raise ValueError(f"Complex MILP clearing failed: {mip_result.message}")

    lp_bounds = list(zip(lower_bounds, upper_bounds, strict=True))
    for variable_index, variable in enumerate(variables):
        if variable["bid_type"] == "BB":
            fixed = float(round(float(mip_result.x[variable_index])))
            lp_bounds[variable_index] = (fixed, fixed)
    lp_result = linprog(
        c=objective,
        A_ub=np.vstack(child_constraints) if child_constraints else None,
        b_ub=np.zeros(len(child_constraints)) if child_constraints else None,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=lp_bounds,
        method="highs",
    )
    if not lp_result.success or lp_result.x is None:
        raise ValueError(f"Complex LP price recovery failed: {lp_result.message}")
    if lp_result.eqlin.marginals is None:
        raise ValueError("Complex LP did not return balance-constraint dual prices.")

    accepted_supply = {offer.identifier: 0.0 for offer in supply_offers}
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
    # HiGHS reports equality marginals with the opposite sign to the EOM
    # balance-price convention used here (demand minus supply equals zero).
    # Negating them yields the hourly uniform settlement prices.
    prices = {
        product: -float(lp_result.eqlin.marginals[product_rows[product]])
        for product in products
    }
    return accepted_supply, accepted_demand, prices


def _lowest_negative_parent_bid(
    supply_offers: list[SupplyOffer],
    accepted_supply: dict[str, float],
    prices: dict[tuple, float],
) -> str | None:
    """Return the parent/standalone BB to remove in this re-clearing round."""

    blocks: dict[str, list[SupplyOffer]] = defaultdict(list)
    children: dict[str, list[SupplyOffer]] = defaultdict(list)
    for offer in supply_offers:
        if offer.bid_type == "BB":
            blocks[offer.complex_identifier].append(offer)
        elif offer.bid_type == "LB" and offer.parent_bid_id is not None:
            children[offer.parent_bid_id].append(offer)

    negative: list[tuple[float, str]] = []
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
    demand_bids: list[DemandBid],
    supply_offers: list[SupplyOffer],
    accepted_supply: dict[str, float],
    accepted_demand: list[float],
    prices: dict[tuple, float],
    products: tuple[tuple, ...],
) -> tuple[MarketClearingResult, ...]:
    """Project a jointly solved opening back into one result per product."""

    results: list[MarketClearingResult] = []
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
                **_demand_totals(cleared_demands),
                cleared_energy_mwh=sum(
                    item.accepted_energy_mwh for item in cleared_offers
                ),
                clearing_price_eur_per_mwh=prices[product],
                offers=cleared_offers,
                demand_bids=cleared_demands,
                marginal_unit_name=None,
                pricing_method="dual",
            )
        )
    return tuple(results)
