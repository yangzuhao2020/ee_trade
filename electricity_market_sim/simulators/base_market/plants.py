"""发电机组可用出力与报价生成、策略配置校验及交付后的运行状态更新。"""

from collections import defaultdict
from ...bidding import (
    available_power_mw, heuristic_block_offers,
    heuristic_segment_specs, naive_offer, segment_supply_offers,
)
from ...errors import InputValidationError
from .demand import _exchange_import_offer

_POWER_TOLERANCE_MW = 1e-9
_LINKED_REQUIRED_FIELDS = frozenset({"bid_type", "min_acceptance_ratio", "parent_bid_id"})


def _validate_strategy_configuration(
    settings,
    plants,
):
    """有关联报价机组时，检查出清机制和报价字段配置。
    若任一机组使用 powerplant_energy_heuristic_linked 策略，则要求：
    1. market_mechanism 为 complex_clearing；
    2. additional_fields 包含 bid_type、min_acceptance_ratio、parent_bid_id。
    不满足要求时抛出 InputValidationError；没有关联报价机组时直接返回。
    """

    if not any(
        plant.bidding_strategy == "powerplant_energy_heuristic_linked"
        for plant in plants
    ):
        return
    if settings.market_mechanism != "complex_clearing":
        raise InputValidationError(
            "powerplant_energy_heuristic_linked requires "
            "market_mechanism: complex_clearing."
        )
    missing_fields = sorted(_LINKED_REQUIRED_FIELDS - settings.additional_fields)
    if missing_fields:
        raise InputValidationError(
            "powerplant_energy_heuristic_linked requires "
            "markets_config.EOM.additional_fields to include: "
            + ", ".join(missing_fields)
            + "."
        )


def _record_runtime_states(
    market_result,
    runtime_states,
):
    """仅在产品的交付时段结束后，才将实际调度结果写入机组运行状态。"""
    accepted_by_plant = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.unit_name in runtime_states:
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    for plant_name in runtime_states:
        runtime_states[plant_name] = (
            accepted_by_plant[plant_name] / market_result.duration_hours > _POWER_TOLERANCE_MW
        )


def _offers_for_opening(
    *,
    opening,
    plants,
    runtime_states,
    availability_profiles,
    marginal_costs,
    fuel_price_profiles,
    price_forecasts,
    exchange_unit,
    exchange_profiles,
    learning_session=None,
):
    """根据机组可用出力、成本及报价策略生成各交付时段的卖单。
    启发式策略结合运行状态和预测电价，学习机组由学习会话生成报价。
    启用电力交换时追加进口卖单，返回供市场出清使用的卖单列表。
    """

    offers = []
    for plant in plants:
        available_powers = {
            delivery_start: available_power_mw(
                plant, delivery_start, availability_profiles
            )
            for delivery_start, _ in opening.products
        }

        def marginal_cost_at(delivery_start, current_plant=plant):
            if fuel_price_profiles is None:
                return marginal_costs[current_plant.name]
            return current_plant.marginal_cost(fuel_price_profiles[delivery_start])

        if learning_session is not None and plant.name == learning_session.plant.name:
            start, end = opening.products[0]
            offers.extend(learning_session.offers_for_product(
                delivery_start=start,
                delivery_end=end,
                available_power_mw=available_powers[start],
                marginal_cost_eur_per_mwh=marginal_cost_at(start),
            ))
            continue

        if plant.bidding_strategy == "powerplant_energy_naive":
            offers.extend(
                naive_offer(
                    plant,
                    start,
                    end,
                    available_powers[start],
                    marginal_cost_at(start),
                )
                for start, end in opening.products
            )
        elif plant.bidding_strategy == "powerplant_energy_heuristic_flexable":
            for start, end in opening.products:
                # 计算当前交付时段的边际成本。
                marginal_cost = marginal_cost_at(start)
                # 根据运行状态和预测电价计算分段报价，生成并汇总最低出力段与灵活出力段卖单。
                offers.extend(
                    segment_supply_offers(
                        plant, start, end,
                        heuristic_segment_specs(
                            plant, runtime_states[plant.name], start, end,
                            available_powers[start], marginal_cost, price_forecasts,
                        ),
                        marginal_cost,
                    )
                )
        elif plant.bidding_strategy in {
            "powerplant_energy_heuristic_block", "powerplant_energy_heuristic_linked"
        }:
            marginal_cost = marginal_cost_at(opening.products[0][0])
            offers.extend(
                heuristic_block_offers(
                    plant,
                    runtime_states[plant.name],
                    opening.products,
                    available_powers,
                    marginal_cost,
                    price_forecasts,
                    opening.opening_time,
                    linked=(
                        plant.bidding_strategy == "powerplant_energy_heuristic_linked"
                    ),
                )
            )
        else:
            raise InputValidationError(
                f"{plant.bidding_strategy} requires a learning session."
            )
    if exchange_unit is not None:
        for delivery_start, delivery_end in opening.products:
            offers.append(
                _exchange_import_offer(
                    exchange_unit, exchange_profiles[delivery_start], delivery_start, delivery_end
                )
            )
    return offers
