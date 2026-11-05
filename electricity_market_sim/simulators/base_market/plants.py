"""发电机组可用出力与报价生成、策略配置校验、启动成本结算及交付后的运行状态更新。"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING
from ...bidding import (
    available_power_mw, heuristic_block_offers,
    heuristic_segment_specs, naive_offer, segment_supply_offers,
)
from ...errors import InputValidationError
from ...models import (
    ExchangeUnit,
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    PowerPlant,
    SupplyOffer,
)
from .demand import _exchange_import_offer

if TYPE_CHECKING:
    from ...deep_learning.learning import LearningEpisodeSession

_POWER_TOLERANCE_MW = 1e-9
_LINKED_REQUIRED_FIELDS = frozenset({"bid_type", "min_acceptance_ratio", "parent_bid_id"})


def _validate_strategy_configuration(
    settings: MarketSettings,
    plants: tuple[PowerPlant, ...],
) -> None:
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


def _apply_startup_costs(
    market_result: MarketClearingResult,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, bool],
) -> MarketClearingResult:
    """已成交的产品开始交付时，为需要启动的机组计入启动成本。"""

    accepted_by_plant: dict[str, float] = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.offer_type == "power_plant":
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    started_names = {
        name
        for name, is_running in runtime_states.items()
        if not is_running and accepted_by_plant[name] > _POWER_TOLERANCE_MW
    }
    if not started_names:
        return market_result
    plants_by_name = {plant.name: plant for plant in plants}
    has_inflexible_offer = {
        cleared.offer.unit_name
        for cleared in market_result.offers
        if cleared.offer.offer_segment
        in {"inflexible", "block_inflexible", "learning_minimum"}
    }
    applied_names: set[str] = set()
    cleared_offers = []
    for cleared in market_result.offers:
        offer = cleared.offer
        startup_cost = 0.0
        if (
            offer.unit_name in started_names
            and offer.unit_name not in applied_names
            and offer.offer_type == "power_plant"
            and (
                offer.offer_segment
                in {"inflexible", "block_inflexible", "learning_minimum"}
                or offer.unit_name not in has_inflexible_offer
            )
        ):
            startup_cost = plants_by_name[offer.unit_name].start_cost_eur
            applied_names.add(offer.unit_name)
        cleared_offers.append(replace(cleared, startup_cost_eur=startup_cost))
    return replace(market_result, offers=tuple(cleared_offers))


def _record_runtime_states(
    market_result: MarketClearingResult,
    runtime_states: dict[str, bool],
) -> None:
    """仅在产品的交付时段结束后，才将实际调度结果写入机组运行状态。"""

    accepted_by_plant: dict[str, float] = defaultdict(float)
    for cleared in market_result.offers:
        if cleared.offer.unit_name in runtime_states:
            accepted_by_plant[cleared.offer.unit_name] += cleared.accepted_energy_mwh
    for plant_name in runtime_states:
        runtime_states[plant_name] = (
            accepted_by_plant[plant_name] / market_result.duration_hours > _POWER_TOLERANCE_MW
        )


def _offers_for_opening(
    *,
    opening: MarketOpening,
    plants: tuple[PowerPlant, ...],
    runtime_states: dict[str, bool],
    availability_profiles: dict[str, dict[datetime, float]],
    marginal_costs: dict[str, float],
    fuel_price_profiles: dict[datetime, dict[str, float]] | None,
    price_forecasts: dict[datetime, float],
    exchange_unit: ExchangeUnit | None,
    exchange_profiles: dict[datetime, tuple[float, float]],
    learning_session: LearningEpisodeSession | None = None,
) -> list[SupplyOffer]:
    """机组卖单；启用 Exchange 时加入进口卖单"""

    offers: list[SupplyOffer] = []
    for plant in plants:
        available_powers = {
            delivery_start: available_power_mw(
                plant, delivery_start, availability_profiles
            )
            for delivery_start, _ in opening.products
        }

        def marginal_cost_at(
            delivery_start: datetime, current_plant: PowerPlant = plant
        ) -> float:
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
                marginal_cost = marginal_cost_at(start)
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
