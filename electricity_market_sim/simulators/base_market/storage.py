"""储能报价、已成交交付时段的电量推算，以及实际交付后的 SOC 记录。"""

from datetime import datetime
from ...bidding import StorageRuntimeState, storage_heuristic_orders
from ...errors import InputValidationError
from ...market_models import DemandBid, MarketClearingResult, SupplyOffer
from ...models import (
    MarketOpening, MarketSettings, StorageClearingContext,
    StorageDispatchResult, StorageUnit,
)
from ...utils import format_timestamp

_POWER_TOLERANCE_MW = 1e-9


def _validate_storage_configuration(
    settings: MarketSettings,
    storages: tuple[StorageUnit, ...],
) -> None:
    """有储能参与时，检查以下市场配置要求：
    1. opening_frequency >= product_duration * product_count，
       避免不同开市覆盖的交付时段重叠。
    2. product_count > 1 时，market_mechanism 必须为 complex_clearing，
       以便处理跨时段荷电状态（SOC）约束。
    3. 采用 complex_clearing 时，所有储能的 natural_inflow_mw
       必须 <= _POWER_TOLERANCE_MW，当前复杂出清不支持超过容差的自然流入。
    不满足要求时抛出 InputValidationError；没有储能时直接返回。
    自然流入：即使没有通过用电抽水，储存的能量也可能增加。
    """
    if not storages:
        return

    product_horizon = settings.product_duration * settings.product_count
    if settings.opening_frequency < product_horizon:
        raise InputValidationError(
            "储能各次开市覆盖的交付时段不能重叠：EOM opening_frequency "
            f"（{settings.opening_frequency}）不得小于本次开市全部交付时段的总长度"
            f"（{product_horizon}）。"
        )

    if settings.product_count > 1 and settings.market_mechanism != "complex_clearing":
        raise InputValidationError(
            "储能一次参与多个交付时段的交易时，必须设置 "
            "market_mechanism: complex_clearing，以满足跨时段 SOC 约束。"
        )
    if settings.market_mechanism == "complex_clearing" and any(
        storage.natural_inflow_mw > _POWER_TOLERANCE_MW for storage in storages
    ):
        raise InputValidationError(
            "非零的自然流入（natural_inflow）目前仅支持 "
            "product_count = 1 的 pay_as_clear 市场。"
        )

def _storage_market_quantities(
    market_result: MarketClearingResult,
    storage_name: str,
) -> tuple[float, float, float | None, float, float, float | None]:
    """统计指定储能单元在一个交付时段中的充放电报价电量、成交电量和报价价格。

    返回值依次为：充电报价电量、充电成交电量、充电报价价格、
    放电报价电量、放电成交电量、放电报价价格；没有对应订单时，报价价格为 None。
    """

    charge_orders = [
        cleared
        for cleared in market_result.demand_bids
        if cleared.bid.unit_name == storage_name
        and cleared.bid.demand_type == "storage_charge"
    ]
    discharge_orders = [
        cleared
        for cleared in market_result.offers
        if cleared.offer.unit_name == storage_name
        and cleared.offer.offer_type == "storage_discharge"
    ]
    offered_charge = sum(item.bid.volume_mwh for item in charge_orders)
    accepted_charge = sum(item.accepted_energy_mwh for item in charge_orders)
    offered_discharge = sum(item.offer.offered_energy_mwh for item in discharge_orders)
    accepted_discharge = sum(item.accepted_energy_mwh for item in discharge_orders)
    charge_price = charge_orders[0].bid.price_eur_per_mwh if charge_orders else None
    discharge_price = (
        discharge_orders[0].offer.bid_price_eur_per_mwh if discharge_orders else None
    )
    return (
        offered_charge,
        accepted_charge,
        charge_price,
        offered_discharge,
        accepted_discharge,
        discharge_price,
    )

def _project_storage_energies(
    *,
    opening: MarketOpening,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
    scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
) -> dict[str, float]:
    """根据当前电量和已成交的交付安排，推算本次开市首个交付时段开始前的储能量。

    计入开市后至首个交付时段开始时结束的交易，依次考虑自然流入、充放电成交电量和效率。
    推算结果超出 SOC 允许范围时抛出 InputValidationError。
    返回以储能名称为键、预计储能量（MWh）为值的字典。
    """

    first_delivery = opening.products[0][0]
    pending_before_delivery = sorted(
        (
            result
            for result in scheduled_results.values()
            if result.delivery_end > opening.opening_time
            and result.delivery_end <= first_delivery
        ),
        key=lambda result: (result.delivery_end, result.delivery_start),
    )
    projected: dict[str, float] = {}
    for storage in storages:
        energy = storage_states[storage.name].energy_mwh
        for market_result in pending_before_delivery:
            (
                _,
                accepted_charge,
                _,
                _,
                accepted_discharge,
                _,
            ) = _storage_market_quantities(market_result, storage.name)
            energy = min(
                storage.max_energy_mwh,
                energy + storage.natural_inflow_mw * market_result.duration_hours,
            )
            energy += accepted_charge * storage.efficiency_charge
            energy -= accepted_discharge / storage.efficiency_discharge
        tolerance = 1e-7
        if not (
            storage.min_energy_mwh - tolerance
            <= energy
            <= storage.max_energy_mwh + tolerance
        ):
            raise InputValidationError(
                f"根据已成交交易推算，储能 {storage.name!r} 在本次开市首个交付时段"
                f"开始前（{format_timestamp(first_delivery)}）的预计电量超出了 SOC 允许范围。"
            )
        projected[storage.name] = min(
            storage.max_energy_mwh, max(storage.min_energy_mwh, energy)
        )
    return projected

def _storage_orders_for_opening(
    *,
    opening: MarketOpening,
    storages: tuple[StorageUnit, ...],
    initial_energies_mwh: dict[str, float],
    price_forecasts: dict[datetime, float],
    forecast_start: datetime,
    forecast_end: datetime,
) -> tuple[
    list[DemandBid],
    list[SupplyOffer],
    tuple[StorageClearingContext, ...],
]:
    """根据各储能单元的预计交付初始电量和电价预测，生成本次开市各产品的充电买单与放电卖单。
    同时整理初始电量、电量上下限和充放电效率，供复杂出清约束实际成交后的储能电量变化。
    返回充电买单列表、放电卖单列表，以及各储能单元的物理约束数据元组。
    """

    charge_bids: list[DemandBid] = []
    discharge_offers: list[SupplyOffer] = []
    contexts: list[StorageClearingContext] = []
    for storage in storages:
        initial_energy = initial_energies_mwh[storage.name]
        charges, discharges = storage_heuristic_orders(
            storage,
            initial_energy,
            opening.products,
            price_forecasts,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        charge_bids.extend(charges)
        discharge_offers.extend(discharges)
        contexts.append(
            StorageClearingContext(
                unit_name=storage.name,
                initial_energy_mwh=initial_energy,
                min_energy_mwh=storage.min_energy_mwh,
                max_energy_mwh=storage.max_energy_mwh,
                efficiency_charge=storage.efficiency_charge,
                efficiency_discharge=storage.efficiency_discharge,
            )
        )
    return charge_bids, discharge_offers, tuple(contexts)

def _record_storage_states(
    market_result: MarketClearingResult,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
) -> list[StorageDispatchResult]:
    """在交付时段结束后，根据自然流入和实际成交的充放电量更新各储能单元的电量。

    返回交付前后电量、充放电报价与成交电量，以及交易收支等记录，供 SOC 核对和结果输出使用。
    """

    opening_time = market_result.opening_time or market_result.delivery_start
    records: list[StorageDispatchResult] = []
    for storage in storages:
        (
            offered_charge,
            accepted_charge,
            charge_price,
            offered_discharge,
            accepted_discharge,
            discharge_price,
        ) = _storage_market_quantities(market_result, storage.name)
        before, after = storage_states[storage.name].record_dispatch(
            storage,
            accepted_charge,
            accepted_discharge,
            market_result.duration_hours,
        )
        records.append(
            StorageDispatchResult(
                opening_time=opening_time,
                delivery_start=market_result.delivery_start,
                delivery_end=market_result.delivery_end,
                unit_name=storage.name,
                operator=storage.operator,
                technology=storage.technology,
                energy_before_mwh=before,
                energy_after_mwh=after,
                capacity_mwh=storage.capacity_mwh,
                offered_charge_mwh=offered_charge,
                accepted_charge_mwh=accepted_charge,
                charge_bid_price_eur_per_mwh=charge_price,
                offered_discharge_mwh=offered_discharge,
                accepted_discharge_mwh=accepted_discharge,
                discharge_bid_price_eur_per_mwh=discharge_price,
                clearing_price_eur_per_mwh=(market_result.clearing_price_eur_per_mwh),
                charge_payment_eur=(
                    accepted_charge * market_result.clearing_price_eur_per_mwh
                ),
                discharge_revenue_eur=(
                    accepted_discharge * market_result.clearing_price_eur_per_mwh
                ),
                additional_charge_cost_eur=(
                    accepted_charge * storage.additional_cost_charge_eur_per_mwh
                ),
                additional_discharge_cost_eur=(
                    accepted_discharge * storage.additional_cost_discharge_eur_per_mwh
                ),
            )
        )
    return records
