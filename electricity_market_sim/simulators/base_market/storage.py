"""储能运行状态管理、已成交交付时段的电量推算，以及实际交付后的 SOC 记录。"""

from dataclasses import dataclass
from datetime import datetime
from ...errors import InputValidationError
from ...models import (
    MarketClearingResult,
    MarketOpening,
    MarketSettings,
    StorageDispatchResult,
    StorageUnit,
)
from ...utils import format_timestamp

_POWER_TOLERANCE_MW = 1e-9


@dataclass
class StorageRuntimeState:
    """记录储能单元的实际电量，仅在交付结束后更新，未来成交产品的影响另行推算。"""

    energy_mwh: float

    def record_dispatch(
        self,
        storage: StorageUnit,
        accepted_charge_mwh: float,
        accepted_discharge_mwh: float,
        duration_hours: float,
    ) -> tuple[float, float]:
        """执行一个已交付产品的调度，并返回调度前后的储能量。"""

        before = self.energy_mwh
        after = (
            before
            + storage.natural_inflow_mw * duration_hours
            + accepted_charge_mwh * storage.efficiency_charge
            - accepted_discharge_mwh / storage.efficiency_discharge
        )
        tolerance = 1e-7
        if after < storage.min_energy_mwh - tolerance:
            raise ValueError(
                f"Storage {storage.name!r} dispatch falls below min_soc: "
                f"{after:.9f} MWh < {storage.min_energy_mwh:.9f} MWh."
            )
        # 自然流入超过剩余储能容量的部分会被弃用。
        # 提交的充电订单电量已在计入自然流入后确定。
        self.energy_mwh = min(
            storage.max_energy_mwh, max(storage.min_energy_mwh, after)
        )
        return before, self.energy_mwh


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


def _project_storage_energies(
    *,
    opening: MarketOpening,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
    scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
) -> dict[str, float]:
    """根据当前电量及首个交付时段开始前会完成的已成交交易，计入自然流入和充放电效率，推算各储能单元的交付初始电量。"""

    first_delivery = opening.products[0][0]
    pending_before_delivery = sorted(
        (
            result
            for result in scheduled_results.values()
            if result.delivery_end <= first_delivery
        ),
        key=lambda result: (result.delivery_end, result.delivery_start),
    )
    projected = {}
    for storage in storages:
        # 从当前实际电量开始推算。
        energy = storage_states[storage.name].energy_mwh
        for market_result in pending_before_delivery:
            # 计入该交付时段的自然流入，超过容量的部分弃用。
            energy = min(
                storage.max_energy_mwh,
                energy + storage.natural_inflow_mw * market_result.duration_hours,
            )
            # 充电成交量乘以效率，得到实际存入电量。
            energy += sum(
                item.accepted_energy_mwh for item in market_result.demand_bids
                if item.bid.unit_name == storage.name and item.bid.demand_type == "storage_charge"
            ) * storage.efficiency_charge
            # 放电成交量除以效率，得到实际消耗电量。
            energy -= sum(
                item.accepted_energy_mwh for item in market_result.offers
                if item.offer.unit_name == storage.name and item.offer.offer_type == "storage_discharge"
            ) / storage.efficiency_discharge
            
        tolerance = 1e-7
        if not (
            storage.min_energy_mwh - tolerance
            <= energy
            <= storage.max_energy_mwh + tolerance
        ): # 最小电量 - 容差 <= 当前电量 <= 最大电量 + 容差
            raise InputValidationError(
                f"根据已成交交易推算，储能 {storage.name!r} 在本次开市首个交付时段"
                f"开始前（{format_timestamp(first_delivery)}）的预计电量超出了 SOC 允许范围。"
            )
        projected[storage.name] = min(
            storage.max_energy_mwh, max(storage.min_energy_mwh, energy))
    return projected


def _record_storage_states(
    market_result: MarketClearingResult,
    storages: tuple[StorageUnit, ...],
    storage_states: dict[str, StorageRuntimeState],
) -> list[StorageDispatchResult]:
    """交付结束后，根据自然流入和充放电成交量更新储能电量，并返回供状态核对和结果输出使用的调度记录。"""

    records: list[StorageDispatchResult] = []
    for storage in storages:
        charge_orders = [
            item for item in market_result.demand_bids
            if item.bid.unit_name == storage.name and item.bid.demand_type == "storage_charge"
        ]
        discharge_orders = [
            item for item in market_result.offers
            if item.offer.unit_name == storage.name and item.offer.offer_type == "storage_discharge"
        ]
        accepted_charge = sum(item.accepted_energy_mwh for item in charge_orders)
        accepted_discharge = sum(item.accepted_energy_mwh for item in discharge_orders)
        before, after = storage_states[storage.name].record_dispatch(
            storage,
            accepted_charge,
            accepted_discharge,
            market_result.duration_hours,
        )
        records.append(
            StorageDispatchResult(
                opening_time=market_result.opening_time,
                delivery_start=market_result.delivery_start,
                delivery_end=market_result.delivery_end,
                unit=storage,
                energy_before_mwh=before,
                energy_after_mwh=after,
                offered_charge_mwh=sum(item.bid.volume_mwh for item in charge_orders),
                accepted_charge_mwh=accepted_charge,
                charge_bid_price_eur_per_mwh=(
                    charge_orders[0].bid.price_eur_per_mwh if charge_orders else None),
                offered_discharge_mwh=sum(item.offer.offered_energy_mwh for item in discharge_orders),
                accepted_discharge_mwh=accepted_discharge,
                discharge_bid_price_eur_per_mwh=(
                    discharge_orders[0].offer.bid_price_eur_per_mwh if discharge_orders else None),
                clearing_price_eur_per_mwh=(market_result.clearing_price_eur_per_mwh),
            )
        )
    return records
