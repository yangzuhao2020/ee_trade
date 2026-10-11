"""V2 的发电可用容量、状态跟踪和报价策略。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .market_models import DemandBid, SupplyOffer
from .models import PowerPlant, StorageUnit
from .time_utils import format_timestamp, hours_between

_POWER_TOLERANCE_MW = 1e-9
_FORECAST_HOURS = 12


@dataclass
class PlantRuntimeState:
    """``powerplant_energy_heuristic_flexable`` 所需的时序运行状态。

    V2 约定机组初始处于停机状态，出力为零。本版本的调度约束
    有意不包含爬坡约束、热电联产（CHP）的供热承诺，
    以及最小连续运行和停机时间约束。
    """

    is_running: bool
    elapsed_periods: int = 0
    operating_periods: int = 0

    @classmethod
    def initially_off(cls) -> PlantRuntimeState:
        return cls(is_running=False)

    @property
    def average_operating_time(self) -> float:
        """V2 中历史运行时段占总时段的比例，而非连续运行时长。"""

        if self.elapsed_periods == 0:
            return 0.0
        return self.operating_periods / self.elapsed_periods

    def record_dispatch(self, accepted_power_mw: float) -> None:
        """在对应交付时段结束后更新运行状态。"""

        self.elapsed_periods += 1
        self.is_running = accepted_power_mw > _POWER_TOLERANCE_MW
        if self.is_running:
            self.operating_periods += 1


@dataclass
class StorageRuntimeState:
    """单个储能单元在已完成交付后的实际储能量。

    未来已成交的产品由仿真调度器保存，并单独推算其影响。
    此状态仅在交付时段结束后更新。
    """

    energy_mwh: float

    @classmethod
    def from_initial_soc(cls, storage: StorageUnit) -> StorageRuntimeState:
        return cls(energy_mwh=storage.initial_energy_mwh)

    def energy_after_dispatch(
        self,
        storage: StorageUnit,
        accepted_charge_mwh: float,
        accepted_discharge_mwh: float,
        duration_hours: float = 1.0,
    ) -> float:
        """根据电网侧成交的充放电量，计算下一时刻的内部储能量。"""

        return (
            self.energy_mwh
            + storage.natural_inflow_mw * duration_hours
            + accepted_charge_mwh * storage.efficiency_charge
            - accepted_discharge_mwh / storage.efficiency_discharge
        )

    def record_dispatch(
        self,
        storage: StorageUnit,
        accepted_charge_mwh: float,
        accepted_discharge_mwh: float,
        duration_hours: float = 1.0,
    ) -> tuple[float, float]:
        """执行一个已交付产品的调度，并返回调度前后的储能量。"""

        before = self.energy_mwh
        after = self.energy_after_dispatch(
            storage, accepted_charge_mwh, accepted_discharge_mwh, duration_hours
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


def storage_heuristic_orders(
    storage: StorageUnit,
    initial_energy_mwh: float,
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    *,
    forecast_start: datetime | None = None,
    forecast_end: datetime | None = None,
) -> tuple[list[DemandBid], list[SupplyOffer]]:
    """构建每小时互斥的充电或放电订单。

    参考储能量轨迹假设每个提交的订单均全额成交，仅用于确定合理的报价电量。
    复杂出清会另行约束由*实际成交*电量形成的储能量轨迹。
    在仿真边界处，以交付时刻为中心的价格窗口会被截断到可用预测区间，
    与 ASSUME 的边界处理方式一致。
    """

    charge_bids: list[DemandBid] = []
    discharge_offers: list[SupplyOffer] = []
    reference_energy = initial_energy_mwh
    available_start = forecast_start or min(price_forecast)
    available_end = forecast_end or max(price_forecast)
    for delivery_start, delivery_end in products:
        duration_hours = hours_between(delivery_start, delivery_end)
        reference_energy = min(
            storage.max_energy_mwh,
            reference_energy + storage.natural_inflow_mw * duration_hours,
        )
        window_times = [
            delivery_start + timedelta(hours=offset)
            for offset in range(-_FORECAST_HOURS, _FORECAST_HOURS + 1)
            if available_start
            <= delivery_start + timedelta(hours=offset)
            <= available_end
        ]
        window = [price_forecast[time] for time in window_times]
        average_price = sum(window) / len(window)
        current_price = price_forecast[delivery_start]
        timestamp = delivery_start.isoformat()

        if current_price < average_price:
            headroom_mwh = max(0.0, storage.max_energy_mwh - reference_energy)
            charge_energy_mwh = min(
                storage.max_power_charge_mw * duration_hours,
                headroom_mwh / storage.efficiency_charge,
            )
            if charge_energy_mwh > _POWER_TOLERANCE_MW:
                # 与 ASSUME 的储能启发式策略一致：额外充电成本
                # 影响实际利润，但不影响提交的报价。
                bid_price = average_price * storage.efficiency_charge
                charge_bids.append(
                    DemandBid(
                        unit_name=storage.name,
                        operator=storage.operator,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        volume_mwh=charge_energy_mwh,
                        price_eur_per_mwh=bid_price,
                        bid_type="storage_charge",
                        bid_id=f"{storage.name}::charge::{timestamp}",
                        demand_type="storage_charge",
                    )
                )
                reference_energy += (
                    charge_energy_mwh * storage.efficiency_charge
                )
        elif current_price > average_price:
            available_internal_mwh = max(
                0.0, reference_energy - storage.min_energy_mwh
            )
            discharge_energy_mwh = min(
                storage.max_power_discharge_mw * duration_hours,
                available_internal_mwh * storage.efficiency_discharge,
            )
            if discharge_energy_mwh > _POWER_TOLERANCE_MW:
                # 与 ASSUME 的储能启发式策略一致：额外放电成本
                # 在调度后结算，不在此处加价。
                bid_price = average_price / storage.efficiency_discharge
                identifier = f"{storage.name}::discharge::{timestamp}"
                discharge_offers.append(
                    SupplyOffer(
                        unit_name=storage.name,
                        operator=storage.operator,
                        technology=storage.technology,
                        delivery_start=delivery_start,
                        delivery_end=delivery_end,
                        offered_power_mw=discharge_energy_mwh / duration_hours,
                        offered_energy_mwh=discharge_energy_mwh,
                        bid_price_eur_per_mwh=bid_price,
                        marginal_cost_eur_per_mwh=(
                            storage.additional_cost_discharge_eur_per_mwh
                        ),
                        offer_type="storage_discharge",
                        offer_id=identifier,
                        offer_segment="storage_discharge",
                        bid_id=identifier,
                    )
                )
                reference_energy -= (
                    discharge_energy_mwh / storage.efficiency_discharge
                )

    return charge_bids, discharge_offers


def available_power_mw(
    plant: PowerPlant,
    delivery_start: datetime,
    availability_profiles: dict[str, dict[datetime, float]],
) -> float:
    """返回按可选可用率系数折算后的机组可用容量。"""

    profile = availability_profiles.get(plant.name)
    if profile is None:
        return plant.max_power_mw
    try:
        return plant.max_power_mw * profile[delivery_start]
    except KeyError as exc:
        raise KeyError(
            f"availability_df.csv has no hourly profile for "
            f"{format_timestamp(delivery_start)} and "
            f"plant {plant.name!r}."
        ) from exc


def naive_offer(
    plant: PowerPlant,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power: float,
    marginal_cost: float,
) -> SupplyOffer:
    """按 V2 的可用容量构建 V1 的单段边际成本报价。"""

    duration_hours = hours_between(delivery_start, delivery_end)
    offer_id = f"{plant.name}::single::{delivery_start.isoformat()}"
    return SupplyOffer(
        unit_name=plant.name,
        operator=plant.operator,
        technology=plant.technology,
        delivery_start=delivery_start,
        delivery_end=delivery_end,
        offered_power_mw=available_power,
        offered_energy_mwh=available_power * duration_hours,
        bid_price_eur_per_mwh=marginal_cost,
        marginal_cost_eur_per_mwh=marginal_cost,
        offer_id=offer_id,
        bid_id=offer_id,
    )


def segment_supply_offers(
    plant: PowerPlant,
    delivery_start: datetime,
    delivery_end: datetime,
    segment_specs: Iterable[tuple[str, float, float]],
    marginal_cost: float,
) -> list[SupplyOffer]:
    """为每个非零出力段构建一条标识为 ``{plant}::{segment}::{timestamp}`` 的报价。"""

    duration_hours = hours_between(delivery_start, delivery_end)
    timestamp = delivery_start.isoformat()
    offers: list[SupplyOffer] = []
    for segment, power_mw, price in segment_specs:
        if power_mw <= _POWER_TOLERANCE_MW:
            continue
        identifier = f"{plant.name}::{segment}::{timestamp}"
        offers.append(
            SupplyOffer(
                unit_name=plant.name,
                operator=plant.operator,
                technology=plant.technology,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                offered_power_mw=power_mw,
                offered_energy_mwh=power_mw * duration_hours,
                bid_price_eur_per_mwh=price,
                marginal_cost_eur_per_mwh=marginal_cost,
                offer_id=identifier,
                offer_segment=segment,
                bid_id=identifier,
            )
        )
    return offers


def heuristic_flexible_offers(
    plant: PowerPlant,
    state: PlantRuntimeState,
    delivery_start: datetime,
    delivery_end: datetime,
    available_power: float,
    marginal_cost: float,
    price_forecast: dict[datetime, float],
) -> list[SupplyOffer]:
    """将符合条件的机组拆分为 V2 的非灵活出力段和灵活出力段报价。

    灵活出力段始终按边际成本报价。停机机组将启动成本分摊到最低稳定出力段。
    对于运行中的机组，若指定的 13 个时间点的预测表明维持运行的未来价值
    非负，则该出力段可以按零价格报价。
    """

    duration_hours = hours_between(delivery_start, delivery_end)
    # 可用容量不足以达到最低稳定出力的机组不提交报价。
    # 这与出清规则不同：已提交的最低出力段仍可能仅部分成交。
    if available_power < plant.min_power_mw:
        return []

    inflexible_power = plant.min_power_mw
    flexible_power = available_power - inflexible_power

    if inflexible_power > _POWER_TOLERANCE_MW and state.is_running:
        future_prices = [
            price_forecast[delivery_start + timedelta(hours=hour)]
            for hour in range(_FORECAST_HOURS + 1)
        ]
        future_expected_profit = sum(price - marginal_cost for price in future_prices)
        # 简单预测未来是否盈利。
        current_forecast = future_prices[0]
        inflexible_price = (
            0.0
            if current_forecast < marginal_cost and future_expected_profit >= 0.0
            else marginal_cost
        ) # 机组已经处于运行状态时，最低稳定出力段 inflexible 应该报多少钱。
    elif inflexible_power > _POWER_TOLERANCE_MW:
        expected_operating_time = max(
            state.average_operating_time,
            plant.min_operating_time_hours,
            duration_hours,
        )
        inflexible_price = marginal_cost + plant.start_cost_eur / (
            expected_operating_time * inflexible_power
        )

    segment_specs = []
    if inflexible_power > _POWER_TOLERANCE_MW:
        segment_specs.append(("inflexible", inflexible_power, inflexible_price))
    if flexible_power > _POWER_TOLERANCE_MW:
        segment_specs.append(("flexible", flexible_power, marginal_cost))
    return segment_supply_offers(
        plant, delivery_start, delivery_end, segment_specs, marginal_cost
    )


def heuristic_block_offers(
    plant: PowerPlant,
    state: PlantRuntimeState,
    products: tuple[tuple[datetime, datetime], ...],
    available_powers_mw: dict[datetime, float],
    marginal_cost: float,
    price_forecast: dict[datetime, float],
    opening_time: datetime,
    linked: bool,
) -> list[SupplyOffer]:
    """为一次开市创建一个最低出力块报价（BB）及其逐小时灵活出力报价。

    构建报价时有意不更新 ``state``：V2 使用报价截止时的状态快照，
    为同一次日前开市的所有产品定价。
    """

    inflexible_legs: list[SupplyOffer] = []
    flexible_legs: list[SupplyOffer] = []
    for delivery_start, delivery_end in products:
        hourly = heuristic_flexible_offers(
            plant,
            state,
            delivery_start,
            delivery_end,
            available_powers_mw[delivery_start],
            marginal_cost,
            price_forecast,
        )
        for offer in hourly:
            if offer.offer_segment == "inflexible":
                inflexible_legs.append(offer)
            else:
                flexible_legs.append(offer)

    opening_id = opening_time.isoformat()
    parent_bid_id = f"{plant.name}::BB::{opening_id}"
    has_parent = bool(inflexible_legs)
    offers: list[SupplyOffer] = []
    if has_parent:
        total_energy = sum(leg.offered_energy_mwh for leg in inflexible_legs)
        block_price = sum(
            leg.bid_price_eur_per_mwh * leg.offered_energy_mwh
            for leg in inflexible_legs
        ) / total_energy
        offers.extend(
            replace(
                leg,
                offer_id=f"{parent_bid_id}::{leg.delivery_start.isoformat()}",
                bid_id=parent_bid_id,
                bid_type="BB",
                min_acceptance_ratio=1.0,
                parent_bid_id=None,
                bid_price_eur_per_mwh=block_price,
                offer_segment="block_inflexible",
            )
            for leg in inflexible_legs
        )

    for leg in flexible_legs:
        child_id = f"{plant.name}::{'LB' if linked and has_parent else 'SB'}::{opening_id}::{leg.delivery_start.isoformat()}"
        offers.append(
            replace(
                leg,
                offer_id=child_id,
                bid_id=child_id,
                bid_type="LB" if linked and has_parent else "SB",
                parent_bid_id=parent_bid_id if linked and has_parent else None,
                offer_segment="linked_flexible" if linked and has_parent else "flexible",
            )
        )
    return offers
