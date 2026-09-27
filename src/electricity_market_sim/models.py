"""Small, explicit data structures used by the electricity-market simulator.
例如：
- 市场配置：MarketSettings
- 发电机组与需求单元：PowerPlant、DemandUnit
- 买单和卖单：DemandBid、SupplyOffer
- 单时段出清结果：MarketClearingResult
- 完整仿真结果：SimulationResult
它还提供成交电量转成交功率、收入、成本、利润等基础计算，并检查能量守恒。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .market_models import (
    ClearedDemandBid,
    ClearedSupplyOffer,
    DemandBid,
    MarketClearingResult,
    SupplyOffer,
    Trade,
)

__all__ = [
    "ClearedDemandBid",
    "ClearedSupplyOffer",
    "DemandBid",
    "DemandUnit",
    "ExchangeSchedule",
    "ExchangeUnit",
    "HouseholdDispatchResult",
    "HouseholdFlexibilityResult",
    "HouseholdPlan",
    "HouseholdUnit",
    "IndustrialDevice",
    "IndustrialDispatchResult",
    "IndustrialFlexibilityResult",
    "IndustrialOptimizationWindowResult",
    "IndustrialPlan",
    "IndustrialUnit",
    "LearningConfig",
    "LearningStepResult",
    "LearningTransition",
    "MarketClearingResult",
    "MarketOpening",
    "MarketSettings",
    "PowerPlant",
    "SimulationResult",
    "StorageClearingContext",
    "StorageDispatchResult",
    "StorageUnit",
    "SupplyOffer",
    "Trade",
]


@dataclass(frozen=True)
class LearningConfig:
    """Validated Version 5 single-agent MATD3 settings."""

    learning_mode: bool
    algorithm: str
    learning_rate: float
    training_episodes: int
    initial_experience_episodes: int
    replay_buffer_size: int
    batch_size: int
    gamma: float
    train_frequency_steps: int
    gradient_steps: int
    validation_interval: int
    exploration_noise_std: float
    noise_sigma: float
    noise_scale: float
    noise_dt: float
    action_noise_schedule: str
    tau: float
    policy_delay: int
    target_policy_noise: float
    target_noise_clip: float
    actor_architecture: str = "mlp"
    device: str = "cpu"
    continue_learning: bool = False
    trained_policies_save_path: str | None = None
    trained_policies_load_path: str | None = None
    max_bid_price: float = 100.0


@dataclass(frozen=True)
class MarketSettings:
    """The EOM settings shared by simple and complex V2 market openings."""

    start: datetime
    end: datetime
    time_step: timedelta
    opening_frequency: timedelta
    opening_duration: timedelta
    product_duration: timedelta
    product_count: int
    first_delivery: timedelta
    maximum_bid_price: float
    minimum_bid_price: float
    market_mechanism: str
    market_id: str = "EOM"
    exchange_units_file: str | None = None
    industrial_dsm_units_file: str | None = None
    additional_fields: frozenset[str] = field(default_factory=frozenset)
    learning_config: LearningConfig | None = None
    seed: int | None = None


@dataclass(frozen=True)
class MarketOpening:
    """All delivery products offered together at one market opening."""

    opening_time: datetime
    products: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class DemandUnit:
    """An EOM demand participant, optionally generated from elasticity inputs."""

    name: str
    operator: str
    bidding_strategy: str = "demand_energy_naive"
    profile_column: str | None = None
    max_power_mw: float | None = None
    elasticity: float | None = None
    elasticity_model: str | None = None
    max_price_eur_per_mwh: float | None = None
    num_bids: int | None = None
    price_eur_per_mwh: float | None = None

    @property
    def is_elastic(self) -> bool:
        return self.bidding_strategy == "demand_energy_heuristic_elastic"


@dataclass(frozen=True)
class ExchangeUnit:
    """The single virtual participant representing planned imports and exports."""

    name: str
    operator: str
    price_import_eur_per_mwh: float
    price_export_eur_per_mwh: float


@dataclass(frozen=True)
class ExchangeSchedule:
    """One delivery hour's positive planned import and export powers."""

    import_power_mw: float
    export_power_mw: float


@dataclass(frozen=True)
class PowerPlant:
    """A dispatchable plant and the parameters used by its bidding strategy."""

    name: str
    operator: str
    technology: str
    bidding_strategy: str
    fuel_type: str
    emission_factor: float
    max_power_mw: float
    min_power_mw: float
    efficiency: float
    additional_cost_eur_per_mwh: float
    # These optional V2 parameters retain the V2 example defaults when their
    # columns are absent from ``powerplant_units.csv``.
    start_cost_eur: float = 0.0
    min_operating_time_hours: float = 1.0
    min_down_time_hours: float = 1.0

    def marginal_cost(self, fuel_prices: dict[str, float]) -> float:
        """Return the ASSUME-compatible variable marginal cost.

        Input emission factors are interpreted per unit of fuel input, hence both
        fuel and CO2 costs are divided by the electrical efficiency.
        """

        # Variable renewables in the supplied V2 examples have no fuel-price
        # column; their fuel component is conventionally zero.
        fuel_price = fuel_prices.get(self.fuel_type, 0.0)
        co2_price = fuel_prices["co2"]
        return (
            (fuel_price + co2_price * self.emission_factor) / self.efficiency
            + self.additional_cost_eur_per_mwh
        )


@dataclass(frozen=True)
class StorageUnit:
    """A storage participant and the physical limits used by its EOM strategy."""

    name: str
    operator: str
    technology: str
    bidding_strategy: str
    max_power_charge_mw: float
    max_power_discharge_mw: float
    efficiency_charge: float
    efficiency_discharge: float
    min_soc: float
    max_soc: float
    capacity_mwh: float
    initial_soc: float
    additional_cost_charge_eur_per_mwh: float
    additional_cost_discharge_eur_per_mwh: float
    natural_inflow_mw: float = 0.0

    @property
    def min_energy_mwh(self) -> float:
        return self.min_soc * self.capacity_mwh

    @property
    def max_energy_mwh(self) -> float:
        return self.max_soc * self.capacity_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.capacity_mwh


@dataclass(frozen=True)
class HouseholdUnit:
    """One building participant with a heat pump and an internal battery."""

    name: str
    operator: str
    node: str
    bidding_strategy: str
    objective: str
    flexibility_measure: str
    cost_tolerance_percent: float
    is_prosumer: bool
    fixed_power_mw: float
    heat_pump_max_power_mw: float
    heat_pump_min_power_mw: float
    heat_pump_ramp_up_mw: float
    heat_pump_ramp_down_mw: float
    cop: float
    battery_capacity_mwh: float
    battery_min_soc: float
    battery_max_soc: float
    battery_initial_soc: float
    battery_efficiency_charge: float
    battery_efficiency_discharge: float
    battery_max_charge_power_mw: float
    battery_max_discharge_power_mw: float
    battery_ramp_up_mw: float
    battery_ramp_down_mw: float
    battery_loss_rate: float = 0.0
    bid_price_eur_per_mwh: float = 3000.0

    @property
    def initial_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_initial_soc

    @property
    def min_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_min_soc

    @property
    def max_energy_mwh(self) -> float:
        return self.battery_capacity_mwh * self.battery_max_soc


@dataclass(frozen=True)
class HouseholdPlan:
    """A household's forecast-based schedule for one delivery product."""

    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    planned_heat_pump_power_mw: float
    planned_battery_charge_power_mw: float
    planned_battery_discharge_power_mw: float
    planned_soc_after: float


@dataclass(frozen=True)
class HouseholdDispatchResult:
    """Actual building operation after the corresponding buy order clears."""

    opening_time: datetime
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    heat_demand_mw_th: float
    fixed_power_mw: float
    planned_grid_power_mw: float
    heat_pump_power_mw: float
    battery_charge_power_mw: float
    battery_discharge_power_mw: float
    soc_before: float
    soc_after: float
    unmet_electricity_mwh: float
    unmet_heat_mwh_th: float


@dataclass(frozen=True)
class HouseholdFlexibilityResult:
    """Cost-tolerant grid-power bounds for one planned household product."""

    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    minimum_grid_power_mw: float
    maximum_grid_power_mw: float


@dataclass(frozen=True)
class IndustrialDevice:
    """One physical device inside an industrial steel plant."""

    technology: str
    fuel_type: str
    max_power_mw: float
    min_power_mw: float
    ramp_up_mw: float
    ramp_down_mw: float
    efficiency: float = 0.0
    specific_dri_demand: float = 0.0
    specific_electricity_consumption: float = 0.0
    specific_hydrogen_consumption: float = 0.0
    specific_iron_ore_consumption: float = 0.0
    specific_lime_demand: float = 0.0
    lime_co2_factor: float = 0.0
    min_operating_time_hours: float = 0.0
    min_down_time_hours: float = 0.0


@dataclass(frozen=True)
class IndustrialUnit:
    """One steel-plant participant assembled from three device rows."""

    name: str
    operator: str
    node: str
    bidding_strategy: str
    objective: str
    flexibility_measure: str
    cost_tolerance_percent: float
    demand_t: float
    load_profile_deviation: float
    horizon_mode: str
    look_ahead: timedelta
    commit_horizon: timedelta
    rolling_step: timedelta
    forecast_price_column: str
    electrolyser: IndustrialDevice
    dri_plant: IndustrialDevice
    eaf: IndustrialDevice


@dataclass(frozen=True)
class IndustrialPlan:
    """Forecast-based industrial schedule for one delivery product."""

    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    forecast_price_eur_per_mwh: float
    reference_power_mw: float
    electrolyser_power_mw: float
    hydrogen_output_mwh: float
    dri_power_mw: float
    dri_output_t: float
    eaf_power_mw: float
    planned_steel_output_t: float
    planned_grid_power_mw: float
    planned_energy_mwh: float
    forecast_cost_eur: float


@dataclass(frozen=True)
class IndustrialDispatchResult:
    """Planned and actual steel production after an industrial bid clears."""

    opening_time: datetime
    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    electrolyser_power_mw: float
    hydrogen_output_mwh: float
    dri_power_mw: float
    dri_output_t: float
    eaf_power_mw: float
    planned_steel_output_t: float
    planned_grid_power_mw: float
    planned_energy_mwh: float
    actual_grid_power_mw: float
    actual_steel_output_t: float
    forecast_cost_eur: float


@dataclass(frozen=True)
class IndustrialFlexibilityResult:
    """Pointwise cost-tolerant power bounds for one committed product."""

    window_id: str
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    baseline_power_mw: float
    minimum_power_mw: float
    maximum_power_mw: float

    @property
    def flex_up_mw(self) -> float:
        return self.maximum_power_mw - self.baseline_power_mw

    @property
    def flex_down_mw(self) -> float:
        return self.baseline_power_mw - self.minimum_power_mw


@dataclass(frozen=True)
class IndustrialOptimizationWindowResult:
    """One rolling window's non-repeated cost totals."""

    window_id: str
    unit_name: str
    optimization_start: datetime
    optimization_end: datetime
    commit_start: datetime
    commit_end: datetime
    baseline_variable_cost_eur: float
    maximum_flexible_variable_cost_eur: float


@dataclass(frozen=True)
class SimulationResult:
    """The complete, in-memory outcome of one simulation run."""

    settings: MarketSettings
    market_results: tuple[MarketClearingResult, ...]
    storage_results: tuple[StorageDispatchResult, ...] = ()
    household_results: tuple[HouseholdDispatchResult, ...] = ()
    household_flexibility_results: tuple[HouseholdFlexibilityResult, ...] = ()
    industry_results: tuple[IndustrialDispatchResult, ...] = ()
    industry_flexibility_results: tuple[IndustrialFlexibilityResult, ...] = ()
    industry_optimization_windows: tuple[IndustrialOptimizationWindowResult, ...] = ()
    learning_steps: tuple[LearningStepResult, ...] = ()
    learning_transitions: tuple[LearningTransition, ...] = ()
    learning_load_base_mw: float | None = None


@dataclass(frozen=True)
class LearningStepResult:
    """Auditable action, dispatch, and reward for one learning delivery."""

    delivery_start: datetime
    action: tuple[float, float]
    available_power_mw: float
    accepted_power_mw: float
    clearing_price_eur_per_mwh: float
    profit_eur: float
    regret_eur: float
    reward: float


@dataclass(frozen=True)
class LearningTransition:
    """One replay-buffer item using the unsorted executed action."""

    delivery_start: datetime
    state: tuple[float, ...]
    action: tuple[float, float]
    reward: float
    next_state: tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class StorageClearingContext:
    """Physical storage data required by one joint complex-clearing opening."""

    unit_name: str
    initial_energy_mwh: float
    min_energy_mwh: float
    max_energy_mwh: float
    efficiency_charge: float
    efficiency_discharge: float


@dataclass(frozen=True)
class StorageDispatchResult:
    """One storage unit's accepted dispatch and SOC transition for one product."""

    opening_time: datetime
    delivery_start: datetime
    delivery_end: datetime
    unit_name: str
    operator: str
    technology: str
    energy_before_mwh: float
    energy_after_mwh: float
    capacity_mwh: float
    offered_charge_mwh: float
    accepted_charge_mwh: float
    charge_bid_price_eur_per_mwh: float | None
    offered_discharge_mwh: float
    accepted_discharge_mwh: float
    discharge_bid_price_eur_per_mwh: float | None
    clearing_price_eur_per_mwh: float
    charge_payment_eur: float
    discharge_revenue_eur: float
    additional_charge_cost_eur: float
    additional_discharge_cost_eur: float

    @property
    def soc_before(self) -> float:
        return self.energy_before_mwh / self.capacity_mwh

    @property
    def soc_after(self) -> float:
        return self.energy_after_mwh / self.capacity_mwh

    @property
    def net_cash_flow_eur(self) -> float:
        return (
            self.discharge_revenue_eur
            - self.charge_payment_eur
            - self.additional_charge_cost_eur
            - self.additional_discharge_cost_eur
        )
