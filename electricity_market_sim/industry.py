"""Rolling steel-production optimization and post-clearing dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

import numpy as np

try:
    from scipy.optimize import linprog
except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
    linprog = None
    _SCIPY_IMPORT_ERROR: Exception | None = exc
else:
    _SCIPY_IMPORT_ERROR = None

from .errors import InputValidationError
from .market_models import DemandBid, MarketClearingResult
from .models import (
    IndustrialDispatchResult,
    IndustrialFlexibilityResult,
    IndustrialOptimizationWindowResult,
    IndustrialPlan,
    IndustrialUnit,
)


_TOLERANCE = 1e-7


def _require_optimizer() -> None:
    if linprog is None:
        detail = f": {_SCIPY_IMPORT_ERROR}" if _SCIPY_IMPORT_ERROR else ""
        raise InputValidationError(
            "Industrial optimization requires SciPy with scipy.optimize.linprog "
            f"available{detail}."
        )


def _duration_hours(product: tuple[datetime, datetime]) -> float:
    return (product[1] - product[0]).total_seconds() / 3600


@dataclass(frozen=True)
class _ProcessCoefficients:
    dri_t_per_steel_t: float
    hydrogen_mwh_per_steel_t: float
    electrolyser_mwh_per_steel_t: float
    dri_mwh_per_steel_t: float
    eaf_mwh_per_steel_t: float
    grid_mwh_per_steel_t: float
    iron_ore_t_per_steel_t: float
    lime_t_per_steel_t: float
    co2_t_per_steel_t: float


def process_coefficients(unit: IndustrialUnit) -> _ProcessCoefficients:
    """Return the linear material and electricity requirements per tonne steel."""

    dri = unit.eaf.specific_dri_demand
    hydrogen = dri * unit.dri_plant.specific_hydrogen_consumption
    electrolyser_energy = hydrogen / unit.electrolyser.efficiency
    dri_energy = dri * unit.dri_plant.specific_electricity_consumption
    eaf_energy = unit.eaf.specific_electricity_consumption
    lime = unit.eaf.specific_lime_demand
    return _ProcessCoefficients(
        dri_t_per_steel_t=dri,
        hydrogen_mwh_per_steel_t=hydrogen,
        electrolyser_mwh_per_steel_t=electrolyser_energy,
        dri_mwh_per_steel_t=dri_energy,
        eaf_mwh_per_steel_t=eaf_energy,
        grid_mwh_per_steel_t=electrolyser_energy + dri_energy + eaf_energy,
        iron_ore_t_per_steel_t=(
            dri * unit.dri_plant.specific_iron_ore_consumption
        ),
        lime_t_per_steel_t=lime,
        co2_t_per_steel_t=lime * unit.eaf.lime_co2_factor,
    )


def _price(prices: dict[str, float], *names: str) -> float:
    for name in names:
        if name in prices:
            return prices[name]
    return 0.0


def missing_industrial_price_names(
    fuel_price_profiles: dict[datetime, dict[str, float]],
) -> tuple[str, ...]:
    """Return raw-material prices that will use the documented zero default."""

    columns = set(next(iter(fuel_price_profiles.values()), {}))
    missing: list[str] = []
    if not ({"iron ore", "iron_ore"} & columns):
        missing.append("iron ore")
    if "lime" not in columns:
        missing.append("lime")
    return tuple(missing)


def _steel_upper_bound(
    unit: IndustrialUnit, coefficients: _ProcessCoefficients, duration: float
) -> float:
    limits = (
        unit.electrolyser.max_power_mw
        * duration
        / coefficients.electrolyser_mwh_per_steel_t,
        unit.dri_plant.max_power_mw
        * duration
        / coefficients.dri_mwh_per_steel_t,
        unit.eaf.max_power_mw * duration / coefficients.eaf_mwh_per_steel_t,
    )
    return min(limits)


def _power_coefficients(
    coefficients: _ProcessCoefficients, duration: float
) -> tuple[float, float, float, float]:
    electrolyser = coefficients.electrolyser_mwh_per_steel_t / duration
    dri = coefficients.dri_mwh_per_steel_t / duration
    eaf = coefficients.eaf_mwh_per_steel_t / duration
    return electrolyser, dri, eaf, electrolyser + dri + eaf


def maximum_industry_production(
    unit: IndustrialUnit,
    products: tuple[tuple[datetime, datetime], ...],
    initial_powers_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Maximize steel production under device capacity and ramp constraints."""

    if not products:
        return 0.0
    _require_optimizer()
    coefficients = process_coefficients(unit)
    durations = np.array([_duration_hours(product) for product in products])
    bounds = [
        (0.0, _steel_upper_bound(unit, coefficients, duration))
        for duration in durations
    ]
    power_per_tonne = [
        _power_coefficients(coefficients, duration)[:3] for duration in durations
    ]
    rows: list[np.ndarray] = []
    limits: list[float] = []
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    for index in range(len(products)):
        for device_index, device in enumerate(devices):
            coefficient = power_per_tonne[index][device_index]
            increase = np.zeros(len(products))
            increase[index] = coefficient
            decrease = np.zeros(len(products))
            decrease[index] = -coefficient
            if index:
                previous_coefficient = power_per_tonne[index - 1][device_index]
                increase[index - 1] = -previous_coefficient
                decrease[index - 1] = previous_coefficient
                previous_power = None
            else:
                previous_power = initial_powers_mw[device_index]
            rows.append(increase)
            limits.append(
                device.ramp_up_mw
                + (previous_power if previous_power is not None else 0.0)
            )
            rows.append(decrease)
            limits.append(
                device.ramp_down_mw
                - (previous_power if previous_power is not None else 0.0)
            )
    result = linprog(
        -np.ones(len(products)),
        A_ub=np.vstack(rows),
        b_ub=np.array(limits),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise InputValidationError(
            f"Maximum-production calculation failed for {unit.name!r}: "
            f"{result.message}"
        )
    return float(np.sum(result.x))


@dataclass(frozen=True)
class _IndustryProblem:
    products: tuple[tuple[datetime, datetime], ...]
    durations: np.ndarray
    prices: np.ndarray
    references: np.ndarray
    variable_cost_per_tonne: np.ndarray
    grid_power_per_tonne: np.ndarray
    electrolyser_power_per_tonne: np.ndarray
    dri_power_per_tonne: np.ndarray
    eaf_power_per_tonne: np.ndarray
    steel_offset: int
    positive_deviation_offset: int
    negative_deviation_offset: int
    variable_count: int
    objective: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...]


def _build_industry_problem(
    unit: IndustrialUnit,
    products: tuple[tuple[datetime, datetime], ...],
    remaining_products: tuple[tuple[datetime, datetime], ...],
    commit_count: int,
    price_forecast: dict[datetime, float],
    normalized_load_profile: dict[datetime, float],
    fuel_price_profiles: dict[datetime, dict[str, float]],
    demand_to_schedule_t: float,
    initial_powers_mw: tuple[float, float, float],
) -> _IndustryProblem:
    if not products or products != remaining_products[: len(products)]:
        raise InputValidationError(
            "Industrial optimization products must be a non-empty prefix of the "
            "remaining products."
        )
    if not 0 < commit_count <= len(products):
        raise InputValidationError("Industrial commit interval is outside its window.")
    if not isfinite(demand_to_schedule_t) or demand_to_schedule_t < 0:
        raise InputValidationError("Industrial remaining demand must be non-negative.")

    remaining_load_profile: list[float] = []
    for start, _ in remaining_products:
        try:
            normalized = normalized_load_profile[start]
        except KeyError as exc:
            raise InputValidationError(
                "Industrial normalized load profile is missing "
                f"{start.isoformat(sep=' ')} for {unit.name!r}."
            ) from exc
        if not isfinite(normalized) or normalized < 0:
            raise InputValidationError(
                "Industrial normalized load profile must be a non-negative "
                f"finite value at {start.isoformat(sep=' ')} for {unit.name!r}."
            )
        remaining_load_profile.append(normalized)

    coefficients = process_coefficients(unit)
    count = len(products)
    remaining_count = len(remaining_products)
    steel_offset = 0
    positive_offset = remaining_count
    negative_offset = remaining_count + count
    variable_count = remaining_count + 2 * count
    remaining_durations = np.array(
        [_duration_hours(product) for product in remaining_products]
    )
    if np.any(remaining_durations <= 0):
        raise InputValidationError("Industrial products must have positive durations.")
    durations = remaining_durations[:count]
    remaining_power_coefficients = np.array(
        [
            _power_coefficients(coefficients, duration)
            for duration in remaining_durations
        ]
    )

    prices = np.empty(count)
    references = np.empty(count)
    variable_costs = np.empty(count)
    electrolyser_power = remaining_power_coefficients[:count, 0]
    dri_power = remaining_power_coefficients[:count, 1]
    eaf_power = remaining_power_coefficients[:count, 2]
    grid_power = remaining_power_coefficients[:count, 3]
    for index, (start, _) in enumerate(products):
        try:
            prices[index] = price_forecast[start]
            raw_prices = fuel_price_profiles[start]
        except KeyError as exc:
            raise InputValidationError(
                f"Industrial input is missing {start.isoformat(sep=' ')} for "
                f"{unit.name!r}."
            ) from exc
        references[index] = (
            remaining_load_profile[index] * unit.electrolyser.max_power_mw
        )
        variable_costs[index] = (
            coefficients.grid_mwh_per_steel_t * prices[index]
            + coefficients.iron_ore_t_per_steel_t
            * _price(raw_prices, "iron ore", "iron_ore")
            + coefficients.lime_t_per_steel_t * _price(raw_prices, "lime")
            + coefficients.co2_t_per_steel_t * _price(raw_prices, "co2")
        )
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, _steel_upper_bound(unit, coefficients, duration))
        for duration in remaining_durations
    ]
    bounds.extend([(0.0, None)] * (2 * count))

    equality_rows: list[np.ndarray] = []
    equality_values: list[float] = []
    for index in range(count):
        row = np.zeros(variable_count)
        row[steel_offset + index] = grid_power[index]
        row[positive_offset + index] = -1.0
        row[negative_offset + index] = 1.0
        equality_rows.append(row)
        equality_values.append(references[index])

    inequality_rows: list[np.ndarray] = []
    inequality_values: list[float] = []
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    device_coefficients = tuple(
        remaining_power_coefficients[:, index] for index in range(3)
    )
    for index in range(remaining_count):
        for device_index, device in enumerate(devices):
            coefficient = device_coefficients[device_index][index]
            increase = np.zeros(variable_count)
            increase[steel_offset + index] = coefficient
            decrease = np.zeros(variable_count)
            decrease[steel_offset + index] = -coefficient
            if index:
                previous = device_coefficients[device_index][index - 1]
                increase[steel_offset + index - 1] = -previous
                decrease[steel_offset + index - 1] = previous
                initial = 0.0
            else:
                initial = initial_powers_mw[device_index]
            inequality_rows.extend((increase, decrease))
            inequality_values.extend(
                (device.ramp_up_mw + initial, device.ramp_down_mw - initial)
            )

    # Keep a physically reachable production path through every remaining
    # product.  Variables after the look-ahead window carry no objective cost;
    # they only prove that this window leaves enough ramp-feasible capacity.
    row = np.zeros(variable_count)
    row[steel_offset : steel_offset + remaining_count] = 1.0
    equality_rows.append(row)
    equality_values.append(demand_to_schedule_t)

    if commit_count < len(remaining_products):
        remaining_profile_total = sum(remaining_load_profile)
        commit_profile_total = sum(remaining_load_profile[:commit_count])
        ratio = (
            commit_profile_total / remaining_profile_total
            if remaining_profile_total > _TOLERANCE
            else commit_count / len(remaining_products)
        )
        profile_minimum = demand_to_schedule_t * ratio * (
            1.0 - unit.load_profile_deviation
        )
        row = np.zeros(variable_count)
        row[steel_offset : steel_offset + commit_count] = -1.0
        inequality_rows.append(row)
        inequality_values.append(-profile_minimum)

    objective = np.zeros(variable_count)
    objective[steel_offset : steel_offset + count] = variable_costs
    penalty = 10.0 / max(unit.load_profile_deviation, 0.01)
    objective[positive_offset : positive_offset + count] = penalty
    objective[negative_offset : negative_offset + count] = penalty
    return _IndustryProblem(
        products=products,
        durations=durations,
        prices=prices,
        references=references,
        variable_cost_per_tonne=variable_costs,
        grid_power_per_tonne=grid_power,
        electrolyser_power_per_tonne=electrolyser_power,
        dri_power_per_tonne=dri_power,
        eaf_power_per_tonne=eaf_power,
        steel_offset=steel_offset,
        positive_deviation_offset=positive_offset,
        negative_deviation_offset=negative_offset,
        variable_count=variable_count,
        objective=objective,
        A_ub=(np.vstack(inequality_rows) if inequality_rows else None),
        b_ub=(np.array(inequality_values) if inequality_values else None),
        A_eq=np.vstack(equality_rows),
        b_eq=np.array(equality_values),
        bounds=tuple(bounds),
    )


def _solve(
    unit: IndustrialUnit,
    problem: _IndustryProblem,
    objective: np.ndarray,
    *,
    variable_cost_limit: float | None = None,
) -> np.ndarray:
    _require_optimizer()
    A_ub = problem.A_ub
    b_ub = problem.b_ub
    if variable_cost_limit is not None:
        row = np.zeros(problem.variable_count)
        count = len(problem.products)
        row[problem.steel_offset : problem.steel_offset + count] = (
            problem.variable_cost_per_tonne
        )
        A_ub = row.reshape(1, -1) if A_ub is None else np.vstack([A_ub, row])
        limit = variable_cost_limit + _TOLERANCE
        b_ub = np.array([limit]) if b_ub is None else np.append(b_ub, limit)
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        bounds=problem.bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise InputValidationError(
            f"Industrial optimization failed for {unit.name!r}: {result.message}"
        )
    return np.where(np.abs(result.x) < _TOLERANCE, 0.0, result.x)


def optimize_industry_window(
    unit: IndustrialUnit,
    products: tuple[tuple[datetime, datetime], ...],
    remaining_products: tuple[tuple[datetime, datetime], ...],
    commit_count: int,
    price_forecast: dict[datetime, float],
    normalized_load_profile: dict[datetime, float],
    fuel_price_profiles: dict[datetime, dict[str, float]],
    demand_to_schedule_t: float,
    initial_powers_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[
    tuple[IndustrialPlan, ...],
    tuple[IndustrialFlexibilityResult, ...],
    IndustrialOptimizationWindowResult,
]:
    """Solve one rolling baseline and pointwise flexibility bounds."""

    maximum = maximum_industry_production(
        unit, remaining_products, initial_powers_mw
    )
    if demand_to_schedule_t > maximum + _TOLERANCE:
        raise InputValidationError(
            f"Industrial demand is infeasible for {unit.name!r}: remaining "
            f"demand={demand_to_schedule_t:.6f} t, maximum production="
            f"{maximum:.6f} t, shortfall={demand_to_schedule_t - maximum:.6f} t."
        )
    problem = _build_industry_problem(
        unit,
        products,
        remaining_products,
        commit_count,
        price_forecast,
        normalized_load_profile,
        fuel_price_profiles,
        demand_to_schedule_t,
        initial_powers_mw,
    )
    solution = _solve(unit, problem, problem.objective)
    count = len(products)
    steel = solution[problem.steel_offset : problem.steel_offset + count]
    baseline_variable_cost = float(
        np.dot(problem.variable_cost_per_tonne, steel)
    )
    flexible_limit = baseline_variable_cost * (
        1.0 + unit.cost_tolerance_percent / 100.0
    )
    window_id = f"{unit.name}::{products[0][0].isoformat()}"
    coefficients = process_coefficients(unit)
    plans: list[IndustrialPlan] = []
    for index, (start, end) in enumerate(products):
        steel_output = steel[index]
        duration = problem.durations[index]
        grid_power = problem.grid_power_per_tonne[index] * steel_output
        plans.append(
            IndustrialPlan(
                window_id=window_id,
                delivery_start=start,
                delivery_end=end,
                unit_name=unit.name,
                forecast_price_eur_per_mwh=problem.prices[index],
                reference_power_mw=problem.references[index],
                electrolyser_power_mw=(
                    problem.electrolyser_power_per_tonne[index] * steel_output
                ),
                hydrogen_output_mwh=(
                    coefficients.hydrogen_mwh_per_steel_t * steel_output
                ),
                dri_power_mw=problem.dri_power_per_tonne[index] * steel_output,
                dri_output_t=coefficients.dri_t_per_steel_t * steel_output,
                eaf_power_mw=problem.eaf_power_per_tonne[index] * steel_output,
                planned_steel_output_t=steel_output,
                planned_grid_power_mw=grid_power,
                planned_energy_mwh=grid_power * duration,
                forecast_cost_eur=(
                    problem.variable_cost_per_tonne[index] * steel_output
                ),
            )
        )

    flexibility: list[IndustrialFlexibilityResult] = []
    for index, plan in enumerate(plans[:commit_count]):
        minimum_objective = np.zeros(problem.variable_count)
        minimum_objective[problem.steel_offset + index] = (
            problem.grid_power_per_tonne[index]
        )
        minimum_solution = _solve(
            unit,
            problem,
            minimum_objective,
            variable_cost_limit=flexible_limit,
        )
        maximum_objective = -minimum_objective
        maximum_solution = _solve(
            unit,
            problem,
            maximum_objective,
            variable_cost_limit=flexible_limit,
        )
        minimum_power = (
            minimum_solution[problem.steel_offset + index]
            * problem.grid_power_per_tonne[index]
        )
        maximum_power = (
            maximum_solution[problem.steel_offset + index]
            * problem.grid_power_per_tonne[index]
        )
        if minimum_power > plan.planned_grid_power_mw + _TOLERANCE or (
            maximum_power < plan.planned_grid_power_mw - _TOLERANCE
        ):
            raise InputValidationError(
                f"Industrial flexibility bounds exclude the baseline for "
                f"{unit.name!r} at {plan.delivery_start.isoformat(sep=' ')}."
            )
        flexibility.append(
            IndustrialFlexibilityResult(
                window_id=window_id,
                delivery_start=plan.delivery_start,
                delivery_end=plan.delivery_end,
                unit_name=unit.name,
                baseline_power_mw=plan.planned_grid_power_mw,
                minimum_power_mw=minimum_power,
                maximum_power_mw=maximum_power,
            )
        )

    window = IndustrialOptimizationWindowResult(
        window_id=window_id,
        unit_name=unit.name,
        optimization_start=products[0][0],
        optimization_end=products[-1][1],
        commit_start=products[0][0],
        commit_end=products[commit_count - 1][1],
        baseline_variable_cost_eur=baseline_variable_cost,
        maximum_flexible_variable_cost_eur=flexible_limit,
    )
    return tuple(plans), tuple(flexibility), window


def dispatch_industry(
    plan: IndustrialPlan,
    accepted_energy_mwh: float,
    opening_time: datetime,
) -> IndustrialDispatchResult:
    """Scale a planned production chain by its accepted purchase ratio."""

    if accepted_energy_mwh < -_TOLERANCE or (
        accepted_energy_mwh > plan.planned_energy_mwh + _TOLERANCE
    ):
        raise InputValidationError(
            f"Accepted industrial energy for {plan.unit_name!r} must be between "
            "zero and its submitted volume."
        )
    if plan.planned_energy_mwh <= _TOLERANCE:
        ratio = 0.0
    else:
        ratio = max(0.0, accepted_energy_mwh) / plan.planned_energy_mwh
    duration = _duration_hours((plan.delivery_start, plan.delivery_end))
    return IndustrialDispatchResult(
        opening_time=opening_time,
        window_id=plan.window_id,
        delivery_start=plan.delivery_start,
        delivery_end=plan.delivery_end,
        unit_name=plan.unit_name,
        electrolyser_power_mw=plan.electrolyser_power_mw,
        hydrogen_output_mwh=plan.hydrogen_output_mwh,
        dri_power_mw=plan.dri_power_mw,
        dri_output_t=plan.dri_output_t,
        eaf_power_mw=plan.eaf_power_mw,
        planned_steel_output_t=plan.planned_steel_output_t,
        planned_grid_power_mw=plan.planned_grid_power_mw,
        planned_energy_mwh=plan.planned_energy_mwh,
        actual_grid_power_mw=(accepted_energy_mwh / duration),
        actual_steel_output_t=(ratio * plan.planned_steel_output_t),
        forecast_cost_eur=plan.forecast_cost_eur,
    )


class IndustryRollingCoordinator:
    """Maintain committed plans and delivered production across market events."""

    def __init__(
        self,
        units: tuple[IndustrialUnit, ...],
        all_products: tuple[tuple[datetime, datetime], ...],
        price_forecasts: dict[str, dict[datetime, float]],
        normalized_load_profiles: dict[str, dict[datetime, float]],
        fuel_price_profiles: dict[datetime, dict[str, float]],
        maximum_bid_price: float,
    ) -> None:
        self.units = units
        self.all_products = tuple(sorted(set(all_products)))
        self.product_set = frozenset(self.all_products)
        self.price_forecasts = price_forecasts
        self.normalized_load_profiles = normalized_load_profiles
        self.fuel_price_profiles = fuel_price_profiles
        self.maximum_bid_price = maximum_bid_price
        self.plans: dict[tuple[str, datetime], IndustrialPlan] = {}
        self.delivered: set[tuple[str, datetime]] = set()
        self.actual_steel_t = {unit.name: 0.0 for unit in units}
        self.last_actual_powers_mw = {
            unit.name: (0.0, 0.0, 0.0) for unit in units
        }
        self.dispatch_results: list[IndustrialDispatchResult] = []
        self.flexibility_results: list[IndustrialFlexibilityResult] = []
        self.window_results: list[IndustrialOptimizationWindowResult] = []

    @staticmethod
    def _cleared_industry_energy(
        result: MarketClearingResult, unit_name: str
    ) -> float:
        cleared = next(
            (
                item
                for item in result.demand_bids
                if item.bid.unit_name == unit_name
                and item.bid.demand_type == "industrial_load"
            ),
            None,
        )
        return 0.0 if cleared is None else cleared.accepted_energy_mwh

    def _result_for_plan(
        self,
        plan: IndustrialPlan,
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> MarketClearingResult:
        matches = [
            result
            for result in scheduled_results.values()
            if result.delivery_start == plan.delivery_start
            and result.delivery_end == plan.delivery_end
        ]
        if len(matches) != 1:
            raise InputValidationError(
                f"Expected one cleared market result for pending industrial product "
                f"{plan.delivery_start.isoformat(sep=' ')}, found {len(matches)}."
            )
        return matches[0]

    @staticmethod
    def _actual_device_powers(
        plan: IndustrialPlan, accepted_energy_mwh: float
    ) -> tuple[float, float, float]:
        ratio = (
            0.0
            if plan.planned_energy_mwh <= _TOLERANCE
            else accepted_energy_mwh / plan.planned_energy_mwh
        )
        return (
            ratio * plan.electrolyser_power_mw,
            ratio * plan.dri_power_mw,
            ratio * plan.eaf_power_mw,
        )

    def _plan_window(
        self,
        unit: IndustrialUnit,
        first_product: tuple[datetime, datetime],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> None:
        remaining_products = tuple(
            product
            for product in self.all_products
            if product[0] >= first_product[0]
            and (unit.name, product[0]) not in self.plans
        )
        if not remaining_products or remaining_products[0] != first_product:
            raise InputValidationError(
                f"Cannot construct the next industrial window for {unit.name!r} at "
                f"{first_product[0].isoformat(sep=' ')}."
            )
        optimization_end = first_product[0] + unit.look_ahead
        products = tuple(
            product for product in remaining_products if product[0] < optimization_end
        )
        commit_end = first_product[0] + unit.commit_horizon
        commit_count = sum(
            1 for product in products if product[0] < commit_end
        )

        pending: list[tuple[IndustrialPlan, MarketClearingResult, float]] = []
        for (unit_name, start), plan in self.plans.items():
            if unit_name != unit.name or (unit_name, start) in self.delivered:
                continue
            if plan.delivery_start >= first_product[0]:
                continue
            result = self._result_for_plan(plan, scheduled_results)
            accepted = self._cleared_industry_energy(result, unit.name)
            pending.append((plan, result, accepted))
        pending.sort(key=lambda item: item[0].delivery_start)
        pending_steel = sum(
            dispatch_industry(
                plan,
                accepted,
                result.opening_time or result.delivery_start,
            ).actual_steel_output_t
            for plan, result, accepted in pending
        )
        remaining_demand = max(
            0.0, unit.demand_t - self.actual_steel_t[unit.name]
        )
        demand_to_schedule = max(0.0, remaining_demand - pending_steel)
        initial_powers = self.last_actual_powers_mw[unit.name]
        if pending:
            last_plan, _, accepted = pending[-1]
            if last_plan.delivery_end == first_product[0]:
                initial_powers = self._actual_device_powers(last_plan, accepted)

        plans, flexibility, window = optimize_industry_window(
            unit=unit,
            products=products,
            remaining_products=remaining_products,
            commit_count=commit_count,
            price_forecast=self.price_forecasts[unit.name],
            normalized_load_profile=self.normalized_load_profiles[unit.name],
            fuel_price_profiles=self.fuel_price_profiles,
            demand_to_schedule_t=demand_to_schedule,
            initial_powers_mw=initial_powers,
        )
        for plan in plans[:commit_count]:
            self.plans[(unit.name, plan.delivery_start)] = plan
        self.flexibility_results.extend(flexibility)
        self.window_results.append(window)

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> list[DemandBid]:
        """Ensure committed plans exist and create non-negative industrial bids."""

        bids: list[DemandBid] = []
        eligible_products = tuple(
            product for product in products if product in self.product_set
        )
        if not eligible_products:
            return bids
        for unit in self.units:
            missing = next(
                (
                    product
                    for product in eligible_products
                    if (unit.name, product[0]) not in self.plans
                ),
                None,
            )
            if missing is not None:
                self._plan_window(unit, missing, scheduled_results)
            for start, end in eligible_products:
                plan = self.plans[(unit.name, start)]
                if plan.planned_energy_mwh <= _TOLERANCE:
                    continue
                bids.append(
                    DemandBid(
                        unit_name=unit.name,
                        operator=unit.operator,
                        delivery_start=start,
                        delivery_end=end,
                        volume_mwh=plan.planned_energy_mwh,
                        price_eur_per_mwh=self.maximum_bid_price,
                        bid_id=f"{unit.name}::industry::{start.isoformat()}",
                        demand_type="industrial_load",
                    )
                )
        return bids

    def record_delivery(self, result: MarketClearingResult) -> None:
        """Apply one completed product to actual steel production."""

        for unit in self.units:
            key = (unit.name, result.delivery_start)
            plan = self.plans.get(key)
            if plan is None or key in self.delivered:
                continue
            accepted = self._cleared_industry_energy(result, unit.name)
            dispatch = dispatch_industry(
                plan,
                accepted,
                result.opening_time or result.delivery_start,
            )
            self.dispatch_results.append(dispatch)
            self.actual_steel_t[unit.name] += dispatch.actual_steel_output_t
            self.last_actual_powers_mw[unit.name] = self._actual_device_powers(
                plan, accepted
            )
            self.delivered.add(key)

    def validate_completion(self) -> None:
        """Reject a run whose final market deliveries cannot satisfy steel demand."""

        for unit in self.units:
            shortfall = unit.demand_t - self.actual_steel_t[unit.name]
            if shortfall > _TOLERANCE:
                raise InputValidationError(
                    f"Industrial demand is incomplete for {unit.name!r} at the "
                    f"simulation end: remaining demand={shortfall:.6f} t, maximum "
                    f"production=0.000000 t, shortfall={shortfall:.6f} t."
                )
