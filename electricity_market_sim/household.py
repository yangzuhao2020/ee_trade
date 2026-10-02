"""Forecast planning and post-clearing dispatch for V3 household participants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
    Bounds = None
    LinearConstraint = None
    milp = None
    _SCIPY_IMPORT_ERROR: Exception | None = exc
else:
    _SCIPY_IMPORT_ERROR = None

from .errors import InputValidationError
from .models import (
    HouseholdDispatchResult,
    HouseholdFlexibilityResult,
    HouseholdPlan,
    HouseholdUnit,
)


_TOLERANCE = 1e-8


def _require_household_optimizer() -> None:
    if milp is None or Bounds is None or LinearConstraint is None:
        detail = f": {_SCIPY_IMPORT_ERROR}" if _SCIPY_IMPORT_ERROR else ""
        raise InputValidationError(
            "Household optimization requires SciPy with scipy.optimize.milp "
            f"available{detail}."
        )


def _duration_hours(product: tuple[datetime, datetime]) -> float:
    start, end = product
    return (end - start).total_seconds() / 3600


@dataclass(frozen=True)
class _HouseholdProblem:
    durations: np.ndarray
    prices: np.ndarray
    heat_demands: np.ndarray
    heat_powers: np.ndarray
    variable_count: int
    grid_offset: int
    charge_offset: int
    discharge_offset: int
    energy_offset: int
    bounds: Bounds
    integrality: np.ndarray
    matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray


def _build_household_problem(
    household: HouseholdUnit,
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    heat_forecast: dict[datetime, float],
    initial_energy_mwh: float,
    initial_heat_pump_power_mw: float,
    initial_battery_charge_power_mw: float,
    initial_battery_discharge_power_mw: float,
) -> _HouseholdProblem:
    if not products:
        raise InputValidationError("Household optimization requires at least one product.")
    if initial_energy_mwh < household.min_energy_mwh - _TOLERANCE or (
        initial_energy_mwh > household.max_energy_mwh + _TOLERANCE
    ):
        raise InputValidationError(
            f"Household {household.name!r} initial battery energy is outside its "
            "SOC limits."
        )
    initial_powers = (
        initial_heat_pump_power_mw,
        initial_battery_charge_power_mw,
        initial_battery_discharge_power_mw,
    )
    if any(not np.isfinite(value) or value < 0 for value in initial_powers):
        raise InputValidationError(
            f"Household {household.name!r} initial device powers must be "
            "non-negative finite values."
        )
    if initial_heat_pump_power_mw > household.heat_pump_max_power_mw + _TOLERANCE:
        raise InputValidationError(
            f"Household {household.name!r} initial heat-pump power exceeds its limit."
        )
    if (
        initial_battery_charge_power_mw
        > household.battery_max_charge_power_mw + _TOLERANCE
        or initial_battery_discharge_power_mw
        > household.battery_max_discharge_power_mw + _TOLERANCE
    ):
        raise InputValidationError(
            f"Household {household.name!r} initial battery power exceeds its limit."
        )
    if (
        initial_battery_charge_power_mw > _TOLERANCE
        and initial_battery_discharge_power_mw > _TOLERANCE
    ):
        raise InputValidationError(
            f"Household {household.name!r} cannot initially charge and discharge "
            "at the same time."
        )

    count = len(products)
    grid_offset = 0
    charge_offset = count
    discharge_offset = 2 * count
    energy_offset = 3 * count
    mode_offset = 4 * count
    variable_count = 5 * count

    durations = np.array([_duration_hours(product) for product in products])
    if np.any(durations <= 0):
        raise InputValidationError("Household products must have positive durations.")
    prices = np.empty(count)
    heat_demands = np.empty(count)
    heat_powers = np.empty(count)
    for index, (start, _) in enumerate(products):
        try:
            prices[index] = price_forecast[start]
        except KeyError as exc:
            raise InputValidationError(
                f"forecasts_df.csv is missing price_EOM at {start.isoformat(sep=' ')}."
            ) from exc
        try:
            heat_demands[index] = heat_forecast[start]
        except KeyError as exc:
            raise InputValidationError(
                f"forecasts_df.csv is missing {household.name}_heat_demand at "
                f"{start.isoformat(sep=' ')}."
            ) from exc
        if heat_demands[index] < 0:
            raise InputValidationError("Household heat demand cannot be negative.")
        heat_powers[index] = heat_demands[index] / household.cop
        if heat_powers[index] > household.heat_pump_max_power_mw + _TOLERANCE:
            raise InputValidationError(
                f"Household {household.name!r} heat demand at "
                f"{start.isoformat(sep=' ')} exceeds heat-pump capacity."
            )
        if (
            heat_powers[index] > _TOLERANCE
            and heat_powers[index]
            < household.heat_pump_min_power_mw - _TOLERANCE
        ):
            raise InputValidationError(
                f"Household {household.name!r} heat-pump power at "
                f"{start.isoformat(sep=' ')} is below its configured minimum."
            )
        previous_heat_power = (
            heat_powers[index - 1] if index else initial_heat_pump_power_mw
        )
        increase = heat_powers[index] - previous_heat_power
        decrease = previous_heat_power - heat_powers[index]
        if increase > household.heat_pump_ramp_up_mw + _TOLERANCE or (
            decrease > household.heat_pump_ramp_down_mw + _TOLERANCE
        ):
            raise InputValidationError(
                f"Household {household.name!r} heat demand requires an "
                "infeasible heat-pump ramp."
            )

    lower = np.zeros(variable_count)
    upper = np.empty(variable_count)
    upper[grid_offset : grid_offset + count] = (
        household.fixed_power_mw
        + heat_powers
        + household.battery_max_charge_power_mw
    )
    upper[charge_offset : charge_offset + count] = (
        household.battery_max_charge_power_mw
    )
    upper[discharge_offset : discharge_offset + count] = (
        household.battery_max_discharge_power_mw
    )
    lower[energy_offset : energy_offset + count] = household.min_energy_mwh
    upper[energy_offset : energy_offset + count] = household.max_energy_mwh
    upper[mode_offset : mode_offset + count] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[mode_offset : mode_offset + count] = 1

    rows: list[np.ndarray] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: dict[int, float], minimum: float, maximum: float
    ) -> None:
        row = np.zeros(variable_count)
        for variable, coefficient in coefficients.items():
            row[variable] = coefficient
        rows.append(row)
        row_lower.append(minimum)
        row_upper.append(maximum)

    for index in range(count):
        retention = (1.0 - household.battery_loss_rate) ** durations[index]
        fixed_and_heat = household.fixed_power_mw + heat_powers[index]
        add_constraint(
            {
                grid_offset + index: 1.0,
                charge_offset + index: -1.0,
                discharge_offset + index: 1.0,
            },
            fixed_and_heat,
            fixed_and_heat,
        )

        energy_coefficients = {
            energy_offset + index: 1.0,
            charge_offset + index: (
                -household.battery_efficiency_charge * durations[index]
            ),
            discharge_offset + index: (
                durations[index] / household.battery_efficiency_discharge
            ),
        }
        if index:
            energy_coefficients[energy_offset + index - 1] = -retention
            energy_rhs = 0.0
        else:
            energy_rhs = retention * initial_energy_mwh
        add_constraint(energy_coefficients, energy_rhs, energy_rhs)

        # mode=1 permits charging; mode=0 permits discharging.
        add_constraint(
            {
                charge_offset + index: 1.0,
                mode_offset + index: -household.battery_max_charge_power_mw,
            },
            -np.inf,
            0.0,
        )
        add_constraint(
            {
                discharge_offset + index: 1.0,
                mode_offset + index: household.battery_max_discharge_power_mw,
            },
            -np.inf,
            household.battery_max_discharge_power_mw,
        )

        for offset, initial_power in (
            (charge_offset, initial_battery_charge_power_mw),
            (discharge_offset, initial_battery_discharge_power_mw),
        ):
            if index:
                add_constraint(
                    {offset + index: 1.0, offset + index - 1: -1.0},
                    -household.battery_ramp_down_mw,
                    household.battery_ramp_up_mw,
                )
            else:
                add_constraint(
                    {offset: 1.0},
                    max(0.0, initial_power - household.battery_ramp_down_mw),
                    initial_power + household.battery_ramp_up_mw,
                )

    add_constraint(
        {energy_offset + count - 1: 1.0},
        initial_energy_mwh,
        np.inf,
    )
    return _HouseholdProblem(
        durations=durations,
        prices=prices,
        heat_demands=heat_demands,
        heat_powers=heat_powers,
        variable_count=variable_count,
        grid_offset=grid_offset,
        charge_offset=charge_offset,
        discharge_offset=discharge_offset,
        energy_offset=energy_offset,
        bounds=Bounds(lower, upper),
        integrality=integrality,
        matrix=np.vstack(rows),
        constraint_lower=np.array(row_lower),
        constraint_upper=np.array(row_upper),
    )


def _solve_household_problem(
    household: HouseholdUnit,
    problem: _HouseholdProblem,
    objective: np.ndarray,
    extra_rows: tuple[tuple[np.ndarray, float, float], ...] = (),
) -> np.ndarray:
    _require_household_optimizer()
    assert milp is not None and LinearConstraint is not None
    matrix = problem.matrix
    lower = problem.constraint_lower
    upper = problem.constraint_upper
    if extra_rows:
        matrix = np.vstack([matrix, *(row for row, _, _ in extra_rows)])
        lower = np.concatenate(
            [lower, np.array([minimum for _, minimum, _ in extra_rows])]
        )
        upper = np.concatenate(
            [upper, np.array([maximum for _, _, maximum in extra_rows])]
        )
    result = milp(
        objective,
        integrality=problem.integrality,
        bounds=problem.bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise InputValidationError(
            f"Household optimization failed for {household.name!r}: {result.message}"
        )
    return np.where(np.abs(result.x) < _TOLERANCE, 0.0, result.x)


def optimize_household(
    household: HouseholdUnit,
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    heat_forecast: dict[datetime, float],
    initial_energy_mwh: float,
    *,
    initial_heat_pump_power_mw: float = 0.0,
    initial_battery_charge_power_mw: float = 0.0,
    initial_battery_discharge_power_mw: float = 0.0,
) -> tuple[HouseholdPlan, ...]:
    """Minimize forecast purchase cost subject to heat-pump and battery limits."""

    if not products:
        return ()
    _require_household_optimizer()
    problem = _build_household_problem(
        household,
        products,
        price_forecast,
        heat_forecast,
        initial_energy_mwh,
        initial_heat_pump_power_mw,
        initial_battery_charge_power_mw,
        initial_battery_discharge_power_mw,
    )
    count = len(products)
    objective = np.zeros(problem.variable_count)
    # Fixed heat-pump consumption is a constant in the purchase-cost objective.
    # Removing that constant keeps the MILP objective near the small battery
    # scale and prevents solver tolerances from accepting needless cycling.
    objective[problem.charge_offset : problem.charge_offset + count] = (
        problem.prices * problem.durations
    )
    objective[problem.discharge_offset : problem.discharge_offset + count] = (
        -problem.prices * problem.durations
    )
    solution = _solve_household_problem(household, problem, objective)
    plans: list[HouseholdPlan] = []
    for index, (start, end) in enumerate(products):
        energy_after = solution[problem.energy_offset + index]
        plans.append(
            HouseholdPlan(
                delivery_start=start,
                delivery_end=end,
                unit_name=household.name,
                forecast_price_eur_per_mwh=problem.prices[index],
                heat_demand_mw_th=problem.heat_demands[index],
                fixed_power_mw=household.fixed_power_mw,
                planned_grid_power_mw=solution[problem.grid_offset + index],
                planned_heat_pump_power_mw=problem.heat_powers[index],
                planned_battery_charge_power_mw=solution[
                    problem.charge_offset + index
                ],
                planned_battery_discharge_power_mw=solution[
                    problem.discharge_offset + index
                ],
                planned_soc_after=energy_after / household.battery_capacity_mwh,
            )
        )
    return tuple(plans)


def evaluate_household_flexibility(
    household: HouseholdUnit,
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    heat_forecast: dict[datetime, float],
    initial_energy_mwh: float,
    baseline_plans: tuple[HouseholdPlan, ...],
    *,
    initial_heat_pump_power_mw: float = 0.0,
    initial_battery_charge_power_mw: float = 0.0,
    initial_battery_discharge_power_mw: float = 0.0,
) -> tuple[HouseholdFlexibilityResult, ...]:
    """Calculate cost-tolerant minimum and maximum grid power per product."""

    if not products:
        return ()
    _require_household_optimizer()
    if len(baseline_plans) != len(products) or any(
        plan.unit_name != household.name
        or plan.delivery_start != product[0]
        or plan.delivery_end != product[1]
        for plan, product in zip(baseline_plans, products, strict=True)
    ):
        raise InputValidationError(
            f"Household {household.name!r} flexibility requires one matching "
            "baseline plan per product."
        )
    if household.cost_tolerance_percent < 0 or not np.isfinite(
        household.cost_tolerance_percent
    ):
        raise InputValidationError(
            f"Household {household.name!r} cost_tolerance must be non-negative."
        )

    problem = _build_household_problem(
        household,
        products,
        price_forecast,
        heat_forecast,
        initial_energy_mwh,
        initial_heat_pump_power_mw,
        initial_battery_charge_power_mw,
        initial_battery_discharge_power_mw,
    )
    count = len(products)
    baseline_cost = sum(
        problem.prices[index]
        * baseline_plans[index].planned_grid_power_mw
        * problem.durations[index]
        for index in range(count)
    )
    cost_limit = baseline_cost + abs(baseline_cost) * (
        household.cost_tolerance_percent / 100
    )
    cost_row = np.zeros(problem.variable_count)
    cost_row[problem.grid_offset : problem.grid_offset + count] = (
        problem.prices * problem.durations
    )
    cost_constraint = ((cost_row, -np.inf, cost_limit + _TOLERANCE),)

    results: list[HouseholdFlexibilityResult] = []
    for index, (start, end) in enumerate(products):
        minimum_objective = np.zeros(problem.variable_count)
        minimum_objective[problem.grid_offset + index] = 1.0
        minimum_solution = _solve_household_problem(
            household,
            problem,
            minimum_objective,
            cost_constraint,
        )

        maximum_objective = np.zeros(problem.variable_count)
        maximum_objective[problem.grid_offset + index] = -1.0
        maximum_solution = _solve_household_problem(
            household,
            problem,
            maximum_objective,
            cost_constraint,
        )
        minimum = minimum_solution[problem.grid_offset + index]
        maximum = maximum_solution[problem.grid_offset + index]
        baseline = baseline_plans[index].planned_grid_power_mw
        if minimum > baseline + _TOLERANCE or maximum < baseline - _TOLERANCE:
            raise InputValidationError(
                f"Household {household.name!r} flexibility bounds exclude its "
                f"baseline at {start.isoformat(sep=' ')}."
            )
        results.append(
            HouseholdFlexibilityResult(
                delivery_start=start,
                delivery_end=end,
                unit_name=household.name,
                minimum_grid_power_mw=minimum,
                maximum_grid_power_mw=maximum,
            )
        )
    return tuple(results)


def dispatch_household(
    household: HouseholdUnit,
    plans: tuple[HouseholdPlan, ...],
    accepted_grid_energy_mwh: dict[datetime, float],
    initial_energy_mwh: float,
    *,
    initial_heat_pump_power_mw: float = 0.0,
    initial_battery_charge_power_mw: float = 0.0,
    initial_battery_discharge_power_mw: float = 0.0,
) -> tuple[HouseholdDispatchResult, ...]:
    """Apply fixed market acceptances and update physical SOC sequentially."""

    energy = initial_energy_mwh
    previous_heat_pump = initial_heat_pump_power_mw
    previous_charge = initial_battery_charge_power_mw
    previous_discharge = initial_battery_discharge_power_mw
    records: list[HouseholdDispatchResult] = []
    for plan in plans:
        duration = _duration_hours((plan.delivery_start, plan.delivery_end))
        retained_energy = energy * (
            (1.0 - household.battery_loss_rate) ** duration
        )
        accepted_energy = accepted_grid_energy_mwh.get(plan.delivery_start, 0.0)
        grid_power = accepted_energy / duration
        required_power = plan.fixed_power_mw + plan.planned_heat_pump_power_mw

        maximum_discharge = min(
            household.battery_max_discharge_power_mw,
            previous_discharge + household.battery_ramp_up_mw,
            max(0.0, retained_energy - household.min_energy_mwh)
            * household.battery_efficiency_discharge
            / duration,
        )
        if grid_power + _TOLERANCE < required_power:
            discharge = min(
                plan.planned_battery_discharge_power_mw,
                maximum_discharge,
                required_power - grid_power,
            )
        else:
            discharge = 0.0

        available_power = grid_power + discharge
        served_fixed = min(plan.fixed_power_mw, available_power)
        remaining_for_heat = max(0.0, available_power - served_fixed)
        heat_pump_power = min(
            plan.planned_heat_pump_power_mw,
            remaining_for_heat,
            previous_heat_pump + household.heat_pump_ramp_up_mw,
        )
        remaining_after_heat = max(
            0.0,
            available_power - served_fixed - heat_pump_power,
        )

        maximum_charge = min(
            household.battery_max_charge_power_mw,
            previous_charge + household.battery_ramp_up_mw,
            max(0.0, household.max_energy_mwh - retained_energy)
            / household.battery_efficiency_charge
            / duration,
        )
        charge = 0.0 if discharge > _TOLERANCE else min(
            remaining_after_heat,
            maximum_charge,
        )
        unallocated_power = remaining_after_heat - charge
        if unallocated_power > _TOLERANCE:
            raise InputValidationError(
                f"Household {household.name!r} accepted purchase at "
                f"{plan.delivery_start.isoformat(sep=' ')} cannot be allocated "
                "to the heat pump or battery within device constraints."
            )

        energy_before = energy
        next_energy = (
            retained_energy
            + charge * duration * household.battery_efficiency_charge
            - discharge * duration / household.battery_efficiency_discharge
        )
        if next_energy < household.min_energy_mwh - _TOLERANCE:
            raise InputValidationError(
                f"Household {household.name!r} accepted purchase at "
                f"{plan.delivery_start.isoformat(sep=' ')} cannot cover battery "
                "losses while maintaining min_soc."
            )
        if next_energy > household.max_energy_mwh + _TOLERANCE:
            raise InputValidationError(
                f"Household {household.name!r} dispatch at "
                f"{plan.delivery_start.isoformat(sep=' ')} exceeds max_soc."
            )
        energy = min(
            household.max_energy_mwh,
            max(household.min_energy_mwh, next_energy),
        )
        unmet_electricity = max(0.0, plan.fixed_power_mw - served_fixed) * duration
        unmet_heat = max(
            0.0,
            plan.heat_demand_mw_th - heat_pump_power * household.cop,
        ) * duration
        records.append(
            HouseholdDispatchResult(
                delivery_start=plan.delivery_start,
                delivery_end=plan.delivery_end,
                unit_name=household.name,
                forecast_price_eur_per_mwh=plan.forecast_price_eur_per_mwh,
                heat_demand_mw_th=plan.heat_demand_mw_th,
                fixed_power_mw=plan.fixed_power_mw,
                planned_grid_power_mw=plan.planned_grid_power_mw,
                heat_pump_power_mw=heat_pump_power,
                battery_charge_power_mw=charge,
                battery_discharge_power_mw=discharge,
                soc_before=energy_before / household.battery_capacity_mwh,
                soc_after=energy / household.battery_capacity_mwh,
                unmet_electricity_mwh=unmet_electricity,
                unmet_heat_mwh_th=unmet_heat,
            )
        )
        previous_heat_pump = heat_pump_power
        previous_charge = charge
        previous_discharge = discharge
    return tuple(records)
