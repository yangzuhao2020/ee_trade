"""第三版居民参与者的购电计划优化、灵活性评估及出清后设备调度。
根据预测电价和热需求制定购电与电池充放电计划，评估成本容忍度下的购电功率上下界。
出清后按实际成交电量安排热泵和电池运行，更新电池 SOC，并记录未满足的用电与供热需求。
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from .errors import InputValidationError
from .models import (
    HouseholdDispatchResult,
    HouseholdPlan,
    HouseholdUnit,
)
from .utils import hours_between


_TOLERANCE = 1e-8


@dataclass(frozen=True)
class _HouseholdProblem:
    """把一个家庭的优化模型数据保存到同一个对象里"""
    products: tuple[tuple[datetime, datetime], ...]
    durations: np.ndarray
    prices: np.ndarray
    heat_demands: np.ndarray
    heat_powers: np.ndarray
    variable_count: int
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    bounds: Bounds
    integrality: np.ndarray
    matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray


@dataclass(frozen=True)
class _HouseholdVariables:
    """优化变量分块的列索引、取值范围和整数标记。"""
    variable_count: int
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    energy: np.ndarray
    mode: np.ndarray
    bounds: Bounds
    integrality: np.ndarray


def _build_household_problem(
    household: HouseholdUnit,
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    heat_forecast: dict[datetime, float],
    initial_energy_mwh: float,
    initial_battery_charge_power_mw: float,
    initial_battery_discharge_power_mw: float,
) -> _HouseholdProblem:
    """准备预测并组装变量与约束。"""
    durations = np.array([hours_between(*product) for product in products])
    prices = np.array([price_forecast[start] for start, _ in products], dtype=float)
    heat_demands = np.array([heat_forecast[start] for start, _ in products], dtype=float)
    heat_powers = heat_demands / household.cop
    if np.any(heat_powers > household.heat_pump_max_power_mw + _TOLERANCE):
        raise InputValidationError(
            f"家庭用户 {household.name!r} 的热需求所需热泵功率超出功率上限。"
        )
    variables = _build_household_variables(household, heat_powers)
    matrix, constraint_lower, constraint_upper = _build_household_constraints(
        household,
        durations,
        heat_powers,
        variables,
        initial_energy_mwh,
        initial_battery_charge_power_mw,
        initial_battery_discharge_power_mw,
    )
    return _HouseholdProblem(
        products=products,
        durations=durations,
        prices=prices,
        heat_demands=heat_demands,
        heat_powers=heat_powers,
        variable_count=variables.variable_count,
        grid=variables.grid,
        charge=variables.charge,
        discharge=variables.discharge,
        bounds=variables.bounds,
        integrality=variables.integrality,
        matrix=matrix,
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
    )


def _build_household_variables(
    household: HouseholdUnit,
    heat_powers: np.ndarray,
) -> _HouseholdVariables:
    """为 grid 购电、充电、放电、能量和充放电模式定义变量。"""
    count = len(heat_powers)
    grid, charge, discharge, energy, mode = (
        np.arange(k * count, (k + 1) * count) for k in range(5))
    variable_count = 5 * count

    lower = np.zeros(variable_count)
    upper = np.empty(variable_count)
    upper[grid] = heat_powers + household.battery_max_charge_power_mw
    # 购电功率上限 = 热泵用电功率 + 电池最大充电功率
    upper[charge] = household.battery_max_charge_power_mw
    upper[discharge] = household.battery_max_discharge_power_mw
    lower[energy] = household.min_energy_mwh # 电池最低能量界限
    upper[energy] = household.max_energy_mwh # 电池最高能量界限
    upper[mode] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[mode] = 1 # 不同同时充放电。

    return _HouseholdVariables(
        variable_count=variable_count,
        grid=grid,
        charge=charge,
        discharge=discharge,
        energy=energy,
        mode=mode,
        bounds=Bounds(lower, upper),
        integrality=integrality,
    )


def _build_household_constraints(
    household: HouseholdUnit,
    durations: np.ndarray,
    heat_powers: np.ndarray,
    variables: _HouseholdVariables,
    initial_energy_mwh: float,
    initial_battery_charge_power_mw: float,
    initial_battery_discharge_power_mw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把家庭用电和电池运行必须遵守的规则，转换成优化求解器能够使用的线性约束。
    | 规则 | 被约束的表达式 | 下界 | 上界 |
    |---|---|---|---|
    | 功率平衡 | `购电 − 充电 + 放电` | 家庭用电需求 | 家庭用电需求 |
    | 放电模式控制 | `放电 + 最大放电功率 × mode` | 无限制 | 最大放电功率 |
    | 末端电量要求 | `最后时段的电池储能量` | 初始储能量 | 无限制 |
    """
    count = len(durations)
    rows: list[np.ndarray] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: dict[int, float], 
        minimum: float, maximum: float
    ) -> None:
        # minimum ≤ 各变量乘以对应系数后的总和 ≤ maximum
        row = np.zeros(variables.variable_count)
        for variable, coefficient in coefficients.items():
            row[variable] = coefficient

        rows.append(row)
        row_lower.append(minimum)
        row_upper.append(maximum)

    for index in range(count):
        add_constraint(
            {
                variables.grid[index]: 1.0,
                variables.charge[index]: -1.0,
                variables.discharge[index]: 1.0,
            },
            heat_powers[index],
            heat_powers[index],
        )

        _add_battery_energy_constraint(
            add_constraint,
            household,
            variables,
            index,
            durations[index],
            initial_energy_mwh,
        )

        # 充放电模式为 1 时允许充电，为 0 时允许放电。
        add_constraint(
            {
                variables.charge[index]: 1.0,
                variables.mode[index]: -household.battery_max_charge_power_mw,
            }, -np.inf, 0.0,)
        add_constraint(
            {
                variables.discharge[index]: 1.0,
                variables.mode[index]: household.battery_max_discharge_power_mw,
            }, -np.inf, household.battery_max_discharge_power_mw,)

        _add_battery_ramp_constraints(
            add_constraint,
            household,
            variables,
            index,
            initial_battery_charge_power_mw,
            initial_battery_discharge_power_mw,
        )

    add_constraint(
        {variables.energy[-1]: 1.0},
        initial_energy_mwh,
        np.inf,
    )
    return np.vstack(rows), np.array(row_lower), np.array(row_upper)


def _add_battery_energy_constraint(
    add_constraint: Callable[[dict[int, float], float, float], None],
    household: HouseholdUnit,
    variables: _HouseholdVariables,
    index: int,
    duration: float,
    initial_energy_mwh: float,
) -> None:
    """根据充放电效率和自放电率连接相邻时段的电池能量。
    当前时段结束的电量 = 上一时段保留下来的电量 + 充电实际存入的电量− 放电实际消耗的电量
    """
    retention = (1.0 - household.battery_loss_rate) ** duration
    energy_coefficients = {
        variables.energy[index]: 1.0,
        variables.charge[index]: (
            -household.battery_efficiency_charge * duration
        ),
        variables.discharge[index]: (
            duration / household.battery_efficiency_discharge
        ),
    } # 当前电量 − 充电功率 × 时长 × 充电效率 + 放电功率 × 时长 ÷ 放电效率
    if index:
        energy_coefficients[variables.energy[index - 1]] = -retention
        energy_rhs = 0.0
    else:
        energy_rhs = retention * initial_energy_mwh
    add_constraint(energy_coefficients, energy_rhs, energy_rhs)


def _add_battery_ramp_constraints(
    add_constraint: Callable[[dict[int, float], float, float], None],
    household: HouseholdUnit,
    variables: _HouseholdVariables,
    index: int,
    initial_battery_charge_power_mw: float,
    initial_battery_discharge_power_mw: float,
) -> None:
    """限制电池充电、放电功率的变化幅度，避免相邻时段的功率跳变过大。"""
    for indices, initial_power in (
        (variables.charge, initial_battery_charge_power_mw),
        (variables.discharge, initial_battery_discharge_power_mw),
    ):
        if index:
            add_constraint(
                {indices[index]: 1.0, indices[index - 1]: -1.0},
                -household.battery_ramp_down_mw,
                household.battery_ramp_up_mw,
            )
        else:
            add_constraint(
                {indices[0]: 1.0},
                initial_power - household.battery_ramp_down_mw,
                initial_power + household.battery_ramp_up_mw,
            )


def _solve_household_problem(
    household: HouseholdUnit,
    problem: _HouseholdProblem,
    objective: np.ndarray,
    constraint: LinearConstraint,
) -> np.ndarray:
    """在满足变量范围和运行约束的前提下求解优化问题，返回各变量的取值。"""
    # x 是全部优化变量的取值；milp 寻找使 objective @ x 最小的可行方案。
    # 具体目标由调用者传入，例如降低购电费用或最小化某个时段的购电功率。
    result = milp(
        objective,  # 目标函数中各变量的系数，并非变量的实际取值。
        integrality=problem.integrality,  # 0 表示连续变量，1 表示整数变量。
        bounds=problem.bounds,  # 单个变量的上下界，如充电功率和电池储能量范围。
        constraints=constraint,  # 变量之间的线性关系，如功率平衡和电量变化。
    )
    # 求解失败或未得到变量取值时，抛出包含家庭名称和求解器说明的异常。
    if not result.success or result.x is None:
        raise InputValidationError(
            f"家庭用户 {household.name!r} 优化求解失败：{result.message}"
        )
    # result.x 按变量索引保存各时段的购电、充放电功率、储能量和模式。
    # 将数值计算产生的接近零的小数归零，其他变量取值保持原值。
    return np.where(np.abs(result.x) < _TOLERANCE, 0.0, result.x)


def _baseline_objective(problem: _HouseholdProblem) -> np.ndarray:
    # 固定负荷和热泵费用是常数，只计算电池充放电引起的费用变化。
    objective = np.zeros(problem.variable_count)
    objective[problem.charge] = problem.prices * problem.durations
    objective[problem.discharge] = -problem.prices * problem.durations
    return objective


def _household_plans(
    household: HouseholdUnit,
    problem: _HouseholdProblem,
    solution: np.ndarray, # 求解器计算出的各个优化变量的值
) -> tuple[HouseholdPlan, ...]:
    # 把优化求解器输出的数值数组，按交付时段整理成家庭运行计划。
    return tuple(
        HouseholdPlan(
            delivery_start=start,
            delivery_end=end,
            unit_name=household.name,
            forecast_price_eur_per_mwh=problem.prices[index],
            heat_demand_mw_th=problem.heat_demands[index],
            planned_grid_power_mw=solution[problem.grid[index]],
            planned_heat_pump_power_mw=problem.heat_powers[index],
            planned_battery_discharge_power_mw=solution[
                problem.discharge[index]
            ],
        )
        for index, (start, end) in enumerate(problem.products)
    )


def _flexibility_bounds(
    household: HouseholdUnit,
    problem: _HouseholdProblem,
    baseline_plans: tuple[HouseholdPlan, ...],
) -> tuple[tuple[float, float], ...]:
    """在满足设备运行约束、且总购电费用不超过
    基准费用加上成本容忍度允许增量的条件下，
    计算各时段购电功率的最小值和最大值。
    """
    baseline_cost = sum(
        problem.prices[index] * plan.planned_grid_power_mw * problem.durations[index]
        for index, plan in enumerate(baseline_plans)
    )

    cost_limit = baseline_cost + abs(baseline_cost) * (
        household.cost_tolerance_percent / 100)
    cost_row = np.zeros(problem.variable_count)
    cost_row[problem.grid] = problem.prices * problem.durations
    constraint = LinearConstraint(
        np.vstack((problem.matrix, cost_row)),
        np.append(problem.constraint_lower, -np.inf),
        np.append(problem.constraint_upper, cost_limit + _TOLERANCE),
    )
    # 根据基准总费用和成本容忍度计算费用上限，
    # 并向优化模型添加“整个计划周期的总购电费用不得超过该上限”的约束。

    results: list[tuple[float, float]] = []
    # 在设备约束和总费用上限内，计算每个时段购电功率的上下限，并整理返回。
    for index in range(len(problem.products)):
        bounds: list[float] = []
        for direction in (1.0, -1.0):
            objective = np.zeros(problem.variable_count)
            objective[problem.grid[index]] = direction
            bounds.append(_solve_household_problem(
                    household,
                    problem,
                    objective,
                    constraint,
                )[problem.grid[index]])
            
        results.append((bounds[0], bounds[1]))
    return tuple(results)


def plan_household_opening(
    households: tuple[HouseholdUnit, ...],
    products: tuple[tuple[datetime, datetime], ...],
    price_forecast: dict[datetime, float],
    forecasts: dict[str, dict[datetime, float]],
    initial_energy_mwh: dict[str, float],
    initial_battery_charge_power_mw: dict[str, float],
    initial_battery_discharge_power_mw: dict[str, float],
    *,
    evaluate_flexibility: bool = True,
) -> tuple[
    dict[str, tuple[HouseholdPlan, ...]],
    dict[str, tuple[tuple[float, float], ...] | None],
]:
    """家庭购电计划优化与灵活性评估的主求解函数。
    1. 定义问题：根据家庭设备参数、预测数据和初始状态构建优化模型。
    2. 设置目标：以整个计划周期的总购电费用最小为目标。
    3. 设置约束：设置功率平衡、电池能量和设备运行限制等约束。
    4. 执行求解：求出最低费用方案，并整理为各时段的基准运行计划。
    5. 评估灵活性：复用模型，加入成本容忍度对应的总费用上限，
    分别求解各时段购电功率的最小值和最大值。
    6. 返回结果：返回按家庭名称组织的基准计划和购电功率上下限。
    evaluate_flexibility=False 时跳过第 5 步，各家庭的上下界为 None。
    """
    plans_by_household: dict[str, tuple[HouseholdPlan, ...]] = {}
    bounds_by_household: dict[str, tuple[tuple[float, float], ...] | None] = {}
    for household in households:
        problem = _build_household_problem(
            household,
            products,
            price_forecast,
            forecasts[f"{household.name}_heat_demand"],
            initial_energy_mwh[household.name],
            initial_battery_charge_power_mw[household.name],
            initial_battery_discharge_power_mw[household.name],
        ) # 准备模型数据
        plans_by_household[household.name] = _household_plans(
            household,
            problem,
            _solve_household_problem(
                household,
                problem,
                _baseline_objective(problem),
                LinearConstraint(
                    problem.matrix, problem.constraint_lower, problem.constraint_upper
                ),
            ), 
        )

        bounds_by_household[household.name] = (
            _flexibility_bounds(household, problem, plans_by_household[household.name])
            if evaluate_flexibility else None
        )

    return plans_by_household, bounds_by_household


def dispatch_household(
    household: HouseholdUnit,
    plans: tuple[HouseholdPlan, ...],
    accepted_grid_energy_mwh: dict[datetime, float],
    initial_energy_mwh: float,
    *,
    grid_power_bounds_mw: tuple[tuple[float, float], ...] | None,
    initial_heat_pump_power_mw: float = 0.0,
    initial_battery_charge_power_mw: float = 0.0,
    initial_battery_discharge_power_mw: float = 0.0,
) -> tuple[HouseholdDispatchResult, ...]:
    """结合原定运行计划和市场实际成交电量，按时间顺序模拟家庭设备运行，并更新电池 SOC。
    1. 优化求解 → 得到最低费用运行计划 plans
    2. 提交购电订单并出清 → 得到实际成交电量
    3. 调用 dispatch_household() → 根据计划和成交电量模拟实际运行
    """
    energy = initial_energy_mwh
    previous_heat_pump = initial_heat_pump_power_mw
    previous_charge = initial_battery_charge_power_mw
    previous_discharge = initial_battery_discharge_power_mw
    records: list[HouseholdDispatchResult] = []
    
    for index, plan in enumerate(plans): # 按时间顺序处理每个交付时段的计划
        duration = hours_between(plan.delivery_start, plan.delivery_end)
        retained_energy = energy * ((1.0 - household.battery_loss_rate) ** duration)
        # 计算电池扣除自放电损耗后的剩余能量
        accepted_energy = accepted_grid_energy_mwh[plan.delivery_start]
        # 获取当前实际成交的购电量
        grid_power = accepted_energy / duration # 将成交电量转换为该时段的平均购电功率
        required_power = plan.planned_heat_pump_power_mw
        # 计算家庭计划所需的总用电功率
        discharge_ramp_limit_mw = (
            previous_discharge + household.battery_ramp_up_mw)
        available_discharge_energy_mwh = max(
            0.0, retained_energy - household.min_energy_mwh)
        discharge_energy_limit_mw = (
            available_discharge_energy_mwh
            * household.battery_efficiency_discharge
            / duration
        )
        maximum_discharge = min(
            household.battery_max_discharge_power_mw,
            discharge_ramp_limit_mw,
            discharge_energy_limit_mw,
        ) # 当前时段允许的最大电池放电功率
        if grid_power + _TOLERANCE < required_power:
            discharge = min(
                plan.planned_battery_discharge_power_mw,
                maximum_discharge,
                required_power - grid_power,
            )
        else:
            discharge = 0.0
        # 当电网购电不足以满足用电需求时，确定电池实际放电功率；购电充足时不放电。

        available_power = grid_power + discharge # 计算当前可用的总功率
        heat_pump_power = min(
            plan.planned_heat_pump_power_mw,
            household.heat_pump_max_power_mw,
            available_power,
            previous_heat_pump + household.heat_pump_ramp_up_mw,
        ) # 确定热泵实际用电功率
        remaining_after_heat = available_power - heat_pump_power
        # 计算供应热泵后还剩多少功率
        charge_ramp_limit_mw = previous_charge + household.battery_ramp_up_mw
        # 爬坡限制对应的充电功率上限
        available_charge_capacity_mwh = max(0.0, household.max_energy_mwh - retained_energy)
        # 电池还能存入多少能量
        charge_energy_limit_mw = (
            available_charge_capacity_mwh
            / household.battery_efficiency_charge
            / duration)
        maximum_charge = min(
            household.battery_max_charge_power_mw,
            charge_ramp_limit_mw,
            charge_energy_limit_mw)
        charge = 0.0 if discharge > _TOLERANCE else min(
            remaining_after_heat, maximum_charge,)
        unallocated_power = remaining_after_heat - charge
        if unallocated_power > _TOLERANCE:
            raise InputValidationError(
                f"家庭用户 {household.name!r} 在 {plan.delivery_start.isoformat(sep=' ')} "
                "时刻的成交购电量无法在设备约束内分配给热泵或电池。")

        energy_before = energy
        # 计入效率后，充电实际存入电池的能量。
        charge_energy_mwh = (charge * duration * household.battery_efficiency_charge)
        # 计入效率后，放电实际消耗的电池能量。
        discharge_energy_mwh = (discharge * duration / household.battery_efficiency_discharge)
        next_energy = retained_energy + charge_energy_mwh - discharge_energy_mwh
        
        if next_energy < household.min_energy_mwh - _TOLERANCE:
            raise InputValidationError(
                f"家庭用户 {household.name!r} 在 {plan.delivery_start.isoformat(sep=' ')} "
                "时刻的成交购电量不足以弥补电池损耗并维持最低 SOC。")
        energy = min(household.max_energy_mwh,
            max(household.min_energy_mwh, next_energy),)
        unmet_heat = max(
            0.0,
            plan.heat_demand_mw_th - heat_pump_power * household.cop,
        ) * duration
        records.append(
            HouseholdDispatchResult(
                plan=plan,
                heat_pump_power_mw=heat_pump_power,
                battery_charge_power_mw=charge,
                battery_discharge_power_mw=discharge,
                soc_before=energy_before / household.battery_capacity_mwh,
                soc_after=energy / household.battery_capacity_mwh,
                unmet_heat_mwh_th=unmet_heat,
                grid_power_bounds_mw=(
                    grid_power_bounds_mw[index] if grid_power_bounds_mw is not None else None
                ),
            )
        )
        previous_heat_pump = heat_pump_power
        previous_charge = charge
        previous_discharge = discharge
    return tuple(records)
