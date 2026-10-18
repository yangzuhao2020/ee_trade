"""第四版工业需求响应：钢铁生产滚动优化、购电功率灵活性评估及出清后执行记录。
根据预测价格、参考负荷和生产约束制定计划，并按实际成交电量更新钢产量与设备出力。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import numpy as np
from scipy.optimize import linprog
from .errors import InputValidationError
from .models import (
    DemandBid,
    IndustrialDevice,
    IndustrialDispatchResult,
    IndustrialFlexibilityResult,
    IndustrialOptimizationWindowResult,
    IndustrialPlan,
    IndustrialUnit,
    MarketClearingResult,
)
from .utils import hours_between


_TOLERANCE = 1e-7


@dataclass(frozen=True)
class _ProcessCoefficients:
    dri_t_per_steel_t: float
    hydrogen_mwh_per_steel_t: float
    electrolyser_mwh_per_steel_t: float
    dri_mwh_per_steel_t: float
    eaf_mwh_per_steel_t: float
    grid_mwh_per_steel_t: float 
    # 电解槽 + DRI 电费（直接还原铁的需求量） + 电弧炉电费
    iron_ore_t_per_steel_t: float
    lime_t_per_steel_t: float
    co2_t_per_steel_t: float


@dataclass(frozen=True)
class _IndustryProblem:
    products: tuple[tuple[datetime, datetime], ...]
    durations: np.ndarray
    variable_cost_per_tonne: np.ndarray
    grid_power_per_tonne: np.ndarray
    electrolyser_power_per_tonne: np.ndarray
    dri_power_per_tonne: np.ndarray
    eaf_power_per_tonne: np.ndarray
    steel: np.ndarray
    variable_count: int
    objective: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: tuple[tuple[float | None, float | None], ...]
    A_ub_limited: np.ndarray


def process_coefficients(unit: IndustrialUnit) -> _ProcessCoefficients:
    """工艺约束关系：返回每吨钢的线性原料需求和电力需求系数。
    生产 1 吨钢需要多少原料、氢能和电能”，供后续生产优化使用"""
    # 以下均按生产 1 吨钢计算。
    dri = unit.eaf.specific_dri_demand  # DRI 需求
    hydrogen = dri * unit.dri_plant.specific_hydrogen_consumption  # 氢能需求
    electrolyser_energy = hydrogen / unit.electrolyser.efficiency  # 制氢耗电
    dri_energy = dri * unit.dri_plant.specific_electricity_consumption  # DRI 装置耗电
    eaf_energy = unit.eaf.specific_electricity_consumption  # 电弧炉耗电
    lime = unit.eaf.specific_lime_demand  # 石灰需求

    return _ProcessCoefficients(
        dri_t_per_steel_t=dri,
        hydrogen_mwh_per_steel_t=hydrogen,
        electrolyser_mwh_per_steel_t=electrolyser_energy,
        dri_mwh_per_steel_t=dri_energy,
        eaf_mwh_per_steel_t=eaf_energy,
        grid_mwh_per_steel_t=electrolyser_energy + dri_energy + eaf_energy,
        iron_ore_t_per_steel_t=(
            dri * unit.dri_plant.specific_iron_ore_consumption),
        lime_t_per_steel_t=lime,
        co2_t_per_steel_t=lime * unit.eaf.lime_co2_factor,
    )


def _steel_upper_bound(
    unit: IndustrialUnit, 
    coefficients: _ProcessCoefficients, 
    duration: float
) -> float:
    """约束函数：三台设备的最大功率分别能支撑多少吨钢，并取其中最小值作为整条生产链的产量上限。"""
    # 时段最大用电量（MWh）= 最大功率 × 时长。
    electrolyser_max_energy_mwh = unit.electrolyser.max_power_mw * duration
    dri_max_energy_mwh = unit.dri_plant.max_power_mw * duration
    eaf_max_energy_mwh = unit.eaf.max_power_mw * duration

    # 各环节钢产量上限（t）= 最大用电量 ÷ 每吨钢耗电量
    electrolyser_steel_limit_t = (
        electrolyser_max_energy_mwh / coefficients.electrolyser_mwh_per_steel_t)
    dri_steel_limit_t = dri_max_energy_mwh / coefficients.dri_mwh_per_steel_t
    eaf_steel_limit_t = eaf_max_energy_mwh / coefficients.eaf_mwh_per_steel_t

    limits = (electrolyser_steel_limit_t, dri_steel_limit_t, eaf_steel_limit_t)
    return min(limits)


def _constraint_row(
    coefficients: dict[int, float],
    variable_count: int,
) -> np.ndarray:
    """将 {变量索引: 系数} 转换为 linprog 使用的一行约束系数。"""
    row = np.zeros(variable_count)
    for variable, coefficient in coefficients.items():
        row[variable] = coefficient
    return row


def _ramp_constraint_rows(
    devices: tuple[IndustrialDevice, IndustrialDevice, IndustrialDevice],
    power_coefficients: tuple[Sequence[float], ...],
    initial_powers_mw: tuple[float, float, float],
    *,
    steel: np.ndarray,
    row_width: int,
) -> tuple[list[np.ndarray], list[float]]:
    """约束函数：每台设备、每个时段生成两条约束：功率上升不能太快，功率下降也不能太快，并返回给线性规划求解器使用。"""
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for index in range(len(power_coefficients[0])):
        for device_index, device in enumerate(devices):
            coefficient = power_coefficients[device_index][index]  # 功率系数（MW/吨钢）
            increase = {steel[index]: coefficient}  # 上升：本期功率 - 上期功率
            decrease = {steel[index]: -coefficient}  # 下降：上期功率 - 本期功率
            if index:
                previous = power_coefficients[device_index][index - 1]
                increase[steel[index - 1]] = -previous
                decrease[steel[index - 1]] = previous
                initial = 0.0  # 上期功率已计入左侧
            else:
                initial = initial_powers_mw[device_index]  # 首时段使用初始功率
            # 保存约束系数及右侧上限。
            rows.extend((
                _constraint_row(increase, row_width),
                _constraint_row(decrease, row_width),
            ))
            limits.extend(
                (device.ramp_up_mw + initial, device.ramp_down_mw - initial))
    return rows, limits


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
    """构建一个滚动窗口的工业线性规划问题。

    目标：最小化窗口内生产成本与购电功率偏差惩罚之和。
    生产成本 = 每吨钢可变成本 × 钢产量，含电费、铁矿石、石灰和碳排放；
    偏差惩罚 = penalty × (正功率偏差 + 负功率偏差)，以参考购电功率为基准。
    前瞻窗口外的产量变量不计成本，仅用于确保本窗口计划
    为后续生产保留足够且满足爬坡约束的产能。
    """
    # ---- 0. 输入校验 ----
    if not products or products != remaining_products[: len(products)]:
        raise InputValidationError(
            "工业优化窗口的交付产品必须非空，并且是剩余交付产品序列的起始部分。")
    if not 0 < commit_count <= len(products):
        raise InputValidationError("工业生产计划的承诺区间超出优化窗口范围。")
    if not isfinite(demand_to_schedule_t) or demand_to_schedule_t < 0:
        raise InputValidationError("工业剩余需求必须为非负有限值。")

    # ---- 1. 变量布局：x = [剩余时段钢产量 | 窗口内正偏差 | 窗口内负偏差] ----
    count = len(products)
    remaining_count = len(remaining_products)
    steel = np.arange(remaining_count)      # 所有剩余时段的钢产量变量索引
    positive = np.arange(remaining_count, remaining_count + count)   # 正偏差
    negative = np.arange(remaining_count + count, remaining_count + 2 * count)  # 负偏差
    variable_count = remaining_count + 2 * count

    # ---- 2. 数据准备：时长、功率系数、参考功率、可变成本与变量边界 ----
    coefficients = process_coefficients(unit)
    remaining_load_profile = tuple(
        normalized_load_profile[start] for start, _ in remaining_products)
    remaining_durations = np.array(
        [hours_between(*product) for product in remaining_products])
    remaining_power_coefficients = np.array([
        _power_coefficients(coefficients, duration)
        for duration in remaining_durations])

    prices = np.empty(count)
    references = np.empty(count)
    variable_costs = np.empty(count)
    for index, (start, _) in enumerate(products):
        prices[index] = price_forecast[start]
        raw_prices = fuel_price_profiles[start]
        references[index] = (
            remaining_load_profile[index] * unit.electrolyser.max_power_mw)
        variable_costs[index] = (
            coefficients.grid_mwh_per_steel_t * prices[index] # 该时段每生产一吨钢的购电成本
            + coefficients.iron_ore_t_per_steel_t
            * _price(raw_prices, "iron ore", "iron_ore") # 每生产一吨钢所需的铁矿石成本
            + coefficients.lime_t_per_steel_t * _price(raw_prices, "lime")
            + coefficients.co2_t_per_steel_t * _price(raw_prices, "co2")
            # 每吨钢的石灰成本和碳排放成本。
        )

    # 钢产量、正负偏差变量的上下界。
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, _steel_upper_bound(unit, coefficients, duration))
        for duration in remaining_durations]
    bounds.extend([(0.0, None)] * (2 * count))

    # ---- 3. 目标函数：生产成本 + 偏差惩罚 ----
    penalty = 10.0 / max(unit.load_profile_deviation, 0.01)

    objective = np.concatenate([
        variable_costs,                      # 窗口内钢产量：每吨钢成本
        np.zeros(remaining_count - count),   # 窗口外钢产量：本次不计成本
        np.full(count, penalty),             # 正偏差：惩罚系数
        np.full(count, penalty),             # 负偏差：惩罚系数
    ])

    # ---- 4. 约束构造 ----
    grid_power = remaining_power_coefficients[:count, 3]
    equality_rows: list[np.ndarray] = []
    equality_values: list[float] = []
    inequality_rows: list[np.ndarray] = []  # 不等式左侧系数
    inequality_values: list[float] = []  # 不等式右侧上限

    # 4.1 逐时段建立：计划购电功率 - 正偏差 + 负偏差 = 参考功率。
    for index in range(count):
        row = _constraint_row(
            {
                steel[index]: grid_power[index],  # 钢产量的功率系数（MW/吨钢）
                positive[index]: -1.0,
                negative[index]: 1.0,
            },
            variable_count,
        )
        equality_rows.append(row)
        equality_values.append(references[index])  # 右侧参考功率（MW）

    # 4.2 各设备功率升降约束：设备功率 = 功率系数 × 该时段钢产量。
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    device_coefficients = tuple(
        remaining_power_coefficients[:, index] for index in range(3))
    ramp_rows, ramp_limits = _ramp_constraint_rows(
        devices,
        device_coefficients,
        initial_powers_mw,
        steel=steel,
        row_width=variable_count,
    )
    inequality_rows.extend(ramp_rows)
    inequality_values.extend(ramp_limits)

    # 4.3 剩余各时段产量之和 = 剩余需求。
    row = _constraint_row({variable: 1.0 for variable in steel}, variable_count)
    equality_rows.append(row)
    equality_values.append(demand_to_schedule_t)

    # 4.4 承诺区产量下限：-承诺区产量 ≤ -最低产量。
    if commit_count < remaining_count:
        remaining_profile_total = sum(remaining_load_profile)
        commit_profile_total = sum(remaining_load_profile[:commit_count])
        # 按参考负荷占比分配；总负荷近零时按时段数分配。
        ratio = (
            commit_profile_total / remaining_profile_total
            if remaining_profile_total > _TOLERANCE
            else commit_count / remaining_count)
        # 按允许偏差下调承诺区最低产量。
        profile_minimum = demand_to_schedule_t * ratio * (
            1.0 - unit.load_profile_deviation)
        row = _constraint_row(
            {variable: -1.0 for variable in steel[:commit_count]},
            variable_count)
        inequality_rows.append(row)
        inequality_values.append(-profile_minimum)

    # 可选的成本上限行，供限定可变成本时追加到不等式矩阵末尾。
    cost_limit_row = np.zeros(variable_count)
    cost_limit_row[steel[:count]] = variable_costs

    # ---- 5. 组装问题 ----
    window_power_coefficients = remaining_power_coefficients[:count]
    return _IndustryProblem(
        products=products,
        durations=remaining_durations[:count],
        variable_cost_per_tonne=variable_costs,
        grid_power_per_tonne=window_power_coefficients[:, 3],
        electrolyser_power_per_tonne=window_power_coefficients[:, 0],
        dri_power_per_tonne=window_power_coefficients[:, 1],
        eaf_power_per_tonne=window_power_coefficients[:, 2],
        steel=steel,
        variable_count=variable_count,
        objective=objective,
        A_eq=np.vstack(equality_rows),
        b_eq=np.array(equality_values),
        A_ub=(np.vstack(inequality_rows) if inequality_rows else None),
        b_ub=(np.array(inequality_values) if inequality_values else None),
        bounds=tuple(bounds),
        A_ub_limited=np.vstack([*inequality_rows, cost_limit_row]),
    )


def _solve(
    unit: IndustrialUnit,
    problem: _IndustryProblem,
    objective: np.ndarray,
    *,
    variable_cost_limit: float | None = None,
) -> np.ndarray:
    """真正运行优化求解，找到满足所有生产约束、并使目标最小的变量取值
    """
    # 读取原始不等式约束：A_ub @ x <= b_ub，x 为完整决策变量向量。
    A_ub = problem.A_ub
    b_ub = problem.b_ub
    if variable_cost_limit is not None:
        # 扩展矩阵的最后一行已包含窗口可变成本系数，补充对应的成本上限。
        A_ub = problem.A_ub_limited
        # 上限加上数值容差，允许微小的浮点误差。
        limit = variable_cost_limit + _TOLERANCE
        # 原约束为空时创建右端向量，否则在末尾追加成本上限。
        b_ub = np.array([limit]) if b_ub is None else np.append(b_ub, limit)

    # 求解 min(objective @ x)，同时满足不等式、等式约束和变量上下界。
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
            f"工业单元 {unit.name!r} 优化失败：{result.message}")

    # 将接近零的正负值统一置零，避免数值噪声进入后续计划与调度。
    return np.where(np.abs(result.x) < _TOLERANCE, 0.0, result.x)


def _build_industry_plans(
    unit: IndustrialUnit,
    problem: _IndustryProblem,
    steel: np.ndarray,
    window_id: str,
) -> tuple[IndustrialPlan, ...]:
    """按窗口钢产量生成逐时段生产计划，折算设备功率、产量与预测成本。"""
    coefficients = process_coefficients(unit)
    plans: list[IndustrialPlan] = []
    for index, (start, end) in enumerate(problem.products):
        steel_output = steel[index]
        duration = problem.durations[index]
        grid_power = problem.grid_power_per_tonne[index] * steel_output
        plans.append(
            IndustrialPlan(
                window_id=window_id,
                delivery_start=start,
                delivery_end=end,
                unit_name=unit.name,
                electrolyser_power_mw=(
                    problem.electrolyser_power_per_tonne[index] * steel_output),
                hydrogen_output_mwh=(
                    coefficients.hydrogen_mwh_per_steel_t * steel_output),
                dri_power_mw=problem.dri_power_per_tonne[index] * steel_output,
                dri_output_t=coefficients.dri_t_per_steel_t * steel_output,
                eaf_power_mw=problem.eaf_power_per_tonne[index] * steel_output,
                planned_steel_output_t=steel_output,
                planned_grid_power_mw=grid_power,
                planned_energy_mwh=grid_power * duration,
                forecast_cost_eur=(
                    problem.variable_cost_per_tonne[index] * steel_output),
            )
        )
    return tuple(plans)


def _calculate_industry_flexibility(
    unit: IndustrialUnit,
    problem: _IndustryProblem,
    committed_plans: tuple[IndustrialPlan, ...],
    variable_cost_limit: float,
) -> tuple[IndustrialFlexibilityResult, ...]:
    """在窗口可变成本上限内，逐承诺时段求购电功率的可达上下界。
    committed_plans 为窗口起始的承诺计划，序号与决策变量中的钢产量一一对应。
    """
    flexibility: list[IndustrialFlexibilityResult] = []
    for index, plan in enumerate(committed_plans):
        minimum_objective = np.zeros(problem.variable_count)
        minimum_objective[problem.steel[index]] = (
            problem.grid_power_per_tonne[index]
        )
        minimum_solution = _solve(
            unit,
            problem,
            minimum_objective,
            variable_cost_limit=variable_cost_limit,
        )
        maximum_objective = -minimum_objective
        maximum_solution = _solve(
            unit,
            problem,
            maximum_objective,
            variable_cost_limit=variable_cost_limit,
        )
        minimum_power = (
            minimum_solution[problem.steel[index]]
            * problem.grid_power_per_tonne[index]
        )
        maximum_power = (
            maximum_solution[problem.steel[index]]
            * problem.grid_power_per_tonne[index]
        )
        if minimum_power > plan.planned_grid_power_mw + _TOLERANCE or (
            maximum_power < plan.planned_grid_power_mw - _TOLERANCE):
            raise InputValidationError(
                f"工业单元 {unit.name!r} 在 {plan.delivery_start.isoformat(sep=' ')} "
                "时刻的灵活性上下界未包含基准计划功率。")
        flexibility.append(
            IndustrialFlexibilityResult(
                window_id=plan.window_id,
                delivery_start=plan.delivery_start,
                delivery_end=plan.delivery_end,
                unit_name=unit.name,
                baseline_power_mw=plan.planned_grid_power_mw,
                minimum_power_mw=minimum_power,
                maximum_power_mw=maximum_power,
            )
        )
    return tuple(flexibility)


def _power_coefficients(
    coefficients: _ProcessCoefficients, duration: float
) -> tuple[float, float, float, float]:
    """"按时段时长，将每吨钢耗电量换算为设备及总购电功率系数。"""
    electrolyser = coefficients.electrolyser_mwh_per_steel_t / duration
    dri = coefficients.dri_mwh_per_steel_t / duration
    eaf = coefficients.eaf_mwh_per_steel_t / duration
    return electrolyser, dri, eaf, electrolyser + dri + eaf


def maximum_industry_production(
    unit: IndustrialUnit,
    products: tuple[tuple[datetime, datetime], ...],
    initial_powers_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """在设备容量和爬坡约束下最大化钢产量。"""
    if not products:
        return 0.0
    coefficients = process_coefficients(unit)
    durations = np.array([hours_between(*product) for product in products])
    bounds = [(0.0, _steel_upper_bound(unit, coefficients, duration))
        for duration in durations]
    power_per_tonne = [
        _power_coefficients(coefficients, duration)[:3] for duration in durations]
    device_coefficients = tuple(
        tuple(per_tonne[device_index] for per_tonne in power_per_tonne)
        for device_index in range(3))
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    rows, limits = _ramp_constraint_rows(
        devices,
        device_coefficients,
        initial_powers_mw,
        steel=np.arange(len(products)),
        row_width=len(products),
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
            f"工业单元 {unit.name!r} 的最大产量计算失败："
            f"{result.message}"
        )
    return float(np.sum(result.x))


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
    """优化函数：求解一个滚动窗口的基准计划，以及逐时段的灵活性上下界。"""
    maximum = maximum_industry_production(
        unit, remaining_products, initial_powers_mw
    )
    if demand_to_schedule_t > maximum + _TOLERANCE:
        raise InputValidationError(
            f"工业单元 {unit.name!r} 的需求无法满足："
            f"剩余需求={demand_to_schedule_t:.6f} 吨，最大产量={maximum:.6f} 吨，"
            f"缺口={demand_to_schedule_t - maximum:.6f} 吨。"
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
    steel = solution[problem.steel[:count]]
    baseline_variable_cost = float(
        np.dot(problem.variable_cost_per_tonne, steel))
    flexible_limit = baseline_variable_cost * (
        1.0 + unit.cost_tolerance_percent / 100.0)
    
    window_id = f"{unit.name}::{products[0][0].isoformat()}"
    plans = _build_industry_plans(unit, problem, steel, window_id)
    
    flexibility = _calculate_industry_flexibility(
        unit, problem, plans[:commit_count], flexible_limit)
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
    return plans, flexibility, window


def _actual_steel_output(
    plan: IndustrialPlan, accepted_energy_mwh: float
) -> float:
    """按成交电量占申报电量的比例折算实际钢产量；成交电量越界时抛错。"""
    if accepted_energy_mwh < -_TOLERANCE or (
        accepted_energy_mwh > plan.planned_energy_mwh + _TOLERANCE
    ):
        raise InputValidationError(
            f"工业单元 {plan.unit_name!r} 的成交电量必须介于零与申报电量之间。")
    if plan.planned_energy_mwh <= _TOLERANCE:
        return 0.0
    ratio = max(0.0, accepted_energy_mwh) / plan.planned_energy_mwh
    return ratio * plan.planned_steel_output_t


def dispatch_industry(
    plan: IndustrialPlan,
    accepted_energy_mwh: float,
) -> IndustrialDispatchResult:
    """按购电量的成交比例缩放计划生产链的产量。"""
    # 先校验成交电量范围，再计算功率。
    actual_steel = _actual_steel_output(plan, accepted_energy_mwh)
    duration = hours_between(plan.delivery_start, plan.delivery_end)
    return IndustrialDispatchResult(
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
        actual_steel_output_t=actual_steel,
        forecast_cost_eur=plan.forecast_cost_eur,
    )


class IndustryRollingCoordinator:
    """协调工业用户跨市场事件的滚动生产计划、购电报价与实际交付。
    开市时调用 optimize_industry_window 优化生产，仅锁定承诺区间内的计划；
    交付结束后按成交电量更新实际钢产量和设备功率，供后续窗口衔接。
    前序已出清但尚未交付的计划计入预计产量，避免重复安排生产。
    """
    def __init__(
        self,
        units: tuple[IndustrialUnit, ...],
        all_products: tuple[tuple[datetime, datetime], ...],
        price_forecasts: dict[str, dict[datetime, float]],
        normalized_load_profiles: dict[str, dict[datetime, float]],
        fuel_price_profiles: dict[datetime, dict[str, float]],
        maximum_bid_price: float,
    ) -> None:
        """保存工业用户、交付时段和预测输入，初始化计划与交付状态。
        预测电价用于优化生产成本，maximum_bid_price 用于购电申报。
        """
        self.units = units
        # 按交付时间去重排序；集合用于筛选本次开市的有效产品。
        self.all_products = tuple(sorted(set(all_products)))
        self.product_set = frozenset(self.all_products)
        self.price_forecasts = price_forecasts
        self.normalized_load_profiles = normalized_load_profiles
        self.fuel_price_profiles = fuel_price_profiles
        self.maximum_bid_price = maximum_bid_price
        # 以（用户名称，交付开始时间）关联承诺计划与已完成的交付。
        self.plans: dict[tuple[str, datetime], IndustrialPlan] = {}
        self.delivered: set[tuple[str, datetime]] = set()
        # 实际钢产量（吨）和设备功率（MW）作为后续窗口的初始状态。
        self.actual_steel_t = {unit.name: 0.0 for unit in units}
        self.last_actual_powers_mw = {
            unit.name: (0.0, 0.0, 0.0) for unit in units
        }
        # 累积交付、灵活性和窗口成本记录，仿真结束时汇入结果。
        self.dispatch_results: list[IndustrialDispatchResult] = []
        self.flexibility_results: list[IndustrialFlexibilityResult] = []
        self.window_results: list[IndustrialOptimizationWindowResult] = []

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> list[DemandBid]:
        """为本次开市生成工业购电订单，缺少承诺计划时先滚动优化。
        申报电量取计划购电量，价格取市场最高限价；零电量不申报。
        """
        bids: list[DemandBid] = []
        eligible_products = tuple(
            product for product in products if product in self.product_set)

        if not eligible_products:
            return bids

        for unit in self.units:
            # 首个未承诺产品作为新滚动窗口的起点。
            missing_product: tuple[datetime, datetime] | None = None
            for product in eligible_products:
                delivery_start = product[0]
                plan_key = (unit.name, delivery_start)
                has_committed_plan = plan_key in self.plans

                if not has_committed_plan:
                    missing_product = product
                    break

            if missing_product is not None:
                self._plan_window(unit, missing_product, scheduled_results)
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

    def _plan_window(
        self,
        unit: IndustrialUnit,
        first_product: tuple[datetime, datetime],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> None:
        """从首个未承诺产品开始优化，并保存本轮承诺计划及评估结果。"""
        # ---- 1. 确定窗口：从首个未承诺产品起，按前瞻/承诺时域切片 ----
        # 已承诺产品不参与重排，保留其余时段供本轮检查剩余产能。
        remaining_products = tuple(
            product
            for product in self.all_products
            if product[0] >= first_product[0]
            and (unit.name, product[0]) not in self.plans
        )
        if not remaining_products or remaining_products[0] != first_product:
            raise InputValidationError(
                f"无法为工业单元 {unit.name!r} 构建从 "
                f"{first_product[0].isoformat(sep=' ')} 开始的下一个优化窗口。"
            )
        # 前瞻时长决定优化窗口，承诺时长决定其中需锁定的产品数量。
        optimization_end = first_product[0] + unit.look_ahead
        products = tuple(
            product for product in remaining_products if product[0] < optimization_end)
        commit_end = first_product[0] + unit.commit_horizon
        commit_count = sum(
            1 for product in products if product[0] < commit_end)

        # ---- 2. 待安排量 = 总需求 - 已交付产量 - 已成交待交付预计产量 ----
        pending_steel, initial_powers = self._pending_state(
            unit, first_product, scheduled_results)
        demand_to_schedule = max(
            0.0, unit.demand_t - self.actual_steel_t[unit.name] - pending_steel)

        # ---- 3. 求解滚动窗口优化 ----
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

        # ---- 4. 锁定承诺区计划，记录评估结果 ----
        # 承诺区间供后续开市复用；其余前瞻计划留待下一轮重算。
        for plan in plans[:commit_count]:
            self.plans[(unit.name, plan.delivery_start)] = plan
        self.flexibility_results.extend(flexibility)
        self.window_results.append(window)

    def record_delivery(self, result: MarketClearingResult) -> None:
        """交付结束时按成交比例累计钢产量，并更新后续优化所需的设备功率。"""
        for unit in self.units:
            key = (unit.name, result.delivery_start)
            plan = self.plans.get(key)
            # 无计划或已记账的产品直接跳过，避免重复累计产量。
            if plan is None or key in self.delivered:
                continue
            accepted = self._cleared_industry_energy(result, unit.name)
            dispatch = dispatch_industry(plan, accepted)
            self.dispatch_results.append(dispatch)
            self.actual_steel_t[unit.name] += dispatch.actual_steel_output_t
            self.last_actual_powers_mw[unit.name] = self._actual_device_powers(
                plan, accepted
            )
            self.delivered.add(key)

    def validate_completion(self) -> None:
        """全部交付结束后检查钢需求，产量缺口超过数值容差时抛出异常。"""
        for unit in self.units:
            shortfall = unit.demand_t - self.actual_steel_t[unit.name]
            if shortfall > _TOLERANCE:
                raise InputValidationError(
                    f"仿真结束时，工业单元 {unit.name!r} 的需求尚未完成："
                    f"剩余需求={shortfall:.6f} 吨，最大产量=0.000000 吨，"
                    f"缺口={shortfall:.6f} 吨。"
                )

    def _pending_state(
        self,
        unit: IndustrialUnit,
        first_product: tuple[datetime, datetime],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> tuple[float, tuple[float, float, float]]:
        """计算前序已成交但未交付的预计钢产量，以及衔接本窗口的初始设备功率。
        用前序未交付计划的成交电量预估产量，不提前计入实际完成量；
        若末段待交付与新窗口起点相邻，以其成交后功率作为爬坡约束起点。
        """
        pending: list[tuple[IndustrialPlan, float]] = []
        for (unit_name, start), plan in self.plans.items():
            if unit_name != unit.name or (unit_name, start) in self.delivered:
                continue
            if plan.delivery_start >= first_product[0]:
                continue
            result = self._result_for_plan(plan, scheduled_results)
            accepted = self._cleared_industry_energy(result, unit.name)
            pending.append((plan, accepted))

        # 按交付时间排序后累加，保证求和顺序稳定；末段用于初始功率衔接。
        pending.sort(key=lambda item: item[0].delivery_start)
        pending_steel = sum(
            _actual_steel_output(plan, accepted) for plan, accepted in pending)
        initial_powers = self.last_actual_powers_mw[unit.name]
        # 前序待交付产品若与新窗口相邻，以其成交后功率衔接爬坡约束。
        if pending:
            last_plan, accepted = pending[-1]
            if last_plan.delivery_end == first_product[0]:
                initial_powers = self._actual_device_powers(last_plan, accepted)
        return pending_steel, initial_powers

    def _result_for_plan(
        self,
        plan: IndustrialPlan,
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> MarketClearingResult:
        """按交付起止时间查找计划对应的出清结果，要求恰好匹配一个。"""
        matches: list[MarketClearingResult] = []
        for result in scheduled_results.values():
            is_start_match = result.delivery_start == plan.delivery_start
            is_end_match = result.delivery_end == plan.delivery_end

            if is_start_match and is_end_match:
                matches.append(result)

        if len(matches) != 1:
            raise InputValidationError(
                f"待交付的工业产品 {plan.delivery_start.isoformat(sep=' ')} "
                f"应恰好对应一个市场出清结果，实际找到 {len(matches)} 个。"
            )
        return matches[0]

    @staticmethod
    def _cleared_industry_energy(
        result: MarketClearingResult, unit_name: str
    ) -> float:
        """读取指定工业用户的成交电量（MWh）；未找到对应订单时返回零。"""
        for cleared_bid in result.demand_bids:
            bid = cleared_bid.bid
            is_target_unit = bid.unit_name == unit_name
            is_industrial_load = bid.demand_type == "industrial_load"

            if is_target_unit and is_industrial_load:
                return cleared_bid.accepted_energy_mwh

        return 0.0

    @staticmethod
    def _actual_device_powers(
        plan: IndustrialPlan, accepted_energy_mwh: float
    ) -> tuple[float, float, float]:
        """按成交比例缩放设备功率（MW），计划电量近零时返回全零。
        返回顺序为电解槽、直接还原装置、电弧炉。
        """
        ratio = (0.0
            if plan.planned_energy_mwh <= _TOLERANCE
            else accepted_energy_mwh / plan.planned_energy_mwh
        )
        return (
            ratio * plan.electrolyser_power_mw,
            ratio * plan.dri_power_mw,
            ratio * plan.eaf_power_mw,
        )


def _price(prices: dict[str, float], *names: str) -> float:
    for name in names:
        if name in prices:
            return prices[name]
    return 0.0


def missing_industrial_price_names(
    fuel_price_profiles: dict[datetime, dict[str, float]],
) -> tuple[str, ...]:
    """返回缺少价格数据、将按文档规定默认取 0 的原料名称。"""
    columns = set(next(iter(fuel_price_profiles.values()), {}))
    missing: list[str] = []
    if not ({"iron ore", "iron_ore"} & columns):
        missing.append("iron ore")
    if "lime" not in columns:
        missing.append("lime")
    return tuple(missing)


