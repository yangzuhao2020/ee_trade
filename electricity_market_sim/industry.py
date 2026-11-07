"""第四版工业需求响应：钢铁生产滚动优化、购电功率灵活性评估及出清后执行记录。
根据预测价格、参考负荷和生产约束制定计划，并按实际成交电量更新钢产量与设备出力。"""

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
import numpy as np
from scipy.optimize import linprog
from .errors import InputValidationError
from .models import (
    DemandBid,
    IndustrialDevice,
    IndustrialResult,
    Industrial_Optimization_Window_Result,
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
    variable_cost_per_tonne: np.ndarray
    grid_power_per_tonne: np.ndarray
    electrolyser_power_per_tonne: np.ndarray
    dri_power_per_tonne: np.ndarray
    eaf_power_per_tonne: np.ndarray
    objective: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
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


def _steel_lower_bound(
    unit: IndustrialUnit,
    coefficients: _ProcessCoefficients,
    duration: float,
) -> float:
    """三台设备同时运行时，取各自最低功率对应钢产量的最大值。"""
    return max(
        unit.electrolyser.min_power_mw * duration / coefficients.electrolyser_mwh_per_steel_t,
        unit.dri_plant.min_power_mw * duration / coefficients.dri_mwh_per_steel_t,
        unit.eaf.min_power_mw * duration / coefficients.eaf_mwh_per_steel_t,
    )


def _steel_upper_bound(
    unit: IndustrialUnit, 
    coefficients: _ProcessCoefficients, 
    duration: float
) -> float:
    """约束函数：三台设备的最大功率分别能支撑多少吨钢，并取其中最小值作为整条生产链的产量上限。"""
    # 时段最大用电量= 最大功率 × 时长。
    # 各环节钢产量上限（t）= 最大用电量 ÷ 每吨钢耗电量
    return min(
        unit.electrolyser.max_power_mw * duration / coefficients.electrolyser_mwh_per_steel_t,
        unit.dri_plant.max_power_mw * duration / coefficients.dri_mwh_per_steel_t,
        unit.eaf.max_power_mw * duration / coefficients.eaf_mwh_per_steel_t,
    )


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
    devices,
    power_coefficients,
    initial_powers_mw,
    *,
    row_width,
):
    """约束：设备功率按固定工艺比例联动，每时段只保留最严格的升、降约束。"""
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for index, coefficients in enumerate(power_coefficients):
        row = np.zeros(row_width)
        row[index] = 1.0
        initial = initial_powers_mw
        if index:
            # 除以本期功率系数后，上期产量按两时段的时长比换算。
            row[index - 1] = -power_coefficients[index - 1, 0] / coefficients[0]
            initial = (0.0, 0.0, 0.0)
        rows.extend((row, -row))
        limits.extend((
            min((device.ramp_up_mw + power) / coefficient
                for device, power, coefficient in zip(devices, initial, coefficients)),
            min((device.ramp_down_mw - power) / coefficient
                for device, power, coefficient in zip(devices, initial, coefficients)),
        ))
    return rows, limits


def _commit_production_constraint_rows(
    unit,
    remaining_products,
    commit_count,
    remaining_load_profile,
    bounds,
    demand_to_schedule_t,
    row_width,
):
    """按参考负荷占比，为本轮和后续承诺区间建立产量上下限。
    先预留各时段最低产量，再按负荷占比分配剩余任务并允许指定偏差。
    最后一个区间的产量由总需求等式和前面区间的产量确定。
    """
    rows = []
    limits = []
    remaining_count = len(remaining_products)
    commit_start = 0
    commit_end = commit_count
    
    while commit_end < remaining_count:
        remaining_profile_total = sum(remaining_load_profile[commit_start:])
        commit_profile_total = sum(remaining_load_profile[commit_start:commit_end])
        # 按参考负荷占比分配；总负荷近零时按时段数分配。
        ratio = (
            commit_profile_total / remaining_profile_total
            if remaining_profile_total > _TOLERANCE
            else (commit_end - commit_start) / (remaining_count - commit_start))
        minimum_commit = sum(bound[0] for bound in bounds[commit_start:commit_end])
        minimum_remaining = sum(bound[0] for bound in bounds[commit_start:remaining_count])

        for sign, allowance in ((-1.0, 1.0 - unit.load_profile_deviation),
                                (1.0, 1.0 + unit.load_profile_deviation)):
            factor = ratio * allowance
            # 先预留最低产量，再分配剩余任务；前面区间的计划产量移到左侧。
            terms = {variable: sign * factor for variable in range(commit_start)}
            terms.update({variable: sign for variable in range(commit_start, commit_end)})
            rows.append(_constraint_row(terms, row_width))
            limits.append(
                sign * (minimum_commit + (demand_to_schedule_t - minimum_remaining) * factor))

        commit_start = commit_end
        next_commit_end = remaining_products[commit_start][0] + unit.commit_horizon
        while commit_end < remaining_count and remaining_products[commit_end][0] < next_commit_end:
            commit_end += 1

    return rows, limits


def _power_coefficients(
    coefficients: _ProcessCoefficients, duration: float
) -> tuple[float, float, float, float]:
    """"按时段时长，将每吨钢耗电量换算为设备及总购电功率系数。"""
    electrolyser = coefficients.electrolyser_mwh_per_steel_t / duration
    dri = coefficients.dri_mwh_per_steel_t / duration
    eaf = coefficients.eaf_mwh_per_steel_t / duration
    return electrolyser, dri, eaf, electrolyser + dri + eaf


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
    """定义问题：构建一个滚动窗口的工业线性规划问题。

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
    # 所有剩余时段的钢产量变量索引为 0 到 remaining_count - 1。
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

    variable_costs = np.empty(count)
    for index, (start, _) in enumerate(products):
        raw_prices = fuel_price_profiles[start]
        variable_costs[index] = (
            coefficients.grid_mwh_per_steel_t * price_forecast[start] # 该时段每生产一吨钢的购电成本
            + coefficients.iron_ore_t_per_steel_t
            * _price(raw_prices, "iron ore", "iron_ore") # 每生产一吨钢所需的铁矿石成本
            + coefficients.lime_t_per_steel_t * _price(raw_prices, "lime")
            + coefficients.co2_t_per_steel_t * raw_prices["co2"]
            # 每吨钢的石灰成本和碳排放成本。
        )

    # 钢产量、正负偏差变量的上下界。
    bounds: list[tuple[float | None, float | None]] = [
        (_steel_lower_bound(unit, coefficients, duration),
         _steel_upper_bound(unit, coefficients, duration))
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

    # 4.1 逐时段建立：计划购电功率 - 正偏差 + 负偏差 = 参考功率。
    for index in range(count):
        row = _constraint_row(
            {
                index: grid_power[index],  # 钢产量的功率系数（MW/吨钢）
                remaining_count + index: -1.0,  # 正偏差
                remaining_count + count + index: 1.0,  # 负偏差
            },
            variable_count,
        )
        equality_rows.append(row)
        equality_values.append(
            remaining_load_profile[index] * unit.electrolyser.max_power_mw
        )  # 右侧参考功率

    # 4.2 各设备功率升降约束：设备功率 = 功率系数 × 该时段钢产量。
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    # 不等式左侧系数
    # 不等式右侧上限
    inequality_rows, inequality_values = _ramp_constraint_rows(
        devices,
        remaining_power_coefficients[:, :3],
        initial_powers_mw,
        row_width=variable_count,
    )

    # 4.3 剩余各时段产量之和 = 剩余需求。
    row = _constraint_row({variable: 1.0 for variable in range(remaining_count)}, variable_count)
    equality_rows.append(row)
    equality_values.append(demand_to_schedule_t)

    # 4.4 按参考负荷占比约束各承诺区间的产量。
    # commit_rows：不等式左侧的系数行；commit_limits：各行对应的右侧上限。
    # 两者一一对应：commit_rows[i] @ x <= commit_limits[i]，x 为完整决策变量向量。
    commit_rows, commit_limits = _commit_production_constraint_rows(
        unit,
        remaining_products,
        commit_count,
        remaining_load_profile,
        bounds,
        demand_to_schedule_t,
        row_width=variable_count,
    ) # 最少产量 和 最多产量
    inequality_rows.extend(commit_rows)
    inequality_values.extend(commit_limits)

    # 可选的成本上限行，供限定可变成本时追加到不等式矩阵末尾。
    cost_limit_row = np.zeros(variable_count)
    cost_limit_row[:count] = variable_costs

    # ---- 5. 组装问题 ----
    window_power_coefficients = remaining_power_coefficients[:count]
    return _IndustryProblem(
        products=products,
        variable_cost_per_tonne=variable_costs,
        grid_power_per_tonne=window_power_coefficients[:, 3],
        electrolyser_power_per_tonne=window_power_coefficients[:, 0],
        dri_power_per_tonne=window_power_coefficients[:, 1],
        eaf_power_per_tonne=window_power_coefficients[:, 2],
        objective=objective,
        A_eq=np.vstack(equality_rows),
        b_eq=np.array(equality_values),
        A_ub=np.vstack(inequality_rows),
        b_ub=np.array(inequality_values),
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
    """把前面准备好的目标和约束交给优化求解器，计算出满足约束、并让目标值最小的一组变量。
    """
    # 读取原始不等式约束：A_ub @ x <= b_ub，x 为完整决策变量向量。
    A_ub = problem.A_ub
    b_ub = problem.b_ub
    if variable_cost_limit is not None:
        # 扩展矩阵的最后一行已包含窗口可变成本系数，补充对应的成本上限。
        A_ub = problem.A_ub_limited
        # 上限加上数值容差，允许微小的浮点误差。
        limit = variable_cost_limit + _TOLERANCE
        # 在原约束右端向量末尾追加成本上限。
        b_ub = np.append(b_ub, limit)

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
    power_bounds_mw: tuple[tuple[float, float], ...],
) -> tuple[IndustrialResult, ...]:
    """将各承诺时段的优化钢产量换算为设备功率、氢气和 DRI 产量及预测可变成本，并附上购电功率上下界，返回计划元组。"""
    coefficients = process_coefficients(unit)
    plans: list[IndustrialResult] = []
    for index, (minimum_power, maximum_power) in enumerate(power_bounds_mw):
        start, end = problem.products[index]
        steel_output = steel[index]
        plans.append(
            IndustrialResult(
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
                forecast_cost_eur=(
                    problem.variable_cost_per_tonne[index] * steel_output),
                minimum_power_mw=minimum_power,
                maximum_power_mw=maximum_power,
            )
        )
    return tuple(plans)


def _calculate_industry_flexibility(
    unit: IndustrialUnit,
    problem: _IndustryProblem,
    committed_steel: np.ndarray,
    variable_cost_limit: float,
) -> tuple[tuple[float, float], ...]:
    """在生产约束和窗口总可变成本上限内，分别求各承诺时段的最低、最高购电功率，各时段极值不保证可同时实现。20% """
    bounds = []
    for index, steel in enumerate(committed_steel):
        # 仅以当前时段购电功率为目标；其他时段仍参与求解并满足原有约束。
        minimum_objective = np.zeros(problem.objective.size)
        minimum_objective[index] = (
            problem.grid_power_per_tonne[index]
        )
        minimum_solution = _solve(
            unit,
            problem,
            minimum_objective,
            variable_cost_limit=variable_cost_limit,
        )
        # 求解器执行最小化，目标取负即求当前时段购电功率的最大值。
        maximum_objective = -minimum_objective
        maximum_solution = _solve(
            unit,
            problem,
            maximum_objective,
            variable_cost_limit=variable_cost_limit,
        )
        minimum_power = (
            minimum_solution[index]
            * problem.grid_power_per_tonne[index]
        )
        maximum_power = (
            maximum_solution[index]
            * problem.grid_power_per_tonne[index]
        )
        # 基准计划应位于可达范围内，比较时允许数值容差。
        planned_power = problem.grid_power_per_tonne[index] * steel
        if minimum_power > planned_power + _TOLERANCE or (
            maximum_power < planned_power - _TOLERANCE):
            raise InputValidationError(
                f"工业单元 {unit.name!r} 在 {problem.products[index][0].isoformat(sep=' ')} "
                "时刻的灵活性上下界未包含基准计划功率。")
        bounds.append((minimum_power, maximum_power))
    return tuple(bounds)


def maximum_industry_production(
    unit: IndustrialUnit,
    products: tuple[tuple[datetime, datetime], ...],
    initial_powers_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """在设备容量和爬坡约束下最大化钢产量，传入全部时段的最大钢产量总和。"""
    if not products:
        return 0.0
    coefficients = process_coefficients(unit)
    durations = np.array([hours_between(*product) for product in products])
    bounds = [(_steel_lower_bound(unit, coefficients, duration),
               _steel_upper_bound(unit, coefficients, duration))
        for duration in durations]
    power_per_tonne = np.array([
        _power_coefficients(coefficients, duration)[:3] for duration in durations])
    devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
    rows, limits = _ramp_constraint_rows(
        devices,
        power_per_tonne,
        initial_powers_mw,
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
    tuple[IndustrialResult, ...],
    Industrial_Optimization_Window_Result,
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
    # 提取优化窗口内各时段的钢产量。
    count = len(products)
    steel = solution[:count]
    baseline_variable_cost = float(
        np.dot(problem.variable_cost_per_tonne, steel))
    flexible_limit = baseline_variable_cost * (
        1.0 + unit.cost_tolerance_percent / 100.0)
    
    window_id = f"{unit.name}::{products[0][0].isoformat()}"
    power_bounds = _calculate_industry_flexibility(
        unit, problem, steel[:commit_count], flexible_limit)
    plans = _build_industry_plans(unit, problem, steel, window_id, power_bounds)
    window = Industrial_Optimization_Window_Result(
        window_id=window_id,
        unit_name=unit.name,
        optimization_start=products[0][0],
        optimization_end=products[-1][1],
        commit_start=products[0][0],
        commit_end=products[commit_count - 1][1],
        baseline_variable_cost_eur=baseline_variable_cost,
        maximum_flexible_variable_cost_eur=flexible_limit,
    )
    return plans, window


def _actual_production(plan, accepted_energy_mwh):
    """按成交电量换算实际钢产量和购电功率，返回执行结果及电解槽、DRI、电弧炉的实际功率。"""
    planned_energy = plan.planned_energy_mwh
    ratio = 0.0 if planned_energy <= _TOLERANCE else accepted_energy_mwh / planned_energy
    duration = hours_between(plan.delivery_start, plan.delivery_end)
    dispatch = replace(
        plan,
        actual_grid_power_mw=(accepted_energy_mwh / duration),
        actual_steel_output_t=max(0.0, ratio) * plan.planned_steel_output_t,
    )
    actual_powers = (
        ratio * plan.electrolyser_power_mw,
        ratio * plan.dri_power_mw,
        ratio * plan.eaf_power_mw,
    )
    return dispatch, actual_powers


class IndustryRollingCoordinator:
    """管理工业用户的滚动生产计划、购电报价及交付状态更新。"""
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
        # 上游已完成交付时段的唯一性校验和排序。
        self.all_products = all_products
        self.price_forecasts = price_forecasts
        self.normalized_load_profiles = normalized_load_profiles
        self.fuel_price_profiles = fuel_price_profiles
        self.maximum_bid_price = maximum_bid_price
        # 以（用户名称，交付开始时间）保存未交付的承诺计划。
        self.plans: dict[tuple[str, datetime], IndustrialResult] = {}
        # 实际钢产量（吨）和设备功率作为后续窗口的初始状态。
        self.actual_steel_t = {unit.name: 0.0 for unit in units}
        # 连续生产的设备在仿真开始前已按整条生产链的最低负荷运行。
        self.last_actual_powers_mw: dict[str, tuple[float, float, float]] = {}
        for unit in units:
            coefficients = process_coefficients(unit)
            minimum_steel = _steel_lower_bound(unit, coefficients, 1.0)
            self.last_actual_powers_mw[unit.name] = tuple(
                minimum_steel * power
                for power in _power_coefficients(coefficients, 1.0)[:3]
            )
        # 累积交付、灵活性和窗口成本记录，仿真结束时汇入结果。
        self.dispatch_results: list[IndustrialResult] = []
        self.window_results: list[Industrial_Optimization_Window_Result] = []

    def bids_for_products(
        self,
        products: tuple[tuple[datetime, datetime], ...],
        scheduled_results: dict[tuple[datetime, datetime, datetime], MarketClearingResult],
    ) -> list[DemandBid]:
        """
        申报电量取计划购电量，价格取市场最高限价；零电量不申报。
        """
        bids: list[DemandBid] = []
        for unit in self.units:
            # 首个未承诺产品作为新滚动窗口的起点。
            missing_product = next(
                (product for product in products
                 if (unit.name, product[0]) not in self.plans),
                None,
            )
            if missing_product is not None:
                self._plan_window(unit, missing_product, scheduled_results)
            for start, end in products:
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
        plans, window = optimize_industry_window(
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
        for plan in plans:
            self.plans[(unit.name, plan.delivery_start)] = plan
        self.window_results.append(window)
        
    def _pending_state(
        self,
        unit: IndustrialUnit,
        first_product: tuple[datetime, datetime],
        scheduled_results,
    ):
        """计算前序未交付计划的预计钢产量，并确定本轮优化的起始设备功率。"""
        pending = []
        for (unit_name, _), plan in self.plans.items():
            if unit_name != unit.name:
                continue
            if plan.delivery_start >= first_product[0]:
                continue
            result = self._result_for_plan(plan, scheduled_results)
            accepted = self._cleared_industry_energy(result, unit.name)
            pending.append(_actual_production(plan, accepted))

        # 按交付时间排序后累加，保证求和顺序稳定；末段用于初始功率衔接。
        pending.sort(key=lambda item: item[0].delivery_start)
        pending_steel = sum(
            dispatch.actual_steel_output_t for dispatch, _ in pending)
        initial_powers = self.last_actual_powers_mw[unit.name]
        # 前序待交付产品若与新窗口相邻，以其成交后功率衔接爬坡约束。
        
        if pending:
            last_dispatch, actual_powers = pending[-1]
            if last_dispatch.delivery_end == first_product[0]:
                initial_powers = actual_powers
        return pending_steel, initial_powers


    def record_delivery(self, result: MarketClearingResult):
        """交付结束时按成交比例累计钢产量，并更新后续优化所需的设备功率。"""
        for unit in self.units:
            key = (unit.name, result.delivery_start)
            plan = self.plans.get(key)
            # 已记账计划会被移除；无待交付计划的产品直接跳过，避免重复累计产量。
            if plan is None:
                continue
            accepted = self._cleared_industry_energy(result, unit.name)
            dispatch, actual_powers = _actual_production(plan, accepted)
            devices = (unit.electrolyser, unit.dri_plant, unit.eaf)
            for device, power, previous in zip(
                devices, actual_powers, self.last_actual_powers_mw[unit.name]):

                if power < device.min_power_mw - _TOLERANCE:
                    raise InputValidationError(
                        f"工业单元 {unit.name!r} 在 {plan.delivery_start} 的 "
                        f"{device.technology} 实际功率 {power:.6f} MW "
                        f"低于最低生产功率 {device.min_power_mw:.6f} MW。")
                if (power - previous > device.ramp_up_mw + _TOLERANCE
                        or previous - power > device.ramp_down_mw + _TOLERANCE):
                    raise InputValidationError(
                        f"工业单元 {unit.name!r} 在 {plan.delivery_start} 的 "
                        f"{device.technology} 实际功率变化超过爬坡限制。")

            self.dispatch_results.append(dispatch)
            self.actual_steel_t[unit.name] += dispatch.actual_steel_output_t
            self.last_actual_powers_mw[unit.name] = actual_powers
            del self.plans[key]


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


    def _result_for_plan(self, plan: IndustrialResult, scheduled_results):
        """按交付起止时间查找对应的出清结果；入口已保证产品唯一。"""
        return next(
            result for result in scheduled_results.values()
            if result.delivery_start == plan.delivery_start
            and result.delivery_end == plan.delivery_end
        )

    @staticmethod
    def _cleared_industry_energy(
        result: MarketClearingResult, unit_name: str
    ) -> float:
        """从出清结果中查出指定钢厂实际买到的电量，未找到对应订单时返回零。"""
        for cleared_bid in result.demand_bids:
            bid = cleared_bid.bid
            if bid.unit_name == unit_name and bid.demand_type == "industrial_load":
                return cleared_bid.accepted_energy_mwh

        return 0.0


def _price(prices: dict[str, float], *names: str):
    """按候选字段名依次读取铁矿石或石灰价格，均未找到时返回零。"""
    for name in names:
        if name in prices:
            return prices[name]
    return 0.0


def missing_industrial_price_names(
    fuel_price_profiles: dict[datetime, dict[str, float]],
) -> tuple[str, ...]:
    """返回缺少价格数据、将按文档规定默认取 0 的原料名称。"""
    columns = set(next(iter(fuel_price_profiles.values())))
    missing: list[str] = []
    if not ({"iron ore", "iron_ore"} & columns):
        missing.append("iron ore")
    if "lime" not in columns:
        missing.append("lime")
    return tuple(missing)

