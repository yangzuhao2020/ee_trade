"""电力市场仿真的 CSV 输入加载与校验，覆盖发电、储能、需求、居民和工业参与者。
支持原始分辨率、逐小时均值及按产品时长对齐的时间序列读取。"""

import csv
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
import warnings
from ..errors import InputValidationError
from ..config import parse_duration
from ..models import (
    DemandUnit,
    ExchangeSchedule,
    ExchangeUnit,
    HouseholdUnit,
    IndustrialDevice,
    IndustrialUnit,
    PowerPlant,
    StorageUnit,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise InputValidationError(f"{path} must contain a header row.")
        rows = list(reader)
    if not rows:
        raise InputValidationError(f"{path} contains no data rows.")
    return rows


def _require(row: dict[str, str], column: str, path: Path, row_number: int) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise InputValidationError(
            f"{path.name}, row {row_number}: required column {column!r} is empty."
        )
    return value


def _parse_float(raw: str, context: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise InputValidationError(
            f"{context} must be numeric, got {raw!r}."
        ) from exc
    if not isfinite(value):
        raise InputValidationError(f"{context} must be finite.")
    return value


def _float(row: dict[str, str], column: str, path: Path, row_number: int) -> float:
    raw = _require(row, column, path, row_number)
    return _parse_float(raw, f"{path.name}, row {row_number}: {column!r}")


def _optional_float(
    row: dict[str, str],
    column: str,
    default: float,
    path: Path,
    row_number: int,
) -> float:
    """读取可选的数值型 CSV 列；列缺失或为空时返回文档规定的 V2 默认值。"""

    raw = row.get(column)
    if raw is None or not raw.strip():
        return default
    return _parse_float(raw, f"{path.name}, row {row_number}: {column!r}")


def load_powerplants(path: Path) -> tuple[PowerPlant, ...]:
    """从 `powerplant_units.csv` 读取 V1 简单报价（naïve）、V2 启发式报价和 V5 学习报价的发电机组。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    rows = _read_rows(path)
    plants: list[PowerPlant] = []
    names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        name = _require(row, "name", path, row_number)
        if name in names:
            raise InputValidationError(f"{path.name}: duplicate plant name {name!r}.")
        names.add(name)

        strategy = _require(row, "bidding_EOM", path, row_number)
        if strategy not in {
            "powerplant_energy_naive",
            "powerplant_energy_heuristic_flexable",
            "powerplant_energy_heuristic_block",
            "powerplant_energy_heuristic_linked",
            "powerplant_energy_learning",
        }:
            raise InputValidationError(
                f"{path.name}, row {row_number}: unsupported bidding_EOM strategy "
                f"{strategy!r}."
            )

        plant = PowerPlant(
            name=name,
            operator=_require(row, "unit_operator", path, row_number),
            technology=_require(row, "technology", path, row_number),
            bidding_strategy=strategy,
            fuel_type=_require(row, "fuel_type", path, row_number),
            emission_factor=_float(row, "emission_factor", path, row_number),
            max_power_mw=_float(row, "max_power", path, row_number),
            min_power_mw=_float(row, "min_power", path, row_number),
            efficiency=_float(row, "efficiency", path, row_number),
            additional_cost_eur_per_mwh=_float(row, "additional_cost", path, row_number),
            start_cost_eur=_optional_float(row, "start_cost", 0.0, path, row_number),
            min_operating_time_hours=_optional_float(
                row, "min_operating_time", 1.0, path, row_number
            ),
            min_down_time_hours=_optional_float(
                row, "min_down_time", 1.0, path, row_number
            ),
            bid_price_eur_per_mwh=(
                _float(row, "bid_price", path, row_number)
                if (row.get("bid_price") or "").strip()
                else None
            ),
        )
        if (plant.bid_price_eur_per_mwh is not None
                and strategy != "powerplant_energy_naive"):
            raise InputValidationError(
                f"{path.name}, row {row_number}: bid_price is only supported "
                "for powerplant_energy_naive."
            )
        plants.append(plant)
    return tuple(plants)


def load_storage_units(path: Path) -> tuple[StorageUnit, ...]:
    """从 ``storage_units.csv`` 读取可选的 V2 储能参与者。"""

    if not path.is_file():
        return ()

    rows = _read_rows(path)
    storages: list[StorageUnit] = []
    names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        name = _require(row, "name", path, row_number)
        if name in names:
            raise InputValidationError(
                f"{path.name}: duplicate storage unit name {name!r}."
            )
        names.add(name)

        strategy = _require(row, "bidding_EOM", path, row_number)
        if strategy != "storage_energy_heuristic_flexable":
            raise InputValidationError(
                f"{path.name}, row {row_number}: unsupported bidding_EOM strategy "
                f"{strategy!r}."
            )

        min_soc = _float(row, "min_soc", path, row_number)
        max_soc = _float(row, "max_soc", path, row_number)
        storage = StorageUnit(
            name=name,
            operator=_require(row, "unit_operator", path, row_number),
            technology=_require(row, "technology", path, row_number),
            bidding_strategy=strategy,
            max_power_charge_mw=_float(
                row, "max_power_charge", path, row_number
            ),
            max_power_discharge_mw=_float(
                row, "max_power_discharge", path, row_number
            ),
            efficiency_charge=_float(
                row, "efficiency_charge", path, row_number
            ),
            efficiency_discharge=_float(
                row, "efficiency_discharge", path, row_number
            ),
            min_soc=min_soc,
            max_soc=max_soc,
            capacity_mwh=_float(row, "capacity", path, row_number),
            initial_soc=_optional_float(
                row, "initial_soc", min_soc, path, row_number
            ),
            additional_cost_charge_eur_per_mwh=_float(
                row, "additional_cost_charge", path, row_number
            ),
            additional_cost_discharge_eur_per_mwh=_float(
                row, "additional_cost_discharge", path, row_number
            ),
            natural_inflow_mw=_optional_float(
                row, "natural_inflow", 0.0, path, row_number
            ),
        )

        storages.append(storage)

    return tuple(storages)


def load_demand_units(path: Path) -> tuple[DemandUnit, ...]:
    """读取 EOM 需求参与者，忽略只属于 CRM 的行。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    rows = _read_rows(path)
    units: list[DemandUnit] = []
    names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        name = _require(row, "name", path, row_number)
        if name in names:
            raise InputValidationError(f"{path.name}: duplicate demand unit name {name!r}.")
        names.add(name)
        raw_strategy = row.get("bidding_EOM")
        strategy = raw_strategy.strip() if raw_strategy else ""
        # CRM-only participants intentionally have no EOM strategy.
        if not strategy:
            continue
        if strategy not in {
            "demand_energy_naive",
            "demand_energy_heuristic_elastic",
        }:
            raise InputValidationError(
                f"{path.name}, row {row_number}: unsupported bidding_EOM strategy "
                f"{strategy!r}."
            )
        operator = _require(row, "unit_operator", path, row_number)
        if strategy == "demand_energy_naive":
            units.append(
                DemandUnit(
                    name=name,
                    operator=operator,
                    bidding_strategy=strategy,
                    profile_column=name,
                    price_eur_per_mwh=(
                        _float(row, "price", path, row_number)
                        if (row.get("price") or "").strip()
                        else None
                    ),
                )
            )
            continue

        elasticity = _float(row, "elasticity", path, row_number)
        elasticity_model = _require(row, "elasticity_model", path, row_number)
        max_price = _float(row, "max_price", path, row_number)
        num_bids_raw = _float(row, "num_bids", path, row_number)
        if not num_bids_raw.is_integer():
            raise InputValidationError(
                f"{path.name}, row {row_number}: num_bids must be an integer."
            )
        unit = DemandUnit(
            name=name,
            operator=operator,
            bidding_strategy=strategy,
            max_power_mw=_float(row, "max_power", path, row_number),
            elasticity=elasticity,
            elasticity_model=elasticity_model,
            max_price_eur_per_mwh=max_price,
            num_bids=int(num_bids_raw),
        )
        if unit.elasticity_model != "isoelastic":
            raise InputValidationError(
                f"{path.name}, row {row_number}: only 'isoelastic' is supported."
            )
        units.append(unit)
    return tuple(units)


def _group_field(
    rows: list[tuple[int, dict[str, str]]],
    column: str,
    path: Path,
    group: str,
    *,
    required: bool = True,
) -> str | None:
    values = {
        value
        for _, row in rows
        if (value := row.get(column, "").strip())
    }
    if len(values) > 1:
        raise InputValidationError(
            f"{path.name}: rows for one {group} disagree on {column!r}."
        )
    if not values:
        if required:
            raise InputValidationError(
                f"{path.name}: {group.replace(' ', '-')} field {column!r} is not configured."
            )
        return None
    return values.pop()


def _group_by_name(
    raw_rows: list[dict[str, str]], path: Path
) -> dict[str, list[tuple[int, dict[str, str]]]]:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in enumerate(raw_rows, start=2):
        grouped[_require(row, "name", path, row_number)].append((row_number, row))
    return grouped


def _devices_by_technology(
    name: str,
    rows: list[tuple[int, dict[str, str]]],
    required_devices: set[str],
    path: Path,
    group: str,
) -> dict[str, tuple[int, dict[str, str]]]:
    by_technology: dict[str, tuple[int, dict[str, str]]] = {}
    for row_number, row in rows:
        technology = _require(row, "technology", path, row_number)
        if technology in by_technology:
            raise InputValidationError(
                f"{path.name}: {group} {name!r} has duplicate {technology!r} rows."
            )
        by_technology[technology] = (row_number, row)
    missing = required_devices - by_technology.keys()
    if missing:
        raise InputValidationError(
            f"{path.name}: {group} {name!r} is missing device rows: "
            + ", ".join(sorted(missing))
            + "."
        )
    unsupported = by_technology.keys() - required_devices
    if unsupported:
        raise InputValidationError(
            f"{path.name}: {group} {name!r} has unsupported devices: "
            + ", ".join(sorted(unsupported))
            + "."
        )
    return by_technology


def load_household_units(path: Path) -> tuple[HouseholdUnit, ...]:
    """将热泵行和电池行合并为以建筑为单位的 V3 参与者。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    raw_rows = _read_rows(path)
    grouped = _group_by_name(raw_rows, path)

    households: list[HouseholdUnit] = []
    for name, rows in sorted(grouped.items()):
        by_technology = _devices_by_technology(
            name, rows, {"heat_pump", "generic_storage"}, path, "building"
        )

        heat_number, heat = by_technology["heat_pump"]
        battery_number, battery = by_technology["generic_storage"]
        strategy = _group_field(rows, "bidding_EOM", path, "building")
        if strategy != "household_energy_optimization":
            raise InputValidationError(
                f"{path.name}: building {name!r} uses unsupported bidding_EOM "
                f"{strategy!r}."
            )
        prosumer_raw = (_group_field(
            rows, "is_prosumer", path, "building", required=False
        ) or "No").lower()
        if prosumer_raw != "no":
            raise InputValidationError(
                f"{path.name}: version three currently requires is_prosumer=No."
            )
        cost_tolerance_raw = _group_field(rows, "cost_tolerance", path, "building")
        cost_tolerance = _parse_float(
            cost_tolerance_raw, f"{path.name}: building {name!r} cost_tolerance"
        )
        heat_min_operating_time = _optional_float(
            heat, "min_operating_time", 0.0, path, heat_number
        )
        if heat_min_operating_time != 0:
            raise InputValidationError(
                f"{path.name}: building {name!r} non-zero heat-pump "
                "min_operating_time is not supported in version three."
            )

        household = HouseholdUnit(
            name=name,
            operator=_group_field(rows, "unit_operator", path, "building"),
            bidding_strategy=strategy,
            objective=_group_field(rows, "objective", path, "building"),
            flexibility_measure=_group_field(
                rows, "flexibility_measure", path, "building"
            ),
            cost_tolerance_percent=cost_tolerance,
            is_prosumer=False,
            fixed_power_mw=0.0,
            heat_pump_max_power_mw=_float(
                heat, "max_power", path, heat_number
            ),
            heat_pump_min_power_mw=_float(
                heat, "min_power", path, heat_number
            ),
            heat_pump_ramp_up_mw=_float(heat, "ramp_up", path, heat_number),
            heat_pump_ramp_down_mw=_float(
                heat, "ramp_down", path, heat_number
            ),
            cop=_float(heat, "cop", path, heat_number),
            battery_capacity_mwh=_float(
                battery, "capacity", path, battery_number
            ),
            battery_min_soc=_float(battery, "min_soc", path, battery_number),
            battery_max_soc=_optional_float(
                battery, "max_soc", 1.0, path, battery_number
            ),
            battery_initial_soc=_float(
                battery, "initial_soc", path, battery_number
            ),
            battery_efficiency_charge=_float(
                battery, "efficiency_charge", path, battery_number
            ),
            battery_efficiency_discharge=_float(
                battery, "efficiency_discharge", path, battery_number
            ),
            battery_max_charge_power_mw=_float(
                battery, "max_charging_rate", path, battery_number
            ),
            battery_max_discharge_power_mw=_float(
                battery, "max_discharging_rate", path, battery_number
            ),
            battery_ramp_up_mw=_float(
                battery, "ramp_up", path, battery_number
            ),
            battery_ramp_down_mw=_float(
                battery, "ramp_down", path, battery_number
            ),
            battery_loss_rate=_optional_float(
                battery, "storage_loss_rate", 0.0, path, battery_number
            ),
        )
        if household.objective != "min_variable_cost":
            raise InputValidationError(
                f"{path.name}: building {name!r} objective must be min_variable_cost."
            )
        if household.flexibility_measure != "cost_based_load_shift":
            raise InputValidationError(
                f"{path.name}: building {name!r} flexibility_measure must be "
                "cost_based_load_shift."
            )
        households.append(household)
    return tuple(households)


def load_industrial_units(path: Path) -> tuple[IndustrialUnit, ...]:
    """将电解槽、DRI（直接还原铁）和 EAF（电弧炉）行合并为钢厂参与者。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    raw_rows = _read_rows(path)
    grouped = _group_by_name(raw_rows, path)

    def plant_field(
        rows: list[tuple[int, dict[str, str]]],
        column: str,
    ) -> str | None:
        return _group_field(rows, column, path, "steel plant")

    def numeric_plant_field(
        rows: list[tuple[int, dict[str, str]]], column: str
    ) -> float:
        raw = plant_field(rows, column)
        return _parse_float(raw, f"{path.name}: steel-plant field {column!r}")

    units: list[IndustrialUnit] = []
    for name, rows in sorted(grouped.items()):
        by_technology = _devices_by_technology(
            name, rows, {"electrolyser", "dri_plant", "eaf"}, path, "steel plant"
        )

        if plant_field(rows, "unit_type") != "steel_plant":
            raise InputValidationError(
                f"{path.name}: industrial unit {name!r} must use unit_type=steel_plant."
            )
        strategy = plant_field(rows, "bidding_EOM")
        if strategy != "industry_energy_optimization":
            raise InputValidationError(
                f"{path.name}: steel plant {name!r} uses unsupported bidding_EOM "
                f"{strategy!r}."
            )
        objective = plant_field(rows, "objective")
        if objective != "min_variable_cost":
            raise InputValidationError(
                f"{path.name}: steel plant {name!r} objective must be "
                "min_variable_cost."
            )
        flexibility = plant_field(rows, "flexibility_measure")
        if flexibility != "cost_based_load_shift":
            raise InputValidationError(
                f"{path.name}: steel plant {name!r} flexibility_measure must be "
                "cost_based_load_shift."
            )
        horizon_mode = plant_field(rows, "horizon_mode")
        if horizon_mode != "rolling_horizon":
            raise InputValidationError(
                f"{path.name}: steel plant {name!r} horizon_mode must be "
                "rolling_horizon."
            )

        def device(technology: str) -> IndustrialDevice:
            row_number, row = by_technology[technology]
            result = IndustrialDevice(
                technology=technology,
                fuel_type=(row.get("fuel_type") or "").strip(),
                max_power_mw=_float(row, "max_power", path, row_number),
                min_power_mw=_float(row, "min_power", path, row_number),
                ramp_up_mw=_float(row, "ramp_up", path, row_number),
                ramp_down_mw=_float(row, "ramp_down", path, row_number),
                efficiency=_optional_float(row, "efficiency", 0.0, path, row_number),
                specific_dri_demand=_optional_float(
                    row, "specific_dri_demand", 0.0, path, row_number
                ),
                specific_electricity_consumption=_optional_float(
                    row,
                    "specific_electricity_consumption",
                    0.0,
                    path,
                    row_number,
                ),
                specific_hydrogen_consumption=_optional_float(
                    row,
                    "specific_hydrogen_consumption",
                    0.0,
                    path,
                    row_number,
                ),
                specific_iron_ore_consumption=_optional_float(
                    row,
                    "specific_iron_ore_consumption",
                    0.0,
                    path,
                    row_number,
                ),
                specific_lime_demand=_optional_float(
                    row, "specific_lime_demand", 0.0, path, row_number
                ),
                lime_co2_factor=_optional_float(
                    row, "lime_co2_factor", 0.0, path, row_number
                ),
                min_operating_time_hours=_optional_float(
                    row, "min_operating_time", 0.0, path, row_number
                ),
                min_down_time_hours=_optional_float(
                    row, "min_down_time", 0.0, path, row_number
                ),
            )
            if result.min_operating_time_hours != 0 or result.min_down_time_hours != 0:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: version four supports "
                    "industrial min_operating_time=min_down_time=0 only."
                )
            return result

        electrolyser = device("electrolyser")
        dri_plant = device("dri_plant")
        eaf = device("eaf")
        if dri_plant.fuel_type != "hydrogen":
            raise InputValidationError(
                f"{path.name}: steel plant {name!r} DRI fuel_type must be hydrogen."
            )

        cost_tolerance = numeric_plant_field(rows, "cost_tolerance")
        demand = numeric_plant_field(rows, "demand")
        deviation = numeric_plant_field(rows, "load_profile_deviation")
        look_ahead = parse_duration(
            plant_field(rows, "look_ahead_horizon"), "look_ahead_horizon"
        )
        commit_horizon = parse_duration(
            plant_field(rows, "commit_horizon"), "commit_horizon"
        )
        rolling_step = parse_duration(
            plant_field(rows, "rolling_step"), "rolling_step"
        )
        if rolling_step != commit_horizon:
            raise InputValidationError(
                f"{path.name}: version four requires rolling_step=commit_horizon."
            )

        units.append(
            IndustrialUnit(
                name=name,
                operator=plant_field(rows, "unit_operator"),
                bidding_strategy=strategy,
                objective=objective,
                flexibility_measure=flexibility,
                cost_tolerance_percent=cost_tolerance,
                demand_t=demand,
                load_profile_deviation=deviation,
                horizon_mode=horizon_mode,
                look_ahead=look_ahead,
                commit_horizon=commit_horizon,
                rolling_step=rolling_step,
                forecast_price_column=(
                    plant_field(rows, "forecast_electricity_price_update")
                ),
                electrolyser=electrolyser,
                dri_plant=dri_plant,
                eaf=eaf,
            )
        )
    return tuple(units)


def _read_time_series(
    path: Path,
    select_columns: Callable[[list[str]], Sequence[str]],
) -> tuple[Sequence[str], dict[datetime, dict[str, float]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            raise InputValidationError(
                f"{path.name} must include a 'datetime' column."
            )
        columns = select_columns(reader.fieldnames)
        rows: dict[datetime, dict[str, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row["datetime"], path, row_number)
            if timestamp in rows:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: duplicate timestamp "
                    f"{timestamp.isoformat(sep=' ')}."
                )
            rows[timestamp] = {
                column: _float(row, column, path, row_number) for column in columns
            }
    return columns, rows


def _by_column(
    rows: dict[datetime, dict[str, float]], columns: Sequence[str]
) -> dict[str, dict[datetime, float]]:
    return {
        column: {timestamp: values[column] for timestamp, values in rows.items()}
        for column in columns
    }


def _hourly_means(
    rows: dict[datetime, dict[str, float]], columns: Sequence[str]
) -> dict[str, dict[datetime, float]]:
    """对每个小时内的可用样本按相同权重求平均。
    允许不完整的小时和不规则采样。只有一个样本时，该样本代表整个小时；没有样本的
    小时不会出现在结果中。这既不是按时间加权的平均，也不同于对齐曲线所用的
    “时段数据必须完整”规则。
    """

    buckets: dict[str, dict[datetime, list[float]]] = {
        column: defaultdict(list) for column in columns
    }
    for timestamp, values in rows.items():
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        for column in columns:
            buckets[column][hour].append(values[column])
    return {
        column: {hour: sum(values) / len(values) for hour, values in hourly.items()}
        for column, hourly in buckets.items()
    }


def load_time_series_profiles(
    path: Path,
    required_columns: tuple[str, ...],
    *,
    warn_unused_columns: bool = False,
) -> dict[str, dict[datetime, float]]:
    """按原始分辨率读取指定的时间序列列，不做小时聚合。
    返回结构为“列名 → 时间戳 → 数值”。时间戳为 datetime 对象，数值为 float，
    例如读取 demand_EOM 列后得到：
        {
            "demand_EOM": {
                datetime(2024, 1, 1, 0, 15): 100.0,
                datetime(2024, 1, 1, 0, 30): 120.0,
            }
        }
    若源数据每 15 分钟一条，则保留各条记录，不计算小时平均值。
    """
    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")

    def select(fieldnames: list[str]) -> tuple[str, ...]:
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing columns: {', '.join(missing)}."
            )
        if warn_unused_columns:
            unused = [
                column
                for column in fieldnames
                if column != "datetime" and column not in required_columns
            ]
            if unused:
                warnings.warn(
                    f"{path.name}: ignored unused columns: {', '.join(unused)}.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        return required_columns

    _, rows = _read_time_series(path, select)
    return _by_column(rows, required_columns)


def load_aligned_time_series_profiles(
    path: Path,
    required_columns: tuple[str, ...],
    product_duration: timedelta,
) -> dict[str, dict[datetime, float]]:
    """按产品分辨率读取数值曲线，更细粒度的数据取平均值。
    源数据的时间间隔必须固定，并且能整除市场产品时长。只有当某个产品时段内
    预期的源数据点全部存在时才保留该时段，这样后续按产品查找时会报告输入缺失，
    而不是对不完整的区间求平均。
    """

    exact = load_time_series_profiles(path, required_columns)
    return _align_time_series_profiles(path, exact, product_duration)


def _align_time_series_profiles(
    path: Path,
    exact: dict[str, dict[datetime, float]],
    product_duration: timedelta,
) -> dict[str, dict[datetime, float]]:
    """将已读取的各列曲线按完整产品时段对齐，不再读取文件。"""

    if not exact:
        return exact
    timestamps = sorted(next(iter(exact.values())))
    if len(timestamps) < 2:
        raise InputValidationError(
            f"{path.name} requires at least two timestamps to determine its cadence."
        )
    steps = {
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    }
    if len(steps) != 1:
        raise InputValidationError(f"{path.name} timestamps must use a regular cadence.")
    source_step = steps.pop()
    if source_step > product_duration:
        raise InputValidationError(
            f"{path.name} cadence {source_step} cannot be aligned to product "
            f"duration {product_duration}."
        )
    ratio = product_duration.total_seconds() / source_step.total_seconds()
    expected_count = round(ratio)
    if abs(ratio - expected_count) > 1e-9:
        raise InputValidationError(
            f"{path.name} cadence must divide the product duration exactly."
        )
    if expected_count == 1:
        return exact

    product_seconds = int(product_duration.total_seconds())
    epoch = datetime(1970, 1, 1)
    buckets: dict[datetime, list[datetime]] = defaultdict(list)
    for timestamp in timestamps:
        elapsed = int((timestamp - epoch).total_seconds())
        bucket = epoch + timedelta(seconds=(elapsed // product_seconds) * product_seconds)
        buckets[bucket].append(timestamp)

    aligned = {column: {} for column in exact}
    for bucket, members in buckets.items():
        expected = [bucket + index * source_step for index in range(expected_count)]
        if members != expected:
            continue
        for column in exact:
            aligned[column][bucket] = sum(exact[column][time] for time in members) / len(
                members
            )
    return aligned


def load_aligned_fuel_price_profiles(
    path: Path, product_duration: timedelta
) -> dict[datetime, dict[str, float]]:
    """按配置的产品分辨率读取所有燃料价格列。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")

    def select(fieldnames: list[str]) -> tuple[str, ...]:
        if fieldnames[0].strip() != "datetime":
            raise InputValidationError(
                f"{path.name} must use the time-series layout beginning with datetime."
            )
        columns = tuple(
            column
            for raw_column in fieldnames[1:]
            if (column := raw_column.strip())
        )
        if "co2" not in columns:
            raise InputValidationError(f"{path.name} must provide a 'co2' column.")
        return columns

    columns, rows = _read_time_series(path, select)
    by_column = _align_time_series_profiles(path, _by_column(rows, columns), product_duration)
    timestamps = sorted(by_column[columns[0]])
    return {
        timestamp: {column: by_column[column][timestamp] for column in columns}
        for timestamp in timestamps
    }


def load_fuel_price_profiles(path: Path) -> dict[datetime, dict[str, float]]:
    """读取 15 分钟场景使用的、按时间索引的燃料价格和 CO2 价格。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")

    def select(fieldnames: list[str]) -> list[str]:
        price_columns = [column for column in fieldnames if column != "datetime"]
        if "co2" not in price_columns:
            raise InputValidationError(f"{path.name} must provide a 'co2' column.")
        return price_columns

    _, profiles = _read_time_series(path, select)
    return profiles


def load_exact_availability_profiles(
    path: Path, plants: tuple[PowerPlant, ...]
) -> dict[str, dict[datetime, float]]:
    """按原始分辨率读取以机组名称为列的可用率曲线；没有曲线的机组可用率按 1.0 处理。"""

    if not path.is_file():
        return {}
    plant_names = {plant.name for plant in plants}
    columns, rows = _read_time_series(
        path,
        lambda fieldnames: [column for column in fieldnames if column in plant_names],
    )
    return _by_column(rows, columns)


def load_exchange_unit(path: Path) -> ExchangeUnit:
    """从 `exchange_units.csv` 读取唯一一个 V1 外部电网（Exchange）参与者。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    rows = _read_rows(path)
    if len(rows) != 1:
        raise InputValidationError(
            f"{path.name}: version one supports exactly one Exchange unit."
        )

    row_number = 2
    row = rows[0]
    strategy = _require(row, "bidding_EOM", path, row_number)
    if strategy != "exchange_energy_naive":
        raise InputValidationError(
            f"{path.name}, row {row_number}: only 'exchange_energy_naive' is supported."
        )
    return ExchangeUnit(
        name=_require(row, "name", path, row_number),
        operator=_require(row, "unit_operator", path, row_number),
        price_import_eur_per_mwh=_float(row, "price_import", path, row_number),
        price_export_eur_per_mwh=_float(row, "price_export", path, row_number),
    )


def load_fuel_prices(path: Path) -> dict[str, float]:
    """将 V1 静态格式的 `fuel_prices_df.csv` 读取为“燃料 → 价格”映射。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or not rows[0]:
        raise InputValidationError(
            f"{path.name} must contain a fuel header row and one price row."
        )
    headers = [header.strip() for header in rows[0]]
    values = [value.strip() for value in rows[1]]
    if headers[0] != "fuel":
        raise InputValidationError(
            f"{path.name} uses an unsupported format; the first column must be 'fuel'."
        )
    if len(values) != len(headers):
        raise InputValidationError(f"{path.name}: header and price row lengths differ.")

    prices: dict[str, float] = {}
    for fuel, raw_price in zip(headers[1:], values[1:]):
        if not fuel:
            raise InputValidationError(f"{path.name}: a fuel column has no name.")
        try:
            prices[fuel] = float(raw_price)
        except ValueError as exc:
            raise InputValidationError(
                f"{path.name}: price for {fuel!r} must be numeric, got {raw_price!r}."
            ) from exc
    if "co2" not in prices:
        raise InputValidationError(f"{path.name} must provide a 'co2' price column.")
    return prices


def _parse_datetime(raw: str, path: Path, row_number: int) -> datetime:
    value = raw.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for date_format in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                continue
        raise InputValidationError(
            f"{path.name}, row {row_number}: invalid datetime {raw!r}."
        )


def load_hourly_demand_profiles(
    path: Path, demand_units: tuple[DemandUnit, ...]
) -> dict[str, dict[datetime, float]]:
    """读取需求功率（MW）数据，按小时对可用样本求等权平均。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    profiled_units = [unit for unit in demand_units if unit.profile_column is not None]

    def select(fieldnames: list[str]) -> list[str]:
        missing = [
            unit.profile_column
            for unit in profiled_units
            if unit.profile_column not in fieldnames
        ]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing demand profile columns: {', '.join(missing)}."
            )
        return [unit.profile_column for unit in profiled_units]

    columns, rows = _read_time_series(path, select)
    hourly = _hourly_means(rows, columns)
    return {unit.name: hourly[unit.profile_column] for unit in profiled_units}


def load_hourly_availability_profiles(
    path: Path, plants: tuple[PowerPlant, ...]
) -> dict[str, dict[datetime, float]]:
    """读取可选的 0～1 可用率系数，按小时对可用样本求等权平均。
    没有对应列的机组可用率为 1.0。有对应列的机组必须提供仿真及其价格预测
    用到的每一个交付小时。
    """

    if not path.is_file():
        return {}
    plant_names = {plant.name for plant in plants}

    def select(fieldnames: list[str]) -> list[str]:
        profile_columns = [
            column for column in fieldnames if column != "datetime" and column in plant_names
        ]
        unknown_columns = [
            column
            for column in fieldnames
            if column != "datetime" and column not in plant_names
        ]
        if unknown_columns:
            raise InputValidationError(
                f"{path.name} contains no matching plant for availability columns: "
                + ", ".join(unknown_columns)
            )
        if not profile_columns:
            raise InputValidationError(
                f"{path.name} must contain at least one power-plant availability column."
            )
        return profile_columns

    columns, rows = _read_time_series(path, select)
    return _hourly_means(rows, columns)


def load_hourly_exchange_profiles(
    path: Path, exchange_unit: ExchangeUnit
) -> dict[datetime, ExchangeSchedule]:
    """读取外部电网交换功率（MW），按小时对可用样本求等权平均。"""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")

    import_column = f"{exchange_unit.name}_import"
    export_column = f"{exchange_unit.name}_export"

    def select(fieldnames: list[str]) -> list[str]:
        missing = [
            column
            for column in (import_column, export_column)
            if column not in fieldnames
        ]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing exchange profile columns: {', '.join(missing)}."
            )
        return [import_column, export_column]

    columns, rows = _read_time_series(path, select)
    hourly = _hourly_means(rows, columns)
    return {
        hour: ExchangeSchedule(
            import_power_mw=import_power,
            export_power_mw=hourly[export_column][hour],
        )
        for hour, import_power in hourly[import_column].items()
    }
