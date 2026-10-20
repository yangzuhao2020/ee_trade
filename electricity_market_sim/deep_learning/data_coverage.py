"""训练前检查三个数据时段的覆盖；不解析时序数值，不生成预测或市场反馈。"""

import csv
from datetime import datetime
from pathlib import Path

from ..errors import InputValidationError
from ..inputs import load_demand_units, load_exchange_unit, load_powerplants, load_storage_units
from ..inputs.csv_loader import _parse_datetime
from ..models import MarketSettings
from ..simulation import learning_period_settings, market_openings
from ..simulators.base_market.forecasts import _forecast_starts
from ..simulators.base_market.market import _FORECAST_STRATEGIES, _POWER_TOLERANCE_MW
from ..utils import format_timestamp


def _hourly_coverage(
    path: Path, columns: tuple[str, ...] | None = None,
) -> dict[str, set[datetime]]:
    """只保留列名与小时索引；与运行时一致，每小时有一个样本即视为覆盖。"""
    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            raise InputValidationError(f"{path.name} must include a 'datetime' column.")
        selected = (
            tuple(name for name in reader.fieldnames if name != "datetime")
            if columns is None else columns
        )
        missing = set(selected) - set(reader.fieldnames)
        if missing:
            raise InputValidationError(
                f"{path.name} is missing profile columns: {', '.join(sorted(missing))}."
            )
        coverage = {name: set() for name in selected}
        timestamps = set()
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row.get("datetime") or "", path, row_number)
            if timestamp in timestamps:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: duplicate timestamp {timestamp}."
                )
            timestamps.add(timestamp)
            hour = timestamp.replace(minute=0, second=0, microsecond=0)
            for name in selected:
                # 检查单元格存在，不把需求、可用率或交换功率转换为数值。
                if not (row.get(name) or "").strip():
                    raise InputValidationError(
                        f"{path.name}/{name}, row {row_number}: missing data at {timestamp}."
                    )
                coverage[name].add(hour)
    return coverage


def validate_learning_data_coverage(input_path: Path, settings: MarketSettings) -> None:
    """在训练和检查点写入前预检 train、validation、test 的数据完整性。"""
    plants = load_powerplants(input_path / "powerplant_units.csv")
    demands = load_demand_units(input_path / "demand_units.csv")
    storages = load_storage_units(input_path / "storage_units.csv")
    demand_columns = tuple(unit.profile_column for unit in demands if unit.profile_column is not None)
    profiles = [
        (f"demand_df.csv/{name}", hours)
        for name, hours in _hourly_coverage(input_path / "demand_df.csv", demand_columns).items()
    ]
    availability_path = input_path / "availability_df.csv"
    if availability_path.is_file():
        availability = _hourly_coverage(availability_path)
        unknown = availability.keys() - {plant.name for plant in plants}
        if unknown or not availability:
            raise InputValidationError(
                "availability_df.csv must contain matching power-plant columns; "
                f"unknown columns: {', '.join(sorted(unknown))}."
            )
        profiles.extend((f"availability_df.csv/{name}", hours) for name, hours in availability.items())
    if settings.exchange_units_file is not None:
        exchange = load_exchange_unit(input_path / settings.exchange_units_file)
        exchange_columns = (f"{exchange.name}_import", f"{exchange.name}_export")
        profiles.extend(
            (f"exchanges_df.csv/{name}", hours)
            for name, hours in _hourly_coverage(input_path / "exchanges_df.csv", exchange_columns).items()
        )
    requires_forecast = any(
        plant.bidding_strategy in _FORECAST_STRATEGIES and plant.min_power_mw > _POWER_TOLERANCE_MW
        for plant in plants
    )
    for name, _ in settings.learning_config.periods:
        period_settings = learning_period_settings(settings, name)
        deliveries = {
            start for opening in market_openings(period_settings) for start, _ in opening.products
        }
        required = deliveries | _forecast_starts(
            deliveries, settings=period_settings,
            requires_price_forecast=requires_forecast,
            requires_storage_price_forecast=bool(storages), hard_forecast_horizon=True,
        )
        for source, hours in profiles:
            missing = sorted(required - hours)
            if missing:
                detail = ", ".join(format_timestamp(time) for time in missing[:6])
                raise InputValidationError(
                    f"学习时段 {name} 缺少 {source} 交付/预测数据：{detail}"
                    f"（共 {len(missing)} 个小时）。"
                )
