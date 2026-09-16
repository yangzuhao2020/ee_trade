"""CSV readers and input validation for the first-version data contract."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from math import isfinite
from pathlib import Path

from .errors import InputValidationError
from .models import DemandUnit, ExchangeSchedule, ExchangeUnit, PowerPlant


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise InputValidationError(f"{path} must contain a header row.")
        rows = list(reader)
    if not rows:
        raise InputValidationError(f"{path} contains no data rows.")
    return rows


def _require(row: dict[str, str], column: str, path: Path, row_number: int) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise InputValidationError(
            f"{path.name}, row {row_number}: required column {column!r} is empty."
        )
    return value.strip()


def _float(row: dict[str, str], column: str, path: Path, row_number: int) -> float:
    raw = _require(row, column, path, row_number)
    try:
        value = float(raw)
    except ValueError as exc:
        raise InputValidationError(
            f"{path.name}, row {row_number}: {column!r} must be numeric, got {raw!r}."
        ) from exc
    if not isfinite(value):
        raise InputValidationError(
            f"{path.name}, row {row_number}: {column!r} must be finite."
        )
    return value


def load_powerplants(path: Path) -> tuple[PowerPlant, ...]:
    """Read naïvely bidding power plants from `powerplant_units.csv`."""

    rows = _read_rows(path)
    plants: list[PowerPlant] = []
    names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        name = _require(row, "name", path, row_number)
        if name in names:
            raise InputValidationError(f"{path.name}: duplicate plant name {name!r}.")
        names.add(name)

        strategy = _require(row, "bidding_EOM", path, row_number)
        if strategy != "powerplant_energy_naive":
            raise InputValidationError(
                f"{path.name}, row {row_number}: only 'powerplant_energy_naive' is supported."
            )

        plant = PowerPlant(
            name=name,
            operator=_require(row, "unit_operator", path, row_number),
            technology=_require(row, "technology", path, row_number),
            fuel_type=_require(row, "fuel_type", path, row_number),
            emission_factor=_float(row, "emission_factor", path, row_number),
            max_power_mw=_float(row, "max_power", path, row_number),
            min_power_mw=_float(row, "min_power", path, row_number),
            efficiency=_float(row, "efficiency", path, row_number),
            additional_cost_eur_per_mwh=_float(row, "additional_cost", path, row_number),
        )
        if plant.max_power_mw <= 0:
            raise InputValidationError(f"{path.name}, row {row_number}: max_power must be positive.")
        if plant.min_power_mw < 0 or plant.min_power_mw > plant.max_power_mw:
            raise InputValidationError(
                f"{path.name}, row {row_number}: min_power must be between 0 and max_power."
            )
        if plant.efficiency <= 0:
            raise InputValidationError(f"{path.name}, row {row_number}: efficiency must be positive.")
        if plant.emission_factor < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: emission_factor cannot be negative."
            )
        plants.append(plant)
    return tuple(plants)


def load_demand_units(path: Path) -> tuple[DemandUnit, ...]:
    """Read inelastic demand units from `demand_units.csv`."""

    rows = _read_rows(path)
    units: list[DemandUnit] = []
    names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        name = _require(row, "name", path, row_number)
        if name in names:
            raise InputValidationError(f"{path.name}: duplicate demand unit name {name!r}.")
        names.add(name)
        strategy = _require(row, "bidding_EOM", path, row_number)
        if strategy != "demand_energy_naive":
            raise InputValidationError(
                f"{path.name}, row {row_number}: only 'demand_energy_naive' is supported."
            )
        units.append(
            DemandUnit(
                name=name,
                operator=_require(row, "unit_operator", path, row_number),
                profile_column=name,
            )
        )
    return tuple(units)


def load_exchange_unit(path: Path) -> ExchangeUnit:
    """Read exactly one v1 Exchange participant from `exchange_units.csv`."""

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
    """Read the v1 static `fuel_prices_df.csv` layout into a price mapping."""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or len(rows[0]) < 2:
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
    for fuel, raw_price in zip(headers[1:], values[1:], strict=True):
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
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise InputValidationError(
            f"{path.name}, row {row_number}: invalid datetime {raw!r}."
        ) from exc


def _hour_start(timestamp: datetime) -> datetime:
    return timestamp.replace(minute=0, second=0, microsecond=0)


def load_hourly_demand_profiles(
    path: Path, demand_units: tuple[DemandUnit, ...]
) -> dict[str, dict[datetime, float]]:
    """Load 15-minute (or hourly) MW data and resample each column by hourly mean."""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            raise InputValidationError(f"{path.name} must include a 'datetime' column.")
        missing = [unit.profile_column for unit in demand_units if unit.profile_column not in reader.fieldnames]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing demand profile columns: {', '.join(missing)}."
            )

        buckets: dict[str, dict[datetime, list[float]]] = {
            unit.name: defaultdict(list) for unit in demand_units
        }
        seen_timestamps: set[datetime] = set()
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row["datetime"], path, row_number)
            if timestamp in seen_timestamps:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: duplicate timestamp {timestamp.isoformat(sep=' ')}."
                )
            seen_timestamps.add(timestamp)
            bucket = _hour_start(timestamp)
            for unit in demand_units:
                value = _float(row, unit.profile_column, path, row_number)
                if value < 0:
                    raise InputValidationError(
                        f"{path.name}, row {row_number}: demand cannot be negative."
                    )
                buckets[unit.name][bucket].append(value)

    return {
        unit_name: {
            hour: sum(values) / len(values) for hour, values in hourly_values.items()
        }
        for unit_name, hourly_values in buckets.items()
    }


def load_hourly_exchange_profiles(
    path: Path, exchange_unit: ExchangeUnit
) -> dict[datetime, ExchangeSchedule]:
    """Load positive 15-minute exchange plans and resample them to hourly MW."""

    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")

    import_column = f"{exchange_unit.name}_import"
    export_column = f"{exchange_unit.name}_export"
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            raise InputValidationError(f"{path.name} must include a 'datetime' column.")
        missing = [
            column
            for column in (import_column, export_column)
            if column not in reader.fieldnames
        ]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing exchange profile columns: {', '.join(missing)}."
            )

        import_buckets: dict[datetime, list[float]] = defaultdict(list)
        export_buckets: dict[datetime, list[float]] = defaultdict(list)
        seen_timestamps: set[datetime] = set()
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row["datetime"], path, row_number)
            if timestamp in seen_timestamps:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: duplicate timestamp "
                    f"{timestamp.isoformat(sep=' ')}."
                )
            seen_timestamps.add(timestamp)
            import_power = _float(row, import_column, path, row_number)
            export_power = _float(row, export_column, path, row_number)
            if import_power < 0 or export_power < 0:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: import and export powers must be non-negative."
                )
            hour = _hour_start(timestamp)
            import_buckets[hour].append(import_power)
            export_buckets[hour].append(export_power)

    if import_buckets.keys() != export_buckets.keys():
        raise InputValidationError(
            f"{path.name}: import and export profiles do not cover the same hours."
        )
    return {
        hour: ExchangeSchedule(
            import_power_mw=sum(import_values) / len(import_values),
            export_power_mw=sum(export_buckets[hour]) / len(export_buckets[hour]),
        )
        for hour, import_values in import_buckets.items()
    }


def validate_fuel_coverage(plants: tuple[PowerPlant, ...], fuel_prices: dict[str, float]) -> None:
    missing = sorted({plant.fuel_type for plant in plants} - fuel_prices.keys())
    if missing:
        raise InputValidationError(
            "fuel_prices_df.csv is missing prices for: " + ", ".join(missing)
        )
