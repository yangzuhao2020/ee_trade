"""CSV readers and input validation for the supported V1/V2 data contract."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from math import isfinite
from pathlib import Path

from .errors import InputValidationError
from .models import DemandUnit, ExchangeSchedule, ExchangeUnit, PowerPlant, StorageUnit


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputValidationError(f"Missing required input file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
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


def _optional_float(
    row: dict[str, str],
    column: str,
    default: float,
    path: Path,
    row_number: int,
) -> float:
    """Read an optional numeric CSV column, using the documented V2 default."""

    raw = row.get(column)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
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
    """Read V1 naïve and V2 heuristic plants from `powerplant_units.csv`."""

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
        if plant.start_cost_eur < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: start_cost cannot be negative."
            )
        if plant.min_operating_time_hours <= 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: min_operating_time must be positive."
            )
        if plant.min_down_time_hours < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: min_down_time cannot be negative."
            )
        plants.append(plant)
    return tuple(plants)


def load_storage_units(path: Path) -> tuple[StorageUnit, ...]:
    """Read optional V2 storage participants from ``storage_units.csv``."""

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

        if storage.max_power_charge_mw < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: max_power_charge cannot be negative."
            )
        if storage.max_power_discharge_mw < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: max_power_discharge cannot be negative."
            )
        if (
            storage.max_power_charge_mw == 0
            and storage.max_power_discharge_mw == 0
        ):
            raise InputValidationError(
                f"{path.name}, row {row_number}: at least one storage power limit "
                "must be positive."
            )
        if not 0 < storage.efficiency_charge <= 1:
            raise InputValidationError(
                f"{path.name}, row {row_number}: efficiency_charge must be in (0, 1]."
            )
        if not 0 < storage.efficiency_discharge <= 1:
            raise InputValidationError(
                f"{path.name}, row {row_number}: efficiency_discharge must be in (0, 1]."
            )
        if not 0 <= storage.min_soc <= storage.max_soc <= 1:
            raise InputValidationError(
                f"{path.name}, row {row_number}: SOC limits must satisfy "
                "0 <= min_soc <= max_soc <= 1."
            )
        if not storage.min_soc <= storage.initial_soc <= storage.max_soc:
            raise InputValidationError(
                f"{path.name}, row {row_number}: initial_soc must satisfy "
                "min_soc <= initial_soc <= max_soc."
            )
        if storage.capacity_mwh <= 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: capacity must be positive."
            )
        if storage.additional_cost_charge_eur_per_mwh < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: additional_cost_charge "
                "cannot be negative."
            )
        if storage.additional_cost_discharge_eur_per_mwh < 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: additional_cost_discharge "
                "cannot be negative."
            )
        if storage.natural_inflow_mw != 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: non-zero natural_inflow is not "
                "supported in version two."
            )
        storages.append(storage)

    return tuple(storages)


def load_demand_units(path: Path) -> tuple[DemandUnit, ...]:
    """Read EOM demand participants and ignore rows belonging only to CRM."""

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
        if unit.max_power_mw is None or unit.max_power_mw <= 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: elastic max_power must be positive."
            )
        if unit.elasticity is None or unit.elasticity >= 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: elasticity must be negative."
            )
        if unit.elasticity_model != "isoelastic":
            raise InputValidationError(
                f"{path.name}, row {row_number}: only 'isoelastic' is supported."
            )
        if unit.max_price_eur_per_mwh is None or unit.max_price_eur_per_mwh <= 0:
            raise InputValidationError(
                f"{path.name}, row {row_number}: max_price must be positive."
            )
        if unit.num_bids is None or unit.num_bids < 2:
            raise InputValidationError(
                f"{path.name}, row {row_number}: num_bids must be at least 2."
            )
        units.append(unit)
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
        profiled_units = [unit for unit in demand_units if unit.profile_column is not None]
        missing = [unit.profile_column for unit in profiled_units if unit.profile_column not in reader.fieldnames]
        if missing:
            raise InputValidationError(
                f"{path.name} is missing demand profile columns: {', '.join(missing)}."
            )

        buckets: dict[str, dict[datetime, list[float]]] = {
            unit.name: defaultdict(list) for unit in profiled_units
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
            for unit in profiled_units:
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


def load_hourly_availability_profiles(
    path: Path, plants: tuple[PowerPlant, ...]
) -> dict[str, dict[datetime, float]]:
    """Load optional 0--1 availability factors and resample them by hourly mean.

    A plant with no column uses availability 1.0.  A plant with a column must
    provide every delivery hour that the simulation or its price forecast uses.
    """

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            raise InputValidationError(f"{path.name} must include a 'datetime' column.")
        plant_names = {plant.name for plant in plants}
        profile_columns = [
            column for column in reader.fieldnames if column != "datetime" and column in plant_names
        ]
        unknown_columns = [
            column
            for column in reader.fieldnames
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

        buckets: dict[str, dict[datetime, list[float]]] = {
            column: defaultdict(list) for column in profile_columns
        }
        seen_timestamps: set[datetime] = set()
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row["datetime"], path, row_number)
            if timestamp in seen_timestamps:
                raise InputValidationError(
                    f"{path.name}, row {row_number}: duplicate timestamp "
                    f"{timestamp.isoformat(sep=' ')}."
                )
            seen_timestamps.add(timestamp)
            hour = _hour_start(timestamp)
            for column in profile_columns:
                value = _float(row, column, path, row_number)
                if not 0.0 <= value <= 1.0:
                    raise InputValidationError(
                        f"{path.name}, row {row_number}: availability for {column!r} "
                        "must be between 0 and 1."
                    )
                buckets[column][hour].append(value)

    return {
        plant_name: {
            hour: sum(values) / len(values) for hour, values in hourly_values.items()
        }
        for plant_name, hourly_values in buckets.items()
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
    missing = sorted(
        {
            plant.fuel_type
            for plant in plants
            if plant.fuel_type != "renewable"
        }
        - fuel_prices.keys()
    )
    if missing:
        raise InputValidationError(
            "fuel_prices_df.csv is missing prices for: " + ", ".join(missing)
        )
