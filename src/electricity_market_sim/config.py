"""YAML parsing and validation for the supported single-node market design."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any

import yaml

from .errors import InputValidationError
from .models import MarketSettings


_DURATION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>min|m|h|d)$")


def parse_duration(value: Any, field_name: str) -> timedelta:
    """Parse the compact duration notation used by ASSUME-style YAML files."""

    if not isinstance(value, str):
        raise InputValidationError(f"{field_name} must be a duration such as '1h'.")

    match = _DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        raise InputValidationError(
            f"{field_name}={value!r} is invalid; use a value such as '1h' or '15m'."
        )

    magnitude = float(match.group("value"))
    unit = match.group("unit")
    if magnitude <= 0:
        raise InputValidationError(f"{field_name} must be greater than zero.")
    if unit in {"m", "min"}:
        return timedelta(minutes=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    return timedelta(days=magnitude)


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise InputValidationError(f"{field_name} must be an ISO-like date and time.")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputValidationError(
            f"{field_name}={value!r} cannot be parsed as a date and time."
        ) from exc


def load_market_settings(config_path: Path, scenario: str = "base") -> MarketSettings:
    """Read a supported scenario and reject incompatible market designs."""

    supported_scenarios = {"base", "base_with_exchanges"}
    if scenario not in supported_scenarios:
        raise InputValidationError(
            "Version one supports only 'base' or 'base_with_exchanges'; "
            f"received scenario {scenario!r}."
        )
    if not config_path.is_file():
        raise InputValidationError(f"Missing configuration file: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(f"Cannot parse YAML file {config_path}: {exc}") from exc

    if not isinstance(loaded, dict) or scenario not in loaded:
        raise InputValidationError(f"Scenario {scenario!r} is missing from {config_path}.")
    data = loaded[scenario]
    if not isinstance(data, dict):
        raise InputValidationError(f"Scenario {scenario!r} must be a YAML mapping.")

    raw_exchange_units = data.get("exchange_units")
    if raw_exchange_units is None:
        exchange_units_file = None
    elif isinstance(raw_exchange_units, str) and raw_exchange_units.strip():
        exchange_units_file = raw_exchange_units.strip()
    else:
        raise InputValidationError(
            "exchange_units must be null or the non-empty name of a CSV file."
        )

    markets = data.get("markets_config")
    if not isinstance(markets, dict) or "EOM" not in markets:
        raise InputValidationError("The base scenario must define markets_config.EOM.")
    eom = markets["EOM"]
    if not isinstance(eom, dict):
        raise InputValidationError("markets_config.EOM must be a mapping.")

    products = eom.get("products")
    if not isinstance(products, list) or len(products) != 1 or not isinstance(products[0], dict):
        raise InputValidationError("Version one requires exactly one EOM product definition.")
    product = products[0]

    try:
        settings = MarketSettings(
            start=_parse_datetime(data["start_date"], "start_date"),
            end=_parse_datetime(data["end_date"], "end_date"),
            time_step=parse_duration(data["time_step"], "time_step"),
            opening_frequency=parse_duration(
                eom["opening_frequency"], "markets_config.EOM.opening_frequency"
            ),
            opening_duration=parse_duration(
                eom["opening_duration"], "markets_config.EOM.opening_duration"
            ),
            product_duration=parse_duration(
                product["duration"], "markets_config.EOM.products[0].duration"
            ),
            product_count=int(product["count"]),
            first_delivery=parse_duration(
                product["first_delivery"], "markets_config.EOM.products[0].first_delivery"
            ),
            maximum_bid_price=float(eom["maximum_bid_price"]),
            minimum_bid_price=float(eom["minimum_bid_price"]),
            market_mechanism=str(eom["market_mechanism"]),
            exchange_units_file=exchange_units_file,
        )
    except KeyError as exc:
        raise InputValidationError(f"Missing required configuration field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid EOM configuration: {exc}") from exc

    if settings.end <= settings.start:
        raise InputValidationError("end_date must be later than start_date.")
    if settings.time_step != timedelta(hours=1):
        raise InputValidationError("Version one supports only time_step: 1h.")
    if settings.opening_frequency != timedelta(hours=1):
        raise InputValidationError("Version one supports hourly market openings only.")
    if settings.opening_duration != timedelta(hours=1):
        raise InputValidationError("Version one requires opening_duration: 1h.")
    if settings.product_duration != timedelta(hours=1) or settings.product_count != 1:
        raise InputValidationError("Version one supports exactly one 1-hour product per opening.")
    if settings.first_delivery != timedelta(hours=1):
        raise InputValidationError("Version one requires first_delivery: 1h.")
    if settings.market_mechanism != "pay_as_clear":
        raise InputValidationError("Version one supports only market_mechanism: pay_as_clear.")
    if settings.minimum_bid_price > settings.maximum_bid_price:
        raise InputValidationError("minimum_bid_price cannot exceed maximum_bid_price.")
    if scenario == "base" and settings.exchange_units_file is not None:
        raise InputValidationError("The 'base' scenario must not configure exchange_units.")
    if scenario == "base_with_exchanges" and settings.exchange_units_file is None:
        raise InputValidationError(
            "The 'base_with_exchanges' scenario requires an exchange_units CSV file."
        )

    return settings
