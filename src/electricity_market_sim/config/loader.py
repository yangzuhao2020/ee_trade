"""YAML parsing and validation for the supported single-node market design."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any

import yaml

from ..errors import InputValidationError
from ..models import MarketSettings


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


def _parse_additional_fields(eom: dict[str, Any]) -> frozenset[str]:
    """Read optional ASSUME order metadata fields without silently discarding them."""

    raw_fields = eom.get("additional_fields", [])
    if not isinstance(raw_fields, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw_fields
    ):
        raise InputValidationError(
            "markets_config.EOM.additional_fields must be a list of non-empty strings."
        )
    return frozenset(value.strip() for value in raw_fields)


def load_market_settings(config_path: Path, scenario: str = "base") -> MarketSettings:
    """Read one scenario's EOM settings, including V2 complex openings."""
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
    additional_fields = _parse_additional_fields(eom)
    if "start_date" in eom:
        market_start = _parse_datetime(
            eom["start_date"], "markets_config.EOM.start_date"
        )
    else:
        market_start = _parse_datetime(data["start_date"], "start_date")

    products = eom.get("products")
    if not isinstance(products, list) or len(products) != 1 or not isinstance(products[0], dict):
        raise InputValidationError("Version one requires exactly one EOM product definition.")
    product = products[0]

    try:
        settings = MarketSettings(
            start=market_start,
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
            additional_fields=additional_fields,
        )
    except KeyError as exc:
        raise InputValidationError(f"Missing required configuration field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid EOM configuration: {exc}") from exc

    if settings.end <= settings.start:
        raise InputValidationError("end_date must be later than start_date.")
    is_pay_as_bid = settings.market_mechanism == "pay_as_bid"
    if not is_pay_as_bid and settings.time_step != timedelta(hours=1):
        raise InputValidationError("Version one and two support only time_step: 1h.")
    if settings.opening_frequency <= timedelta(0):
        raise InputValidationError("EOM opening_frequency must be positive.")
    if settings.opening_duration <= timedelta(0):
        raise InputValidationError("EOM opening_duration must be positive.")
    if not is_pay_as_bid and settings.product_duration != timedelta(hours=1):
        raise InputValidationError("Version two supports only 1-hour EOM products.")
    if is_pay_as_bid and settings.product_duration != settings.time_step:
        raise InputValidationError(
            "pay_as_bid requires product duration to equal the scenario time_step."
        )
    if settings.product_count <= 0:
        raise InputValidationError("EOM product count must be positive.")
    if is_pay_as_bid:
        if settings.time_step != timedelta(minutes=15):
            raise InputValidationError(
                "Version three pay_as_bid requires a 15-minute time_step."
            )
        if settings.opening_frequency != timedelta(hours=24):
            raise InputValidationError(
                "Version three pay_as_bid requires a 24-hour opening_frequency."
            )
        if settings.first_delivery != timedelta(minutes=15):
            raise InputValidationError(
                "Version three pay_as_bid requires first_delivery: 15min."
            )
        if settings.product_count > 96:
            raise InputValidationError(
                "Version three pay_as_bid supports at most 96 products per opening."
            )
    if settings.market_mechanism not in {
        "pay_as_clear",
        "complex_clearing",
        "pay_as_bid",
    }:
        raise InputValidationError(
            "EOM market_mechanism must be pay_as_clear, complex_clearing, "
            "or pay_as_bid."
        )
    if settings.minimum_bid_price > settings.maximum_bid_price:
        raise InputValidationError("minimum_bid_price cannot exceed maximum_bid_price.")
    if scenario == "base" and settings.exchange_units_file is not None:
        raise InputValidationError("The 'base' scenario must not configure exchange_units.")
    if scenario == "base_with_exchanges" and settings.exchange_units_file is None:
        raise InputValidationError(
            "The 'base_with_exchanges' scenario requires an exchange_units CSV file."
        )

    return settings
