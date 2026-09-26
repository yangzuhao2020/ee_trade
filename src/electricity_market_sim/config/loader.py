"""YAML parsing and validation for the supported single-node market design."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from ..errors import InputValidationError
from ..models import LearningConfig, MarketSettings

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


def _load_learning_config(
    data: dict[str, Any], *, time_step: timedelta
) -> LearningConfig | None:
    raw = data.get("learning_config")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputValidationError("learning_config must be a YAML mapping.")

    def required(name: str) -> Any:
        if name not in raw:
            raise InputValidationError(
                f"Missing required learning_config field: {name}"
            )
        return raw[name]

    def optional_path(name: str) -> str | None:
        value = raw.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(
                f"learning_config.{name} must be null or a non-empty path."
            )
        return value.strip()

    try:
        learning_mode = required("learning_mode")
        if not isinstance(learning_mode, bool):
            raise InputValidationError("learning_config.learning_mode must be boolean.")
        continue_learning = raw.get("continue_learning", False)
        if not isinstance(continue_learning, bool):
            raise InputValidationError(
                "learning_config.continue_learning must be boolean."
            )
        train_frequency = parse_duration(
            required("train_freq"), "learning_config.train_freq"
        )
        frequency_ratio = train_frequency / time_step
        if not float(frequency_ratio).is_integer():
            raise InputValidationError(
                "learning_config.train_freq must be an integer multiple of time_step."
            )
        config = LearningConfig(
            learning_mode=learning_mode,
            algorithm=str(required("algorithm")),
            learning_rate=float(required("learning_rate")),
            training_episodes=int(required("training_episodes")),
            initial_experience_episodes=int(
                required("episodes_collecting_initial_experience")
            ),
            replay_buffer_size=int(required("replay_buffer_size")),
            batch_size=int(required("batch_size")),
            gamma=float(required("gamma")),
            train_frequency_steps=int(frequency_ratio),
            gradient_steps=int(required("gradient_steps")),
            validation_interval=int(required("validation_episodes_interval")),
            exploration_noise_std=float(required("exploration_noise_std")),
            noise_sigma=float(required("noise_sigma")),
            noise_scale=float(required("noise_scale")),
            noise_dt=float(required("noise_dt")),
            action_noise_schedule=str(required("action_noise_schedule")),
            tau=float(required("tau")),
            policy_delay=int(required("policy_delay")),
            target_policy_noise=float(required("target_policy_noise")),
            target_noise_clip=float(required("target_noise_clip")),
            device=str(raw.get("device", "cpu")),
            continue_learning=continue_learning,
            trained_policies_save_path=optional_path("trained_policies_save_path"),
            trained_policies_load_path=optional_path("trained_policies_load_path"),
            max_bid_price=float(raw.get("max_bid_price", 100.0)),
        )
    except InputValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid learning_config: {exc}") from exc

    if config.algorithm != "matd3":
        raise InputValidationError(
            "Version 5 requires learning_config.algorithm: matd3."
        )
    if config.action_noise_schedule != "linear":
        raise InputValidationError(
            "Version 5 requires learning_config.action_noise_schedule: linear."
        )
    positive = {
        "learning_rate": config.learning_rate,
        "training_episodes": config.training_episodes,
        "replay_buffer_size": config.replay_buffer_size,
        "batch_size": config.batch_size,
        "train_freq": config.train_frequency_steps,
        "gradient_steps": config.gradient_steps,
        "validation_episodes_interval": config.validation_interval,
        "noise_dt": config.noise_dt,
        "policy_delay": config.policy_delay,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise InputValidationError(
            "These learning_config fields must be positive: " + ", ".join(invalid)
        )
    if config.initial_experience_episodes < 0:
        raise InputValidationError(
            "learning_config.episodes_collecting_initial_experience cannot be negative."
        )
    if not isfinite(config.max_bid_price) or config.max_bid_price <= 0:
        raise InputValidationError(
            "learning_config.max_bid_price must be a positive finite value."
        )
    if config.replay_buffer_size < config.batch_size:
        raise InputValidationError(
            "learning_config.replay_buffer_size must be at least batch_size."
        )
    if not 0.0 <= config.gamma <= 1.0:
        raise InputValidationError("learning_config.gamma must be in [0, 1].")
    if not 0.0 < config.tau <= 1.0:
        raise InputValidationError("learning_config.tau must be in (0, 1].")
    nonnegative = {
        "exploration_noise_std": config.exploration_noise_std,
        "noise_sigma": config.noise_sigma,
        "noise_scale": config.noise_scale,
        "target_policy_noise": config.target_policy_noise,
        "target_noise_clip": config.target_noise_clip,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise InputValidationError(
            "These learning_config fields cannot be negative: " + ", ".join(invalid)
        )
    return config


def load_market_settings(config_path: Path, scenario: str = "base") -> MarketSettings:
    """Read one scenario's EOM settings, including V2 complex openings."""
    if not config_path.is_file():
        raise InputValidationError(f"Missing configuration file: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(
            f"Cannot parse YAML file {config_path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict) or scenario not in loaded:
        raise InputValidationError(
            f"Scenario {scenario!r} is missing from {config_path}."
        )
    data = loaded[scenario]
    if not isinstance(data, dict):
        raise InputValidationError(f"Scenario {scenario!r} must be a YAML mapping.")

    seed = data.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise InputValidationError("seed must be null or a non-negative integer.")

    raw_exchange_units = data.get("exchange_units")
    if raw_exchange_units is None:
        exchange_units_file = None
    elif isinstance(raw_exchange_units, str) and raw_exchange_units.strip():
        exchange_units_file = raw_exchange_units.strip()
    else:
        raise InputValidationError(
            "exchange_units must be null or the non-empty name of a CSV file."
        )

    raw_industrial_units = data.get("industrial_dsm_units")
    if raw_industrial_units is None:
        industrial_dsm_units_file = None
    elif isinstance(raw_industrial_units, str) and raw_industrial_units.strip():
        industrial_dsm_units_file = raw_industrial_units.strip()
    else:
        raise InputValidationError(
            "industrial_dsm_units must be null or the non-empty name of a CSV file."
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
    if (
        not isinstance(products, list)
        or len(products) != 1
        or not isinstance(products[0], dict)
    ):
        raise InputValidationError(
            "Version one requires exactly one EOM product definition."
        )
    product = products[0]

    try:
        time_step = parse_duration(data["time_step"], "time_step")
        settings = MarketSettings(
            start=market_start,
            end=_parse_datetime(data["end_date"], "end_date"),
            time_step=time_step,
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
                product["first_delivery"],
                "markets_config.EOM.products[0].first_delivery",
            ),
            maximum_bid_price=float(eom["maximum_bid_price"]),
            minimum_bid_price=float(eom["minimum_bid_price"]),
            market_mechanism=str(eom["market_mechanism"]),
            exchange_units_file=exchange_units_file,
            industrial_dsm_units_file=industrial_dsm_units_file,
            additional_fields=additional_fields,
            learning_config=_load_learning_config(data, time_step=time_step),
            seed=seed,
        )
    except KeyError as exc:
        raise InputValidationError(
            f"Missing required configuration field: {exc.args[0]}"
        ) from exc
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
        raise InputValidationError(
            "The 'base' scenario must not configure exchange_units."
        )
    if scenario == "base_with_exchanges" and settings.exchange_units_file is None:
        raise InputValidationError(
            "The 'base_with_exchanges' scenario requires an exchange_units CSV file."
        )

    return settings
