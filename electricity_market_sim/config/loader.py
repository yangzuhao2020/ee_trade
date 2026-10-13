"""从 YAML 读取并校验单节点市场配置、时间参数及第五版学习设置。"""

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
    """解析 ASSUME 风格 YAML 文件中使用的简写时长（如 '1h'、'15m'）。"""

    if not isinstance(value, str):
        raise InputValidationError(f"{field_name} 必须是时长字符串，例如 '1h'。")

    match = _DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        raise InputValidationError(
            f"{field_name}={value!r} 无效，请使用 '1h' 或 '15m' 这类格式。"
        )

    magnitude = float(match.group("value"))
    unit = match.group("unit")
    if magnitude <= 0:
        raise InputValidationError(f"{field_name} 必须大于 0。")
    if unit in {"m", "min"}:
        return timedelta(minutes=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    return timedelta(days=magnitude)


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise InputValidationError(f"{field_name} 必须是 ISO 格式的日期时间。")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputValidationError(
            f"{field_name}={value!r} 无法解析为日期时间。"
        ) from exc


def _parse_additional_fields(eom: dict[str, Any]) -> frozenset[str]:
    """读取可选的 ASSUME 订单元数据字段（additional_fields），不会静默丢弃这些字段。"""

    raw_fields = eom.get("additional_fields", [])
    if not isinstance(raw_fields, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw_fields
    ):
        raise InputValidationError(
            "markets_config.EOM.additional_fields 必须是由非空字符串组成的列表。"
        )
    return frozenset(value.strip() for value in raw_fields)


def _load_learning_config(
    data: dict[str, Any], *, time_step: timedelta
) -> LearningConfig | None:
    raw = data.get("learning_config")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputValidationError("learning_config 必须是 YAML 映射。")

    def required(name: str) -> Any:
        if name not in raw:
            raise InputValidationError(
                f"learning_config 缺少必填字段：{name}"
            )
        return raw[name]

    def optional_path(name: str) -> str | None:
        value = raw.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(
                f"learning_config.{name} 必须为 null 或非空路径。"
            )
        return value.strip()

    try:
        learning_mode = required("learning_mode")
        if not isinstance(learning_mode, bool):
            raise InputValidationError("learning_config.learning_mode 必须是布尔值。")
        continue_learning = raw.get("continue_learning", False)
        if not isinstance(continue_learning, bool):
            raise InputValidationError(
                "learning_config.continue_learning 必须是布尔值。"
            )
        train_frequency = parse_duration(
            required("train_freq"), "learning_config.train_freq"
        )
        frequency_ratio = train_frequency / time_step
        if not float(frequency_ratio).is_integer():
            raise InputValidationError(
                "learning_config.train_freq 必须是 time_step 的整数倍。"
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
            actor_architecture=str(raw.get("actor_architecture", "mlp")),
            device=str(raw.get("device", "cpu")),
            continue_learning=continue_learning,
            trained_policies_save_path=optional_path("trained_policies_save_path"),
            trained_policies_load_path=optional_path("trained_policies_load_path"),
            max_bid_price=float(raw.get("max_bid_price", 100.0)),
        )
    except InputValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"learning_config 无效：{exc}") from exc

    if config.algorithm != "matd3":
        raise InputValidationError(
            "第五版要求 learning_config.algorithm: matd3。"
        )
    if config.actor_architecture != "mlp":
        raise InputValidationError(
            "第五版仅支持 learning_config.actor_architecture: mlp。"
        )
    if config.action_noise_schedule != "linear":
        raise InputValidationError(
            "第五版要求 learning_config.action_noise_schedule: linear。"
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
            "以下 learning_config 字段必须为正数：" + ", ".join(invalid)
        )
    if config.initial_experience_episodes < 0:
        raise InputValidationError(
            "learning_config.episodes_collecting_initial_experience 不能为负数。"
        )
    if not isfinite(config.max_bid_price) or config.max_bid_price <= 0:
        raise InputValidationError(
            "learning_config.max_bid_price 必须是有限的正数。"
        )
    if config.replay_buffer_size < config.batch_size:
        raise InputValidationError(
            "learning_config.replay_buffer_size 不能小于 batch_size。"
        )
    if not 0.0 <= config.gamma <= 1.0:
        raise InputValidationError("learning_config.gamma 必须在 [0, 1] 范围内。")
    if not 0.0 < config.tau <= 1.0:
        raise InputValidationError("learning_config.tau 必须在 (0, 1] 范围内。")
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
            "以下 learning_config 字段不能为负数：" + ", ".join(invalid)
        )
    return config


def load_market_settings(config_path: Path, scenario: str = "base") -> MarketSettings:
    """读取一个场景的 EOM 市场配置"""
    if not config_path.is_file():
        raise InputValidationError(f"缺少配置文件：{config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(
            f"无法解析 YAML 文件 {config_path}：{exc}"
        ) from exc

    if not isinstance(loaded, dict) or scenario not in loaded:
        raise InputValidationError(
            f"{config_path} 中缺少场景 {scenario!r}。"
        )
    data = loaded[scenario]
    if not isinstance(data, dict):
        raise InputValidationError(f"场景 {scenario!r} 必须是 YAML 映射。")

    seed = data.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise InputValidationError("seed 必须为 null 或非负整数。")

    raw_exchange_units = data.get("exchange_units")
    if raw_exchange_units is None:
        exchange_units_file = None
    elif isinstance(raw_exchange_units, str) and raw_exchange_units.strip():
        exchange_units_file = raw_exchange_units.strip()
    else:
        raise InputValidationError(
            "exchange_units 必须为 null 或非空的 CSV 文件名。"
        )

    raw_industrial_units = data.get("industrial_dsm_units")
    if raw_industrial_units is None:
        industrial_dsm_units_file = None
    elif isinstance(raw_industrial_units, str) and raw_industrial_units.strip():
        industrial_dsm_units_file = raw_industrial_units.strip()
    else:
        raise InputValidationError(
            "industrial_dsm_units 必须为 null 或非空的 CSV 文件名。"
        )

    markets = data.get("markets_config")
    if not isinstance(markets, dict) or "EOM" not in markets:
        raise InputValidationError("场景必须定义 markets_config.EOM。")
    eom = markets["EOM"]
    if not isinstance(eom, dict):
        raise InputValidationError("markets_config.EOM 必须是映射。")
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
            "第一版要求恰好定义一个 EOM 产品。"
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
            f"缺少必填配置字段：{exc.args[0]}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"EOM 配置无效：{exc}") from exc

    if settings.end <= settings.start:
        raise InputValidationError("end_date 必须晚于 start_date。")
    is_pay_as_bid = settings.market_mechanism == "pay_as_bid"
    if not is_pay_as_bid and settings.time_step != timedelta(hours=1):
        raise InputValidationError("第一版和第二版仅支持 time_step: 1h。")
    if settings.opening_frequency <= timedelta(0):
        raise InputValidationError("EOM opening_frequency 必须大于 0。")
    if settings.opening_duration <= timedelta(0):
        raise InputValidationError("EOM opening_duration 必须大于 0。")
    if not is_pay_as_bid and settings.product_duration != timedelta(hours=1):
        raise InputValidationError("第二版仅支持 1 小时的 EOM 产品。")
    if is_pay_as_bid and settings.product_duration != settings.time_step:
        raise InputValidationError(
            "pay_as_bid 要求产品时长等于场景的 time_step。"
        )
    if settings.product_count <= 0:
        raise InputValidationError("EOM 产品的 count 必须为正数。")
    if is_pay_as_bid:
        if settings.time_step != timedelta(minutes=15):
            raise InputValidationError(
                "第三版 pay_as_bid 要求 time_step 为 15 分钟。"
            )
        if settings.opening_frequency != timedelta(hours=24):
            raise InputValidationError(
                "第三版 pay_as_bid 要求 opening_frequency 为 24 小时。"
            )
        if settings.first_delivery != timedelta(minutes=15):
            raise InputValidationError(
                "第三版 pay_as_bid 要求 first_delivery: 15min。"
            )
        if settings.product_count > 96:
            raise InputValidationError(
                "第三版 pay_as_bid 每次开市最多支持 96 个产品。"
            )
    if settings.market_mechanism not in {
        "pay_as_clear",
        "complex_clearing",
        "pay_as_bid",
    }:
        raise InputValidationError(
            "EOM market_mechanism 必须是 pay_as_clear、complex_clearing "
            "或 pay_as_bid。"
        )
    if settings.minimum_bid_price > settings.maximum_bid_price:
        raise InputValidationError("minimum_bid_price 不能大于 maximum_bid_price。")
    if scenario == "base" and settings.exchange_units_file is not None:
        raise InputValidationError(
            "'base' 场景不能配置 exchange_units。"
        )
    if scenario == "base_with_exchanges" and settings.exchange_units_file is None:
        raise InputValidationError(
            "'base_with_exchanges' 场景必须配置 exchange_units CSV 文件。"
        )

    return settings
