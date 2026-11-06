"""从 YAML 读取并校验单节点市场配置、时间参数及第五版学习设置。"""

import re
import warnings
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path
import yaml
from ..errors import InputValidationError
from ..models import LEARNING_PERIOD_NAMES, LearningConfig, MarketSettings

# 匹配数值与时间单位，如 15min、1.5h。
_DURATION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>min|m|h|d)$")
# 同目录下的算法默认配置，场景配置可覆盖。
_MATD3_DEFAULTS_PATH = Path(__file__).with_name("matd3.yaml")
_GRPO_DEFAULTS_PATH = Path(__file__).with_name("grpo.yaml")
# 场景专属设置，包括学习开关、时段和策略路径，禁止写入公共默认配置。
_LEARNING_SCENE_FIELDS = frozenset({
    "learning_mode", "periods",
    "trained_policies_save_path", "trained_policies_load_path",
})
# 通用及 MATD3 训练参数字段，与场景字段合并后用于识别未知配置项。
_COMMON_AND_MATD3_PARAMETER_FIELDS = frozenset({
    "algorithm", "actor_architecture", "device", "learning_rate",
    "training_episodes", "episodes_collecting_initial_experience",
    "replay_buffer_size", "batch_size", "gamma", "train_freq", "gradient_steps",
    "validation_episodes_interval", "max_bid_price", "exploration_noise_std",
    "action_noise_schedule", "noise_sigma", "noise_scale", "noise_dt", "tau",
    "policy_delay", "target_policy_noise", "target_noise_clip",
})
_GRPO_PARAMETER_FIELDS = frozenset({
    "validation_groups_interval",
    "training_steps", "group_size", "sampling_workers", "update_epochs", "minibatch_size",
    "clip_epsilon", "target_kl", "initial_std", "min_std", "max_std", "advantage_threshold",
    "max_grad_norm", "weight_decay", "min_lr_ratio",
})
_LEARNING_FIELDS = _COMMON_AND_MATD3_PARAMETER_FIELDS | _LEARNING_SCENE_FIELDS | _GRPO_PARAMETER_FIELDS


def _warn_unknown_learning_fields(raw, source):
    """仅由 _load_learning_config 调用，警告位置指向 load_market_settings 的调用方。"""
    unknown = raw.keys() - _LEARNING_FIELDS
    if unknown:
        warnings.warn(
            f"{source} 中的未知强化学习字段将被忽略："
            + ", ".join(sorted(map(str, unknown))),
            UserWarning,
            stacklevel=4,
        )


def _load_learning_defaults(path):
    """读取与加载器同目录的公共训练参数；场景专属设置不能作为默认值。"""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InputValidationError(f"无法读取公共强化学习配置 {path}：{exc}") from exc
    if not isinstance(loaded, dict) or set(loaded) != {"learning_config"}:
        raise InputValidationError(f"{path} 必须只包含 learning_config 映射。")
    raw = loaded["learning_config"]
    forbidden = raw.keys() & _LEARNING_SCENE_FIELDS
    if forbidden:
        raise InputValidationError(
            f"{path} 不能包含场景专属字段：" + ", ".join(sorted(forbidden))
        )
    return raw


def parse_duration(value, field_name):
    """解析 YAML 文件中使用的简写时长（如 '1h'、'15m'）。"""
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        raise InputValidationError(
            f"{field_name}={value!r} 无效，请使用 '1h' 或 '15m' 这类格式。"
        )

    magnitude = float(match.group("value"))
    unit = match.group("unit")
    if unit in {"m", "min"}:
        duration = timedelta(minutes=magnitude)
    elif unit == "h":
        duration = timedelta(hours=magnitude)
    else:
        duration = timedelta(days=magnitude)
    if duration <= timedelta(0):
        raise InputValidationError(f"{field_name} 必须大于 0。")
    return duration


def _parse_integer(value, field_name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise InputValidationError(f"{field_name} 必须是不小于 {minimum} 的整数。")
    return value


def _parse_datetime(value, field_name):
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputValidationError(
            f"{field_name}={value!r} 无法解析为日期时间。"
        ) from exc


def _load_learning_periods(raw, *, required):
    """读取按交付时间定义的训练、验证、测试时段，并检查顺序与重叠。"""
    if raw is None:
        if required:
            raise InputValidationError(
                "学习模式必须在 learning_config.periods 中配置 train、validation、"
                "test 三个交付时段。"
            )
        return None
    if not isinstance(raw, dict) or set(raw) != set(LEARNING_PERIOD_NAMES):
        raise InputValidationError(
            "learning_config.periods 必须恰好包含 train、validation、test 三个时段。"
        )
    periods = {}
    for name in LEARNING_PERIOD_NAMES:
        value = raw[name]
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            raise InputValidationError(
                f"learning_config.periods.{name} 必须只包含 start 和 end。"
            )
        start = _parse_datetime(value["start"], f"learning_config.periods.{name}.start")
        end = _parse_datetime(value["end"], f"learning_config.periods.{name}.end")
        if end <= start:
            raise InputValidationError(
                f"learning_config.periods.{name} 的 end 必须晚于 start。"
            )
        periods[name] = (start, end)
    ordered = list(periods.items())
    for (earlier_name, earlier), (later_name, later) in zip(ordered, ordered[1:]):
        if earlier[1] > later[0]:
            raise InputValidationError(
                f"learning_config.periods.{earlier_name} 必须在 {later_name} "
                "开始之前结束，三个时段不能重叠。"
            )
    return periods


def _validate_learning_period_alignment(settings):
    """检查各时段边界都落在交付产品的整数倍时刻上。"""
    for name, (start, end) in settings.learning_config.periods.items():
        for label, moment in (("start", start), ("end", end)):
            since_midnight = moment - moment.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if since_midnight % settings.product_duration:
                raise InputValidationError(
                    f"learning_config.periods.{name}.{label} 必须与 "
                    "交付产品时长对齐。"
                )


def _load_learning_config(data, *, time_step, source):
    raw = data.get("learning_config")
    if raw is None:
        return None
    if raw.get("algorithm", "matd3") not in {"matd3", "grpo"}:
        raise InputValidationError("learning_config.algorithm 仅支持 matd3 或 grpo。")

    _warn_unknown_learning_fields(raw, f"{source} 的 learning_config")
    if raw.get("learning_mode") is True:
        defaults_path = (_MATD3_DEFAULTS_PATH if raw.get("algorithm", "matd3") == "matd3"
                         else _GRPO_DEFAULTS_PATH)
        defaults = _load_learning_defaults(defaults_path)
        _warn_unknown_learning_fields(defaults, str(defaults_path))
        raw = {**defaults, **raw}
    nulls = sorted(
        name for name in _COMMON_AND_MATD3_PARAMETER_FIELDS | _GRPO_PARAMETER_FIELDS
        if name in raw and raw[name] is None
    )
    if nulls:
        raise InputValidationError(
            f"{source} 的 learning_config 参数不能为 null（需要公共默认值时请省略该字段）："
            + ", ".join(nulls))

    def required(name):
        if name not in raw:
            raise InputValidationError(
                f"learning_config 缺少必填字段：{name}"
            )
        return raw[name]

    def optional_path(name):
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
        algorithm = required("algorithm")
        grpo = algorithm == "grpo"
        if str(raw.get("actor_architecture", "mlp")) != "mlp":
            raise InputValidationError("第五版仅支持 learning_config.actor_architecture: mlp。")
        interval_name = "validation_groups_interval" if grpo else "validation_episodes_interval"
        common = {
            "learning_mode": learning_mode,
            "algorithm": algorithm,
            "learning_rate": float(required("learning_rate")),
            "gamma": float(required("gamma")),
            "validation_interval": _parse_integer(required(interval_name), f"learning_config.{interval_name}"),
            "device": str(raw.get("device", "cpu")),
            "trained_policies_save_path": optional_path("trained_policies_save_path"),
            "trained_policies_load_path": optional_path("trained_policies_load_path"),
            "max_bid_price": float(raw.get("max_bid_price", 100.0)),
            "periods": _load_learning_periods(raw.get("periods"), required=learning_mode),
        }
        # GRPO 不读取 TD3 专属字段；旧配置仍保持原必填校验。
        if grpo:
            config = LearningConfig(
                **common,
                group_size=_parse_integer(required("group_size"), "learning_config.group_size", minimum=2),
                **{name: _parse_integer(required(name), f"learning_config.{name}") for name in (
                    "training_steps", "sampling_workers", "update_epochs", "minibatch_size"
                )},
                **{name: float(required(name)) for name in (
                    "clip_epsilon", "target_kl", "initial_std", "min_std", "max_std",
                    "advantage_threshold",
                    "max_grad_norm", "weight_decay", "min_lr_ratio",
                )},
            )
            _validate_grpo_config(config)
            return config
        train_frequency = parse_duration(
            required("train_freq"), "learning_config.train_freq"
        )
        frequency_ratio = train_frequency / time_step
        if not frequency_ratio.is_integer():
            raise InputValidationError(
                "learning_config.train_freq 必须是 time_step 的整数倍。"
            )
        if str(required("action_noise_schedule")) != "linear":
            raise InputValidationError("第五版要求 learning_config.action_noise_schedule: linear。")
        config = LearningConfig(
            **common,
            training_episodes=_parse_integer(required("training_episodes"), "learning_config.training_episodes"),
            initial_experience_episodes=_parse_integer(
                required("episodes_collecting_initial_experience"),
                "learning_config.episodes_collecting_initial_experience", minimum=0,
            ),
            replay_buffer_size=_parse_integer(required("replay_buffer_size"), "learning_config.replay_buffer_size"),
            batch_size=_parse_integer(required("batch_size"), "learning_config.batch_size"),
            train_frequency_steps=int(frequency_ratio),
            gradient_steps=_parse_integer(required("gradient_steps"), "learning_config.gradient_steps"),
            exploration_noise_std=float(required("exploration_noise_std")),
            noise_sigma=float(required("noise_sigma")),
            noise_scale=float(required("noise_scale")),
            noise_dt=float(required("noise_dt")),
            tau=float(required("tau")),
            policy_delay=_parse_integer(required("policy_delay"), "learning_config.policy_delay"),
            target_policy_noise=float(required("target_policy_noise")),
            target_noise_clip=float(required("target_noise_clip")),
            training_steps=_parse_integer(raw.get("training_steps", 0), "learning_config.training_steps", minimum=0),
        )
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"learning_config 无效：{exc}") from exc

    positive = {
        "learning_rate": config.learning_rate,
        "noise_dt": config.noise_dt,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise InputValidationError(
            "以下 learning_config 字段必须为正数：" + ", ".join(invalid)
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


def _validate_grpo_config(config):
    positive = (
        "learning_rate",
        "target_kl", "initial_std", "min_std", "max_std", "advantage_threshold", "max_bid_price",
        "max_grad_norm",
    )
    if any(not isfinite(getattr(config, name)) or getattr(config, name) <= 0 for name in positive):
        raise InputValidationError("GRPO 的学习率、标准差、报价上限和阈值必须为有限正数。")
    if not config.min_std <= config.initial_std <= config.max_std:
        raise InputValidationError("GRPO 必须满足 min_std <= initial_std <= max_std。")
    if not 0 <= config.gamma <= 1:
        raise InputValidationError("GRPO 要求 gamma 在 [0, 1] 内。")


def load_market_settings(config_path, scenario="base"):
    """读取一个场景的 EOM 市场配置"""
    if not config_path.is_file():
        raise InputValidationError(f"缺少配置文件：{config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(loaded, dict) or scenario not in loaded:
        raise InputValidationError(
            f"{config_path} 中缺少场景 {scenario!r}。"
        )
    data = loaded[scenario]

    seed = data.get("seed")
    if seed is not None:
        seed = _parse_integer(seed, "seed", minimum=0)

    exchange_units_file = data.get("exchange_units")
    if exchange_units_file is not None:
        exchange_units_file = exchange_units_file.strip()

    industrial_dsm_units_file = data.get("industrial_dsm_units")
    if industrial_dsm_units_file is not None:
        industrial_dsm_units_file = industrial_dsm_units_file.strip()

    markets = data.get("markets_config")
    if not isinstance(markets, dict) or "EOM" not in markets:
        raise InputValidationError("场景必须定义 markets_config.EOM。")
    eom = markets["EOM"]
    additional_fields = frozenset(value.strip() for value in eom.get("additional_fields", []))
    if "start_date" in eom:
        market_start = _parse_datetime(
            eom["start_date"], "markets_config.EOM.start_date"
        )
    else:
        market_start = _parse_datetime(data["start_date"], "start_date")

    products = eom.get("products")
    if len(products) != 1:
        raise InputValidationError(
            "第一版要求恰好定义一个 EOM 产品。")
    product = products[0]

    try:
        time_step = parse_duration(data["time_step"], "time_step")
        # 开市持续时间尚未用于事件调度，只校验输入。
        parse_duration(eom["opening_duration"], "markets_config.EOM.opening_duration")
        settings = MarketSettings(
            start=market_start,
            end=_parse_datetime(data["end_date"], "end_date"),
            time_step=time_step,
            opening_frequency=parse_duration(
                eom["opening_frequency"], "markets_config.EOM.opening_frequency"
            ),
            product_duration=parse_duration(
                product["duration"], "markets_config.EOM.products[0].duration"
            ),
            product_count=_parse_integer(product["count"], "markets_config.EOM.products[0].count"),
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
            learning_config=_load_learning_config(
                data, time_step=time_step, source=f"{config_path} 场景 {scenario!r}"
            ),
            seed=seed,
        )
    except KeyError as exc:
        raise InputValidationError(
            f"缺少必填配置字段：{exc.args[0]}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"EOM 配置无效：{exc}") from exc

    is_learning = (
        settings.learning_config is not None and settings.learning_config.learning_mode
    )
    # 学习模式的实际仿真范围由 learning_config.periods 决定，原有起止日期不生效。
    if not is_learning and settings.end <= settings.start:
        raise InputValidationError("end_date 必须晚于 start_date。")
    is_pay_as_bid = settings.market_mechanism == "pay_as_bid"
    if not is_pay_as_bid and settings.time_step != timedelta(hours=1):
        raise InputValidationError("第一版和第二版仅支持 time_step: 1h。")
    if not is_pay_as_bid and settings.product_duration != timedelta(hours=1):
        raise InputValidationError("第二版仅支持 1 小时的 EOM 产品。")
    if is_pay_as_bid and settings.product_duration != settings.time_step:
        raise InputValidationError(
            "pay_as_bid 要求产品时长等于场景的 time_step。"
        )
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
    if settings.learning_config is not None and settings.learning_config.periods:
        _validate_learning_period_alignment(settings)

    return settings
