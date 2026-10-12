"""共享工具：时间格式化、交付时长计算及参与者与燃料价格校验。"""

from __future__ import annotations
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING

from .errors import InputValidationError

if TYPE_CHECKING:
    from .models import PowerPlant


def format_timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


def hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600


def validate_fuel_coverage(plants: tuple[PowerPlant, ...], fuel_prices: dict[str, float]) -> None:
    """检查各机组所需的燃料价格是否齐全，可再生能源机组除外。
    参数：
        plants：待检查的发电机组，燃料类型为 renewable 的机组无需提供燃料价格。
        fuel_prices：燃料类型到价格的映射，键应覆盖其余机组使用的全部燃料类型。
    异常：
        InputValidationError：存在未提供价格的燃料类型时抛出，并列出缺失项。
    """
    # 对所需燃料类型去重，减去已有价格的燃料类型，再排序得到缺失项。
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


def validate_unique_names(names: Iterable[str], scope: str) -> None:
    """检查指定范围内的参与者名称是否唯一，无重名时正常返回。
    参数：
        names：待检查的参与者名称序列或其他可迭代对象。
        scope：检查范围的说明，用于错误消息，例如所有市场单位。
    异常：
        InputValidationError：存在重名时抛出，并按名称排序列出重复项。
    """
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise InputValidationError(
            f"Participant names must be unique across {scope}: "
            + ", ".join(duplicates)
            + "."
        )
