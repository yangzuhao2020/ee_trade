"""Shared timestamp formatting and delivery-duration conversion."""

from datetime import datetime


def format_timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="minutes")


def hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600
