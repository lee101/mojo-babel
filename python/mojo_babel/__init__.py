"""Locale-aware number and date formatting accelerated by Mojo."""

from . import dates, numbers
from ._lib import build
from .dates import (
    format_date,
    format_date_batch,
    format_datetime,
    format_datetime_batch,
    format_interval,
    format_skeleton,
    format_time,
    format_time_batch,
    format_timedelta,
)
from .numbers import (
    format_compact_currency,
    format_compact_decimal,
    format_currency,
    format_currency_batch,
    format_decimal,
    format_decimal_batch,
    format_percent,
    format_percent_batch,
    format_scientific,
)

__version__ = "0.1.0"
__all__ = [
    "format_decimal",
    "format_currency",
    "format_percent",
    "format_scientific",
    "format_compact_decimal",
    "format_compact_currency",
    "format_date",
    "format_time",
    "format_datetime",
    "format_timedelta",
    "format_interval",
    "format_skeleton",
    "format_decimal_batch",
    "format_currency_batch",
    "format_percent_batch",
    "format_date_batch",
    "format_time_batch",
    "format_datetime_batch",
    "numbers",
    "dates",
    "build",
]
