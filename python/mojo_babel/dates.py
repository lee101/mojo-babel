"""Babel-compatible date formatting backed by a Mojo pattern interpreter."""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from babel import dates as _babel
from babel.core import Locale

from ._lib import addr, lib

LC_TIME = _babel.LC_TIME


class _UnsupportedPattern(ValueError):
    pass


@dataclass(frozen=True)
class _Program:
    ops: np.ndarray
    names: np.ndarray
    offsets: np.ndarray
    literals: np.ndarray


def _locale(value) -> Locale:
    return Locale.parse(value or LC_TIME)


@lru_cache(maxsize=256)
def _compile(pattern: str, locale_id: str) -> _Program:
    locale = Locale.parse(locale_id)
    names: list[str] = []
    operations: list[tuple[int, int, int]] = []
    literal_data = bytearray()

    def name_group(values: Iterable[str]) -> int:
        base = len(names)
        names.extend(values)
        return base

    def month_group(width: str, context: str) -> int:
        values = _babel.get_month_names(width, context, locale)
        return name_group(values[i] for i in range(1, 13))

    def day_group(width: str, context: str) -> int:
        values = _babel.get_day_names(width, context, locale)
        return name_group(values[i] for i in range(7))

    def period_group(width: str) -> int:
        widths = [width, "wide", "narrow", "abbreviated"]
        values = None
        for candidate in widths:
            periods = _babel.get_period_names(candidate, "format", locale)
            if "am" in periods and "pm" in periods:
                values = (periods["am"], periods["pm"])
                break
        if values is None:
            raise _UnsupportedPattern("locale has no AM/PM names")
        return name_group(values)

    for token_type, token in _babel.tokenize_pattern(pattern):
        if token_type == "chars":
            encoded = token.encode("utf-8")
            start = len(literal_data)
            literal_data.extend(encoded)
            operations.append((0, len(encoded), start))
            continue

        char, width = token
        if width > 18 and char in ("y", "u", "M", "L", "d", "H", "h", "K", "k", "m", "s"):
            raise _UnsupportedPattern(f"numeric field width {width} exceeds int64")
        if char in ("y", "u"):
            operations.append((1, width, 0))
        elif char in ("M", "L"):
            if width <= 2:
                operations.append((2, width, 0))
            elif width <= 5:
                name_width = {3: "abbreviated", 4: "wide", 5: "narrow"}[width]
                context = "format" if char == "M" else "stand-alone"
                operations.append((3, width, month_group(name_width, context)))
            else:
                raise _UnsupportedPattern(f"unsupported month width {width}")
        elif char == "d":
            operations.append((4, width, 0))
        elif char in ("E", "e", "c"):
            if char in ("e", "c") and width < 3:
                raise _UnsupportedPattern("numeric local weekday")
            effective = max(3, width)
            if effective not in (3, 4, 5, 6):
                raise _UnsupportedPattern(f"unsupported weekday width {width}")
            name_width = {
                3: "abbreviated",
                4: "wide",
                5: "narrow",
                6: "short",
            }[effective]
            context = "stand-alone" if char == "c" else "format"
            operations.append((5, width, day_group(name_width, context)))
        elif char == "H":
            operations.append((6, width, 0))
        elif char == "h":
            operations.append((7, width, 0))
        elif char == "K":
            operations.append((8, width, 0))
        elif char == "k":
            operations.append((9, width, 0))
        elif char == "m":
            operations.append((10, width, 0))
        elif char == "s":
            operations.append((11, width, 0))
        elif char == "S":
            if width != 6:
                # Babel rounds through a binary Python float. Preserve exact
                # compatibility instead of approximating that at the ABI.
                raise _UnsupportedPattern("fractional seconds require Babel rounding")
            operations.append((12, width, 0))
        elif char == "a":
            if width > 5:
                raise _UnsupportedPattern(f"unsupported period width {width}")
            name_width = {3: "abbreviated", 4: "wide", 5: "narrow"}[
                max(3, width)
            ]
            operations.append((13, width, period_group(name_width)))
        elif char == "G":
            if width > 5:
                raise _UnsupportedPattern(f"unsupported era width {width}")
            name_width = {3: "abbreviated", 4: "wide", 5: "narrow"}[
                max(3, width)
            ]
            eras = _babel.get_era_names(name_width, locale)
            operations.append((14, width, name_group((eras[0], eras[1]))))
        else:
            raise _UnsupportedPattern(f"unsupported CLDR field {char!r}")

    name_bytes = bytearray()
    offsets = []
    for value in names:
        start = len(name_bytes)
        name_bytes.extend(value.encode("utf-8"))
        offsets.append((start, len(name_bytes)))
    if not name_bytes:
        name_bytes.append(0)
    if not offsets:
        offsets.append((0, 0))
    if not literal_data:
        literal_data.append(0)

    return _Program(
        np.ascontiguousarray(operations, dtype=np.int64),
        np.frombuffer(bytes(name_bytes), dtype=np.uint8),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.frombuffer(bytes(literal_data), dtype=np.uint8),
    )


def _render(fields: np.ndarray, pattern: str, locale: Locale) -> list[str]:
    fields = np.ascontiguousarray(fields, dtype=np.int64).reshape(-1, 8)
    if not len(fields):
        return []
    program = _compile(pattern, str(locale))
    stride = 256
    output = np.empty((len(fields), stride), dtype=np.uint8)
    lengths = np.empty(len(fields), dtype=np.int64)
    status = lib().mb_format_dates(
        addr(fields),
        len(fields),
        addr(program.ops),
        len(program.ops),
        addr(program.names),
        len(program.names),
        addr(program.offsets),
        len(program.offsets),
        addr(program.literals),
        len(program.literals),
        addr(output),
        stride,
        addr(lengths),
    )
    if status == -1:
        raise OverflowError("formatted date exceeds the Mojo output stride")
    if status:
        raise RuntimeError("Mojo rejected an invalid date buffer or program")
    return [
        output[i, : lengths[i]].tobytes().decode("utf-8")
        for i in range(len(fields))
    ]


def _date_fields(values: Iterable[_datetime.date]) -> np.ndarray:
    rows = []
    for value in values:
        if isinstance(value, _datetime.datetime):
            value = value.date()
        if not isinstance(value, _datetime.date):
            raise TypeError(f"expected date or datetime, got {type(value).__name__}")
        rows.append(
            (value.year, value.month, value.day, value.weekday(), 0, 0, 0, 0)
        )
    return np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 8)


def _time_fields(values: Iterable[_datetime.time | _datetime.datetime]) -> np.ndarray:
    rows = []
    for value in values:
        if isinstance(value, _datetime.datetime):
            pass
        elif isinstance(value, _datetime.time):
            pass
        else:
            raise TypeError(f"expected time or datetime, got {type(value).__name__}")
        rows.append((1, 1, 1, 0, value.hour, value.minute, value.second, value.microsecond))
    return np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 8)


def _datetime_fields(values: Iterable[_datetime.datetime]) -> np.ndarray:
    rows = []
    for value in values:
        if not isinstance(value, _datetime.datetime):
            raise TypeError(f"expected datetime, got {type(value).__name__}")
        rows.append(
            (
                value.year,
                value.month,
                value.day,
                value.weekday(),
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
            )
        )
    return np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 8)


def _date_pattern(format, locale: Locale) -> str:
    if format in ("full", "long", "medium", "short"):
        return _babel.get_date_format(format, locale).pattern
    return str(_babel.parse_pattern(format))


def _time_pattern(format, locale: Locale) -> str:
    if format in ("full", "long", "medium", "short"):
        return _babel.get_time_format(format, locale).pattern
    return str(_babel.parse_pattern(format))


def format_date(
    date: _datetime.date | None = None,
    format: str = "medium",
    locale: Locale | str | None = None,
) -> str:
    if date is None:
        date = _datetime.date.today()
    elif isinstance(date, _datetime.datetime):
        date = date.date()
    locale = _locale(locale)
    pattern = _date_pattern(format, locale)
    try:
        return _render(_date_fields([date]), pattern, locale)[0]
    except _UnsupportedPattern:
        return _babel.format_date(date, format, locale)


def format_date_batch(
    dates: Iterable,
    format: str = "medium",
    locale: Locale | str | None = None,
) -> list[str]:
    locale = _locale(locale)
    pattern = _date_pattern(format, locale)
    if isinstance(dates, np.ndarray) and np.issubdtype(dates.dtype, np.datetime64):
        flat = dates.astype("datetime64[D]").astype(np.int64).ravel()
        if not len(flat):
            return []
        if np.any(flat == np.iinfo(np.int64).min):
            raise ValueError("NaT cannot be formatted")
        days = np.ascontiguousarray(flat, dtype=np.int64)
        fields = np.empty((len(days), 8), dtype=np.int64)
        status = lib().mb_decompose_days(addr(days), len(days), addr(fields))
        if status:
            raise ValueError("invalid buffer passed to Mojo date decomposition")
    else:
        values = list(dates)
        fields = _date_fields(values)
    try:
        return _render(fields, pattern, locale)
    except _UnsupportedPattern:
        if isinstance(dates, np.ndarray):
            values = [
                _datetime.date.fromisoformat(str(value.astype("datetime64[D]")))
                for value in dates.ravel()
            ]
        return [_babel.format_date(value, format, locale) for value in values]


def format_time(
    time: _datetime.time | _datetime.datetime | float | None = None,
    format: str = "medium",
    tzinfo: _datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> str:
    if tzinfo is not None:
        return _babel.format_time(time, format, tzinfo, locale)
    value = _babel._get_time(time, None)
    locale = _locale(locale)
    pattern = _time_pattern(format, locale)
    try:
        return _render(_time_fields([value]), pattern, locale)[0]
    except _UnsupportedPattern:
        return _babel.format_time(value, format, None, locale)


def format_time_batch(
    times: Iterable[_datetime.time | _datetime.datetime],
    format: str = "medium",
    tzinfo: _datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> list[str]:
    values = list(times)
    if tzinfo is not None:
        return [_babel.format_time(value, format, tzinfo, locale) for value in values]
    locale = _locale(locale)
    pattern = _time_pattern(format, locale)
    try:
        return _render(_time_fields(values), pattern, locale)
    except _UnsupportedPattern:
        return [_babel.format_time(value, format, None, locale) for value in values]


def format_datetime(
    datetime=None,
    format: str = "medium",
    tzinfo: _datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> str:
    value = _babel._ensure_datetime_tzinfo(_babel._get_datetime(datetime), tzinfo)
    locale = _locale(locale)
    if format in ("full", "long", "medium", "short"):
        template = (
            _babel.get_datetime_format(format, locale)
            .replace("'", "")
        )
        return (
            template.replace("{0}", format_time(value, format, None, locale))
            .replace("{1}", format_date(value, format, locale))
        )
    pattern = str(_babel.parse_pattern(format))
    try:
        return _render(_datetime_fields([value]), pattern, locale)[0]
    except _UnsupportedPattern:
        return _babel.format_datetime(value, format, None, locale)


def format_datetime_batch(
    datetimes: Iterable[_datetime.datetime],
    format: str = "medium",
    tzinfo: _datetime.tzinfo | None = None,
    locale: Locale | str | None = None,
) -> list[str]:
    values = list(datetimes)
    if tzinfo is not None:
        return [
            _babel.format_datetime(value, format, tzinfo, locale) for value in values
        ]
    locale = _locale(locale)
    if format in ("full", "long", "medium", "short"):
        template = _babel.get_datetime_format(format, locale).replace("'", "")
        dates = format_date_batch(values, format, locale)
        times = format_time_batch(values, format, None, locale)
        return [
            template.replace("{0}", time).replace("{1}", date)
            for date, time in zip(dates, times)
        ]
    pattern = str(_babel.parse_pattern(format))
    try:
        return _render(_datetime_fields(values), pattern, locale)
    except _UnsupportedPattern:
        return [_babel.format_datetime(value, format, None, locale) for value in values]


format_timedelta = _babel.format_timedelta
format_interval = _babel.format_interval
format_skeleton = _babel.format_skeleton
get_date_format = _babel.get_date_format
get_time_format = _babel.get_time_format
get_datetime_format = _babel.get_datetime_format
get_month_names = _babel.get_month_names
get_day_names = _babel.get_day_names
get_timezone = _babel.get_timezone
