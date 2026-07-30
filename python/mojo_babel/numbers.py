"""Babel-compatible number formatting with a Mojo digit renderer."""

from __future__ import annotations

import decimal
import re
from collections.abc import Iterable

import numpy as np
from babel import numbers as _babel
from babel.core import Locale
from babel.numbers import NumberPattern

from ._lib import addr, bytes_array, lib

LC_NUMERIC = _babel.LC_NUMERIC
LC_MONETARY = _babel.LC_MONETARY
UnknownCurrencyFormatError = _babel.UnknownCurrencyFormatError
UnsupportedNumberingSystemError = _babel.UnsupportedNumberingSystemError
_INT64_MAX = np.iinfo(np.int64).max


def _locale(value, category: str) -> Locale:
    default = LC_MONETARY if category == "money" else LC_NUMERIC
    return Locale.parse(value or default)


def _pattern(value) -> NumberPattern:
    return _babel.parse_pattern(value)


def _core(
    scaled_values: np.ndarray,
    decimals: int,
    min_frac: int,
    min_int: int,
    grouping: tuple[int, int],
    group_separator: bool,
    locale: Locale,
    numbering_system: str,
) -> list[str]:
    if not len(scaled_values):
        return []
    group = bytes_array(
        _babel.get_group_symbol(locale, numbering_system=numbering_system)
    )
    decimal_symbol = bytes_array(
        _babel.get_decimal_symbol(locale, numbering_system=numbering_system)
    )
    values = np.ascontiguousarray(scaled_values, dtype=np.int64)
    max_integer = int(values.max()) // (10**decimals)
    integer_width = max(len(str(max_integer)), min_int)
    group_slots = 0
    primary, secondary = grouping
    if group_separator and primary > 0 and integer_width > primary:
        group_slots = 1
        if secondary > 0:
            group_slots += (integer_width - primary - 1) // secondary
    stride = (
        integer_width
        + group_slots * len(group)
        + (len(decimal_symbol) + decimals if decimals else 0)
    )
    output = np.empty((len(values), stride), dtype=np.uint8)
    lengths = np.empty(len(values), dtype=np.int64)
    status = lib().mb_format_scaled(
        addr(values),
        len(values),
        decimals,
        min_frac,
        min_int,
        grouping[0],
        grouping[1],
        int(group_separator),
        addr(group),
        len(group),
        addr(decimal_symbol),
        len(decimal_symbol),
        addr(output),
        stride,
        addr(lengths),
    )
    if status == -1:
        raise OverflowError("formatted number exceeds the Mojo output stride")
    if status:
        raise RuntimeError("Mojo rejected an invalid number buffer or configuration")
    return [
        output[i, : lengths[i]].tobytes().decode("utf-8")
        for i in range(len(values))
    ]


def _decimal_value(number) -> decimal.Decimal:
    if isinstance(number, decimal.Decimal):
        return number
    return decimal.Decimal(str(number))


def _replace_affixes(
    text: str,
    currency: str | None,
    value: decimal.Decimal,
    locale: Locale,
) -> str:
    if currency is not None and "¤" in text:
        if "¤¤¤" in text:
            text = text.replace("¤¤¤", _babel.get_currency_name(currency, value, locale))
        elif "¤¤" in text:
            text = text.replace("¤¤", currency.upper())
        else:
            text = text.replace("¤", _babel.get_currency_symbol(currency, locale))
    if "'" in text:
        text = re.sub(r"'([^']*)'", lambda match: match.group(1) or "'", text)
    return text


def _render_many(
    numbers: Iterable,
    pattern: NumberPattern,
    locale: Locale,
    *,
    currency: str | None = None,
    currency_digits: bool = True,
    decimal_quantization: bool = True,
    group_separator: bool = True,
    numbering_system: str = "latn",
) -> list[str]:
    raw = list(numbers)
    if (
        pattern.exp_prec
        or "@" in pattern.pattern
        or pattern.number_pattern == ""
        or pattern.int_prec[0] > 18
    ):
        return [
            pattern.apply(
                value,
                locale,
                currency=currency,
                currency_digits=currency_digits,
                decimal_quantization=decimal_quantization,
                group_separator=group_separator,
                numbering_system=numbering_system,
            )
            for value in raw
        ]

    dynamic_currency_name = currency is not None and any(
        "¤¤¤" in affix for affix in pattern.prefix + pattern.suffix
    )
    if currency and currency_digits:
        fraction = (_babel.get_currency_precision(currency),) * 2
    else:
        fraction = pattern.frac_prec

    prepared: list[tuple[decimal.Decimal, int]] = []
    scaled_values = np.empty(len(raw), dtype=np.int64)
    fallback = False
    base_max_frac = fraction[1]
    base_quantum = _babel.get_decimal_quantum(base_max_frac)
    for index, number in enumerate(raw):
        value = _decimal_value(number)
        if pattern.scale:
            value = value.scaleb(pattern.scale)
        if not value.is_finite():
            fallback = True
            break
        negative = int(value.is_signed())
        value = abs(value)
        if not decimal_quantization or dynamic_currency_name:
            value = value.normalize()
        value_max_frac = base_max_frac
        if not decimal_quantization:
            value_max_frac = max(value_max_frac, _babel.get_decimal_precision(value))
        if value_max_frac > 18:
            fallback = True
            break
        quantum = (
            base_quantum
            if value_max_frac == base_max_frac
            else _babel.get_decimal_quantum(value_max_frac)
        )
        rounded = value.quantize(quantum)
        scaled = int(rounded.scaleb(value_max_frac))
        if scaled > _INT64_MAX:
            fallback = True
            break
        prepared.append((value, negative))
        scaled_values[index] = scaled

    if fallback:
        return [
            pattern.apply(
                value,
                locale,
                currency=currency,
                currency_digits=currency_digits,
                decimal_quantization=decimal_quantization,
                group_separator=group_separator,
                numbering_system=numbering_system,
            )
            for value in raw
        ]

    max_frac = fraction[1]
    if not decimal_quantization and prepared:
        max_frac = max(
            max_frac,
            max(_babel.get_decimal_precision(value) for value, _ in prepared),
        )
        quantum = _babel.get_decimal_quantum(max_frac)
        rescales: list[int] = []
        for index, (value, _) in enumerate(prepared):
            scaled = int(value.quantize(quantum).scaleb(max_frac))
            if scaled > _INT64_MAX:
                return [
                    pattern.apply(
                        item,
                        locale,
                        currency=currency,
                        currency_digits=currency_digits,
                        decimal_quantization=decimal_quantization,
                        group_separator=group_separator,
                        numbering_system=numbering_system,
                    )
                    for item in raw
                ]
            rescales.append(scaled)
        scaled_values[:] = rescales

    cores = _core(
        scaled_values,
        max_frac,
        fraction[0],
        pattern.int_prec[0],
        pattern.grouping,
        group_separator,
        locale,
        numbering_system,
    )
    if dynamic_currency_name:
        prefixes = pattern.prefix
        suffixes = pattern.suffix
    else:
        zero = decimal.Decimal(0)
        prefixes = tuple(
            _replace_affixes(affix, currency, zero, locale)
            for affix in pattern.prefix
        )
        suffixes = tuple(
            _replace_affixes(affix, currency, zero, locale)
            for affix in pattern.suffix
        )
    rendered = []
    for (value, negative), core in zip(prepared, cores):
        text = prefixes[negative] + core + suffixes[negative]
        if dynamic_currency_name:
            text = _replace_affixes(text, currency, value, locale)
        rendered.append(text)
    return rendered


def format_decimal(
    number,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> str:
    locale = _locale(locale, "number")
    pattern = _pattern(locale.decimal_formats[None] if format is None else format)
    return _render_many(
        [number],
        pattern,
        locale,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )[0]


def format_decimal_batch(
    numbers: Iterable,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> list[str]:
    locale = _locale(locale, "number")
    pattern = _pattern(locale.decimal_formats[None] if format is None else format)
    return _render_many(
        numbers,
        pattern,
        locale,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )


def format_currency(
    number,
    currency: str,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    currency_digits: bool = True,
    format_type: str = "standard",
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> str:
    locale = _locale(locale, "money")
    if format_type == "name":
        return _babel.format_currency(
            number,
            currency,
            format,
            locale,
            currency_digits,
            format_type,
            decimal_quantization,
            group_separator,
            numbering_system=numbering_system,
        )
    if format is None:
        try:
            format = locale.currency_formats[format_type]
        except KeyError:
            raise UnknownCurrencyFormatError(
                f"{format_type!r} is not a known currency format type"
            ) from None
    pattern = _pattern(format)
    return _render_many(
        [number],
        pattern,
        locale,
        currency=currency,
        currency_digits=currency_digits,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )[0]


def format_currency_batch(
    numbers: Iterable,
    currency: str,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    currency_digits: bool = True,
    format_type: str = "standard",
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> list[str]:
    locale = _locale(locale, "money")
    if format_type == "name":
        return [
            format_currency(
                value,
                currency,
                format,
                locale,
                currency_digits,
                format_type,
                decimal_quantization,
                group_separator,
                numbering_system=numbering_system,
            )
            for value in numbers
        ]
    if format is None:
        try:
            format = locale.currency_formats[format_type]
        except KeyError:
            raise UnknownCurrencyFormatError(
                f"{format_type!r} is not a known currency format type"
            ) from None
    return _render_many(
        numbers,
        _pattern(format),
        locale,
        currency=currency,
        currency_digits=currency_digits,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )


def format_percent(
    number,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> str:
    locale = _locale(locale, "number")
    pattern = _pattern(locale.percent_formats[None] if not format else format)
    return _render_many(
        [number],
        pattern,
        locale,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )[0]


def format_percent_batch(
    numbers: Iterable,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    decimal_quantization: bool = True,
    group_separator: bool = True,
    *,
    numbering_system: str = "latn",
) -> list[str]:
    locale = _locale(locale, "number")
    pattern = _pattern(locale.percent_formats[None] if not format else format)
    return _render_many(
        numbers,
        pattern,
        locale,
        decimal_quantization=decimal_quantization,
        group_separator=group_separator,
        numbering_system=numbering_system,
    )


def format_scientific(
    number,
    format: str | NumberPattern | None = None,
    locale: Locale | str | None = None,
    decimal_quantization: bool = True,
    *,
    numbering_system: str = "latn",
) -> str:
    return _babel.format_scientific(
        number,
        format,
        locale,
        decimal_quantization,
        numbering_system=numbering_system,
    )


format_compact_decimal = _babel.format_compact_decimal
format_compact_currency = _babel.format_compact_currency
parse_pattern = _babel.parse_pattern
get_currency_name = _babel.get_currency_name
get_currency_symbol = _babel.get_currency_symbol
get_currency_precision = _babel.get_currency_precision
get_decimal_symbol = _babel.get_decimal_symbol
get_group_symbol = _babel.get_group_symbol
get_plus_sign_symbol = _babel.get_plus_sign_symbol
get_minus_sign_symbol = _babel.get_minus_sign_symbol
