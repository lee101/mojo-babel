import decimal

import pytest
from babel import numbers as reference

from mojo_babel import numbers


LOCALES = ["en_US", "de_DE", "fr_FR", "hi_IN", "ar_EG", "ja_JP", "ru_RU"]
VALUES = [0, -0.0, 1.2, -1234.567, 999999.995, decimal.Decimal("2.3455")]


@pytest.mark.parametrize("locale", LOCALES)
def test_decimal_default_parity(locale):
    for value in VALUES:
        assert numbers.format_decimal(value, locale=locale) == reference.format_decimal(
            value, locale=locale
        )


@pytest.mark.parametrize("locale", LOCALES)
def test_currency_default_parity(locale):
    for currency in ("USD", "EUR", "JPY"):
        for value in VALUES:
            assert numbers.format_currency(
                value, currency, locale=locale
            ) == reference.format_currency(value, currency, locale=locale)


@pytest.mark.parametrize("locale", LOCALES)
def test_percent_default_parity(locale):
    for value in VALUES:
        assert numbers.format_percent(value, locale=locale) == reference.format_percent(
            value, locale=locale
        )


@pytest.mark.parametrize("locale", ["en_US", "de_DE", "ar_EG"])
def test_numbering_system_parity(locale):
    for system in ("latn", "default"):
        assert numbers.format_decimal(
            12345.678, locale=locale, numbering_system=system
        ) == reference.format_decimal(
            12345.678, locale=locale, numbering_system=system
        )


def test_custom_decimal_patterns():
    patterns = ["00000.0000", "#,##0.00;(#,##0.00)", "'about' #,##0.0"]
    for pattern in patterns:
        for value in (12.3, -1234.56):
            assert numbers.format_decimal(
                value, pattern, "en_US"
            ) == reference.format_decimal(value, pattern, "en_US")


def test_quantization_and_group_switches():
    for quantize in (True, False):
        for grouping in (True, False):
            assert numbers.format_decimal(
                decimal.Decimal("12345.678901"),
                locale="fr_FR",
                decimal_quantization=quantize,
                group_separator=grouping,
            ) == reference.format_decimal(
                decimal.Decimal("12345.678901"),
                locale="fr_FR",
                decimal_quantization=quantize,
                group_separator=grouping,
            )


def test_accounting_and_currency_digits():
    for currency_digits in (True, False):
        assert numbers.format_currency(
            -1234.567,
            "JPY",
            locale="en_US",
            format_type="accounting",
            currency_digits=currency_digits,
        ) == reference.format_currency(
            -1234.567,
            "JPY",
            locale="en_US",
            format_type="accounting",
            currency_digits=currency_digits,
        )


@pytest.mark.parametrize("pattern", ["¤ #,##0.00", "¤¤ #,##0.00", "¤¤¤ #,##0.00"])
def test_currency_affix_variants(pattern):
    values = [1, 2, -12.5]
    expected = [
        reference.format_currency(value, "USD", pattern, locale="en_US")
        for value in values
    ]
    assert numbers.format_currency_batch(
        values, "USD", pattern, locale="en_US"
    ) == expected


def test_wide_integer_pattern_fallback():
    pattern = "0" * 180
    assert numbers.format_decimal(12, pattern, "en_US") == reference.format_decimal(
        12, pattern, "en_US"
    )


def test_special_values():
    for value in ("NaN", "Infinity", "-Infinity"):
        assert numbers.format_decimal(
            decimal.Decimal(value), locale="de_DE"
        ) == reference.format_decimal(decimal.Decimal(value), locale="de_DE")


def test_scientific_and_compact_compatibility():
    assert numbers.format_scientific(
        12345.67, "##0.##E00", "de_DE"
    ) == reference.format_scientific(12345.67, "##0.##E00", "de_DE")
    assert numbers.format_compact_decimal(
        1234567, locale="ja_JP"
    ) == reference.format_compact_decimal(1234567, locale="ja_JP")
    assert numbers.format_compact_currency(
        12345, "USD", locale="en_US"
    ) == reference.format_compact_currency(12345, "USD", locale="en_US")


@pytest.mark.parametrize("locale", LOCALES)
def test_decimal_batch_parity(locale):
    expected = [reference.format_decimal(v, locale=locale) for v in VALUES]
    assert numbers.format_decimal_batch(VALUES, locale=locale) == expected


def test_decimal_batch_simd_tail():
    values = [decimal.Decimal(index).scaleb(-2) for index in range(-16, 17)]
    expected = [reference.format_decimal(value, locale="en_US") for value in values]
    assert numbers.format_decimal_batch(values, locale="en_US") == expected


@pytest.mark.parametrize("locale", ["en_US", "fr_FR", "hi_IN", "ar_EG"])
def test_currency_batch_parity(locale):
    expected = [reference.format_currency(v, "EUR", locale=locale) for v in VALUES]
    assert numbers.format_currency_batch(
        VALUES, "EUR", locale=locale
    ) == expected


def test_percent_batch_parity():
    expected = [reference.format_percent(v, locale="ru_RU") for v in VALUES]
    assert numbers.format_percent_batch(VALUES, locale="ru_RU") == expected


def test_invalid_numbering_system_matches_exception():
    with pytest.raises(reference.UnsupportedNumberingSystemError):
        numbers.format_decimal(1, locale="en_US", numbering_system="nope")


def test_nonquantized_mixed_precision_avoids_int64_narrowing():
    values = [decimal.Decimal("9000000000000000"), decimal.Decimal("0.000001")]
    expected = [
        reference.format_decimal(value, locale="en_US", decimal_quantization=False)
        for value in values
    ]
    assert numbers.format_decimal_batch(
        values, locale="en_US", decimal_quantization=False
    ) == expected
