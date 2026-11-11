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


def test_percent_vector_preparation_half_even_and_tail():
    values = [float(index) / 200 for index in range(-517, 520)]
    expected = [reference.format_percent(value, locale="de_DE") for value in values]
    assert numbers.format_percent_batch(values, locale="de_DE") == expected


@pytest.mark.parametrize("size", [255, 256])
def test_percent_vector_preparation_threshold(size):
    values = [12.345] * size
    assert numbers.format_percent_batch(values, locale="en_US") == ["1,234%"] * size


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


# Batch results are compared with upstream's scalar API on the same inputs.
# Both sides render locale text, so parity means string identity: there is no
# numeric tolerance to apply to a rendered string. Inputs that pass through a
# binary float are compared exactly because Babel itself rounds them via
# Decimal(str(value)) with ROUND_HALF_EVEN.
def _assert_strings_match_many(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got == want


def _outcome(call):
    """Render, or capture the exception class upstream would raise."""
    try:
        return ("value", call())
    except Exception as error:  # noqa: BLE001 - the class is the assertion
        return ("error", type(error).__name__)


@pytest.mark.parametrize("locale", LOCALES)
def test_empty_batches_return_empty_list(locale):
    assert numbers.format_decimal_batch([], locale=locale) == []
    assert numbers.format_currency_batch([], "USD", locale=locale) == []
    assert numbers.format_percent_batch([], locale=locale) == []


@pytest.mark.parametrize("locale", LOCALES)
def test_single_element_batches_match_upstream(locale):
    _assert_strings_match_many(
        numbers.format_decimal_batch([1.2], locale=locale),
        [reference.format_decimal(1.2, locale=locale)],
    )
    _assert_strings_match_many(
        numbers.format_currency_batch([-1234.567], "EUR", locale=locale),
        [reference.format_currency(-1234.567, "EUR", locale=locale)],
    )
    _assert_strings_match_many(
        numbers.format_percent_batch([-0.0], locale=locale),
        [reference.format_percent(-0.0, locale=locale)],
    )


@pytest.mark.parametrize("locale", ["en_US", "hi_IN", "ar_EG"])
def test_duplicate_inputs_match_upstream(locale):
    values = [999999.995] * 9
    _assert_strings_match_many(
        numbers.format_decimal_batch(values, locale=locale),
        [reference.format_decimal(value, locale=locale) for value in values],
    )


@pytest.mark.parametrize("special", ["NaN", "sNaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("locale", ["en_US", "de_DE", "ar_EG"])
def test_special_values_scalar_and_batch_parity(special, locale):
    value = decimal.Decimal(special)
    # Babel raises InvalidOperation for signalling NaN, so compare outcomes.
    assert _outcome(
        lambda: numbers.format_decimal_batch([value], locale=locale)
    ) == _outcome(lambda: [reference.format_decimal(value, locale=locale)])
    assert _outcome(
        lambda: numbers.format_percent_batch([value], locale=locale)
    ) == _outcome(lambda: [reference.format_percent(value, locale=locale)])
    assert _outcome(
        lambda: numbers.format_currency_batch([value, value], "USD", locale=locale)
    ) == _outcome(lambda: [reference.format_currency(value, "USD", locale=locale)] * 2)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        decimal.Decimal("-0"),
        decimal.Decimal("0E-20"),
        decimal.Decimal("1E+400"),
        decimal.Decimal("9" * 40),
        2**63,
        -(2**63),
        1e-300,
        1e300,
    ],
)
def test_extreme_and_odd_types_match_upstream(value):
    # bools and out-of-range magnitudes raise upstream; parity covers both
    # the rendered value and the exception class.
    for locale in ("en_US", "fr_FR"):
        assert _outcome(
            lambda: numbers.format_decimal_batch([value], locale=locale)
        ) == _outcome(lambda: [reference.format_decimal(value, locale=locale)])
        assert _outcome(
            lambda: numbers.format_currency_batch([value], "EUR", locale=locale)
        ) == _outcome(lambda: [reference.format_currency(value, "EUR", locale=locale)])


def test_pyfloats_near_half_even_boundaries_match_upstream():
    # The vectorised float lane is only taken at >= 256 values; these inputs sit
    # within a few ULPs of a half-even boundary and must agree with Babel.
    values = [0.125] * 130 + [0.375] * 130 + [2.675] * 130
    _assert_strings_match_many(
        numbers.format_decimal_batch(values, locale="en_US"),
        [reference.format_decimal(value, locale="en_US") for value in values],
    )
    _assert_strings_match_many(
        numbers.format_percent_batch(values, locale="de_DE"),
        [reference.format_percent(value, locale="de_DE") for value in values],
    )


@pytest.mark.parametrize("locale", LOCALES)
def test_currency_name_format_parity(locale):
    # format_type="name" defers to Babel; exercise it per locale, where the
    # currency name is also inflected by the value.
    values = (1, -1234.5, decimal.Decimal("0"))
    _assert_strings_match_many(
        [numbers.format_currency(v, "USD", format_type="name", locale=locale) for v in values],
        [reference.format_currency(v, "USD", format_type="name", locale=locale) for v in values],
    )
    _assert_strings_match_many(
        numbers.format_currency_batch(
            [1, -1234.5], "USD", format_type="name", locale=locale
        ),
        [
            reference.format_currency(value, "USD", format_type="name", locale=locale)
            for value in (1, -1234.5)
        ],
    )


def test_unknown_currency_format_raises_upstream_error():
    with pytest.raises(reference.UnknownCurrencyFormatError):
        numbers.format_currency(
            1, "USD", format_type="bogus", locale="en_US"
        )


def test_batch_invalid_numbering_system_matches_exception():
    with pytest.raises(reference.UnsupportedNumberingSystemError):
        numbers.format_decimal_batch([1], locale="en_US", numbering_system="nope")


@pytest.mark.parametrize("pattern", ["#,##0.00", "00000.0000", "'#'0", "#,##0;(#,##0)"])
def test_numberpattern_instances_match_upstream(pattern):
    ours = numbers.parse_pattern(pattern)
    theirs = reference.parse_pattern(pattern)
    assert numbers.format_decimal(12.3456, ours, "en_US") == reference.format_decimal(
        12.3456, theirs, "en_US"
    )
