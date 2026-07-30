import datetime as dt

import numpy as np
import pytest
from babel import dates as reference

from mojo_babel import dates


LOCALES = ["en_US", "de_DE", "fr_FR", "hi_IN", "ar_EG", "ja_JP", "ru_RU"]
WIDTHS = ["short", "medium", "long", "full"]
DATE = dt.date(2007, 4, 1)
TIME = dt.time(15, 30, 45, 987654)
DATETIME = dt.datetime(2007, 4, 1, 15, 30, 45, 987654)


@pytest.mark.parametrize("locale", LOCALES)
def test_date_widths(locale):
    for width in WIDTHS:
        assert dates.format_date(DATE, width, locale) == reference.format_date(
            DATE, width, locale
        )


@pytest.mark.parametrize("locale", LOCALES)
def test_time_widths(locale):
    for width in WIDTHS:
        assert dates.format_time(TIME, width, locale=locale) == reference.format_time(
            TIME, width, locale=locale
        )


@pytest.mark.parametrize("locale", LOCALES)
def test_datetime_widths(locale):
    for width in WIDTHS:
        assert dates.format_datetime(
            DATETIME, width, locale=locale
        ) == reference.format_datetime(DATETIME, width, locale=locale)


@pytest.mark.parametrize(
    "pattern",
    [
        "EEE, MMM d, ''yy",
        "yyyy.MM.dd G 'at' HH:mm:ss",
        "hh 'o''clock' a",
        "yyyy-MM-dd HH:mm:ss.S",
        "yyyy-MM-dd HH:mm:ss.SSS",
        "yyyy-MM-dd HH:mm:ss.SSSSSS",
    ],
)
def test_custom_patterns(pattern):
    assert dates.format_datetime(
        DATETIME, pattern, locale="en_US"
    ) == reference.format_datetime(DATETIME, pattern, locale="en_US")


def test_timezone_fallback_parity():
    zone = reference.get_timezone("Europe/Paris")
    assert dates.format_datetime(
        DATETIME, "full", zone, "fr_FR"
    ) == reference.format_datetime(DATETIME, "full", zone, "fr_FR")
    assert dates.format_time(
        TIME, "full", zone, "fr_FR"
    ) == reference.format_time(TIME, "full", zone, "fr_FR")


@pytest.mark.parametrize("locale", LOCALES)
def test_date_batch_objects(locale):
    values = [dt.date(1970, 1, 1), dt.date(2000, 2, 29), DATE, dt.date(2026, 7, 30)]
    expected = [reference.format_date(value, locale=locale) for value in values]
    assert dates.format_date_batch(values, locale=locale) == expected


@pytest.mark.parametrize("locale", LOCALES)
def test_date_batch_numpy_calendar_decomposition(locale):
    values = np.array(
        ["1900-03-01", "1969-12-31", "1970-01-01", "2000-02-29", "2400-02-29"],
        dtype="datetime64[D]",
    )
    python_dates = [dt.date.fromisoformat(str(value)) for value in values]
    expected = [reference.format_date(value, "full", locale) for value in python_dates]
    assert dates.format_date_batch(values, "full", locale) == expected


def test_time_batch():
    values = [dt.time(hour, 7, 9, 123456) for hour in range(24)]
    expected = [reference.format_time(value, locale="en_US") for value in values]
    assert dates.format_time_batch(values, locale="en_US") == expected


def test_datetime_batch():
    values = [DATETIME + dt.timedelta(days=i, hours=i) for i in range(40)]
    expected = [reference.format_datetime(value, locale="de_DE") for value in values]
    assert dates.format_datetime_batch(values, locale="de_DE") == expected


def test_timedelta_interval_and_skeleton_compatibility():
    delta = dt.timedelta(days=3)
    assert dates.format_timedelta(delta, locale="fr_FR") == reference.format_timedelta(
        delta, locale="fr_FR"
    )
    end = DATETIME + dt.timedelta(hours=3)
    assert dates.format_interval(
        DATETIME, end, locale="en_US"
    ) == reference.format_interval(DATETIME, end, locale="en_US")
    assert dates.format_skeleton(
        "yMMMd", DATETIME, locale="de_DE"
    ) == reference.format_skeleton("yMMMd", DATETIME, locale="de_DE")


def test_nat_rejected():
    with pytest.raises(ValueError, match="NaT"):
        dates.format_date_batch(np.array(["NaT"], dtype="datetime64[D]"))


def test_numpy_datetime_temporary_is_retained_and_input_is_unchanged():
    values = np.arange(
        np.datetime64("1969-12-20"), np.datetime64("1970-01-20")
    )[::2]
    original = values.copy()
    expected = [
        reference.format_date(dt.date.fromisoformat(str(value)), locale="en_US")
        for value in values
    ]
    assert dates.format_date_batch(values, locale="en_US") == expected
    np.testing.assert_array_equal(values, original)


def test_native_entry_points_reject_null_buffers():
    assert dates.lib().mb_decompose_days(0, 1, 0) == -2
    assert dates.lib().mb_format_dates(
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 0
    ) == -2


@pytest.mark.parametrize("microsecond", [950000, 999999])
@pytest.mark.parametrize("pattern", ["S", "SS", "SSS"])
def test_fractional_second_rounding_fallback_parity(microsecond, pattern):
    value = DATETIME.replace(microsecond=microsecond)
    assert dates.format_datetime(value, pattern, locale="en_US") == (
        reference.format_datetime(value, pattern, locale="en_US")
    )
