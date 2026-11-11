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
    # the trailing argument is the row-loop worker count (0 = serial)
    assert dates.lib().mb_decompose_days(0, 1, 0, 0) == -2
    assert dates.lib().mb_format_dates(
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 0, 0
    ) == -2


@pytest.mark.parametrize("microsecond", [950000, 999999])
@pytest.mark.parametrize("pattern", ["S", "SS", "SSS"])
def test_fractional_second_rounding_fallback_parity(microsecond, pattern):
    value = DATETIME.replace(microsecond=microsecond)
    assert dates.format_datetime(value, pattern, locale="en_US") == (
        reference.format_datetime(value, pattern, locale="en_US")
    )


# babel.dates takes (value, format, tzinfo, locale). A rendered timestamp is text,
# so parity means exact string identity: fields are compared with zero tolerance
# and any clock or calendar drift fails the assertion.


def _outcome(call):
    """Render, or capture the exception class upstream would raise."""
    try:
        return ("value", call())
    except Exception as error:  # noqa: BLE001 - the class is the assertion
        return ("error", type(error).__name__)


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("width", WIDTHS)
def test_time_with_tzinfo_matches_upstream(locale, width):
    zone = reference.get_timezone("Europe/Paris")
    # Every width selects a timezone variant from the datetime's own instant.
    assert dates.format_time(DATETIME, width, zone, locale) == reference.format_time(
        DATETIME, width, zone, locale
    )


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("width", WIDTHS)
def test_datetime_with_tzinfo_matches_upstream(locale, width):
    zone = reference.get_timezone("Europe/Paris")
    assert dates.format_datetime(
        DATETIME, width, zone, locale
    ) == reference.format_datetime(DATETIME, width, zone, locale)


@pytest.mark.parametrize("locale", ["en_US", "fr_FR", "de_DE"])
def test_datetime_aware_input_and_shift_match_upstream(locale):
    zone = reference.get_timezone("Europe/Paris")
    aware = DATETIME.replace(tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    for width in ("full", "long"):
        assert dates.format_datetime(
            aware, width, None, locale
        ) == reference.format_datetime(aware, width, None, locale)
        assert dates.format_datetime(
            aware, width, zone, locale
        ) == reference.format_datetime(aware, width, zone, locale)


@pytest.mark.parametrize("locale", LOCALES)
def test_time_batch_with_tzinfo_matches_upstream(locale):
    zone = reference.get_timezone("Europe/Paris")
    values = [TIME, DATETIME, DATETIME + dt.timedelta(days=200)]
    for width in WIDTHS:
        expected = [
            reference.format_time(value, width, zone, locale) for value in values
        ]
        assert dates.format_time_batch(values, width, zone, locale) == expected


@pytest.mark.parametrize("locale", LOCALES)
def test_datetime_batch_with_tzinfo_matches_upstream(locale):
    zone = reference.get_timezone("Europe/Paris")
    values = [DATETIME, DATETIME + dt.timedelta(days=200)]
    for width in WIDTHS:
        expected = [
            reference.format_datetime(value, width, zone, locale) for value in values
        ]
        assert dates.format_datetime_batch(values, width, zone, locale) == expected


@pytest.mark.parametrize("locale", LOCALES)
def test_empty_batches_return_empty_list(locale):
    assert dates.format_date_batch([], locale=locale) == []
    assert dates.format_date_batch(
        np.array([], dtype="datetime64[D]"), locale=locale
    ) == []
    assert dates.format_time_batch([], locale=locale) == []
    assert dates.format_datetime_batch([], locale=locale) == []


@pytest.mark.parametrize("locale", LOCALES)
def test_single_element_batches_match_upstream(locale):
    assert dates.format_date_batch([DATE], locale=locale) == [
        reference.format_date(DATE, locale=locale)
    ]
    assert dates.format_time_batch([TIME], locale=locale) == [
        reference.format_time(TIME, locale=locale)
    ]
    assert dates.format_datetime_batch([DATETIME], locale=locale) == [
        reference.format_datetime(DATETIME, locale=locale)
    ]


@pytest.mark.parametrize("locale", LOCALES)
def test_batch_matches_port_scalar_api(locale):
    # The batch kernel and the scalar entry point must not diverge.
    dates_in = [DATE, dt.date(2000, 2, 29), DATE + dt.timedelta(days=365)]
    assert dates.format_date_batch(dates_in, "full", locale) == [
        dates.format_date(value, "full", locale) for value in dates_in
    ]
    assert dates.format_time_batch([TIME], "full", locale=locale) == [
        dates.format_time(TIME, "full", locale=locale)
    ]
    assert dates.format_datetime_batch([DATETIME], locale=locale) == [
        dates.format_datetime(DATETIME, locale=locale)
    ]


@pytest.mark.parametrize("tzinfo", [None, dt.timezone(dt.timedelta(hours=5, minutes=30))])
def test_timezone_offset_field_stays_within_one_second(tzinfo):
    # The rendered +HH:MM offset is a number: parse it back and compare with the
    # instant Babel formats, allowing less than one second of difference.
    value = DATETIME.replace(tzinfo=dt.timezone(dt.timedelta(hours=-3)))
    rendered = dates.format_datetime(value, "ZZZZZ", tzinfo, "en_US")
    expected = reference.format_datetime(value, "ZZZZZ", tzinfo, "en_US")
    assert rendered == expected
    sign = -1.0 if rendered.startswith("-") else 1.0
    hours, minutes = (int(part) for part in rendered.lstrip("+-").split(":")[:2])
    offset_seconds = sign * (hours * 60 + minutes) * 60
    want = value.utcoffset() if tzinfo is None else value.astimezone(tzinfo).utcoffset()
    assert abs(offset_seconds - want.total_seconds()) < 1.0


def test_duplicate_inputs_match_upstream():
    values = [DATE] * 7
    assert dates.format_date_batch(values, "full", "en_US") == [
        reference.format_date(value, "full", "en_US") for value in values
    ]
    moments = [DATETIME] * 5
    assert dates.format_datetime_batch(moments, locale="de_DE") == [
        reference.format_datetime(value, locale="de_DE") for value in moments
    ]


@pytest.mark.parametrize(
    "value",
    [dt.date(1, 1, 1), dt.date(9999, 12, 31), dt.date(2000, 2, 29), dt.date(1900, 3, 1)],
)
def test_calendar_boundaries_match_upstream(value):
    for locale in ("en_US", "de_DE", "ja_JP", "ar_EG"):
        assert dates.format_date(value, "full", locale) == reference.format_date(
            value, "full", locale
        )


def test_numpy_calendar_boundaries_match_upstream():
    values = np.array(["0001-01-01", "9999-12-31", "1969-12-31"], dtype="datetime64[D]")
    expected = [
        reference.format_date(dt.date.fromisoformat(str(value)), "full", "en_US")
        for value in values
    ]
    assert dates.format_date_batch(values, "full", "en_US") == expected


def test_date_with_time_pattern_matches_upstream_error():
    # Upstream's DateTimeFormat reads .hour off a date and raises AttributeError.
    assert _outcome(lambda: dates.format_date(DATE, "HH:mm")) == _outcome(
        lambda: reference.format_date(DATE, "HH:mm")
    )
    assert _outcome(
        lambda: dates.format_date_batch([DATE], "HH:mm", "en_US")
    ) == _outcome(
        lambda: [reference.format_date(DATE, "HH:mm", "en_US")]
    )


def test_date_wrong_type_matches_upstream_error():
    assert _outcome(lambda: dates.format_date(123)) == _outcome(
        lambda: reference.format_date(123)
    )
    # The batch entry point rejects non-dates before rendering.
    assert _outcome(lambda: dates.format_date_batch([123], "full", "en_US")) == (
        "error",
        "TypeError",
    )


def test_time_accepts_timestamps_and_none_like_upstream():
    # Timestamps convert through the same UTC pipeline upstream uses.
    for value in (0, 0.0, 1400000000):
        assert dates.format_time(value, locale="en_US") == reference.format_time(
            value, locale="en_US"
        )
    assert _outcome(lambda: dates.format_time("x")) == _outcome(
        lambda: reference.format_time("x")
    )


@pytest.mark.parametrize("skeleton", ["yMMMd", "Hm", "yM", "MMMMd", "jms"])
def test_skeleton_variants_match_upstream(skeleton):
    assert dates.format_skeleton(
        skeleton, DATETIME, locale="en_US"
    ) == reference.format_skeleton(skeleton, DATETIME, locale="en_US")


@pytest.mark.parametrize("delta", [dt.timedelta(0), dt.timedelta(days=-1)])
def test_timedelta_edge_values_match_upstream(delta):
    for width in ("long", "short", "narrow"):
        assert dates.format_timedelta(
            delta, width, locale="en_US"
        ) == reference.format_timedelta(delta, width, locale="en_US")
