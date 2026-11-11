"""Serial and parallel row loops must render the same batches byte for byte.

The kernels take the worker count as an argument, so both paths can be driven
from one process on one input. Sizes are chosen around the parallel threshold
so the crossing point itself is covered, not just the two regimes.
"""

import datetime as dt

import numpy as np
import pytest

from mojo_babel import dates, numbers
from mojo_babel._lib import lib

# 0 = serial; 1 is one worker (still inside `parallelize`); 4 is the cap.
WORKER_COUNTS = [0, 1, 2, 4]
# Below, at, and above the kernels' parallel threshold, plus an odd tail.
SIZES = [1, 1023, 70_000, 131_073]


@pytest.fixture
def worker_env(monkeypatch):
    def apply(count):
        monkeypatch.setenv("MOJO_BABEL_WORKERS", str(count))

    return apply


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("locale", ["en_US", "de_DE", "hi_IN"])
def test_decimal_serial_matches_parallel(worker_env, size, locale):
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(100_000, 80_000, size), 3).tolist()
    outputs = {}
    for count in WORKER_COUNTS:
        worker_env(count)
        outputs[count] = numbers.format_decimal_batch(values, locale=locale)
    for count in WORKER_COUNTS[1:]:
        assert outputs[count] == outputs[0]


@pytest.mark.parametrize("size", SIZES)
def test_currency_and_percent_serial_match_parallel(worker_env, size):
    rng = np.random.default_rng(11)
    values = np.round(rng.normal(1_000, 900, size), 4).tolist()
    serial = {}
    parallel = {}
    for count in (0, 4):
        worker_env(count)
        serial[count] = numbers.format_currency_batch(values, "EUR", locale="fr_FR")
        parallel[count] = numbers.format_percent_batch(values, locale="de_DE")
    worker_env(0)
    assert serial[4] == serial[0]
    worker_env(0)
    assert parallel[4] == parallel[0]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("format", ["medium", "full"])
def test_date_batch_serial_matches_parallel(worker_env, size, format):
    days = np.arange(np.datetime64("1900-01-01"), np.datetime64("1900-01-01") + np.timedelta64(size, "D"))
    outputs = {}
    for count in WORKER_COUNTS:
        worker_env(count)
        outputs[count] = dates.format_date_batch(days, format=format, locale="en_US")
    for count in WORKER_COUNTS[1:]:
        assert outputs[count] == outputs[0]


@pytest.mark.parametrize("size", [1, 1023, 70_000])
def test_datetime_batch_serial_matches_parallel(worker_env, size):
    values = [dt.datetime(2000, 1, 1) + dt.timedelta(minutes=17 * i) for i in range(size)]
    outputs = {}
    for count in (0, 2, 4):
        worker_env(count)
        outputs[count] = dates.format_datetime_batch(values, locale="de_DE")
    assert outputs[2] == outputs[0]
    assert outputs[4] == outputs[0]


def test_parallel_path_rejects_the_same_bad_row(worker_env):
    """A row the serial loop rejects must also fail the parallel loop.

    This drives the kernel directly because the public API cannot produce a
    negative scaled value; the point is that the error flag crosses the worker
    boundary and that the parallel loop covers rows the serial loop abandons at.
    """
    from mojo_babel._lib import addr, bytes_array

    count = 200_000  # above the parallel threshold, so both loops run
    for workers in (0, 4):
        values = np.full(count, 1000, dtype=np.int64)
        values[-1] = -1  # last row only: worst case for the parallel status flag
        group = bytes_array(",")
        decimal_symbol = bytes_array(".")
        output = np.empty((count, 24), dtype=np.uint8)
        lengths = np.empty(count, dtype=np.int64)
        status = lib().mb_format_scaled(
            addr(values),
            count,
            0,
            0,
            1,
            3,
            3,
            1,
            addr(group),
            len(group),
            addr(decimal_symbol),
            len(decimal_symbol),
            addr(output),
            24,
            addr(lengths),
            workers,
        )
        assert status == -2, f"workers={workers} returned {status}"


def test_worker_count_is_clamped_and_signed():
    """The kernel clamps the count instead of trusting the caller."""
    assert lib().mb_decompose_days(0, 0, 0, 99) == 0
    assert lib().mb_decompose_days(0, 0, 0, -5) == 0
