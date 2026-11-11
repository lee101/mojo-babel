"""Public-API A/B at several batch sizes, one worker count per process.

Uses MOJO_BABEL_WORKERS so the row-loop path under test is whatever the library
would use by default; everything else (glue, Decimal lane, unpack) is identical
between the two runs.

    MOJO_BABEL_WORKERS=0 python bench/sizebench.py 262144 5
"""

from __future__ import annotations

import datetime as dt
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_babel as mb  # noqa: E402
from mojo_babel._lib import workers  # noqa: E402


def _time(call, reps: int) -> float:
    call()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    size = int(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(100_000, 80_000, size), 3).tolist()
    days = np.arange(
        np.datetime64("1900-01-01"),
        np.datetime64("1900-01-01") + np.timedelta64(size, "D"),
    )
    datetimes = [
        dt.datetime(2000, 1, 1) + dt.timedelta(minutes=17 * index)
        for index in range(size)
    ]
    cases = [
        ("decimal en_US", lambda: mb.format_decimal_batch(values, locale="en_US")),
        ("date en_US", lambda: mb.format_date_batch(days, locale="en_US")),
        ("datetime de_DE", lambda: mb.format_datetime_batch(datetimes, locale="de_DE")),
    ]
    mb.format_decimal_batch(values[:10], locale="en_US")
    print(f"workers={workers()} size={size} reps={reps}")
    for name, call in cases:
        call()
    for name, call in cases:
        print(f"  {name:16s} {_time(call, reps) * 1e3:.1f} ms")


main()
