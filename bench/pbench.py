"""Batch-level A/B: the public API with the kernel row loop serial vs parallel.

One worker count per process (MOJO_BABEL_WORKERS), same vectors as bench.py,
allocation-only input so the timed region is the whole public call. Run each
worker count in its own process and compare the medians.

    taskset -c 2-5 python bench/pbench.py 4 [reps]
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
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(100_000, 80_000, 50_000), 3).tolist()
    dates_np = np.arange(
        np.datetime64("1900-01-01"),
        np.datetime64("1900-01-01") + np.timedelta64(50_000, "D"),
    )
    datetimes = [
        dt.datetime(2000, 1, 1) + dt.timedelta(minutes=17 * index)
        for index in range(50_000)
    ]

    cases = [
        ("decimal en_US", lambda: mb.format_decimal_batch(values, locale="en_US")),
        ("decimal hi_IN", lambda: mb.format_decimal_batch(values, locale="hi_IN")),
        (
            "currency fr_FR",
            lambda: mb.format_currency_batch(values, "EUR", locale="fr_FR"),
        ),
        ("percent de_DE", lambda: mb.format_percent_batch(values, locale="de_DE")),
        ("date en_US", lambda: mb.format_date_batch(dates_np, locale="en_US")),
        (
            "date ja_JP full",
            lambda: mb.format_date_batch(dates_np, format="full", locale="ja_JP"),
        ),
        ("datetime de_DE", lambda: mb.format_datetime_batch(datetimes, locale="de_DE")),
    ]
    mb.format_decimal_batch(values[:10], locale="en_US")
    print(f"workers={workers()} reps={reps}")
    for name, call in cases:  # warm every path before timing
        call()
    for name, call in cases:
        print(f"  {name:16s} {_time(call, reps) * 1e3:.1f} ms")


main()
