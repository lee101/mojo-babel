"""Batch formatting benchmarks against Babel's public scalar API."""

from __future__ import annotations

import datetime as dt
import math
import os
import platform
import sys
import time

import numpy as np
from babel import dates as babel_dates
from babel import numbers as babel_numbers

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_babel as mb  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(100_000, 80_000, 50_000), 3).tolist()
    dates_np = np.arange(
        np.datetime64("1900-01-01"),
        np.datetime64("1900-01-01") + np.timedelta64(50_000, "D"),
    )
    dates_py = [dt.date.fromisoformat(str(value)) for value in dates_np]
    datetimes = [
        dt.datetime(2000, 1, 1) + dt.timedelta(minutes=17 * index)
        for index in range(50_000)
    ]

    cases = [
        (
            "decimal, 50k, en_US",
            lambda: mb.format_decimal_batch(values, locale="en_US"),
            lambda: [babel_numbers.format_decimal(v, locale="en_US") for v in values],
        ),
        (
            "decimal, 50k, hi_IN",
            lambda: mb.format_decimal_batch(values, locale="hi_IN"),
            lambda: [babel_numbers.format_decimal(v, locale="hi_IN") for v in values],
        ),
        (
            "currency, 50k, fr_FR",
            lambda: mb.format_currency_batch(values, "EUR", locale="fr_FR"),
            lambda: [
                babel_numbers.format_currency(v, "EUR", locale="fr_FR") for v in values
            ],
        ),
        (
            "percent, 50k, de_DE",
            lambda: mb.format_percent_batch(values, locale="de_DE"),
            lambda: [babel_numbers.format_percent(v, locale="de_DE") for v in values],
        ),
        (
            "date, 50k datetime64, en_US",
            lambda: mb.format_date_batch(dates_np, locale="en_US"),
            lambda: [babel_dates.format_date(v, locale="en_US") for v in dates_py],
        ),
        (
            "date, 50k datetime64, ja_JP",
            lambda: mb.format_date_batch(dates_np, format="full", locale="ja_JP"),
            lambda: [
                babel_dates.format_date(v, format="full", locale="ja_JP")
                for v in dates_py
            ],
        ),
        (
            "datetime, 50k, de_DE",
            lambda: mb.format_datetime_batch(datetimes, locale="de_DE"),
            lambda: [
                babel_dates.format_datetime(v, locale="de_DE") for v in datetimes
            ],
        ),
    ]

    mb.format_decimal_batch(values[:10], locale="en_US")
    print(f"Machine: {cpu_name()} ({platform.machine()}, {os.cpu_count()} logical CPUs)")
    print()
    print("| case | mojo-babel | Babel | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in cases:
        mojo_seconds = timeit(ours)
        babel_seconds = timeit(reference)
        ratio = babel_seconds / mojo_seconds
        label = f"{ratio:.2f}x" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.1f} ms | "
            f"{babel_seconds * 1000:.1f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
