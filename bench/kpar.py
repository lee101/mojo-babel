"""Kernel-only A/B: serial vs `parallelize` row loops, interleaved in one process.

Each timed region is exactly one ctypes call into the kernel, so there is no
"first-loaded library wins" bias to control for: the worker count is a per-call
argument. Blocks of every worker count are interleaved (the host is shared), and
the median block is reported.

The scaled kernel rewrites `values` in place, so the input is restored before
every call, the way a fresh batch reaches it from Python. That restore is inside
the timed region for every worker count alike, so it does not favour either side.

    taskset -c 2-5 python bench/kpar.py scaled 50000 0 1 2 4
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import zlib

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojo_babel._lib import addr, bytes_array, lib  # noqa: E402


def make_case(kind: str, n: int):
    if kind == "scaled":
        rng = np.random.default_rng(7)
        raw = np.round(rng.normal(100_000, 80_000, n), 3)
        template = np.ascontiguousarray(np.rint(np.abs(raw) * 1000), dtype=np.int64)
        group = bytes_array(",")
        dec = bytes_array(".")
        out = np.empty((n, 24), dtype=np.uint8)
        lengths = np.empty(n, dtype=np.int64)
        fn = lib().mb_format_scaled
        work = template.copy()

        def call(workers):
            np.copyto(work, template)
            return fn(
                addr(work), n, 3, 0, 1, 3, 3, 1,
                addr(group), len(group), addr(dec), len(dec),
                addr(out), 24, addr(lengths), workers,
            )

        def snapshot():
            return out.tobytes(), lengths.tobytes()

    elif kind == "dates":
        from babel.core import Locale

        from mojo_babel.dates import _compile

        days = np.arange(n, dtype=np.int64) + 25000
        fields = np.empty((n, 8), dtype=np.int64)
        lib().mb_decompose_days(addr(days), n, addr(fields), 0)
        program = _compile("MMM d, y", str(Locale.parse("en_US")))
        out = np.empty((n, 256), dtype=np.uint8)
        lengths = np.empty(n, dtype=np.int64)
        fn = lib().mb_format_dates

        def call(workers):
            return fn(
                addr(fields), n, addr(program.ops), len(program.ops),
                addr(program.names), len(program.names),
                addr(program.offsets), len(program.offsets),
                addr(program.literals), len(program.literals),
                addr(out), 256, addr(lengths), workers,
            )

        def snapshot():
            return out.tobytes(), lengths.tobytes()

    elif kind == "days":
        values = np.arange(n, dtype=np.int64) + 25000
        fields = np.empty((n, 8), dtype=np.int64)
        fn = lib().mb_decompose_days

        def call(workers):
            return fn(addr(values), n, addr(fields), workers)

        def snapshot():
            return (fields.tobytes(),)

    else:  # pragma: no cover
        raise SystemExit(f"unknown kind {kind}")

    return call, snapshot


def main() -> None:
    kind, n = sys.argv[1], int(sys.argv[2])
    worker_sets = [int(value) for value in sys.argv[3:]] or [0, 1, 2, 4]
    blocks, inner = 9, 25
    call, snapshot = make_case(kind, n)

    digests = {}
    for workers in worker_sets:
        status = call(workers)
        if status != 0:
            raise SystemExit(f"workers={workers} returned {status}")
        digests[workers] = zlib.crc32(b"".join(snapshot())) & 0xFFFFFFFF
    unique = set(digests.values())
    if len(unique) != 1:
        raise SystemExit(f"outputs differ across worker counts: {digests}")

    for workers in worker_sets:
        for _ in range(200):
            call(workers)

    samples = {workers: [] for workers in worker_sets}
    for _ in range(blocks):
        for workers in worker_sets:
            start = time.perf_counter()
            for _ in range(inner):
                call(workers)
            samples[workers].append((time.perf_counter() - start) * 1e3 / inner)

    base = worker_sets[0]
    reference = statistics.median(samples[base])
    print(f"{kind} n={n} digest={unique.pop():08x} blocks={blocks}x{inner}")
    for workers in worker_sets:
        values = samples[workers]
        median = statistics.median(values)
        print(
            f"  workers={workers}: median={median:.3f} ms "
            f"min={min(values):.3f} max={max(values):.3f} "
            f"{reference / median:.2f}x vs workers={base}"
        )


main()
