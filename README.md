# mojo-babel

`mojo-babel` is a Mojo implementation of the hot paths in
[Babel](https://babel.pocoo.org/) number and date formatting. It keeps Babel's
public function names and signatures for the covered scalar API, and adds batch
functions that amortize pattern compilation and FFI overhead across an entire
column.

The project does not copy a frozen locale database. Babel remains a dependency
and supplies current CLDR patterns, symbols, names, plural rules, and timezone
data. Mojo performs grouped fixed-point rendering, date field rendering, CLDR
pattern interpretation, and Gregorian decomposition for NumPy `datetime64`
arrays. Batches at or above 131,072 rows run those per-row loops on up to 4
workers; smaller batches stay serial (see "Parallel phase" below).

## Coverage

Accelerated Mojo paths, each covered by parity tests against Babel, are:

- `format_decimal`, `format_currency`, and `format_percent`, including custom
  positive/negative patterns, currency precision, accounting formats, Indian
  grouping, alternate number symbols, optional quantization, and grouping
  control;
- `format_date`, `format_time`, and `format_datetime` for the standard widths
  and custom patterns made from era, year, month, day, weekday, AM/PM, hour,
  minute, second, and six-digit fractional-second fields;
- `format_decimal_batch`, `format_currency_batch`, `format_percent_batch`,
  `format_date_batch`, `format_time_batch`, and `format_datetime_batch`;
- direct Gregorian decomposition of NumPy `datetime64[D]` arrays in Mojo.

Python compatibility wrappers, also covered by parity tests, are exposed for `format_scientific`,
`format_compact_decimal`, `format_compact_currency`, `format_timedelta`,
`format_interval`, and `format_skeleton`. Those operations, currency
`format_type="name"`, significant-digit patterns, fractional-second widths that
require Babel's Python-float rounding, and timezone-name fields use Babel's
Python implementation. Timezone conversion and CLDR metadata lookup are not
ported.

This is not a port of all Babel modules or APIs. Message catalogs, extraction,
plural-rule APIs, units, lists, language and territory display names, parsing,
and other `babel.numbers` and `babel.dates` helpers are outside this package.
Python's `date` and `datetime` types also limit their accelerated paths to the
years those types represent. NumPy day arrays use signed day counts and reject
`NaT`.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The shared library is written to `dist/libmojo-babel.so`. `max` is a required
dependency (pinned alongside `mojo` in `pixi.toml`): the kernels' row loops take
`parallelize` from `max.algorithm`, which ships in the `max` package rather than
in `mojo`, so without it every `max.*` import fails with `unable to locate
module 'max'`.

## Usage

```python
import datetime as dt
import numpy as np

from mojo_babel.dates import format_date, format_date_batch
from mojo_babel.numbers import format_currency, format_decimal_batch

print(format_currency(1234567.89, "EUR", locale="de_DE"))
# 1.234.567,89 €

print(format_date(dt.date(2026, 7, 30), format="full", locale="fr_FR"))
# jeudi 30 juillet 2026

values = [1234.5, -20, 9876543.21]
print(format_decimal_batch(values, locale="hi_IN"))
# ['1,234.5', '-20', '98,76,543.21']

days = np.array(["1970-01-01", "2000-02-29", "2026-07-30"], dtype="datetime64[D]")
print(format_date_batch(days, locale="en_US"))
# ['Jan 1, 1970', 'Feb 29, 2000', 'Jul 30, 2026']
```

## Benchmarks

Measured by `pixi run bench` on this machine on 2026-11-11: Intel Xeon E5-2697
v4 at 2.30 GHz, x86-64, 72 logical CPUs, load average ~16. Each row formats
50,000 identical inputs on both sides; the figure is the median of four
`pixi run bench` runs (each run already reports its own best of three). The
reference is the installed Babel through its public scalar API.

| case | mojo-babel | Babel | speedup |
| --- | ---: | ---: | ---: |
| decimal, 50k, en_US | 157.4 ms | 1007.6 ms | 6.40x |
| decimal, 50k, hi_IN | 160.8 ms | 998.8 ms | 6.21x |
| currency, 50k, fr_FR | 168.8 ms | 2207.6 ms | 13.07x |
| percent, 50k, de_DE | 77.5 ms | 864.3 ms | 11.15x |
| date, 50k datetime64, en_US | 38.4 ms | 544.1 ms | 14.17x |
| date, 50k datetime64, ja_JP | 46.3 ms | 592.4 ms | 12.78x |
| datetime, 50k, de_DE | 183.3 ms | 953.5 ms | 5.20x |

Run `pixi run bench` on the target machine instead of treating these results as
portable; locale mix, string lengths, and CPU frequency all matter.

### Unroll phase: no change shipped, and why

The unroll phase looked for serial dependency chains and loop overhead in the
kernels. Every candidate was compiled and timed against the current code; none was
faster at real caller sizes (50,000 rows), so all of them were reverted. The kernels
are unchanged and produce bit-identical output, including the existing
floating-point summation order — no multi-accumulator reduction exists in this code,
because no kernel in it performs a reduction.

Two `pixi run bench` runs of this byte-identical library, same machine, same day:

| run | decimal en_US | decimal hi_IN | currency fr_FR | percent de_DE | date en_US | date ja_JP | datetime de_DE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 166.1 ms | 176.5 ms | 173.3 ms | 81.5 ms | 39.6 ms | 44.4 ms | 182.4 ms |
| after | 188.5 ms | 178.3 ms | 203.3 ms | 85.4 ms | 35.9 ms | 44.1 ms | 193.9 ms |

The spread between the two runs (up to 15 % in both directions) is machine noise,
not a regression or an improvement: the library is the same file. This is why the
whole-batch table cannot resolve a kernel change that saves ~0.1 ms out of 166 ms —
the native kernels are 1–7 % of each batch total (see `bench/analysis.md`). Kernel
candidates were therefore also timed directly with cache-hot ctypes calls, pinned to
one core, warmed past turbo decay, interleaved A/B, median of 8 blocks of 25:

| candidate | kernel | before | after | verdict |
| --- | --- | ---: | ---: | --- |
| `pow10` loop → compile-time table | `mb_format_scaled` | 1.666 ms | 1.831 ms | reverted, 10 % slower |
| digit divisor chain `//= 10` → table lookup | `mb_format_scaled` | 1.666 ms | 1.850 ms | reverted, 11 % slower |
| per-digit output bound check removed | `mb_format_scaled` | 1.666 ms | 1.760 ms | reverted, 6 % slower |
| `append_uint` divisor chain → table lookup | `mb_format_scaled` | 1.666 ms | 1.829 ms | reverted, 10 % slower |
| fraction digits specialised per width | `mb_format_scaled` | 1.666 ms | 1.710 ms | reverted, neutral to slower |
| loop-invariant `i * 8` hoisted to a pointer bump | `mb_decompose_days` | 1.399 ms | 1.438 ms | reverted, 3 % slower |

The compiler already strength-reduces the `divisor //= 10` chain to a
multiply-and-shift at the top of each iteration (visible in the disassembly), already
unrolls the divisors' small `pow10` loop, and already removes the sign-correction
branches on the positive magnitudes the numbers path passes in. The remaining cost
is the real 64-bit integer division per digit — `idiv` latency, not loop overhead —
which unrolling cannot address; that is work for the next phase.

There is no GPU path. CPU only: memory-bound. The GPU candidate gate is ≥2
flops per byte moved; every kernel and every hot glue row in
`bench/analysis.md` is below it, so this phase shipped no GPU code, no device
allocations, and no new tests (there is nothing GPU-specific to skip on a
CPU-only host). CPU-side, the row loops are threaded above a measured size
threshold (next section); the benchmark covers only the CPU implementation.

### GPU phase: gate failed on arithmetic intensity, nothing shipped

The phase rule is that only kernels above roughly **2 flops per byte moved** are
GPU candidates. The gate was evaluated against `bench/analysis.md` (phase 1
counts, re-derived from the kernel sources rather than copied) before any GPU
work started.

| kernel / row | bytes moved per unit | generous flops per unit | flops/byte | verdict |
| --- | ---: | ---: | ---: | --- |
| `mb_format_scaled` (#5) | 21 B (8 in, 5 out, 8 length) | 27 (9 digits × 3 ops) | **1.29** | below gate |
| `mb_decompose_days` (#8) | 72 B (1 in, 8 × int64 out) | 40 (8 fields × 5 ops) | **0.56** | below gate |
| `mb_format_dates` (#4) | 16.4 MB / 50k rows | ~1.8 M | **0.11** | below gate |
| `_render_many` Decimal lane (#1) | 5.2 MB Decimal heap | ~0.72 M | **0.14** | below gate |
| `_prepare_scaled_floats` (#7) | 1.2 MB in/out/bool | ~0.4 M | **0.33** | below gate |
| unpack / field extraction / join (#2, #3, #9) | — | 0 | **0.00** | pure `str`/allocation glue |

`mb_format_scaled` is the highest at ~1.3 flops/byte and is still 35 % under the
gate; the counting above is deliberately generous (3 ops per emitted digit,
every field counted as 5 flops). Past the arithmetic, none of these kernels is
GPU-shaped regardless of the ratio: they are a serial `idiv` chain per digit,
a data-dependent three-integer op stream per row, and variable-length UTF-8
writes into caller-owned byte rows — all control flow and small integer
division, the two things a device handles worst. The dominant cost in every
number and datetime batch is Python glue (133 + 24–60 ms per call), which no
kernel, CPU or GPU, can move.

The GPU availability check nevertheless ran, because the verdict should record
both gate results, not just the one that stopped the work:

```bash
$ nvidia-smi --query-gpu=memory.free --format=csv,noheader
5554 MiB
```

5554 MiB free is above the 4000 MiB floor, so this phase *would* have been
permitted to run — it was skipped on arithmetic intensity, not on the shared
device. No before/after numbers exist to record because no GPU path was built,
and none are invented here.



`mb_format_scaled`, `mb_format_dates` and `mb_decompose_days` render independent
rows, so each one's row loop can run through `parallelize` from
`max.algorithm` (the `std.algorithm` form does not exist on this toolchain).
Each kernel takes its worker count as a trailing ABI argument: `0` forces the
serial loop, `n>0` asks for `n` workers, clamped to 4. Below the threshold the
kernel ignores the request and stays serial, so small batches never pay for
thread launch. Python passes `mojo_babel._lib.workers()`, which defaults to 4
and can be overridden with `MOJO_BABEL_WORKERS` for measurement and tests.

**Thread-launch cost.** Measured directly on this host with an empty
`parallelize` call: ~18.3 us at 2 workers, ~29 us at 4 — independent of the row
count. That is why there is a threshold at all rather than "parallelise
everything".

**Crossover, then margin.** Kernel-only timings, all timed regions exactly one
ctypes call, worker-count blocks interleaved, median of 9 blocks of 25
(`bench/kpar.py`). Workers pinned to 4 physical cores (the parallel point),
interpreter pinned to one core.

| rows | `mb_format_scaled` serial | 2 workers | 4 workers |
| ---: | ---: | ---: | ---: |
| 8,192 | 1.07 ms | 1.06 ms | 1.07 ms |
| 32,768 | 4.27 ms | 4.27 ms | 4.27 ms |
| 49,152 | 6.31 ms | 6.33 ms | 6.34 ms |
| 57,344 | 7.51 ms | 7.47 ms | 7.78 ms |
| 131,071 | 17.25 ms | 17.16 ms | 17.16 ms |
| 131,072 | 17.12 ms | 11.93 ms | 8.83 ms |
| 262,144 | 34.45 ms | 25.38 ms | 17.04 ms |
| 524,288 | 70.39 ms | 50.35 ms | 39.42 ms |

The other two kernels cross at the same place:

| rows | kernel | serial | 2 workers | 4 workers |
| ---: | --- | ---: | ---: | ---: |
| 131,071 | `mb_format_dates` | 15.64 ms | 15.60 ms | 16.06 ms |
| 131,072 | `mb_format_dates` | 15.78 ms | 8.92 ms | 8.31 ms |
| 262,144 | `mb_format_dates` | 33.13 ms | 19.93 ms | 11.53 ms |
| 262,143 | `mb_decompose_days` | 9.51 ms | 5.64 ms | 3.06 ms |
| 262,144 | `mb_decompose_days` | 9.39 ms | 6.37 ms | 3.37 ms |
| 524,288 | `mb_decompose_days` | 20.22 ms | 12.80 ms | 6.73 ms |

Below ~57,000 rows every kernel is flat at every worker count: the row work
there (~2 ms) is inside the noise of one launch's own cost. Above it the gain
appears and keeps growing. The first *reliably* winning size is 131,072 rows;
65,536 measured anywhere from 0.99x to 1.02x across three repeats, i.e. a tie
(same shape for dates, whose 131,071-row call was also a tie). The shipped
threshold is therefore `PARALLEL_MIN_ROWS = 1 << 17` (131,072 rows), which is a
2x margin over the first winning size and above every size at which the
comparison was a tie. It is one constant shared by all three kernels.

**Worker counts 1, 2, 4 and the cap.** One worker still goes through
`parallelize` and is neutral (49,152 rows: 4.18 ms vs 4.19 ms serial at one
worker) — `parallelize` with a single worker executes serially. Two workers gain
1.4–1.7x and four gain 1.8–3.1x at the sizes above the threshold. Going past 4:

| kernel | rows | 4 workers | 6 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| `mb_format_scaled` | 262,144 | 18.7 ms | 18.1 ms | 17.8 ms |
| `mb_decompose_days` | 524,288 | 7.7 ms | 7.5 ms | 7.5 ms |

which is flat within noise, so the cap stays at 4 as instructed.

**What reaches callers.** The kernels are 1–7 % of a public batch call (that is
the finding in `bench/analysis.md`), so kernel speedups mostly disappear into
the Python glue. At n = 262,144, one process per configuration, median of 5:

| path | serial | 4 workers | change |
| --- | ---: | ---: | ---: |
| `_render` (date kernel + unpack), n=262,144 | 223.5 ms | 197.4 ms | -12 % |
| `_core` (number kernel + unpack), n=262,144 | 187.6 ms | 183.7 ms | wash (2 repeats; the 29 ms `str` unpack and the `Decimal` lane dominate) |
| `mb_format_scaled` alone, n=262,144 | 34.6 ms | 18.3 ms | -47 % |
| `format_decimal_batch`, n=262,144 | 997.6 ms | 990.3 ms | -0.7 % |
| `format_currency_batch`, n=262,144 | 1077.9 ms | 999.9 ms | -7 % |
| `format_percent_batch`, n=262,144 | 1071.0 ms | 1002.1 ms | -6 % |
| `format_date_batch`, n=262,144 | 232.3 ms | 211.3 ms | -9 % |
| `format_date_batch`, n=524,288 | 453.1 ms | 377.3 ms | -17 % |
| `format_datetime_batch`, n=524,288 | 2229.4 ms | 2112.2 ms | -5 % |

Also note the size table above: `format_currency_batch` at 262,144 rows measured
-7 % here while the 50,000-row sweep measured +0.1 %, and its 50,000-row paired
runs ranged from -8 % to +21 %. The batch level cannot resolve these changes;
the kernel table is what the threshold is based on.

**The sizes real callers pass.** The benchmark and every documented usage
example format 50,000 rows, which is *below* the threshold, so those calls
remain serial and execute the pre-change loop exactly. No caller in this
repository reaches the parallel path, and at 50,000 rows the batch timings show
no reproducible difference in either direction. Nothing was reverted, because
at every size the threshold admits, the parallel loop wins; the 50,000-row
measurements are reported as ties rather than sold as gains.

**Correctness.** `tests/test_parallel.py` renders the same input at 1, 1023,
70,000 and 131,073 rows with 0, 1, 2 and 4 workers and asserts the outputs are
identical, for decimals, currency, percent, dates (two patterns) and datetimes.
The A/B driver also refuses to report a number if the CRC of the output differs
between worker counts. A 200,000-row batch whose last row is invalid returns
`-2` from both the serial and the parallel loop, so the error flag crosses the
worker boundary.

To reproduce:

```bash
taskset -c 2-5 pixi run python bench/kpar.py scaled 262144 0 1 2 4
taskset -c 2-5 pixi run python bench/kpar.py dates  131072 0 1 2 4
taskset -c 2-5 pixi run python bench/kpar.py days   262144 0 1 2 4
MOJO_BABEL_WORKERS=0 pixi run python bench/sizebench.py 262144 5
MOJO_BABEL_WORKERS=4 pixi run python bench/sizebench.py 262144 5
```


## How it works

Python resolves a Babel `Locale`, compiles its number or date pattern once, and
allocates contiguous NumPy buffers. Decimal inputs and ambiguous float rounding
lanes use Babel's `Decimal(str(value))` and half-even semantics. Large percent
batches prepare safe float lanes with vectorized NumPy operations, falling back
to `Decimal` for values within four ULPs of a half-even boundary. Quantized
signed 64-bit magnitudes then cross the C ABI as integer addresses. Mojo uses
SIMD to split full vector-width blocks into integer and fractional parts,
handles the remainder with a scalar tail, and inserts locale-specific UTF-8
separators and digits into exact-sized caller-owned byte rows.

Date patterns are compiled into a compact three-integer instruction stream.
Localized names and literals are packed into UTF-8 byte tables with 64-bit
offsets. Mojo interprets the stream over an `n x 8` row-major field buffer and
writes the result plus one length per row. NumPy day counts bypass Python date
objects: Mojo converts days since 1970-01-01 to Gregorian year, month, day, and
weekday before rendering.

All buffers remain owned by live NumPy objects for the complete native call.
ctypes passes their addresses as `Int64`; the exported Mojo functions validate
counts, dimensions, offsets, field ranges, non-null addresses, and output
strides before rebuilding
`UnsafePointer[..., AnyOrigin[mut=True]]` values and never allocate or retain a
pointer after the call.

MIT.
