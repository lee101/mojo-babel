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
arrays.

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

The shared library is written to `dist/libmojo-babel.so`.

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

Measured by `pixi run bench` on this machine on 2026-08-23: Intel Xeon E5-2697
v4 at 2.30 GHz, x86-64, 72 logical CPUs. Each row formats 50,000 values; the
best of three runs is reported. The reference is the installed Babel 2.18.0
through its public scalar API.

| case | mojo-babel | Babel | speedup |
| --- | ---: | ---: | ---: |
| decimal, 50k, en_US | 154.3 ms | 919.9 ms | 5.96x |
| decimal, 50k, hi_IN | 155.6 ms | 980.2 ms | 6.30x |
| currency, 50k, fr_FR | 165.1 ms | 2197.1 ms | 13.30x |
| percent, 50k, de_DE | 76.7 ms | 830.5 ms | 10.83x |
| date, 50k datetime64, en_US | 33.0 ms | 511.6 ms | 15.52x |
| date, 50k datetime64, ja_JP | 45.6 ms | 563.7 ms | 12.37x |
| datetime, 50k, de_DE | 169.4 ms | 972.5 ms | 5.74x |

Run `pixi run bench` on the target machine instead of treating these results as
portable; locale mix, string lengths, and CPU frequency all matter.

There is no GPU path. These kernels perform irregular integer division,
variable-length digit emission, and short UTF-8 writes with less than two
arithmetic operations per byte moved. CPU profiling also shows that the native
row renderer is too short to amortize thread-launch overhead, so it remains
serial; the benchmark covers only the CPU implementation.

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
