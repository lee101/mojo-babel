# ANALYSIS — where the time actually goes

Phase 1 of the port. No algorithm changed, nothing under `src/` or `tests/` touched.
Every number below came from a run in this session on this machine. Where two runs
disagree I say so instead of averaging.

- Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, x86-64, 72 logical CPUs (72 visible; all runs single-threaded).
- Env: Python 3.13.14, numpy 2.5.1, Babel 2.18.0 (pixi env), `dist/libmojo-babel.so` (23,512 bytes).
- Harness: `pixi run bench` (flock `/tmp/mojo-bench.lock`), best of 3.
- Probes: read-only scripts using the same lock and the same input vectors
  (`np.random.default_rng(7)`, 50,000 values / days / datetimes). `bench/probe.py`
  was written and then removed per the phase rule; the commands are recorded below.
- `perf stat` was refused (`perf_event_paranoid=4`, no CAP_PERFMON), so the
  instruction/miss counts come from `valgrind --tool=cachegrind --cache-sim=yes`
  on the kernels alone, and py-spy (leaf frames) on the full harness.

---

## 0. Headline

**The Mojo kernels are between 0.6 % and 3.4 % of each batch call.** Everything
else is Python. The two dominant Python costs are

1. a **per-element `Decimal(str(x))` → `quantize` → `scaleb` loop of 50,000
   iterations** inside `_render_many`, entered for *every* number entry point in
   the benchmark, and
2. a **per-row `tobytes().decode()` loop of 50,000 iterations** that turns the
   native byte cells into `str` objects.

The kernel that everybody would look at first (`mb_format_scaled`, SIMD-shaped)
runs 50,000 rows in **1.70 ms**. It is not the problem.

---

## 1. Public entry points, this session

`pixi run bench`, two full runs (exit 0). Upstream column is the scalar-loop
Babel reference the harness builds. Both runs are reported because the upstream
numbers moved 15–20 % between them on this box; the port numbers moved ≤ 6 %.

| case | port run 1 | Babel run 1 | speedup | port run 2 | Babel run 2 | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decimal, 50k, en_US | 156.5 ms | 953.3 ms | 6.09x | 161.2 ms | 1175.2 ms | 7.29x |
| decimal, 50k, hi_IN | 166.1 ms | 957.8 ms | 5.77x | 166.5 ms | 1186.7 ms | 7.13x |
| currency, 50k, fr_FR | 165.5 ms | 2208.2 ms | 13.34x | 174.6 ms | 2506.2 ms | 14.35x |
| percent, 50k, de_DE | 75.8 ms | 802.5 ms | 10.59x | 82.7 ms | 994.1 ms | 12.02x |
| date, 50k datetime64, en_US | 38.1 ms | 538.5 ms | 14.15x | 32.8 ms | 569.0 ms | 17.32x |
| date, 50k datetime64, ja_JP | 41.7 ms | 582.6 ms | 13.98x | 45.0 ms | 627.8 ms | 13.96x |
| datetime, 50k, de_DE | 184.8 ms | 1015.2 ms | 5.49x | 185.9 ms | 1072.7 ms | 5.77x |

Nothing in the public batch API is slower than upstream. **Rows where the port is
slower exist only on the scalar API** — see §4 and the marked table.

---

## 2. Ranked hotspot table

Times are medians of 7 (or the harness's best-of-3 for whole-entry rows).
"bytes touched" is the traffic the row's own code moves, counted from the code:
list input 8 B/row, int64 8 B/row, date cells `N x 256` (stride the date path
uses), output cells `N x stride` (5 B for `en_US` decimal), lengths 8 B/row,
heap bytes from `tracemalloc` where the row allocates.
"flops" is source-level arithmetic, 3 ops per emitted digit (div, mul, sub) plus
the SIMD split; it is deliberately generous — even so nothing reaches 2 ops/byte.
Bold rows are the top three.

| # | function | input size | time | bytes touched | flops | flops/byte | bound | where the time sits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | **`_render_many` per-row `Decimal(str)`→`quantize`→`scaleb` loop** (`numbers.py:200–229`) | 50,000 floats | **≈133 ms** (165.0 total − 29.3 kernel/unpack − ~3 reassembly) | 5.2 MB Decimal heap + 1.4 MB float str + 400 KB input + 400 KB lens (11.65 MB peak heap, measured `_render_many` alone) | ~0.72 Mops (≈1.4 M ULPs of 28-digit decimal) | **≈0.10** | memory/latency (decimal C lib) | **Python/ctypes glue** — 50,000 Python iterations, no ctypes inside the loop |
| 2 | **row unpack `cells[i,:len].tobytes().decode()`** (`numbers.py:88`, `dates.py:193`) | 50,000 rows | **23.6 ms** (number) / **28.4 ms** (date) | 250 KB out + 400 KB lens → 50,000 `str` (52 B each measured) ≈ 3.0 MB heap | 0 | 0.00 | pure allocation/glue | **Python glue** — 50,000 numpy slices + 50,000 decodes |
| 3 | **`_datetime_fields` + `_time_fields`** (`dates.py:199–253`) | 50,000 py datetimes | **54.2 + 46.3 ms** (×2 inside `format_datetime_batch` = 100.5 ms) | 3.2 MB int64 fields per call + 50,000 attribute reads | 0 | 0.00 | glue | **Python glue** — `isinstance` + 8 attribute reads per element |
| 4 | `mb_format_dates` (native) | 50,000 rows, 256-B stride | **4.43 ms** (en_US) / 5.26 ms (ja_JP full) | 3.2 MB fields + 12.8 MB cells + 400 KB lens = 16.4 MB | ~1.8 Mops (355 Ir/row measured) | **0.11** | memory-bound | **kernel** |
| 5 | `mb_format_scaled` (native) | 50,000 rows | **1.70 ms** (187 Ir/row, 5.4 LLd miss/row measured) | 400 KB values + 250 KB cells + 400 KB lens = 1.05 MB | ~1.5 Mops (≈30 ops/row) | **1.43** | memory-bound | **kernel** |
| 6 | `_core` setup + affix join | 50,000 rows | ~2 ms setup + ~3 ms join (25.6 total − 1.7 native − 23.6 unpack) | 4 B/row | 0 | 0.00 | glue | **Python glue** |
| 7 | `_prepare_scaled_floats` (vectorized NumPy fast lane) | 50,000 floats | 5.84 ms (decimal) / 18.5 ms (percent, scale=2) / 18.6 ms (currency, incl. 4,979 slow rows = 9.56 ms) | 400 KB in, 400 KB out, 400 KB bool | ~0.4 Mops | 0.33 | memory-bound | **glue** — and **not on the public path** (see §3) |
| 8 | `mb_decompose_days` (native) | 50,000 days | **1.62 ms** | 400 KB in + 3.2 MB out = 3.6 MB | ~1.0 Mops (≈20 ops/row) | **0.28** | memory-bound | **kernel** |
| 9 | `datetime` template `zip`/`replace` | 50,000 pairs | 12.7–24.7 ms (two runs) | 2 × 50,000 str | 0 | 0.00 | glue | **Python glue** |
| 10 | ctypes marshalling, scalar API | 1 element | 40–65 µs/call (see §4) | 15 int64 args + 1 row | 0 | 0.00 | glue | **Python/ctypes glue** |

Measured context for scale: empty native call **5 µs**; `np.empty((50000,5))`
**1 µs**; `np.asarray(list of 50k floats)` 2.3 ms; f-string `,.3f` over 50k
**25.3 ms** (faster than the port's number path, see §5).

### Where each batch total goes

| entry point | total | kernel | unpack | python glue | unmeasured/other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `format_decimal_batch` en_US | 165.0 ms | 1.70 ms (1.0 %) | 23.6 ms | ~133 ms Decimal loop + ~6 ms setup/join | ~0 |
| `format_decimal_batch` hi_IN | 168.8 ms | 1.7 ms | 23.6 ms | as above | as above |
| `format_percent_batch` de_DE | 80.5 ms | 1.7 ms (2.1 %) | 23.6 ms | ~52 ms Decimal loop (scaled ×100) | ~3 ms |
| `format_currency_batch` fr_FR | 188.3 ms | 1.7 ms (0.9 %) | 23.6 ms | ~160 ms Decimal loop (incl. 9.6 ms slow-lane retry) | ~3 ms |
| `format_date_batch` np en_US | 36.9 ms | 1.62 + 4.43 = 6.05 ms (16 %) | 28.4 ms | 0.09 ms astype/NaT | ~2 ms |
| `format_date_batch` np ja_JP full | 48.5 ms | 1.62 + 5.26 = 6.88 ms | 28.4 ms | | pattern/build |
| `format_datetime_batch` de_DE | 189.3 ms | 2 × 4.3 = 8.6 ms (4.5 %) | 2 × 29.6 = 59.2 ms | 100.5 ms field extraction + 24.7 ms template | ~0 |

---

## 3. The finding that changes the plan: the fast lane is dead code

`_render_many` (`python/mojo_babel/numbers.py:192`) guards its vectorized NumPy
fast lane with

```python
if decimal_quantization and not dynamic_currency_name and pattern.scale:
```

`pattern.scale` is the *pattern's* exponent scale. Every pattern the benchmark
uses parses with `scale == 0` (`#,##0.###`, `#,##0.00 ¤`, percent `#,##0%`), so the
guard is false and `_prepare_scaled_floats` is **never called through the public
API**. Evidence from this session, all consistent:

- Monkeypatching `_prepare_scaled_floats` to print: `format_decimal_batch` runs twice, prints nothing.
- `sys.settrace` line trace of `_render_many`: L199–L229 execute **50,000 times**; L196–L197 execute once with `fast_prepared = None`.
- Line-level self-time inside `_render_many`: `_decimal_value` 170 ms, `int(scaleb)` 99 ms, `quantize` 65 ms (tracer overhead inflates, ordering is what matters).
- py-spy leaf frames on `bench/bench.py`: `_decimal_value` at `numbers.py:97` is 3.2 % of all on-CPU samples.
- `_core` measured alone is 25.6 ms and `_prepare…` + `_core` measured alone is 40.7 ms, while `_render_many` is 165 ms — the 124 ms gap is exactly the loop.

This is why the port's number path (165 ms/50k) is slower than a plain
`f"{v:,.3f}"` loop (25.3 ms/50k, measured): the slow lane it falls back to does
50,000 Decimal constructions and 100,000 context-dispatched decimal operations
per call.

I did not change this: the guard is an algorithm decision, out of scope for this
phase. It is the #1 hotspot and it must be fixed in glue, not with a kernel phase.

---

## 4. Scalar API is slower than upstream — flagged

Measured as 1,000 calls in a loop (µs/call). Marked as required: the port is
slower on every scalar entry point.

| entry point | mojo-babel | Babel | ratio | marker |
| --- | ---: | ---: | ---: | --- |
| `format_decimal(1234567.891, "en_US")` | 65.5 µs | 20.9 µs | 3.1x slower | **slower than upstream** |
| `format_currency(…, "EUR", "fr_FR")` | 73.9 µs | 44.4 µs | 1.7x slower | **slower than upstream** |
| `format_percent(1234.5, "de_DE")` | 66.3 µs | 16.6 µs | 4.0x slower | **slower than upstream** |
| `format_date(date(2026,7,30), "en_US")` | 47.1 µs | 11.6 µs | 4.1x slower | **slower than upstream** |
| `format_datetime(dt, "de_DE")` | 89.5 µs | 20.9 µs | 4.3x slower | **slower than upstream** |

Breakdown of one scalar call: a 1-row batch (`format_decimal_batch(values[:1])`)
is **62 µs** — so the cost is batch setup, not scale-dependent work. The native
call floor is **5 µs** (empty call) and a 1-row kernel call is 16 µs for
`mb_format_scaled` (the 8 µs difference between a 50,000-row call at 1.70 ms and
the empty call is small per row; the per-call overhead is the call itself, not
arithmetic). The remaining ~40 µs is: `Locale.parse` + `parse_pattern` (~6 µs
measured for the pair), `addr()` validation ×3, two `np.empty`s, ctypes call with
15 int64 arguments, and the row unpack. Per call, this is Python/ctypes glue end
to end; the kernel is 2–3 % of a scalar call.

---

## 5. Python-level loops that re-enter ctypes / allocate per element

1. **No loop re-enters ctypes per element.** All three native symbols are called
   exactly once per batch. That part of the design holds.
2. `_render_many` numbers.py:200–229 — 50,000 iterations, each allocating a
   `Decimal` (5.2 MB heap total), calling `quantize` and `scaleb`. **Hotspot #1.**
3. `_render_many` numbers.py:295–300 — a second 50,000-iteration pass for
   `prefix + core + suffix` and `negative_flags[index]`.
4. `_core` numbers.py:88–91 — 50,000 numpy row slices + 50,000 `tobytes` +
   50,000 `decode` (23.6 ms); `_render` dates.py:193–196 the same (28.4 ms).
5. `_datetime_fields` / `_time_fields` / `_date_fields` dates.py:199–253 —
   per-element `isinstance` chain + 8 attribute accesses (54.2 / 46.3 ms).
6. `format_datetime_batch` dates.py:401–408 — runs (3) and (4) twice (date and
   time) then a third 50,000-element pass for the template `replace` **(24.7 ms)**.
7. `format_time_batch`/`format_date_batch` on object inputs always build a
   Python list first (`list(times)`), a full copy of the caller's iterable.

**Allocations and copies per call (measured):** `tracemalloc` peak — decimal
11.65 MB, percent 7.69 MB, currency 17.49 MB, date-np 19.88 MB, datetime 23.30 MB
for 50,000 rows. The native side allocates nothing; the `np.empty` calls are
~1 µs each. The heap traffic is entirely Python objects (Decimal, str, int64 rows).

---

## 6. Next-phase assignment

"none" below is a verdict too: the next phase must not burn a kernel rewrite on
it. GPU verdicts are backed by the flops/byte column, and every row is far below
the ≈2 flops/byte threshold: `mb_format_scaled` 1.43, `mb_format_dates` 0.11,
`mb_decompose_days` 0.28, and the glue rows are 0.00–0.33.

| hotspot | next phase | why |
| --- | --- | --- |
| #1 Decimal slow-lane loop | **none** (glue fix, not an accel phase) | 133 ms of 165 ms; 0.10 flops/byte; the work is Babel-compatible decimal semantics. Fix the `pattern.scale` gate in Python glue so the existing vectorized lane runs; then re-measure. |
| #2 row unpack | **none** (glue fix) | 24–28 ms; 0 flops/byte, pure `str` construction. |
| #3 field extraction | **none** (glue fix) | 100.5 ms in `format_datetime_batch`; per-object Python attribute reads. |
| #4 `mb_format_dates` | **none** | 4.43 ms with 16.4 MB touched (0.11 flops/byte). Memory-bound; the 256-byte cell stride is the real cost, and that is a buffer-layout decision in glue. |
| #5 `mb_format_scaled` | **none** (simd already present, marginal) | 1.70 ms, 1.43 flops/byte, 187 Ir/row. Faster than `memcpy` of its own output. |
| #6 `_core` setup/join | **none** | ~5 ms total. |
| #7 `_prepare_scaled_floats` | **none** until §3 is fixed | currently unreachable from public entry points; once reachable it is 5.8 ms (decimal) — memory-bound numpy, not an accel target. |
| #8 `mb_decompose_days` | **none** | 1.62 ms, 0.28 flops/byte. |
| #9 template join | **none** (glue fix) | 24.7 ms of `str.replace`; reuse one pass instead of three. |
| #10 scalar ctypes glue | **none** (glue fix; batching already exists) | 40–65 µs/call of marshalling; the scalar API is 1.7–4.3x slower than upstream as a result. |

**No row in this profile justifies `unroll`, `simd`, `parallel`, or `gpu`.**
The kernels already contain the SIMD split (`babel.mojo:168–181`); the date
kernel is scalar per row and is 0.11 flops/byte, so vectorizing it cannot pay.
The measured opportunity is 133 ms (#1) + 24–60 ms (#2, #3, #9) of Python glue
on the number and datetime paths — roughly 80 % of the number batch time and
~70 % of the datetime batch time.

---

## 7. Reproduce

```bash
pixi run bench                                  # the table in §1
flock /tmp/mojo-bench.lock pixi run python bench/probe.py     # phase splits (script removed after this phase)
flock /tmp/mojo-bench.lock pixi run valgrind --tool=cachegrind --cache-sim=yes \
    --cachegrind-out-file=/tmp/cg.out python <kernel-only driver>   # Ir/row, LLd misses
flock /tmp/mojo-bench.lock pixi run py-spy record -r 499 -f raw -o /tmp/p.txt -d 30 -- python bench/bench.py
```

Notes and caveats:

- py-spy reported itself "behind in sampling" on this box; use it for leaf
  attribution, not for absolute percentages.
- Baseline rows (`f-string`, `%.3f`, `np.char.mod`) are not locale-correct and are
  not a substitute for Babel; they exist to show that the current number path
  spends its time in decimal-object work, not in formatting digits.
- Upstream Babel timings varied up to 20 % between the two harness runs. The port
  timings varied ≤ 6 %. Ratios in §1 therefore carry roughly ±15 % uncertainty;
  the phase splits in §2 are from a single probe session and repeated two to three
  times with agreement within 10 %.
- The `pixi run bench` runs are the authoritative end-to-end numbers; the probes
  are diagnostic and never ran concurrently with them (same flock).

---

## 8. Unroll phase: every candidate measured, none kept

Phase 2 targeted technique 1 (multi-accumulator reductions), 2 (compile-time
unrolling of fixed trip counts), 3 (hoisting loop invariants) and 4 (removing
bounds checks the loop bound proves). Source is unchanged; the library is
byte-identical to the phase-1 build. Nothing below shipped.

**Technique 1 does not apply.** There is no reduction in this code base: every kernel
walks rows independently and writes digits. There is no running sum, so there is no
floating-point add chain to break and no summation order to document.

**Technique 2 is not expressible on this pin.** `@parameter for` is rejected
(`'for' statement does not support decorators`), `unroll(range(n))` is an unknown
declaration, comptime recursion in a nested `def` is rejected, and a `@__parameter`
closure cannot be indexed with a runtime value. So a fixed trip count cannot be
unrolled at comptime in the pinned dialect; the only route is a hand-written or
generated straight-line block per width, which measured as a loss (see below).

**Measurement method.** `pixi run bench` cannot resolve these kernels: they are
1–7 % of a batch total, and two runs of the *same* `.so` on this box differed by up to
15 % per row (turbo decay from 3.6 GHz to 2.3 GHz mid-run, plus 2 sockets × 18 cores).
Kernels were therefore timed in isolation — same vectors as the harness (`rng(7)`,
50,000 rows), ctypes with `argtypes` set, `taskset -c 2`, 1.5 s of warm-up past turbo
decay, 8 interleaved A/B blocks of 25, median. Baseline reproduces to ±0.3 %:
`mb_format_scaled` 1.666 ms, `mb_decompose_days` 1.399 ms, `mb_format_dates`
unchanged at 0.11 flops/byte (not attempted). In-process A/B is *not* valid on this
box: the first-loaded library is systematically faster regardless of code, so each
number above comes from one library per process.

| candidate | kernel | baseline | candidate | delta |
| --- | --- | ---: | ---: | --- |
| `pow10` loop → 16-entry comptime `SIMD` table | `mb_format_scaled` | 1.666 ms | 1.831 ms | +10 % |
| digit divisor chain `//= 10` → table lookup by position | `mb_format_scaled` | 1.666 ms | 1.850 ms | +11 % |
| per-digit `if pos >= limit` check removed | `mb_format_scaled` | 1.666 ms | 1.760 ms | +6 % |
| `append_uint` divisor chain → table lookup | `mb_format_scaled` | 1.666 ms | 1.829 ms | +10 % |
| fraction digits written per-width (`write_fraction[1..4]`) | `mb_format_scaled` | 1.666 ms | 1.710 ms | +3 % |
| `i * 8` index arithmetic → running pointer / 8-wide store | `mb_decompose_days` | 1.399 ms | 1.438 ms | +3 % |

**Why every candidate loses.** The disassembly of `format_scaled_row` shows that the
compiler already does the work these techniques target: the `divisor //= 10` chain is
strength-reduced to `mulx 0xcccccccccccccccd` + `shr $3` hoisted to the top of the
iteration, the `pow10(n)` loop is unrolled into a move, and the sign-correction
branches around `/` and `%` are eliminated on the positive magnitudes the numbers
path passes. Replacing those with a table load adds a load and a register copy to a
loop that already runs at 187 instructions/row. Removing the bound check does not
help because the check `and`-merges into an existing compare and the store address is
already computed.

**Where the time actually is.** Per row: `digits10` is a serial `n //= 10` chain
(≤7 iterations, one per integer digit) and each digit costs one *real* 64-bit `idiv`
(latency 32–40 cycles on Broadwell-EP) that cannot be strength-reduced because the
divisor is data-dependent. That division latency — not loop overhead, not the
reduction order, not the bound checks — is the kernel's floor, and it is why
unrolling cannot pay here. Splitting it into per-digit constant divisors is only
possible with SIMD/reciprocal-multiply tricks, which is the next phase, not this one.

**Consequence for the next phase.** Do not re-try 2/3/4 on these kernels; the compiler
already performs them. The payoff is in (a) the `digits10`/`idiv` chain per digit, and
(b) the glue rows of §6 (#1, #2, #3, #9 — 133 + 24–60 ms), which are still untouched
and still dominate every number and datetime batch total.
