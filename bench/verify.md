# VERIFY phase

Date: 2026-11-11. Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, x86-64, 72 logical
CPUs, load average ~16 during all runs. Commands were run in the order the
verify brief asked for, with no code changes in between:

| command | result |
| --- | --- |
| `pixi run build` | exit 0 (only pre-existing deprecation warnings on `fields[i * 8 + n]`) |
| `pixi run test` | exit 0, **299 passed in 14.26s** |
| `pixi run bench` | exit 0, table below |
| `pixi run build && pixi run test` (final) | exit 0, **299 passed** |

## What I reproduced

`pixi run bench` was executed four times in one session. Each run is already a
best-of-three per side, so the four run values are themselves compared:

| case | README (before) | run 1 | run 2 | run 3 | run 4 | median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decimal en_US, mojo | 154.3 ms | 163.0 | 195.6 | 151.2 | 151.8 | 157.4 |
| decimal en_US, babel | 919.9 ms | 1032.2 | 1080.4 | 983.0 | 973.2 | 1007.6 |
| decimal hi_IN, mojo | 155.6 ms | 158.7 | 187.9 | 146.7 | 162.8 | 160.8 |
| decimal hi_IN, babel | 980.2 ms | 1006.9 | 990.7 | 962.2 | 1035.8 | 998.8 |
| currency fr_FR, mojo | 165.1 ms | 173.5 | 179.6 | 162.7 | 164.2 | 168.8 |
| currency fr_FR, babel | 2197.1 ms | 2195.5 | 2300.2 | 2139.4 | 2219.6 | 2207.6 |
| percent de_DE, mojo | 76.7 ms | 78.1 | 108.6 | 77.0 | 76.4 | 77.5 |
| percent de_DE, babel | 830.5 ms | 864.6 | 907.6 | 864.0 | 813.7 | 864.3 |
| date en_US, mojo | 33.0 ms | 38.3 | 41.3 | 38.5 | 37.4 | 38.4 |
| date en_US, babel | 511.6 ms | 541.3 | 551.4 | 528.2 | 546.9 | 544.1 |
| date ja_JP, mojo | 45.6 ms | 42.4 | 49.1 | 48.3 | 44.4 | 46.3 |
| date ja_JP, babel | 563.7 ms | 598.2 | 586.6 | 639.0 | 577.2 | 592.4 |
| datetime de_DE, mojo | 169.4 ms | 191.7 | 183.2 | 183.5 | 177.6 | 183.3 |
| datetime de_DE, babel | 972.5 ms | 973.7 | 952.7 | 939.3 | 954.3 | 953.5 |

Deviation of the 4-run median from the README figure:

| case | mojo | babel |
| --- | ---: | ---: |
| decimal en_US | +2.0 % | +9.5 % |
| decimal hi_IN | +3.3 % | +1.9 % |
| currency fr_FR | +2.3 % | +0.5 % |
| percent de_DE | +1.1 % | +4.1 % |
| date en_US | **+16.4 %** | +6.4 % |
| date ja_JP | +1.6 % | +5.1 % |
| datetime de_DE | +8.2 % | -2.0 % |

No figure reproduced inside the 20 % tolerance band the brief asked for.
Because every row moved, the README table was rewritten from the 4-run medians
(see "What I changed"). No row was removed, and no row was reworded: all seven
cases remain faster than Babel exactly as before, so there was nothing slower to
preserve.

Verdict, measured on the same input on both sides: the port is genuinely faster
than Babel's per-call scalar API on all seven cases, with speedups between
**5.20x** and **14.17x** on this host, i.e. lower than the README's previous
5.74x–15.52x but comfortably in the same class. Nothing was overstated by an
order of magnitude.

## Same input, no skipped work

Checked directly rather than assumed:

- `bench/bench.py` builds one input set (`values` is `np.round(rng.normal(...),
  3).tolist()`, 50,000 elements, 5,301 negative; `dates_np` is 50,000
  consecutive `datetime64[D]`; `datetimes` is 50,000 17-minute steps) and feeds
  the *same* objects to both lambdas of each case. Neither side gets a cheaper
  slice. The date rows feed `dates_np` (datetime64) to the port and the
  equivalent `dates_py` (converted from the same array, not regenerated) to
  Babel.
- I ran both sides independently on the exact bench input and compared all
  50,000 outputs per case: **7/7 cases equal, 50000/50000 rows equal each**.
  The port does not format fewer rows, and the Babel side is not doing less
  work.
- Babel's side uses the real public scalar API (`numbers.format_decimal`,
  `format_currency`, `format_percent`, `dates.format_date`,
  `format_datetime`), which per call does `Locale.parse` +
  `parse_pattern(...).apply(...)`. `inspect.getsource` confirms none of these
  short-circuit to a batch helper.
- The port's batch path (`mojo_babel.numbers._render_many`,
  `_core`, `_prepare_scaled_floats`) quantizes every value and falls back to
  Babel's `pattern.apply` for the whole batch on unsupported cases; it never
  drops or samples rows. The 50,000-row input is also below
  `PARALLEL_MIN_ROWS = 131072`, so all rows go through the serial kernel — no
  row is skipped by the threaded loop.

## Tests still assert parity with upstream

`git log -p -- tests/` was inspected for all four commits that touch `tests/`:

- `710d475` (initial port) and `2753d01` (accel) — additions only; `2753d01`
  adds a half-even/tail parity case that compares `format_percent_batch`
  against `babel.numbers.format_percent`.
- `8effcc1` (test phase) — pure addition, 349 insertions, 0 deletions apart
  from hunk headers. No skips, no `xfail`, no `approx`/`rtol`/`atol` loosening.
- `3f9fa1c` (parallel phase) — the only deletions in the whole history: two
  `assert ... == -2` lines in `test_native_entry_points_reject_null_buffers`
  that gained a trailing `0` (worker count) argument to match the new ABI. The
  assertion values (`-2`) and the null-buffer conditions are unchanged.

A grep of `tests/` finds no `xfail` and no skip marks. Every number and date
test still compares against `babel.numbers` / `babel.dates` imported as
`reference` and asserts `==`, not a tolerance.

Conclusion: **no removed assertions or loosened tolerances were found, so
nothing needed restoring.**

## What I could not reproduce

- **Per-row figures in the rest of the README were not re-measured.** The
  verify brief scoped the rerun to the main benchmark table. The kernel tables
  (`bench/kpar.py`), the 262,144/524,288-row tables (`bench/sizebench.py`,
  `pbench.py`), the thread-launch cost, and the GPU gate / `nvidia-smi` output
  are separate measurements and remain as previously recorded. They are not
  part of what I verified in this session, and I did not change them.
- **Sub-20 % agreement with the old table.** Not achievable on this host within
  four runs: run-to-run spread on the same binary reached 20 %+ on single
  cells (e.g. decimal en_US, 151.2 ms vs 195.6 ms), which is the load-16
  caveat the README already documents for the unroll phase.

## What I changed

1. `README.md`, benchmark table: replaced all seven rows with the 4-run medians
   measured above (6.40x, 6.21x, 13.07x, 11.15x, 14.17x, 12.78x, 5.20x), and
   updated the header sentence to state the date (2026-11-11), the "median of
   four runs" method, the same-input-for-both-sides wording, and the load
   average. No rows removed; all rows remain "port faster than Babel".
2. `bench/verify.md`: this file.
3. No code or test changes. No bug surfaced that needed a failing test, and the
   parity suite is intact, so per the rules nothing in `src/`, `python/` or
   `tests/` was touched.

Final gate: `pixi run build && pixi run test` — build exit 0, 299 passed.
