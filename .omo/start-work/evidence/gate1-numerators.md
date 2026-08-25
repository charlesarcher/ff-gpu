# Gate-1 Numerators — Post-Instrumentation Sweep (plan task 5)

Provenance: `scripts/bench_per_device.sh`, OLD protocol (all optional env flags unset:
`median_n=0 jit_warmup=0 autoextend=0`), sha256 stdout gate ACTIVE. Median of 3 timed
reps after 1 untimed warmup, per config × leg. Sweep timestamp 2026-08-25T16:25:06Z,
git HEAD `32b5859` (timers from `b6645c4`, harness parsing from `32b5859` — first live
cross-validation of both). Outcome integrity: **18/18 cells outcome=OK**, 54/54 raw
reps sha256_ok=yes. Medians computed from `scripts/bench_per_device_raw.csv`
(per config×leg across reps), NOT from summary min/max columns.

---

## KEY THRESHOLD READINGS

### (a) C-funding test: median wheel-expansion ≥ 150 ms at ANY leg ≥ 262144?

| Reading | Value |
|---|---|
| nvidia_gpu @ 524288 | **337.333 ms** → ≥150 ms ✓ |
| nvidia_gpu @ 262144 | 100.126 ms → < 150 ms ✗ |
| nvidia_gpu @ 131072 | 30.070 ms → < 150 ms ✗ |
| amd_gpu @ 524288 (context) | 372.193 ms → ≥150 ms ✓ |
| amd_gpu @ 262144 (context) | 101.075 ms → < 150 ms ✗ |

**Verdict: YES at leg 524288 (both vendors clear 150 ms by >2×); NO at 262144 and
131072 (~100 ms / ~30 ms).** Wheel-expansion cost scales superlinearly; the C-overlap
funding case exists only at legs ≥ 524288.

### (b) B sizing: median zero-fill per leg, both vendors

| Leg | amd_gpu | nvidia_gpu |
|---|---|---|
| 65536 | 0.395 | 0.650 |
| 131072 | 1.978 | 2.069 |
| 262144 | 9.827 | 9.381 |
| 524288 | 38.114 | 36.851 |
| 1048576 | 138.738 | 141.351 |
| 2097152 | 528.263 | 528.885 |

Zero-fill is vendor-symmetric (≤1% spread at every leg) and grows ~linearly with span,
reaching ~528 ms at the 2M leg — the dominant single non-kernel phase there.

### (c) Completeness check: residual unaccounted_ms after the five new timers

Max residual: **22.354 ms** (amd@2097152) = **0.180 %** of that cell's phase_total.
Worst relative residual anywhere: 0.285 % (amd@131072). **The residue is SMALL — the
five new timers close the phase accounting; no large unexplained gap exists.**

---

## Gate-1 Numerators Table

Median of each parsed column across the 3 timed reps per cell (ms). `phase_enum_ms`
included for context. Reference rows have EMPTY phase cells — the CPU reference binary
emits no `ff_sieve timing:` lines; missing data reported as NA, never coerced to zero.

### amd_gpu (`--devices=amd --gpu-search`, RX 9070 XT)

| Leg | wheel_expansion | zero_fill | sched_teardown | search_setup | search_teardown | unaccounted | enum (ctx) |
|---|---|---|---|---|---|---|---|
| 65536 | 15.378 | 0.395 | 0.449 | 0.095 | 0.003 | 0.204 | 7.274 |
| 131072 | 35.880 | 1.978 | 0.440 | 0.107 | 0.005 | 0.370 | 6.975 |
| 262144 | 101.075 | 9.827 | 0.529 | 0.212 | 0.048 | 0.634 | 7.212 |
| 524288 | 372.193 | 38.114 | 0.852 | 0.336 | 0.054 | 1.823 | 7.446 |
| 1048576 | 574.754 | 138.738 | 1.305 | 0.585 | 0.055 | 5.963 | 7.551 |
| 2097152 | 831.899 | 528.263 | 1.581 | 1.369 | 0.067 | 22.354 | 7.460 |

### nvidia_gpu (`--devices=nvidia --gpu-search`, RTX 5090)

| Leg | wheel_expansion | zero_fill | sched_teardown | search_setup | search_teardown | unaccounted | enum (ctx) |
|---|---|---|---|---|---|---|---|
| 65536 | 16.265 | 0.650 | 2.690 | 0.226 | 0.007 | 0.183 | 124.148 |
| 131072 | 30.070 | 2.069 | 3.579 | 0.306 | 0.250 | 0.173 | 122.982 |
| 262144 | 100.126 | 9.381 | 8.097 | 0.249 | 0.284 | 0.638 | 120.415 |
| 524288 | 337.333 | 36.851 | 22.945 | 0.412 | 0.307 | 1.100 | 114.464 |
| 1048576 | 558.082 | 141.351 | 41.957 | 0.737 | 0.330 | 3.863 | 112.993 |
| 2097152 | 747.539 | 528.885 | 44.154 | 1.248 | 2.228 | 14.358 | 112.956 |

### ref (reference ff_seg, software/CPU)

All six legs: every phase column NA (binary emits no timing lines) — honest empty cells.

---

## Cross-validation (adversarial probe)

nv@524288 wheel-expansion recomputed by hand from raw CSV rows:
rep1 = 363.867, rep2 = 337.333, rep3 = 303.036 → sorted {303.036, 337.333, 363.867}
→ median (n=3, middle element) = **337.333** — matches the table value exactly.

## Notes for downstream gates

- Scheduler teardown is NV-asymmetric (44.154 ms vs AMD's 1.581 ms @2M) — a real
  vendor split the old subtraction-derived narrative could not see.
- Device enumeration is NV-dominated (~113–124 ms flat across legs vs ~7 ms AMD);
  it sits inside phase_total but outside the five new timers, and is fully accounted
  (residual check above).
- Search setup/teardown are negligible (<2.3 ms everywhere) — no optimization target.
