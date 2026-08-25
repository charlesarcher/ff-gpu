# Re-Baseline Dual Report — New Protocol vs Old Protocol (plan task 7)

ONE freeze event. Old side: Gate-1 sweep at commit `d01d9dc` (OLD protocol:
`median_n=0 jit_warmup=0 autoextend=0`, sha256 gate active, median-of-3).
New side: single re-baseline sweep of 2026-08-25T16:50:49Z with ACTIVATED defaults
(`sha_gate=on median_n=5 jit_warmup=1 autoextend=1`), same binary tree (`74c5b20` +
knob-default flip only; no source/ changes between sweeps).

## nvidia-persistenced prereq (verbatim transcript)

```
=== persistence_mode BEFORE ===
2026-08-25T16:49:06Z
Disabled
--- systemctl status (pre) ---
disabled
inactive
=== sudo -n systemctl enable --now nvidia-persistenced ===
sudo: a password is required
rc=1
=== persistence_mode AFTER ===
2026-08-25T16:49:16Z
Disabled
--- systemctl status (post) ---
disabled
inactive
```

**Verdict: BLOCKED-sudo** — passwordless sudo unavailable; enablement NOT performed
(not fabricated). persistence_mode remained `Disabled` before AND after the attempt.
The freeze therefore does NOT include any persistenced effect; enabling it later is a
separate protocol change requiring its own re-baseline.

## Wall-clock medians (s): OLD vs NEW per config × leg

delta% = (new − old)/old × 100. Negative = faster under new protocol.
| Config | Leg | old med (s) | new med (s) | Δ% | old reps | new reps | outcome |
|---|---|---|---|---|---|---|---|
| `ref` | 65536 | 0.018 | 0.019 | +5.6% | 3 | 5 | OK |
| `ref` | 131072 | 0.083 | 0.088 | +6.0% | 3 | 7 | OK |
| `ref` | 262144 | 0.429 | 0.422 | -1.6% | 3 | 7 | OK |
| `ref` | 524288 | 2.194 | 2.245 | +2.3% | 3 | 5 | OK |
| `ref` | 1048576 | 9.315 | 9.407 | +1.0% | 3 | 3 | OK |
| `ref` | 2097152 | 41.020 | 41.105 | +0.2% | 3 | 3 | OK |
| `amd_gpu` | 65536 | 0.111 | 0.101 | -9.0% | 3 | 5 | OK |
| `amd_gpu` | 131072 | 0.156 | 0.157 | +0.6% | 3 | 7 | OK |
| `amd_gpu` | 262144 | 0.361 | 0.357 | -1.1% | 3 | 7 | OK |
| `amd_gpu` | 524288 | 1.148 | 1.151 | +0.3% | 3 | 5 | OK |
| `amd_gpu` | 1048576 | 3.320 | 3.259 | -1.8% | 3 | 3 | OK |
| `amd_gpu` | 2097152 | 12.469 | 12.467 | -0.0% | 3 | 3 | OK |
| `nvidia_gpu` | 65536 | 0.254 | 0.252 | -0.8% | 3 | 5 | OK |
| `nvidia_gpu` | 131072 | 0.285 | 0.296 | +3.9% | 3 | 5 | OK |
| `nvidia_gpu` | 262144 | 0.442 | 0.445 | +0.7% | 3 | 5 | OK |
| `nvidia_gpu` | 524288 | 0.982 | 0.983 | +0.1% | 3 | 5 | OK |
| `nvidia_gpu` | 1048576 | 1.984 | 1.984 | +0.0% | 3 | 3 | OK |
| `nvidia_gpu` | 2097152 | 5.469 | 5.528 | +1.1% | 3 | 3 | OK |

### >±10% movers

- none

**nv@65536 callout (pre-registered expectation):** the task expected sub-second
legs to tighten under median-of-5 + JIT warmup. nvidia_gpu@65536 moved -0.8% (0.254 s → 0.252 s) — within ±10%,
consistent with an init-floor-dominated cell (~113–124 ms device enumeration)
where JIT warmup shaves first-block jitter but cannot move the enumeration floor.

## Phase numerators (ms, median over reps): OLD vs NEW

`NA` = timer absent for that cell (reference emits no `ff_sieve timing:` lines).

### amd_gpu

| Leg | metric | old | new | Δ% |
|---|---|---|---|---|
| 65536 | wheel_expansion | 15.378 | 15.618 | +1.6% |
| 65536 | zero_fill | 0.395 | 0.497 | +25.8% **>±10%** |
| 65536 | sched_teardown | 0.449 | 0.368 | -18.0% **>±10%** |
| 65536 | search_setup | 0.095 | 0.125 | +31.6% **>±10%** |
| 65536 | search_teardown | 0.003 | 0.004 | +33.3% **>±10%** |
| 65536 | unaccounted | 0.204 | 0.213 | +4.4% |
| 65536 | enum_ctx | 7.274 | 7.499 | +3.1% |
| 131072 | wheel_expansion | 35.880 | 35.307 | -1.6% |
| 131072 | zero_fill | 1.978 | 2.266 | +14.6% **>±10%** |
| 131072 | sched_teardown | 0.440 | 0.387 | -12.0% **>±10%** |
| 131072 | search_setup | 0.107 | 0.112 | +4.7% |
| 131072 | search_teardown | 0.005 | 0.005 | +0.0% |
| 131072 | unaccounted | 0.370 | 0.335 | -9.5% |
| 131072 | enum_ctx | 6.975 | 7.337 | +5.2% |
| 262144 | wheel_expansion | 101.075 | 97.925 | -3.1% |
| 262144 | zero_fill | 9.827 | 9.894 | +0.7% |
| 262144 | sched_teardown | 0.529 | 0.539 | +1.9% |
| 262144 | search_setup | 0.212 | 0.184 | -13.2% **>±10%** |
| 262144 | search_teardown | 0.048 | 0.050 | +4.2% |
| 262144 | unaccounted | 0.634 | 0.665 | +4.9% |
| 262144 | enum_ctx | 7.212 | 7.446 | +3.2% |
| 524288 | wheel_expansion | 372.193 | 361.397 | -2.9% |
| 524288 | zero_fill | 38.114 | 38.230 | +0.3% |
| 524288 | sched_teardown | 0.852 | 0.863 | +1.3% |
| 524288 | search_setup | 0.336 | 0.362 | +7.7% |
| 524288 | search_teardown | 0.054 | 0.056 | +3.7% |
| 524288 | unaccounted | 1.823 | 1.922 | +5.4% |
| 524288 | enum_ctx | 7.446 | 7.263 | -2.5% |
| 1048576 | wheel_expansion | 574.754 | 549.624 | -4.4% |
| 1048576 | zero_fill | 138.738 | 139.323 | +0.4% |
| 1048576 | sched_teardown | 1.305 | 1.283 | -1.7% |
| 1048576 | search_setup | 0.585 | 0.658 | +12.5% **>±10%** |
| 1048576 | search_teardown | 0.055 | 0.056 | +1.8% |
| 1048576 | unaccounted | 5.963 | 6.090 | +2.1% |
| 1048576 | enum_ctx | 7.551 | 7.360 | -2.5% |
| 2097152 | wheel_expansion | 831.899 | 757.742 | -8.9% |
| 2097152 | zero_fill | 528.263 | 502.527 | -4.9% |
| 2097152 | sched_teardown | 1.581 | 1.568 | -0.8% |
| 2097152 | search_setup | 1.369 | 1.220 | -10.9% **>±10%** |
| 2097152 | search_teardown | 0.067 | 0.068 | +1.5% |
| 2097152 | unaccounted | 22.354 | 24.209 | +8.3% |
| 2097152 | enum_ctx | 7.460 | 8.991 | +20.5% **>±10%** |

### nvidia_gpu

| Leg | metric | old | new | Δ% |
|---|---|---|---|---|
| 65536 | wheel_expansion | 16.265 | 16.021 | -1.5% |
| 65536 | zero_fill | 0.650 | 0.620 | -4.6% |
| 65536 | sched_teardown | 2.690 | 2.757 | +2.5% |
| 65536 | search_setup | 0.226 | 0.137 | -39.4% **>±10%** |
| 65536 | search_teardown | 0.007 | 0.008 | +14.3% **>±10%** |
| 65536 | unaccounted | 0.183 | 0.155 | -15.3% **>±10%** |
| 65536 | enum_ctx | 124.148 | 123.456 | -0.6% |
| 131072 | wheel_expansion | 30.070 | 31.095 | +3.4% |
| 131072 | zero_fill | 2.069 | 2.131 | +3.0% |
| 131072 | sched_teardown | 3.579 | 3.723 | +4.0% |
| 131072 | search_setup | 0.306 | 0.327 | +6.9% |
| 131072 | search_teardown | 0.250 | 0.267 | +6.8% |
| 131072 | unaccounted | 0.173 | 0.197 | +13.9% **>±10%** |
| 131072 | enum_ctx | 122.982 | 123.720 | +0.6% |
| 262144 | wheel_expansion | 100.126 | 101.621 | +1.5% |
| 262144 | zero_fill | 9.381 | 8.781 | -6.4% |
| 262144 | sched_teardown | 8.097 | 8.168 | +0.9% |
| 262144 | search_setup | 0.249 | 0.286 | +14.9% **>±10%** |
| 262144 | search_teardown | 0.284 | 0.287 | +1.1% |
| 262144 | unaccounted | 0.638 | 0.466 | -27.0% **>±10%** |
| 262144 | enum_ctx | 120.415 | 123.559 | +2.6% |
| 524288 | wheel_expansion | 337.333 | 356.224 | +5.6% |
| 524288 | zero_fill | 36.851 | 33.256 | -9.8% |
| 524288 | sched_teardown | 22.945 | 23.635 | +3.0% |
| 524288 | search_setup | 0.412 | 0.458 | +11.2% **>±10%** |
| 524288 | search_teardown | 0.307 | 0.313 | +2.0% |
| 524288 | unaccounted | 1.100 | 1.159 | +5.4% |
| 524288 | enum_ctx | 114.464 | 114.415 | -0.0% |
| 1048576 | wheel_expansion | 558.082 | 565.851 | +1.4% |
| 1048576 | zero_fill | 141.351 | 139.938 | -1.0% |
| 1048576 | sched_teardown | 41.957 | 43.258 | +3.1% |
| 1048576 | search_setup | 0.737 | 0.763 | +3.5% |
| 1048576 | search_teardown | 0.330 | 0.343 | +3.9% |
| 1048576 | unaccounted | 3.863 | 3.951 | +2.3% |
| 1048576 | enum_ctx | 112.993 | 114.800 | +1.6% |
| 2097152 | wheel_expansion | 747.539 | 868.020 | +16.1% **>±10%** |
| 2097152 | zero_fill | 528.885 | 530.066 | +0.2% |
| 2097152 | sched_teardown | 44.154 | 44.381 | +0.5% |
| 2097152 | search_setup | 1.248 | 1.223 | -2.0% |
| 2097152 | search_teardown | 2.228 | 2.264 | +1.6% |
| 2097152 | unaccounted | 14.358 | 14.515 | +1.1% |
| 2097152 | enum_ctx | 112.956 | 114.625 | +1.5% |

### ref

All six legs: every phase column NA both sides (binary emits no timing lines).

### Reading the >±10% phase flags

Flags on sub-5 ms cells (`zero_fill` @65K/131K, `search_setup`, `search_teardown`,
`unaccounted`) are microsecond-scale timer noise — no mechanism attaches to them and
they are invisible in the wall-median table above. Two flags sit at meaningful scale
and are recorded honestly:

- `nvidia_gpu@2097152 wheel_expansion` 747.539 → 868.020 ms (**+16.1%**): median-of-3
  vs median-of-3. Old reps {642.106, 747.539, 872.154}, new reps {866.940, 868.020,
  941.578} — the shift is at the edge of the old sweep's own ±17% rep envelope. The
  C-funding arithmetic (≥150 ms at any leg ≥262144) is unaffected: both vendors clear
  it by >2× under either table.
- `amd_gpu@2097152 wheel_expansion` 831.899 → 757.742 ms (−8.9%): old reps
  {805.343, 831.899, 903.817}, new reps {695.498, 757.742, 993.334} — heavy overlap;
  same high run-to-run variance regime at the 2M leg.

The NEW column is the frozen numerator reference going forward.

## Protocol-deviation disclosure (autoextend bug, found & fixed post-sweep)

The activated autoextend path had never executed before this sweep and crashed on
its first use (`set -u`): it referenced stale global `$med` instead of `$_med`.
Verbatim from `run/rebaseline_sweep.log`:

```
scripts/bench_per_device.sh: line 406: med: unbound variable
scripts/bench_per_device.sh: line 407: ((: == 1 : arithmetic syntax error: operand expected (error token is "== 1 ")
```

Consequences, fully bounded:

- `ref@65536`: autoextend aborted after the crash → plain median-of-5 (intended N).
- Every later cell that entered autoextend compared its spread against the PREVIOUS
  cell's median, not its own → up to 2 extra reps were added by the wrong criterion
  (e.g. `ref@524288` extended despite 4.2 % own-spread). Cells whose spread relative
  to the stale median stayed ≤0.15 correctly did not extend.
- ALL reps — base, median-of-N, and mis-triggered extras — are valid timed reps that
  passed rc==0, solution-count, device-attribution and sha256 gates; every quoted
  median is computed over all collected reps of its own cell. No number above is
  fabricated or extrapolated.
- Fix committed in the same atomic change: `$med` → `$_med`; logic verified in both
  directions post-fix (tight set → over=0, spread set (0.9,1.2,1.0) → over=1).
- ONE-freeze mandate honored: the sweep was NOT re-run; this sweep is the frozen
  reference. Future sweeps get correct autoextend semantics.

## Outcome integrity

- Summary cells outcome=OK: **18/18** (0 DEFECT, 0 EXPECTED_GATE_REFUSAL).
- Raw reps sha256_ok=yes: **86/86** (gate live all sweep long).
- Sweep rc=0; console log preserved at `run/rebaseline_sweep.log` (gitignored scratch).

## sha256 fault-injection demo (adversarial probe: a gate that never fires is decoration)

Setup: pristine copy of `goldens/out_ff_seg_65536.txt` taken to `/tmp/opencode/rebaseline/`
(repo `goldens/` NEVER touched). Gate logic replicated verbatim from the harness:
`sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/' | sha256sum`. The goldens store
timing lines pre-normalized (`Prime time: N μs`), which is what real binary stdout is
normalized onto.

```
$ norm() { sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/' "$1" | sha256sum | cut -d' ' -f1; }
$ GOLDEN_SHA=$(norm pristine.txt); echo "GATE precomputed golden sha256 = $GOLDEN_SHA"
GATE precomputed golden sha256 = bf17f62c9a0efce657d9c0c1e42a42cc5714687d1274a8788d9ffd2bf6d2a24e

# case 1 — pristine copy vs gate
$ P=$(norm pristine.txt); [[ "$P" == "$GOLDEN_SHA" ]] && echo "sha_ok=yes (pass)" || echo "sha_ok=no (DEFECT)"
sha_ok=yes (pass)

# case 2 — simulated run with REAL timing digits ("Prime time: 123456 μs"):
$ S=$(norm simulated_run.txt); [[ "$S" == "$GOLDEN_SHA" ]] && echo "sha_ok=yes (pass)" || echo "sha_ok=no (DEFECT)"
sha_ok=yes (pass — 123456 normalized to N, matches golden)

# case 3 — FAULT INJECTION: one solution digit corrupted (sum = 17 -> 18), diff proof:
  <       1) sum =       17, product =              52, ...
  >       1) sum =       18, product =              52, ...
$ M=$(norm mutated.txt); echo "mutated sha256 = $M"
mutated sha256 = 5fbb0042a947ff930810c11727e08e58d90bb8621604ab6ee61b34b966f7287f
$ [[ "$M" == "$GOLDEN_SHA" ]] && echo "sha_ok=yes (pass)" || { echo "sha_ok=no (DEFECT)"; echo "=> GATE FIRES: outcome would be marked DEFECT"; }
sha_ok=no (DEFECT)
=> GATE FIRES: outcome would be marked DEFECT

# case 4 — extra stdout line appended (line-count change):
$ T=$(norm timing_only.txt); [[ "$T" == "$GOLDEN_SHA" ]] && echo pass || echo "sha_ok=no (DEFECT)"
sha_ok=no (DEFECT)
```

Verdict: the gate discriminates exactly as designed — timing-digit jitter passes,
content corruption and structural changes fire DEFECT.

## Freeze statement

The NEW-protocol numbers above are the frozen ±3 % band reference going forward.
Bands apply to wall-median comparisons between sweeps run under the SAME activated
protocol (median-of-N≥5 sub-second, JIT warmup per block, autoextend ≤2, sha256 gate).
persistenced state at freeze time: **Disabled (BLOCKED-sudo)** — recorded so any future
enablement is recognized as a protocol change demanding a fresh re-baseline.

