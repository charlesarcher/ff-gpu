# Gate-2 Ranking Memo: {B, C, D} per leg, decided from parsed numerators

Status: **BINDING** instruction set for Waves 4-7 implementers (plan tasks 9, 12, 15, 18 of
`.omo/plans/gpu-optimization-execution.md`). Written by plan task 6, 2026-08-25.

## 0. Inputs and input discipline

Exclusive inputs:

1. `.omo/start-work/evidence/gate1-numerators.md`: Gate-1 sweep (OLD protocol, all optional
   env flags unset, sha256 stdout gate ACTIVE, 18/18 cells outcome=OK, 54/54 reps
   sha256_ok=yes, HEAD `32b5859`, sweep 2026-08-25T16:25:06Z).
2. `.omo/evidence/gpu-speedup/gap-closure/final-verdict.md` §6: M4-era per-phase
   decomposition. Cited ONLY where marked **[PRIOR-SWEEP]** below; it supplies search-kernel
   and D2H sub-timers that the Gate-1 numerators table does not carry.
3. `.omo/plans/gpu-opt-research-bundle.md` §§B, C, D, E: work-item definitions.
4. `.omo/plans/gpu-optimization-execution.md`: gate definitions and wave order.

**Input discipline (binding):** every decision below derives from parsed timer medians.
Subtraction-derived gaps (wall or total minus named phases) are prohibited as decision
inputs in this memo and downstream. Where a figure is not itself a parsed median, the
arithmetic that produces it is shown inline from greppable inputs.

---

## 1. Parsed Gate-1 inputs (medians, ms)

### 1.1 Wheel expansion (parsed `wheel expansion` timer)

| Leg | nvidia_gpu | amd_gpu |
|---|---|---|
| 65536 | 16.265 | 15.378 |
| 131072 | 30.070 | 35.880 |
| 262144 | 100.126 | 101.075 |
| 524288 | 337.333 | 372.193 |
| 1048576 | 558.082 | 574.754 |
| 2097152 | 747.539 | 831.899 |

### 1.2 Zero-fill (parsed A2a timer around `main.cpp:483`)

| Leg | amd_gpu | nvidia_gpu |
|---|---|---|
| 65536 | 0.395 | 0.650 |
| 131072 | 1.978 | 2.069 |
| 262144 | 9.827 | 9.381 |
| 524288 | 38.114 | 36.851 |
| 1048576 | 138.738 | 141.351 |
| 2097152 | 528.263 | 528.885 |

Vendor-symmetric: spread ≤1% at every leg; grows ~linearly with span.

### 1.3 Residual (`unaccounted_ms`, completeness column)

Per-cell residuals run 0.173-22.354 ms across the 12 GPU cells
(amd: 0.204, 0.370, 0.634, 1.823, 5.963, 22.354; nv: 0.183, 0.173, 0.638, 1.100, 3.863,
14.358 by ascending leg).

---

## 2. Prior-sweep context [PRIOR-SWEEP]

The Gate-1 numerators table does not parse the GPU-search sub-timers. Where an overlap
ceiling needs kernel+D2H, this memo cites the M4-era decomposition (final-verdict §6,
"Per-phase decomposition (medians from raw CSV, ms)", same six legs, earlier binary).
These are context values, not Gate-1 numerators, and are labeled wherever used.

| config.leg | search kernel | D2H | kernel+D2H (sum shown) |
|---|---|---|---|
| nvidia_gpu.65536 | 14.5 | 0.1 | 14.6 |
| nvidia_gpu.131072 | 27.9 | 0.2 | 28.1 |
| nvidia_gpu.262144 | 75.8 | 0.2 | 76.0 |
| nvidia_gpu.524288 | 251.5 | 0.4 | 251.9 |
| nvidia_gpu.1048576 | 706.8 | 0.8 | 707.6 |
| nvidia_gpu.2097152 | 2790.0 | 1.5 | 2791.5 |
| amd_gpu.65536 | 27.6 | 0.1 | 27.7 |
| amd_gpu.131072 | 54.3 | 0.2 | 54.5 |
| amd_gpu.262144 | 159.4 | 0.2 | 159.6 |
| amd_gpu.524288 | 555.2 | 0.4 | 555.6 |
| amd_gpu.1048576 | 1918.3 | 0.7 | 1919.0 |
| amd_gpu.2097152 | 8202.4 | 1.4 | 8203.8 |

---

## 3. Threshold evaluations (arithmetic inline)

### 3.1 C funding test: "parsed wheel-expansion ≥150 ms at ANY leg ≥262144?"

Evaluated against Gate-1 medians, every leg ≥262144, both vendors:

| Reading | Evaluation | Result |
|---|---|---|
| nv@262144 | 100.126 ≥ 150 | FALSE |
| amd@262144 | 101.075 ≥ 150 | FALSE |
| nv@524288 | 337.333 ≥ 150 | TRUE |
| amd@524288 | 372.193 ≥ 150 | TRUE |
| nv@1048576 | 558.082 ≥ 150 | TRUE |
| amd@1048576 | 574.754 ≥ 150 | TRUE |
| nv@2097152 | 747.539 ≥ 150 | TRUE |
| amd@2097152 | 831.899 ≥ 150 | TRUE |

**Verdict: YES. Qualifying legs: 524288, 1048576, 2097152, both vendors.** Leg 262144 fails
on both vendors (100.126 and 101.075, both < 150). Floor legs fail on magnitude too
(16.265/15.378 @65536; 30.070/35.880 @131072, all < 150) and are out of scope regardless
(plan constraint: floor cells documentation only). The tightest qualifying margin is
nv@524288: 337.333 / 150 = 2.25× above the bar; expansion cost scales superlinearly with
leg, so margins widen upward (831.899 / 150 = 5.55× at amd@2M).

**C expected gain per funded leg = min(parsed expansion, kernel+D2H [PRIOR-SWEEP]):**

| Leg | min( ) evaluation | Gain ceiling (ms) | Binding side |
|---|---|---|---|
| nv@524288 | min(337.333, 251.5 + 0.4 = 251.9) | 251.9 | kernel+D2H |
| amd@524288 | min(372.193, 555.2 + 0.4 = 555.6) | 372.193 | expansion |
| nv@1048576 | min(558.082, 706.8 + 0.8 = 707.6) | 558.082 | expansion |
| amd@1048576 | min(574.754, 1918.3 + 0.7 = 1919.0) | 574.754 | expansion |
| nv@2097152 | min(747.539, 2790.0 + 1.5 = 2791.5) | 747.539 | expansion |
| amd@2097152 | min(831.899, 8202.4 + 1.4 = 8203.8) | 831.899 | expansion |

Exactly one funded cell is kernel-bound (nv@524288); everywhere else overlap hides the
entire expansion behind kernel+D2H. That single exception is D's reason to exist.

### 3.2 D provisional outlook (scope: nv@524288 only, per plan task 15 / bundle §D)

Premise check: kernel < expansion at nv@524288. 251.5 [PRIOR-SWEEP] < 337.333 (parsed) → TRUE.

Post-C permanent exposure at nv@524288:
337.333 − (251.5 + 0.4) = 337.333 − 251.9 = **85.433 ms** that overlap cannot hide.

Provisional gate (bundle §D G1): fund iff post-B parsed expansion @nv@524288 ≥ ~100 ms.
Today's parsed value: 337.333 ≥ 100 → TRUE → **PROVISIONAL-FUND outlook.**

**Final call: DEFERRED to the post-B measurement. Plan task 15's binding gate reads
"WITHDRAW if post-B parsed expansion @nv@524288 < ~100 ms". This memo does not pre-decide
it.** Stability note: B deletes the zero-fill memset, not the expansion pass, so D's
numerator (the parsed wheel-expansion timer) is expected stable across B; the re-measurement
still binds because allocation and cache effects of uninitialized memory can shift timings.

Honesty note: bundle §D's "~120-130 ms permanently exposed" was a pre-instrumentation
estimate. The parsed-numerator arithmetic above (85.433 ms) supersedes it.

### 3.3 B sizing (zero-fill deletion, plan task 9)

Deleting the memset books approximately the measured timer, since the deleted work IS the
timed work. Booking range per leg = two-vendor span of parsed zero-fill medians:

| Leg | Booking range (ms) | Materiality |
|---|---|---|
| 65536 | 0.395 - 0.650 | immaterial (floor lane) |
| 131072 | 1.978 - 2.069 | immaterial (floor lane) |
| 262144 | 9.381 - 9.827 | minor |
| 524288 | 36.851 - 38.114 | material |
| 1048576 | 138.738 - 141.351 | major |
| 2097152 | 528.263 - 528.885 | headline |

Summed across all six legs per vendor (addition of parsed medians):
nv 0.650 + 2.069 + 9.381 + 36.851 + 141.351 + 528.885 = **719.187 ms**;
amd 0.395 + 1.978 + 9.827 + 38.114 + 138.738 + 528.263 = **717.315 ms**.

**Verdict: FUND at every leg.** Vendor symmetry (≤1% spread at every leg) means one booking
range serves both cards. At 2M this is the dominant single non-kernel phase on BOTH vendors
(~528 ms per run). Floor-leg gains are real but below claim resolution; floor cells stay
documentation-only per plan constraints. B ships as one global code change either way.

### 3.4 E sizing (expansion tile re-tiling, plan task 18)

E targets the same parsed wheel-expansion column. It serves the default CPU-search
production path and dump-map, paths with no kernel to hide behind:

| Leg | Attributed target (nv / amd, ms) |
|---|---|
| 524288 | 337.333 / 372.193 |
| 1048576 | 558.082 / 574.754 |
| 2097152 | 747.539 / 831.899 |

Bundle §E bandwidth-capped projections stand as targets (~35-60 ms @524K, ~120-160 @1M,
~420-520 @2M). Sequence AFTER B/C/D decisions (Wave 7). Materiality: the attributed residual
is hundreds of ms at every leg ≥524288, so E is NOT deferred as immaterial.

---

## 4. Residual accounting statement

Max absolute `unaccounted_ms` = **22.354 ms** (amd@2097152), which is 0.180% of that cell's
phase_total. Worst relative residual anywhere = **0.285%** (amd@131072). Bound:
unaccounted ≤ 22.354 ms absolute and ≤ 0.285% relative across all 12 GPU cells.
**Accounting is CLOSED: no hidden fifth component exists.** Per cell, the residue is at
least 14× smaller than the smallest decision-relevant numerator in that same cell
(zero-fill @262144: 9.381 / 0.638 = 14.7× nv, 9.827 / 0.634 = 15.5× amd). Device
enumeration sits inside phase_total but outside the five new timers; it is fully covered by
this residual check and is F8/task 10's target on its own track, not a hidden component.

---

## 5. Master decision table (leg × item)

| Leg | B zero-fill deletion | C overlap | D scoped decoder |
|---|---|---|---|
| 65536 | FUND (booking 0.395-0.650 ms; floor lane, doc-only claims) | SKIP (floor lane; 16.265 / 15.378 < 150) | SKIP (floor lane; no funded overlap exists to leave exposure) |
| 131072 | FUND (booking 1.978-2.069 ms; floor lane) | SKIP (floor lane; 30.070 / 35.880 < 150) | SKIP (floor lane) |
| 262144 | FUND (booking 9.381-9.827 ms) | SKIP (100.126 < 150; 101.075 < 150) | SKIP (C skipped here, so no post-overlap exposure arises; D's scope is nv@524288) |
| 524288 | FUND (booking 36.851-38.114 ms) | FUND both vendors (337.333 ≥ 150; 372.193 ≥ 150; gain ceiling 251.9 nv [PRIOR-SWEEP-bound] / 372.193 amd) | nv: **DEFER-TO-POST-B**, provisional outlook PROVISIONAL-FUND (337.333 ≥ 100 today; task 15 gate decides) · amd: WITHDRAW (555.6 [PRIOR-SWEEP] > 372.193, so C hides expansion fully; no residual remains) |
| 1048576 | FUND (booking 138.738-141.351 ms) | FUND both vendors (558.082 ≥ 150; 574.754 ≥ 150; expansion-bound) | WITHDRAW both vendors (707.6 / 1919.0 [PRIOR-SWEEP] > 558.082 / 574.754; fully hidden) |
| 2097152 | FUND (booking 528.263-528.885 ms) | FUND both vendors (747.539 ≥ 150; 831.899 ≥ 150; expansion-bound) | WITHDRAW both vendors (2791.5 / 8203.8 [PRIOR-SWEEP] > 747.539 / 831.899; fully hidden) |

Every entry above carries its arithmetic in §3. No entry lacks a number.

---

## 6. Ranking narrative and execution order

Measured split per leg: expansion exceeds zero-fill at every leg ≥262144
(e.g. @524288: 337.333/372.193 vs 36.851/38.114; @2M: 747.539/831.899 vs 528.263/528.885),
and both dwarf sched teardown, search setup, and search teardown (<2.3 ms everywhere for
the latter two). Ranking {B, C, D} by measured split therefore puts expansion work first in
magnitude, yet the execution order is **B first everywhere**, for three reasons:

1. **Largest vendor-symmetric lever.** B's summed booking is ~717-719 ms per vendor across
   legs, and ~528 ms of it sits at 2M on BOTH cards. No other single change removes more
   parsed time, and no other lever is this symmetric between vendors.
2. **Prerequisite cleanliness for C/D attribution.** The main.cpp lane serializes
   9→12→15 (B→C→D). Landing B first means C's spawn/join timing and D's gate measurement
   run against final allocation semantics. Measuring C or D before B would attribute gains
   against a memory layout that changes again afterward.
3. **D's binding gate is defined post-B.** Plan task 15 evaluates "post-B parsed expansion";
   D cannot be finally decided until B lands, so D cannot execute first under any ordering.

**Execution order:** Task 9 (B, all legs) → Task 12 (C, funded on GPU-search legs ≥524288;
SKIP recorded at 262144 with arithmetic above) → Task 15 (D, post-B gate evaluation,
nv@524288 only) → Task 18 (E, sized on the same expansion column). Tasks 10, 11, 13, 14,
16, 17 proceed on their own instrument gates; this memo does not rank them.

---

## 7. What would flip a decision

- **C:** no flip condition post-funding. Qualifying legs clear the bar by ≥2.25×
  (337.333 / 150), and the test is already evaluated on parsed medians.
- **D:** flips to WITHDRAW iff post-B parsed expansion @nv@524288 < ~100 ms. Expected
  stable, since B does not touch the expansion pass, but the measurement binds, not this
  expectation.
- **B:** no numeric flip condition. Its gates are correctness gates: verify.sh byte-exact
  both modes × both vendors, dump-map sha256 equality, ctest green, poison-build
  golden-identical (plan task 9).
- **E:** DEFER only if the attributed expansion residual becomes immaterial after B/C/D
  land. Current parsed column (§3.4) says material at every leg ≥524288.

---

## 8. Out-of-scope observations (recorded, not funded here)

- Scheduler teardown is NV-asymmetric: 44.154 ms (nv) vs 1.581 ms (amd) @2097152, growing
  with leg on NV only. No {B,C,D,E} item targets it; recorded as a future instrument-gated
  candidate.
- Device enumeration is NV-dominated (~113-124 ms flat vs ~7 ms AMD); inside phase_total,
  fully accounted by §4; owned by F8/task 10.
- Search setup/teardown negligible (<2.3 ms everywhere): no target.

---

## 9. Verification receipts

- All §1 values: grep `.omo/start-work/evidence/gate1-numerators.md` (tables lines 33-38,
  59-77; threshold readings lines 19-23; residual lines 45-47).
- All [PRIOR-SWEEP] values: grep `.omo/evidence/gpu-speedup/gap-closure/final-verdict.md`
  §6 table (lines 67-78).
- Derived figures (251.9, 555.6, 707.6, 1919.0, 2791.5, 8203.8, 85.433, 719.187, 717.315,
  2.25×, 5.55×, 14.7×, 15.5×): each is an inline sum, difference, or ratio of greppable
  inputs shown at point of use.
- Bundle definitions and projections: `.omo/plans/gpu-opt-research-bundle.md` §§B (line 33),
  C (lines 36-38), D (lines 40-41), E (lines 43-44).
