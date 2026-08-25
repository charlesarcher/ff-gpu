# F2 pre-MR trial screen — counters-first diagnostic (@amd@1048576)

Task 16 (F2) STEP 1 evidence. Numbers-only instrumented build (`-DFF_SEARCH_F2_COUNTERS`,
64-slot hashed atomic counters in `dev_IsPrime`, zeroed/read back per launch in
`SearchKernelRun_<arch>`). **All timings from this build are discarded by design** — every
call pays counter atomics. Stdout verified byte-exact vs `goldens/out_ff_seg_1048576.txt`
under the canonical `(Prime|Freudenthal) time: N` normalization.

## Raw counter line (single search launch, production leg 1048576, --devices amd --gpu-search)

```
ff_sieve f2 counters: total=7904029122 mrEntries=5815229 screenReject=3578074 survive=2237155 survPrime=587458 survComp=1649697 rejT4-7=0,3578074,0,0 survT4-7=0,2237155,0,0
```

## Derived rates

| quantity | value |
|---|---|
| total dev_IsPrime calls | 7,904,029,122 |
| MR-chain entries (n > maxPrimeMapValue = 68,719,476,751 ≈ 2^36) | 5,815,229 (0.0736% of calls) |
| screen {3,5,7,11,13,17,19,23,29,31} rejects | 3,578,074 |
| **marginal reject rate r among MR entries** | **61.54%** (corrected-research band 43–69% ✓) |
| survivors | 2,237,155 = 587,458 prime (26.3%) + 1,649,697 composite liars (73.7%) |
| maxTests histogram | 100% in the maxTests=5 bucket, both rejects and survivors |

Counter semantics note (misleading-success guard): MR-entry rate (0.0736% of calls) and
candidate rate are counted as SEPARATE counters; the reject rate is computed among MR
entries only, never among all candidates. All screen rejects are composite by
construction (p ≤ 31 divides n, n > 2^36 ≫ 31²).

## Decision arithmetic (op model, gfx1201)

Per-MR-entry chain cost actually incurred before first-witness rejection:

- Montgomery setup: `dev_MontMinv` ≈ 15 + `dev_MontR2` = 128 × `MontAddMod` ≈ 512
  + `oneM`/`nm1`/`xM` init = 3 × `MontMul` ≈ 36 → **≈ 563 ops**
- first-witness exponentiation: nDiv2Odd ≈ 35–41 bits (n ∈ [2^36, ~2^42)) → ~53 `MontMul`
  × ~12 ops ≈ 640 + square-root-of-unity scan ≈ 12 → **≈ 652 ops**
- **saved per screen reject ≈ 1215 ops**
- screen cost: 10 software u64 moduli (AMD has no integer divide; SIMT divergence defeats
  short-circuit) ≈ **100–200 ops** (research figure 100–120 at the optimistic end)

Projected net per MR entry = r × saved − cost
= 0.6154 × 1215 − 150 ≈ **+598 ops** (corner band +477 … +700)

Breakeven reject rate at these costs ≈ 100–200 / 1215 ≈ **8–16%**; measured r = 61.5%
≈ 4–7× breakeven. Kernel-level projection: total net ≈ 5.815M × 598 ≈ 3.5e9 ops against
≈ 12.8e9 ops of MR-chain work + ≈ 118e9 ops of in-map fast-path work → **≈ 2–3% of
search-kernel ALU** — material enough to measure, small enough that only an interleaved
A/B may decide adoption.

## Decision (pre-registered rule → outcome)

Rule: DROP iff projected net clearly negative; A/B mandatory otherwise; adoption only on
measured median win. Projection is robustly POSITIVE (≥ 3× above its own corner-band
noise, nowhere near ±20% of breakeven) → **PROCEED TO TWO-BUILD INTERLEAVED A/B**
(clean HEAD vs screened treatment), ≥5 reps/side @1M and @2M amd gpu-search under
`flock /tmp/opencode/ff-gpu.lock`, deciding on parsed `search kernel` timers only.

Verbatim verdict at A/B close-out: see f2-ab-decision section appended below.

## A/B result appendix

Two-build interleaved A/B under `flock /tmp/opencode/ff-gpu.lock`, one untimed warmup per
side per leg, 6 timed reps/side, parsed `search kernel` stderr timers only (raw stderr:
`/tmp/opencode/f2ab_{base,treat}_{1048576,2097152}_{1..6}.err`). Base = clean HEAD
(git b409375); treat = screen-only diff in `dev_IsPrime` MR branch (`n > 961 &&
!(n%3 && … && n%31) → return false`), no counters. Treatment stdout verified byte-exact
(normalized) vs `goldens/out_ff_seg_65536.txt` before timing.

| leg | base median (ms) | treat median (ms) | Δ median | base range | treat range |
|---|---|---|---|---|---|
| @1048576 | 1859.872 | 1883.896 | **+1.29% slower** | 1826.861–1938.818 | 1824.250–1929.216 |
| @2097152 | 8010.146 | 8040.258 | **+0.38% slower** | 7882.686–8180.924 | 7847.694–8466.514 |

Pairwise rep-to-rep deltas split 3–3 on BOTH legs; distributions fully overlap → the
+598-ops/entry projection did NOT materialize as wall-clock win on gfx1201. Mechanism
(consistent with the N=16 occupancy findings): the search kernel is latency/
occupancy-bound, not ALU-bound — rejected lanes' savings evaporate into warp divergence
while the screen's ~10 software u64 mods land on every entry's critical path.

Post-A/B gates on landed state (= HEAD, treatment reverted): incremental
`cmake --build --preset dev` rc=0 zero errors; `ctest -R m4_mr_diff_bin` Passed
(buckets T:E incl. tiny-bound D/E exercised the un-screened reference-exact MR path).

## VERDICT: DROP

The pre-MR trial screen {3,5,7..31} is DROPPED by measurement: no net win at either leg
(medians +1.29% / +0.38% slower, noise-dominated). Counters justified the attempt
(r = 61.5% ≫ 8–16% breakeven) but the kernel's latency-bound regime converts projected
ALU savings into nothing. Tree unchanged beyond this evidence file; no production code
carries the screen or any diagnostic counter.

