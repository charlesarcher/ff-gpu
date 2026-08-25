# Problems — kernel-gap-closure

Unresolved blockers and technical debt discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---
## 2026-08-24 task-10 verdict run — committed ZERO-REGRESSION tier FAIL (4 cells, one structural cause)

Defect routed per task-10 mandate (no remediation inside task 10). Full tables + receipts: `.omo/evidence/gpu-speedup/gap-closure/final-verdict.md` §3/§6; raw data `task-10-kernel-gap-closure/{bench_per_device_raw.csv,rerun-raw.csv}`.

- **nv@524288 wall 0.893 → 0.985 s (+10.3% after one-clean-rerun; sweep-1 1.029)** and **nv@131072 0.270 → 0.292 s (+8.1%)**. STRUCTURAL: the wheel canonical-expansion pass (canonical-at-D2H boundary contract that preserves the dump-map sha256 golden contract) adds ~380 ms @524288 / ~30 ms @131072 inside the total timer on NV — MORE than the sieve saves at those legs (sieve itself IMPROVED 337→101 ms @524288). Same mechanism nets a big WIN at 1M/2M (NV walls −21.9%/−53.7%), so it is leg-dependent, not a kernel regression. Consequence: NV@524288 speedup 2.18x vs prior verdict 2.46x (−11.5%, outside ±3% band).
- **nv@65536 0.240 → 0.249 s (+3.75%)** and **amd@65536 0.104 → 0.112 s (+7.7%)**: floor-cell breaches at noise-scale absolute deltas (+9/+8 ms on init-floor-dominated walls); amd side is partly REAL single-digit cost from task-8's wheel decode (search kernel 21.0→27.6 ms @65K — the price of the 2M unlock), nv side is expansion+floor jitter.
- Candidate remediations for a future plan (NOT evaluated here): (a) skip/slim expansion when hostMap is never consumed post-emit (emission reads GPU results, not hostMap — verify no consumer remains), (b) overlap expansion with search phase instead of post-join serial, (c) chunk-parallel expansion beyond slab grain (task-67 noted this headroom), (d) re-score floor cells against a noise-aware band (±3% of 100–260 ms walls = ±3–8 ms, below measurement resolution).

---

## Corrections (2026-08-25, adversarial research verdict)

An adversarial research round (`.omo/plans/gpu-opt-research-bundle.md`, §Verified facts 1/3/7/10) re-examined the 2026-08-24 entries above. Those entries stand unedited as history; each item below annotates forward.

### C1. Remediation candidate (a): rationale wrong, procedure sound

Candidate (a)'s parenthetical rationale, "emission reads GPU results, not hostMap", is FACTUALLY WRONG. Emission reads the canonical hostMap via `prime.IsPrime` (`source/m4/gpu_search_emission.cpp:35`), and `GpuPrime` wraps `hostMap.data()` at construction (`source/main.cpp:566`). The procedure half, "verify no consumer remains", remains sound: executed honestly, that audit discovers the emitter consumer itself rather than licensing expansion removal. Evidence: bundle §Verified facts 10 and 3.

### C2. Magnitude claim reframed: attribution unresolved, magnitude confirmed on BOTH vendors

The "~380 ms inside NVIDIA mid-leg totals" figure (first bullet above) was subtraction-derived AND vendor-misframed; AMD pays the same slice. Restated claim: "attribution unresolved, magnitude confirmed ~370–1412 ms unattributed slice on BOTH vendors (NV 373.2/750.3/1411.6 ms and AMD 389.3/725.6/1217.0 ms @524K/1M/2M, rep2 of the 2026-08-25 raw CSV); dump receipts bound pure expansion at ≤79% (@1M) and ≤54% (@2M) of the slice; remainder includes hostMap zero-fill and scheduler teardown". Evidence: `scripts/bench_per_device_raw.csv` rep2 rows (slice = phase_total_ms minus enum/budget/sieve/search); bundle §Verified facts 1 and 10.

### C3. Floor-cell regressions: UNADJUDICABLE at current measurement resolution

nv@65536 +3.75%, nv@131072 +8.1%, amd@65536 +7.7% are marked UNADJUDICABLE at current measurement resolution (±3% of 104–292 ms walls vs demonstrated 100+ ms same-day enumeration jitter; block-first-rep inflation documented). NV's jitter is structural: ~85 ms spawn/dual-runtime-link cost outside main's total plus ~116–125 ms device enumeration inside it, against ~28 ms and ~7.5 ms on AMD. The pending harness fix (median-of-N≥5 for sub-second legs, work item A4) is the resolution path. Evidence: bundle §Verified facts 7; §Surviving work items A4.

---
