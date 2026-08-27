# FF-GPU Project Status & Issues

## Overview

This is a GPU port of the Freudenthal prime-search program (`segmentedSieve.C`). The project implements:
- **GPU sieve**: GPU-accelerated prime number sieve on RTX 5090 + RX 9070 XT
- **CPU search**: Multi-threaded Freudenthal search (31 threads, same as reference)
- **GPU search**: GPU-accelerated Freudenthal search (byte-identical to reference on ALL legs, both GPUs)

Both GPUs are driven by ONE HIP-only source set compiled twice per arch
(gfx1201 via HIP_PLATFORM=amd, sm_120 via HIP_PLATFORM=nvidia) — no direct
CUDA calls anywhere in the tree.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ff_sieve binary                         │
├─────────────────────────────────────────────────────────────┤
│  GPU Sieve (correct, byte-identical to CPU)                  │
│  ├── RTX 5090 (CUDA backend)                                │
│  └── RX 9070 XT (HIP/ROCm backend)                          │
├─────────────────────────────────────────────────────────────┤
│  CPU Search (correct, multi-threaded, 31 threads)            │
│  └── FreudenthalThreads pattern from segmentedSieve.C        │
├─────────────────────────────────────────────────────────────┤
│  GPU Search (byte-identical to CPU reference, all legs)      │
│  └── GPU Freudenthal kernel with sum-indexed slots           │
└─────────────────────────────────────────────────────────────┘
```

## Current Performance (authoritative sweep: `scripts/bench_per_device.sh`, median of 3 timed reps after 1 untimed warmup; wheel-30 gap-closure final sweep 2026-08-24 — every figure verbatim from `.omo/evidence/gpu-speedup/gap-closure/final-verdict.md`)

> **Measurement upgrade (2026-08-25)**: four new stderr timers landed
> (`hostmap zero-fill`, `scheduler teardown`, `search device setup`,
> `search device teardown`); the harness parses five phase columns (wheel
> expansion plus those four) plus an `unaccounted_ms` completeness check, and
> gates every rep on a normalized-sha256 stdout match. Wall-clock and speedup
> tables below are untouched (2026-08-24 sweep); only the mechanism narrative
> has been restated from the parsed medians. Numerators archived at
> `.omo/start-work/evidence/gate1-numerators.md`.

### Wall-Clock Time (seconds)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| Original (ff_seg) | 0.019s | 0.085s | 0.441s | 2.240s | 9.473s | 41.340s |
| nvidia_gpu (`--devices=nvidia --gpu-search`) | 0.263s | 0.288s | 0.450s | 1.029s | 1.974s | 5.452s |
| amd_gpu (`--devices=amd --gpu-search`) | 0.113s | 0.154s | 0.362s | 1.193s | 3.283s | 12.483s |

### Speedup vs Reference (>1 = faster; ✗ = honest miss, stated plainly)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| nvidia_gpu | 0.07x | 0.30x | 0.98x | **2.18x ✗**(prior verdict 2.46x, −11.5%) | **4.80x** (prior 3.68x, +30%) | **7.58x** (prior 3.43x, +121%) |
| amd_gpu | 0.17x | 0.55x | **1.22x** | **1.88x** (stretch ≥1.35x ✓) | **2.89x** (bar ≥1.6x ✓, stretch ≥1.8x ✓) | **3.31x** (true GPU-search cell¹) |

Task-10 final verdict: the correctness contract is fully green (byte-identical
everywhere, ctest 8/8, dump-map sha256 intact), and the plan's headline goals
beat their bars — AMD@1M **2.89x** (committed bar ≥1.6x, stretch ≥1.8x both
met), amd@2M converted from a 50.6 s CPU-fallback completion cell into a true
GPU-search cell at **12.483 s** (bar ≤38 s), sieve execution deficit **88.7%
recovered** @1M vs amd-gap-analysis §2.3. Recorded honestly against that:

- The committed ZERO-regression tier is an honest **FAIL**: 14/18 cells clean;
  4 cells beyond ±3% after the one sanctioned re-run (nv@524288 +10.3%,
  nv@131072 +8.1%, amd@65536 +7.7%, nv@65536 +3.75%). Mechanism, restated from
  parsed timers (parsed wheel-expansion timer, Gate-1 sweep 2026-08-25; see
  Measurement upgrade note below): the wheel canonical-expansion pass is paid
  by BOTH vendors, not NV alone. Parsed medians: nv 337.333 / 558.082 /
  747.539 ms and amd 372.193 / 574.754 / 831.899 ms @524K/1M/2M, with phase
  attribution closed to ≤22.354 ms residual (≤0.285%). At the NV mid-legs this
  pass outweighs the sieve savings there (NV sieve itself IMPROVED 337→101 ms
  @524288). The decomposition also surfaced a previously invisible cost:
  hostmap zero-fill measures 0.4→528 ms, vendor-symmetric (~528 ms @2M on both
  cards); its deletion is funded (plan task 9). The two @65536 regressions
  stay listed but are unadjudicable at current resolution: a ±3% band is
  single-digit ms against 104-292 ms walls, below the 100+ ms
  enumeration-jitter scale (parsed enum column: nv ~113-124 ms vs amd
  ~7.5 ms), with config-block first-rep inflation documented in
  `scripts/BENCHMARK_METHODOLOGY.md`.
- NVIDIA band check is a **PARTIAL FAIL**: @524288 2.18x vs prior 2.46x =
  −11.5% ✗; @1M 4.80x and @2M 7.58x exceed their bands in the improvement
  direction (recorded as gains).
- Occupancy split recorded honestly: sieve kernel 16 waves/SIMD ✓; search
  kernel 12 waves by measured decision (N=16 spill-free but +2.17% slower
  @1M). (Superseded 2026-08-25: post-bitmask-diet the tradeoff inverted and
  N=16 is now baked — see Optimization campaign below.)

¹ amd_gpu@2M runs GPU search ENGAGED on the RX 9070 XT post-wheel-30: card
named in stderr, zero fallback notices, residency handoff 0 B H2D, all reps
carry search-kernel sub-timers (impossible under CPU fallback).

### Correctness

Verified 2026-08-23 via `ctest` (5/5) and `scripts/verify.sh --all-legs`
(byte-exact stdout vs `goldens/out_ff_seg_<leg>.txt`, solution counts
asserted, plus solution blocks vs `out_pen_*` / `out_pen2_*`):

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| GPU+CPU search, NVIDIA only | YES | YES | YES | YES | YES | YES |
| GPU+CPU search, AMD only | YES | YES | YES | YES | YES | GATE¹ |
| GPU+CPU search, both GPUs | YES | YES | YES | YES | YES | YES |
| GPU sieve + GPU search, AMD device[0] | YES | YES | YES | YES | YES | GATE¹ |
| GPU sieve + GPU search, NVIDIA device[1] | YES | YES | YES | YES | YES | YES |

¹ At the time of the 2026-08-23 run this cell was an expected capacity-gate
refusal (the canonical 16 GiB map exceeded AMD's default-budget backing).
Post-task-14 the aggregate gate auto-enabled the host overflow tier and the
cell completed rc=0 byte-identical via CPU-search fallback. Post-wheel-30
(task-10 sweep, 2026-08-24) the fallback clause is itself historical: the
compressed internal map (8.53 GiB at 2M) fits the card, so this cell runs
TRUE GPU search — card named, zero fallback notices, residency handoff 0 B
H2D (18/18 cells outcome=OK). Prime-map sha256 is identical across AMD sieve,
NVIDIA sieve, and both search paths (`--dump-map`, leg 1M).

### 2. GPU Search Performance (UPDATED 2026-08-24, wheel-30 gap-closure final sweep)

**Status**: Measured by the authoritative sweep — see Current Performance
above. Headlines: NVIDIA up to **7.58x** @2M / **4.80x** @1M; AMD **2.89x**
@1M (committed + stretch bars met), **1.88x** @524288, and amd@2M now a true
GPU-search cell at **12.483 s** / 3.31x. The pre-wheel "AMD ≥2x misses are
structural" conclusion is OBSOLETE — it was invalidated by wheel-30 map
compression. Honest residual: 4/18 cells regressed beyond ±3% vs the frozen
pre-plan baseline (mechanism decomposed in Current Performance above: the
wheel canonical-expansion pass, paid by both vendors per the parsed Gate-1
timers; the floor-cell deltas are unadjudicable at current resolution),
and NV@524288 sits at 2.18x vs its prior 2.46x verdict.

### 3. Thread Count Mismatch (INFORMATIONAL)

**Status**: Reference uses 31 threads, new binary defaults to 20 (now changed to 31).

**Impact**: Minor performance difference in CPU search phase.

**Fix**: Already done - changed `cpu_search.cpp` default from 20 to 31 threads.

## Hardware Configuration

| Component | Details |
|-----------|---------|
| GPU 1 | NVIDIA RTX 5090 (CUDA, sm_120) - 31.4 GB VRAM |
| GPU 2 | AMD RX 9070 XT (HIP/ROCm, gfx1201) - 17.1 GB VRAM |
| CPU | Multi-threaded, hardware_concurrency detected |
| RAM | 93 GB total, ~84 GB available |
| Disk | 1.7 TB free (/home/archerc/) |

## Files of Interest

| File | Purpose |
|------|---------|
| `source/cpu_search.cpp` | Multi-threaded CPU search (FreudenthalThreads pattern) |
| `source/m4/gpu_search_kernel.h` | GPU Freudenthal search kernel |
| `source/m4/gpu_search_launcher.cpp` | Host-side wrapper for GPU search |
| `source/sieve_slab_engine.cpp` | GPU sieve engine |
| `scripts/bench_per_device.sh` | Authoritative per-device benchmark script |
| `scripts/bench_per_device_results.csv` | Benchmark data (CSV format) |
| `scripts/bench_per_device_report.md` | Benchmark report (markdown) |
| `goldens/` | Golden files for byte-exact verification |

## Next Steps

1. ~~Fix GPU search kernel~~ DONE — root cause was a stale 7-arg extern
   declaration of the 9-arg `SearchKernelRun_*` in `test/source/m4_order.cpp`
   (extern "C" binds silently; the kernel then read garbage for its
   `smallPrimes` pointer → AMD page fault). Test now passes all legs.
2. ~~Test on both architectures~~ DONE — pure-HIP dual build (gfx1201 +
   sm_120), both kernels verified per device.
3. ~~Validate at all 6 legs~~ DONE — see correctness table above.
4. ~~Re-benchmark with the fixed GPU search~~ DONE — authoritative
   `scripts/bench_per_device.sh` sweep run 2026-08-24 (see Current
   Performance above).
5. ~~Benchmark GPU search to validate performance improvement~~ DONE —
   task-17 verdict said AMD ≥2x misses were structural (zero-overhead
   ceilings below bar). SUPERSEDED 2026-08-25: that conclusion was
   invalidated by wheel-30 map compression (item 6) — AMD@1M reached
   **2.89x**.
6. ~~DECISION POINT (user): **wheel-30 map compression**~~ DONE — landed and
   measured (kernel-gap-closure tasks 1–10; final sweep 2026-08-24, see
   Current Performance above).
   - **Density-figure correction**: this bullet previously claimed the
     packing was "~3.75× denser … ~4.3 GiB at 2M". That was an arithmetic
     error (wrong baseline). The measured truth: the wheel-30 layout stores
     `ceil(span/30)` bytes vs canonical `ceil(span/16)` = **1.875× denser**
     (17179869185 B → 9162596899 B = 8.53 GiB at 2M; D2H traffic ÷1.875
     inside the sieve timer).
   - Measured outcome: AMD@1048576 **2.89x** (committed bar 1.6x, stretch
     1.8x both met); amd@2097152 converted from a 50.6 s CPU-fallback
     completion cell into a **true GPU-search cell at 12.483 s** (bar ≤38 s);
     NV@1M 4.80x / @2M 7.58x; sieve execution deficit **88.7% recovered**
     vs amd-gap-analysis §2.3.
    - Honest cost: the committed ZERO-regression tier FAILED on 4/18 cells
      (nv@524288 +10.3%, nv@131072 +8.1%, amd@65536 +7.7%, nv@65536 +3.75%)
      and NV@524288's speedup fell 2.46x → 2.18x (−11.5%). Decomposition
      (parsed wheel-expansion timer, Gate-1 sweep 2026-08-25): the
      canonical-expansion pass costs nv 337.333 / 558.082 / 747.539 ms and
      amd 372.193 / 574.754 / 831.899 ms @524K/1M/2M, on BOTH vendors; the
      earlier NV-only attribution rested on total-minus-phases arithmetic and
      does not survive the parsed timers. Routed to
      `.omo/notepads/kernel-gap-closure/problems.md`; remediation deliberately
      deferred (task-10 mandate).

## Optimization campaign (2026-08-25)

The 20-task optimization plan (`.omo/plans/gpu-optimization-execution.md`)
ran to completion after the sweep above. The wall-clock/speedup tables in
Current Performance are the 2026-08-24 pre-campaign sweep against the
pre-campaign binary; they stand as the frozen-protocol baseline the campaign
measured against (campaign gains below were booked from same-session
interleaved A/Bs and parsed timers, not yet from a fresh authoritative
sweep). Landed outcomes, every number from the evidence trail:

- **Hostmap zero-fill deletion** (9dca37c): `hostmap zero-fill` timer
  525.328 → 0.003 ms @amd@2M. The vendor-symmetric ~528 ms @2M cost the
  Gate-1 decomposition surfaced is deleted outright.
- **Canonical expansion overlapped behind `ensureCanonical()`** (9f51a68):
  amd@1M total 3591 → 2920 ms; @2M 12472 → 11053 ms (join residuals
  0.001 / 0.026 ms).
- **Scoped in-map emit-verdict decoder** (31c77bd): nv@524288 wall
  688.130 → 509.849 ms median (−25.9%); the GPU-success path no longer
  spawns canonical expansion at all.
- **Scratch bitmask diet** (b409375): search kernel −2.71% @1M / −3.19%
  @2M; scratch 544 → 288 B/lane AMD and 512 → 256 B NV stack; AMD occupancy
  rung re-baked N12 → N16 (post-diet the earlier "+2.17% slower @N=16"
  verdict inverted).
- **Expansion superblock tiling**, default CPU-search path + dump-map
  (3ab5d4b): 4.16× / 2.36× / 1.00× @524K/1M/2M; starvation curve
  191.9 → 75.6 ms (2 → 32 threads @524K). Plan bands honestly missed;
  impossibility proofs for single-touch parallelism under the frozen deposit
  layout are on record (`.omo/start-work/evidence/e-tiling.md`).

Recorded with equal prominence, the nulls and rejects:

- Targeted attribute queries: no measurable enum win (nv ~123–126 ms,
  delta noise-level on this stack; amd ~7.5 ms unchanged) — ebd0d63.
- REJECTED by measurement: F2 pre-MR trial screen (+1.29% @1M / +0.38% @2M),
  F3 warp-uniform pulling (+0.84% @2M), F4 plain-OR reorder (+1.72% sieve
  phase @amd@2M); evidence commits b760324 / 82e8c7d / 0ddc359.
- F5 prewarm falsified, F6 fence demotion + hostRegister priced out
  (597766c); F7 profiler track CLOSED on tooling absence (rocprofv3/omniperf
  not installed; threshold verdicts UNMEASURED, rerun recipe archived at
  `.omo/start-work/evidence/f7-profiler/summary.md`).

Environment records (fuller notes in README):

- `__launch_bounds__` second param means MIN_BLOCKS_PER_MULTIPROCESSOR on
  CUDA but MIN_WARPS_PER_EXECUTION_UNIT on HIP/ROCm (CU/WGP-mode formulas
  differ by wave size; raw GNU attribute spelling silently no-ops on ROCm
  7.2, macro form mandatory; the HIP Porting Guide documents the mapping).
- `CUDA_MODULE_LOADING`: unset everywhere, no repo override → driver default
  LAZY verified (OVERRIDDEN-CLEAN audit).
- nvidia-persistenced: installed, service inactive, persistence mode
  Disabled; passwordless enablement BLOCKED-sudo at the task-7 freeze.
  Enabling it later is a protocol change demanding a fresh re-baseline
  (`scripts/BENCHMARK_METHODOLOGY.md`, frozen-reference section).

## Post-campaign fine-grained sweep (32 points, 2026-08-26)

Linear 64 KiB step 65 536 → 2 097 152 (32 legs), same harness (`bench_per_device.sh`, median-of-5, sha256 gate), 96/96 cells OK. Evidence at `.omo/start-work/evidence/fine-sweep-summary.md` and `scripts/bench_per_device_results.csv`.

Wall medians (s) and speedup vs reference — compact table:

| leg | ref | amd_gpu | nvidia_gpu | amd sp | nvidia sp |
|-----|-----|---------|------------|--------|-----------|
| 65536 | 0.019 | 0.085 | 0.226 | 0.22x | 0.08x |
| 131072 | 0.088 | 0.131 | 0.252 | 0.67x | 0.35x |
| 196608 | 0.202 | 0.169 | 0.272 | 1.20x | 0.74x |
| 262144 | 0.419 | 0.243 | 0.335 | 1.72x | 1.25x |
| 524288 | 2.177 | 0.724 | 0.585 | 3.01x | 3.72x |
| 1048576 | 9.282 | 2.457 | 1.331 | 3.78x | 6.97x |
| 2097152 | 40.437 | 10.733 | 4.243 | 3.77x | 9.53x |

Full 32-row table in the evidence file. Crossover: **AMD at 196 608**, **NVIDIA at 262 144**; beyond, monotonic to **3.77× / 9.53× @2M**.

## Benchmark Recurrence

To reproduce results:
```bash
cd /home/archerc/Downloads/ff-gpu
cmake --build --preset dev
bash scripts/bench_per_device.sh
```

Results will be in:
- `scripts/bench_per_device_results.csv`
- `scripts/bench_per_device_raw.csv`
- `scripts/bench_per_device_report.md`