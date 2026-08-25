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
  nv@131072 +8.1%, amd@65536 +7.7%, nv@65536 +3.75%). Mechanism: the wheel
  canonical-expansion pass costs ~380 ms inside NV mid-leg totals,
  outweighing sieve savings there (NV sieve itself IMPROVED 337→101 ms
  @524288); the two @65536 misses are floor-cell noise-scale.
- NVIDIA band check is a **PARTIAL FAIL**: @524288 2.18x vs prior 2.46x =
  −11.5% ✗; @1M 4.80x and @2M 7.58x exceed their bands in the improvement
  direction (recorded as gains).
- Occupancy split recorded honestly: sieve kernel 16 waves/SIMD ✓; search
  kernel 12 waves by measured decision (N=16 spill-free but +2.17% slower
  @1M).

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
pre-plan baseline (NV mid-leg canonical-expansion cost; floor-cell noise),
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
     and NV@524288's speedup fell 2.46x → 2.18x (−11.5%) — the wheel
     canonical-expansion pass costs ~380 ms inside NV mid-leg totals.
     Routed to `.omo/notepads/kernel-gap-closure/problems.md`; remediation
     deliberately deferred (task-10 mandate).

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