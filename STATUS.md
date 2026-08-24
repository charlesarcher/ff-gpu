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

## Current Performance (authoritative sweep: `scripts/bench_per_device.sh`, median of 3 timed reps after 1 untimed warmup; final sweep 2026-08-24 — see `.omo/evidence/gpu-speedup/final-verdict.md`)

### Wall-Clock Time (seconds)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| Original (ff_seg) | 0.019s | 0.084s | 0.427s | 2.198s | 9.227s | 40.559s |
| nvidia_gpu (`--devices=nvidia --gpu-search`) | 0.253s | 0.271s | 0.419s | 0.894s | 2.506s | 11.822s |
| amd_gpu (`--devices=amd --gpu-search`) | 0.104s | 0.218s | 0.618s | 1.878s | 6.296s | 50.596s |

### Speedup vs Reference (ratio >1 = faster; ✗ = misses the plan's bar for that cell)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| nvidia_gpu | 0.08x | 0.31x ✗(≥1.0) | **1.02x ✓** | **2.46x ✓(≥2.0)** | **3.68x ✓(≥2.0)** | **3.43x ✓(≥2.0)** |
| amd_gpu | 0.18x | 0.39x ✗(≥1.0) | 0.69x ✗(≥1.0) | **1.17x ✗(≥2.0 required)** | **1.47x ✗(≥2.0 required)** | 0.80x (completion cell¹) |

Task-17 verdict: the correctness contract is fully green (byte-identical
everywhere, ctest 8/8) and NVIDIA beats its entire speed bar, but the AMD
≥2.0x cells at 524K/1M miss structurally — zero-overhead ceilings are only
**1.23x** / **1.51x**. The ≥1.0x cells at 131K/262K are init-floor-dominated.
¹ amd_gpu@2M is scored as completion only: it completes rc=0 byte-exact by
default via the auto host-tier spill (GPU search falls back to CPU there);
its 0.80x ratio is recorded for transparency, not competitiveness.

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
refusal (AMD backing ~13.2 GiB at default budget < 16 GiB map). Post-task-14
the aggregate gate auto-enables the host overflow tier: the 2026-08-24
task-17 sweep ran this cell by default and it completed rc=0 byte-identical
(18/18 cells outcome=OK), with GPU search falling back to CPU per the
documented capacity notice. Prime-map sha256 is identical across AMD sieve,
NVIDIA sieve, and both search paths (`--dump-map`, leg 1M).

### 2. GPU Search Performance (UPDATED 2026-08-24)

**Status**: Measured by the authoritative sweep — see Current Performance
above. NVIDIA passes its entire speed bar (up to **3.68x** at 1M); the AMD
≥2.0x cells at 524K/1M miss structurally (zero-overhead ceilings
1.23x / 1.51x). The named follow-up lever is wheel-30 map compression
(user decision point — see Next Steps).

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
   task-17 final verdict: correctness fully green; NVIDIA beats its entire
   speed bar; AMD ≥2x misses are structural (zero-overhead ceilings
   1.23x / 1.51x).
6. DECISION POINT (user): **wheel-30 map compression** — a denser packing
   keeping 8 of 30 residues (~3.75× denser map: 16 GiB → ~4.3 GiB at 2M).
   It shrinks sieve marking work, D2H traffic, and the search table
   proportionally (attacking exactly the phases dominating the AMD misses)
   AND brings the 2M map inside AMD's backing, converting the amd@2M
   CPU-search fallback into true GPU search. Cost: changes the canonical
   prime-map bit layout, so the `--dump-map` sha256 contract and slab
   geometry assumptions need a deliberate re-spec.

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