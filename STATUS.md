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

## Current Performance (from `scripts/bench_full.sh`)

### Wall-Clock Time (seconds)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| Original (ff_seg) | 0.034s | 0.111s | 0.474s | 2.423s | 10.070s | 47.309s |
| GPU+CPU RTX 5090 | 0.730s | 0.736s | 1.002s | 2.800s | 11.977s | 63.966s |
| GPU+CPU RX 9070 XT | 0.633s | 0.634s | 1.059s | 4.408s | 22.731s | - |
| GPU All RTX 5090 | 0.735s | 0.822s | 1.097s | 2.106s | 7.346s | 62.836s |
| GPU All RX 9070 XT | 0.682s | 0.699s | 0.990s | 3.461s | 18.064s | - |

### Speedup vs Reference (ratio >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| GPU+CPU RTX 5090 | 0.05x | 0.15x | 0.47x | 0.87x | 0.84x | 0.74x |
| GPU+CPU RX 9070 XT | 0.05x | 0.18x | 0.45x | 0.55x | 0.44x | - |
| GPU All RTX 5090 | 0.05x | 0.14x | 0.43x | **1.15x** | **1.37x** | 0.75x |
| GPU All RX 9070 XT | 0.05x | 0.16x | 0.48x | 0.70x | 0.56x | - |

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

¹ Expected capacity-gate refusal: AMD backing (~13.2 GiB at default budget)
< 16 GiB map. Passes with `--host-tier-cap=auto` (spill path), byte-identical.
Prime-map sha256 is identical across AMD sieve, NVIDIA sieve, and both
search paths (`--dump-map`, leg 1M).

### 2. GPU vs CPU Performance (LOW PRIORITY)

**Status**: GPU split model (sieve only) is slower than CPU reference.

**Reasons**:
1. GPU sieve overhead: device init, map staging to pinned memory
2. CPU search dominates runtime (10-47s at 2M)
3. GPU sieve is fast (~0.5-3s) but doesn't overcome CPU bottleneck

**Opportunity**:
- If GPU search is fixed, GPU All mode at 524K-1M shows **1.15-1.37x speedup**
- This is the main reason to fix the GPU search kernel

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
| `scripts/bench_full.sh` | Comprehensive benchmark script |
| `scripts/bench_full_results.csv` | Benchmark data (CSV format) |
| `scripts/bench_full_report.md` | Benchmark report (markdown) |
| `goldens/` | Golden files for byte-exact verification |

## Next Steps

1. ~~Fix GPU search kernel~~ DONE — root cause was a stale 7-arg extern
   declaration of the 9-arg `SearchKernelRun_*` in `test/source/m4_order.cpp`
   (extern "C" binds silently; the kernel then read garbage for its
   `smallPrimes` pointer → AMD page fault). Test now passes all legs.
2. ~~Test on both architectures~~ DONE — pure-HIP dual build (gfx1201 +
   sm_120), both kernels verified per device.
3. ~~Validate at all 6 legs~~ DONE — see correctness table above.
4. Re-benchmark with the fixed GPU search (open).
5. Benchmark GPU search to validate performance improvement (open).

## Benchmark Recurrence

To reproduce results:
```bash
cd /home/archerc/Downloads/ff-gpu
bash scripts/bench_full.sh
```

Results will be in:
- `scripts/bench_full_results.csv`
- `scripts/bench_full_report.md`