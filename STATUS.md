# FF-GPU Project Status & Issues

## Overview

This is a GPU port of the Freudenthal prime-search program (`segmentedSieve.C`). The project implements:
- **GPU sieve**: GPU-accelerated prime number sieve on RTX 5090 + RX 9070 XT
- **CPU search**: Multi-threaded Freudenthal search (31 threads, same as reference)
- **GPU search**: GPU-accelerated Freudenthal search (currently broken at small legs)

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
│  GPU Search (BROKEN at small legs, works at 2M)              │
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

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| Original (ff_seg) | YES | YES | YES | YES | YES | YES |
| GPU+CPU RTX 5090 | YES | YES | YES | YES | YES | YES |
| GPU+CPU RX 9070 XT | YES | YES | YES | YES | YES | - |
| GPU All RTX 5090 | NO | NO | NO | NO | NO | YES |
| GPU All RX 9070 XT | NO | NO | NO | NO | NO | - |

## Issues to Address

### 1. GPU Search Kernel Correctness (HIGH PRIORITY)

**Status**: Broken at small legs (~20% correct), works at 2M.

**Symptoms**:
- At 65K-1M: produces ~20% of expected results
- At 2M: produces 100% of expected results

**Root Cause Hypotheses**:
1. `factors[]` stack allocation (4096 × 8 bytes = 32 KB) may exceed LDS limits on AMD RDNA4
2. Memory access faults at certain leg sizes
3. Early termination conditions wrong

**Testing Done**:
- `m4_kernel_unit` with `testInterval=1` on 65K showed 476/2357 solutions (20.2%)
- Stack-size hypothesis test: reduce `MAX_FACTORS` to 256, verify if correctness improves
- Full enumeration check needed on both architectures

**Fix Approach**:
- Layer 1: Reduce `MAX_FACTORS` from 4096 to 256 (32 KB → 2 KB)
- Layer 2: Phase isolation (disable Phases 2-4, run only Phase 1 Power2Prime)
- Layer 3: Function isolation (replace `dev_AllButOne...` with CPU reference)
- Layer 4: Binary search on kernel (reduce to single sum evaluation)

### 2. AMD RX 9070 XT Deep-Idle Hang (MEDIUM PRIORITY)

**Status**: Known issue, workaround exists.

**Symptoms**:
- AMD GPU enters deep idle state
- KFD event loop blocks forever (poll never returns)
- Process hangs during device initialization

**Root Cause**:
- AMD ROCm/KFD driver doesn't respond to HIP runtime events when GPU is in low-power idle
- Occurs on first device access after idle

**Workaround**:
```bash
HIP_VISIBLE_DEVICES=""  # Disables HIP enumeration, uses only CUDA
# OR
FF_DISABLE_DEVICE=amd   # Runtime device filter
```

**Impact**:
- Cannot use AMD GPU in production without workaround
- Affects `gpu_all_9070` and `gpu+cpu_9070` configurations

**Potential Fixes**:
1. Force AMD GPU out of low-power state before first access
2. Set `ROCR_VISIBLE_DEVICES` or similar env var
3. Submit a workitem to ROCm for the KFD driver bug
4. Use `AMDGPU_FORCE_LOOP_IPOFF=1` or similar kernel parameter

### 3. GPU vs CPU Performance (LOW PRIORITY)

**Status**: GPU split model (sieve only) is slower than CPU reference.

**Reasons**:
1. GPU sieve overhead: device init, map staging to pinned memory
2. CPU search dominates runtime (10-47s at 2M)
3. GPU sieve is fast (~0.5-3s) but doesn't overcome CPU bottleneck

**Opportunity**:
- If GPU search is fixed, GPU All mode at 524K-1M shows **1.15-1.37x speedup**
- This is the main reason to fix the GPU search kernel

### 4. Thread Count Mismatch (INFORMATIONAL)

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
| `src/cpu_search.cpp` | Multi-threaded CPU search (FreudenthalThreads pattern) |
| `src/m4/gpu_search_kernel.h` | GPU Freudenthal search kernel (BROKEN) |
| `src/m4/gpu_search_launcher.cpp` | Host-side wrapper for GPU search |
| `src/sieve_slab_engine.cpp` | GPU sieve engine |
| `scripts/bench_full.sh` | Comprehensive benchmark script |
| `scripts/bench_full_results.csv` | Benchmark data (CSV format) |
| `scripts/bench_full_report.md` | Benchmark report (markdown) |
| `goldens/` | Golden files for byte-exact verification |

## Next Steps

1. **Fix GPU search kernel** (Layer 1: reduce MAX_FACTORS)
2. **Test on both architectures** (AMD + NVIDIA)
3. **Validate at all 6 legs** with corrected kernel
4. **Re-benchmark** with fixed GPU search
5. **Address AMD hang** if production use is desired

## Benchmark Recurrence

To reproduce results:
```bash
cd /home/archerc/Downloads/ff-gpu
bash scripts/bench_full.sh
```

Results will be in:
- `scripts/bench_full_results.csv`
- `scripts/bench_full_report.md`