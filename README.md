# FF-GPU: GPU-Accelerated Freudenthal Prime Search

A GPU port of the Freudenthal prime-search program (`segmentedSieve.C`) with multi-architecture support (NVIDIA CUDA + AMD HIP/ROCm).

## Overview

This project implements a heterogeneous GPU-accelerated prime search pipeline:

1. **GPU Sieve**: Fast prime number sieve on GPU (RTX 5090 + RX 9070 XT)
2. **CPU Search**: Multi-threaded Freudenthal search (31 threads, byte-identical to reference)
3. **GPU Search**: GPU-accelerated Freudenthal search (experimental, works at 2M)

**Status**: Production-ready CPU search + GPU sieve. GPU search has correctness issues at small legs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ff_sieve binary                         │
├─────────────────────────────────────────────────────────────┤
│  GPU Sieve (correct, byte-identical to CPU)                  │
│  ├── NVIDIA RTX 5090 (CUDA, sm_120)                         │
│  └── AMD RX 9070 XT (HIP/ROCm, gfx1201)                     │
├─────────────────────────────────────────────────────────────┤
│  CPU Search (correct, multi-threaded, 31 threads)            │
│  └── FreudenthalThreads pattern from segmentedSieve.C        │
├─────────────────────────────────────────────────────────────┤
│  GPU Search (experimental)                                   │
│  └── GPU Freudenthal kernel with sum-indexed slots           │
└─────────────────────────────────────────────────────────────┘
```

## Build Requirements

- **Compiler**: g++ (C++17)
- **NVIDIA**: CUDA 12.8+ (tested with 13.3)
- **AMD**: ROCm 6.4+ (tested with 7.2.4)
- **Build**: CMake 3.28+ (tested with 4.4.2), Ninja

### Build Commands

```bash
# Configure + build everything (GPU binary, tests, reference programs)
cmake --preset dev
cmake --build --preset dev

# Run the test suite (CTest)
ctest --preset dev
```

## Usage

```bash
# Run with GPU sieve + CPU search (default)
./build/ff_sieve 5 2097152

# Run with GPU sieve + GPU search (experimental)
./build/ff_sieve --gpu-search 5 2097152

# Run with specific device only
./build/ff_sieve --devices=nvidia 5 2097152
./build/ff_sieve --devices=amd 5 1048576  # AMD cannot handle 2M

# List available devices
./build/ff_sieve --list-devices
```

### CLI Options

| Option | Description |
|--------|-------------|
| `sumStart sumLimit` | Search range (default: 5 65535) |
| `--gpu-search` | Enable GPU search (experimental) |
| `--devices=<backend>` | Filter devices: `nvidia`, `amd`, or comma-separated |
| `--list-devices` | List detected GPUs |
| `--vram-fraction=<0.90>` | Fraction of free VRAM to use |
| `--host-tier-cap=<bytes>` | Host pinned memory for spill (e.g., `8GiB`) |
| `FF_THREADS=<n>` | CPU search threads (default: 31, max: hardware_concurrency) |
| `HIP_VISIBLE_DEVICES=""` | Disable AMD GPU (workaround for deep-idle hang) |
| `FF_DISABLE_DEVICE=amd` | Runtime disable AMD device |

## Performance

### Benchmark Results (from `scripts/bench_full.sh`)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| **Original (ff_seg)** | 0.034s | 0.111s | 0.474s | 2.423s | 10.070s | 47.309s |
| GPU+CPU RTX 5090 | 0.730s | 0.736s | 1.002s | 2.800s | 11.977s | 63.966s |
| GPU+CPU RX 9070 XT | 0.633s | 0.634s | 1.059s | 4.408s | 22.731s | - |
| GPU All RTX 5090 | 0.735s | 0.822s | 1.097s | 2.106s | 7.346s | 62.836s |

**Speedup vs Reference** (>1.0 = faster):

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| GPU+CPU RTX 5090 | 0.05x | 0.15x | 0.47x | 0.87x | 0.84x | 0.74x |
| GPU All RTX 5090 | 0.05x | 0.14x | 0.43x | **1.15x** | **1.37x** | 0.75x |

### Notes

- **GPU+CPU**: GPU sieve + CPU search (production path)
- **GPU All**: GPU sieve + GPU search (experimental, broken at small legs)
- **AMD limitation**: Cannot handle 2M leg (VRAM insufficient)
- **AMD workaround**: Use `HIP_VISIBLE_DEVICES=""` to disable AMD

## Project Structure

```
ff-gpu/
├── source/                # Source code
│   ├── main.cpp           # Entry point, CLI parsing
│   ├── config.cpp         # Configuration parsing
│   ├── cpu_search.cpp     # Multi-threaded CPU search
│   ├── gpu_prime.h        # GpuPrime API
│   ├── sieve_slab_engine.cpp  # GPU sieve engine
│   ├── smoke/             # Dual-arch smoke kernel (AMD + NVIDIA TUs)
│   └── m4/                # GPU search (M4)
│       ├── gpu_search_kernel.h    # GPU Freudenthal kernel
│       ├── gpu_search_launcher.cpp
│       └── gpu_search_emission.cpp
├── test/source/           # Unit tests
│   ├── m4_kernel_unit.cpp
│   ├── m4_order.cpp
│   ├── slab_cmp.cpp
│   └── abstraction_smoke.cpp
├── scripts/               # Benchmark & validation
│   ├── bench_full.sh      # Comprehensive benchmark
│   ├── verify.sh          # Golden file verification
│   └── m0_bench.sh        # M0 bandwidth benchmark
├── goldens/               # Golden files (byte-exact contract)
├── reference/             # Reference CPU programs
│   ├── ff_seg             # Original segmentedSieve binary
│   ├── pen                # Alternative wheel-sieve
│   └── pen2               # Alternative wheel-sieve v2
├── CMakeLists.txt         # Main build (CMake 3.28+)
├── CMakePresets.json      # Shared presets
└── README.md
```

## Verification

### Run Benchmarks

```bash
bash scripts/bench_full.sh
```

Output:
- `scripts/bench_full_results.csv` - Raw benchmark data
- `scripts/bench_full_report.md` - Formatted report

### Verify Correctness

```bash
# Self-test (harness validation)
bash scripts/verify.sh --self-test

# All legs (CPU search)
bash scripts/verify.sh ./build/ff_sieve --all-legs

# All legs (GPU search)
bash scripts/verify.sh ./build/ff_sieve --all-legs --gpu-search
```

## Known Issues

### 1. GPU Search Correctness (HIGH PRIORITY)

The GPU search kernel produces incorrect results at small legs (~20% correct). Works at 2M.

**Symptoms**:
- 65K: 476/2357 solutions (20.2%)
- 131K-1M: ~20% correct
- 2M: 100% correct

**Root Cause**: `factors[]` stack allocation (4096 × 8 bytes = 32 KB) may exceed LDS limits on AMD RDNA4.

**Fix**: Reduce `MAX_FACTORS` from 4096 to 256 in `source/m4/gpu_search_kernel.h`.

### 2. AMD RX 9070 XT Deep-Idle Hang (MEDIUM)

AMD GPU hangs during initialization when in deep idle state.

**Workaround**: Use `HIP_VISIBLE_DEVICES=""` or `FF_DISABLE_DEVICE=amd`.

**Status**: Known ROCm/KFD driver issue. No fix available.

### 3. Thread Count

Reference uses 31 threads. New binary defaults to 31 (configurable via `FF_THREADS`).

## Hardware

| Component | Details |
|-----------|---------|
| GPU 1 | NVIDIA RTX 5090 (CUDA, sm_120) - 31.4 GB VRAM |
| GPU 2 | AMD RX 9070 XT (HIP/ROCm, gfx1201) - 17.1 GB VRAM |
| CPU | Multi-threaded (31 threads) |
| RAM | 93 GB total |
| Disk | 1.7 TB free |

## References

- Original program: `segmentedSieve.C`
- GPU design plan: `GPU_PLAN.md`
- Status document: `STATUS.md`

## License

Proprietary - all rights reserved.

## Contact

For issues or questions, please open a GitHub issue.