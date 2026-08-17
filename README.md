# FF-GPU: GPU-Accelerated Freudenthal Prime Search

A GPU port of the Freudenthal prime-search program (`segmentedSieve.C`) with multi-architecture support (NVIDIA CUDA + AMD HIP/ROCm).

## Overview

This project implements a heterogeneous GPU-accelerated prime search pipeline:

1. **GPU Sieve**: Fast prime number sieve on GPU (RTX 5090 + RX 9070 XT)
2. **CPU Search**: Multi-threaded Freudenthal search (31 threads, byte-identical to reference)
3. **GPU Search**: GPU-accelerated Freudenthal search (byte-identical on all legs)

**Status**: Production-ready. GPU sieve + CPU search is the default path; GPU search is fully functional and byte-identical to the CPU reference on all 6 benchmark legs.

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
│  GPU Search (byte-identical to CPU reference)                │
│  └── GPU Freudenthal kernel with sum-indexed slots           │
└─────────────────────────────────────────────────────────────┘
```

Both GPU vendors are compiled into **one** binary: per-arch object files are built with `hipcc` (HIP for AMD, CUDA backend for NVIDIA) and linked alongside the host code, so a single process runs both runtimes side by side.

## Build Requirements

| Component | Requirement | Tested |
|-----------|-------------|--------|
| Host compiler | g++ with C++17 | GCC 16.2.1 |
| Build system | CMake ≥ 3.28, Ninja | CMake 4.4.2, Ninja 1.13.2 |
| NVIDIA | CUDA ≥ 12.8 | 13.3 (`/usr/local/cuda`) |
| AMD | ROCm ≥ 6.4 | 7.2.4 (`/opt/rocm`) |
| GPU targets | NVIDIA `sm_120`, AMD `gfx1201` | RTX 5090, RX 9070 XT |

The NVIDIA side is compiled through `hipcc`'s CUDA backend (`-x cu -arch=sm_120`), which delegates to the installed `nvcc`; it does not need `nvcc` directly.

## Building

### Quick start

```bash
# Configure (writes build/ with compile_commands.json)
cmake --preset dev

# Build everything (GPU binary, tests, reference programs)
cmake --build --preset dev
```

The `dev` preset (defined in `CMakeUserPresets.json`):

- Uses the **Ninja** generator and `build/` as the binary directory
- Host flags: `-O2 -Wall` (faithful port of the original Makefile flags, **no** `-DNDEBUG`, so assertions stay live)
- Enables developer mode (`ff-gpu_DEVELOPER_MODE=ON`), which builds the CTest suite
- Exports `build/compile_commands.json` for clangd/editor tooling

### What gets built

| Binary | Path | Description |
|--------|------|-------------|
| `ff_sieve` | `build/ff_sieve` | Main GPU program |
| `m0_bench` | `build/m0_bench` | M0 memory-bandwidth benchmark |
| `ff_seg`, `pen`, `pen2` | `reference/` | CPU reference programs (golden source) |
| `abstraction_smoke` | `build/test/abstraction_smoke` | Dual-runtime smoke test |
| `ff_budget_selftest` | `build/test/ff_budget_selftest` | Pure-host budget/geometry test |
| `slab_cmp` | `build/test/slab_cmp` | Slab-kernel CPU-vs-GPU comparison |
| `m4_kernel_unit_bin` | `build/test/m4_kernel_unit_bin` | M4 kernel unit test |
| `m4_order_bin` | `build/test/m4_order_bin` | M4 output-order vs-golden test |

### Building individual targets

```bash
cmake --build --preset dev --target ff_sieve      # main binary only
cmake --build --preset dev --target m4_order_bin  # one test
cmake --build --preset dev --target ff_seg        # one reference program
```

### Customizing the build

Pass extra cache variables on the configure command line (they persist in `build/CMakeCache.txt`):

```bash
cmake --preset dev -DCMAKE_CXX_FLAGS="-O2 -Wall -g"   # add debug info
cmake --preset dev -DCMAKE_BUILD_TYPE=Release          # if you want NDEBUG-style release
```

Note: the Makefile port intentionally leaves `CMAKE_BUILD_TYPE` empty so the original `-O2 -Wall` (asserts enabled) semantics are preserved.

### Clean rebuild

```bash
rm -rf build && cmake --preset dev && cmake --build --preset dev
```

The build directory is fully self-contained; `rm -rf build` is always safe.

### Hard-coded toolchain paths

The build assumes these fixed locations (mirroring the original Makefile):

- `/opt/rocm/bin/hipcc` — the `hipcc` compiler driver for both vendor sides
- `/opt/rocm/include` — ROCm headers
- `/usr/local/cuda/lib64` — CUDA runtime library (for `-lcudart`)
- `/opt/rocm/lib` — ROCm runtime library (for `-lamdhip64`)

If your toolchain lives elsewhere, edit the paths in `CMakeLists.txt` (the AMD/NVIDIA custom-command blocks and the `FF_GPU_RUNTIME_LIBS` link directories).

## Testing

Run the full suite with CTest:

```bash
ctest --preset dev          # all 5 tests, parallel
```

Run one test, or get verbose output on failure:

```bash
ctest --preset dev -R slab_cmp
ctest --preset dev --output-on-failure
```

### Test descriptions

| Test | Verifies | Notes |
|------|----------|-------|
| `abstraction_smoke` | Both CUDA and ROCm runtimes initialize and run a trivial kernel in one process | Fast (~0.5 s) |
| `ff_budget_selftest` | Pure-host budget/geometry/VRAM-capping logic | No GPU runtimes linked |
| `slab_cmp` | Slab sieve kernel: CPU vs GPU output per slab | all pass |
| `m4_kernel_unit_bin` | M4 GPU search kernel unit behavior | Slow (~16 s) |
| `m4_order_bin` | M4 GPU-search emission vs reference `ff_seg` goldens, byte-identical | Slow (~8 s) |

Expected result: **all tests pass**.

## Running `ff_sieve`

```
ff_sieve [options] <sumStart> <sumLimit>
```

- `<sumStart>`, `<sumLimit>` — search range for prime sums (default: `5 65535`). The reference benchmark legs are 65536, 131072, 262144, 524288, 1048576, 2097152.

### Examples

```bash
# Default: GPU sieve + CPU search (production path), full 2M leg
./build/ff_sieve 5 2097152

# GPU sieve + GPU search (experimental, byte-identical)
./build/ff_sieve --gpu-search 5 2097152

# Restrict to one GPU vendor
./build/ff_sieve --devices=nvidia 5 2097152
./build/ff_sieve --devices=amd 5 1048576     # AMD cannot handle 2M (VRAM)

# CPU-only (no GPU sieve, no GPU search)
./build/ff_sieve --no-gpu 5 65536

# List detected GPUs and exit
./build/ff_sieve --list-devices
```

### CLI options

| Option | Description |
|--------|-------------|
| `sumStart sumLimit` | Search range (positional; default `5 65535`) |
| `--gpu-search` | Use GPU search instead of CPU search |
| `--no-gpu` | Disable all GPU work (CPU-only mode) |
| `--devices=<amd\|nvidia>` | Restrict GPU participation to one vendor |
| `--list-devices` | Print detected GPUs and exit |
| `--vram-fraction=<0.10–1.0>` | Fraction of each GPU's free VRAM used for the sieve (default 0.90) |
| `--device-vram-fraction=<spec>` | Per-device fractions, e.g. `amd=0.9,nvidia=0.8` |
| `--vram-budget=<size>` | Hard cap on total VRAM used, e.g. `20GiB` or bytes |
| `--scratch=<size>` | Host scratch buffer size, e.g. `4GiB` |
| `--slab-size=<size>` | Sieve slab size in bytes (must be 8-value aligned) |
| `--host-tier-cap=<size\|auto>` | Host pinned-memory cap for the overflow tier (e.g. `8GiB`); `auto` resolves from system RAM |
| `--no-host-tier` | Force-disable the host overflow tier |
| `--dump-map=<file>` | Dump the prime map to a binary file (used by verification) |

Sizes accept suffixes like `KiB`/`MiB`/`GiB` or raw byte counts.

### Environment variables

| Variable | Effect |
|----------|--------|
| `FF_THREADS=<n>` | CPU search thread count (default 31; max = `hardware_concurrency`) |
| `FF_DISABLE_DEVICE=amd\|nvidia` | Remove a vendor at startup (diagnostics still print) |

## Verification

### Golden-file verification (`scripts/verify.sh`)

The committed `goldens/` files are the byte-exact contract: `ff_sieve` output must match them for every leg, in both search modes.

```bash
# Harness self-test (checks the diff tooling itself; no binary needed)
bash scripts/verify.sh --self-test

# All 6 legs, CPU search (default) — the production path
bash scripts/verify.sh ./build/ff_sieve --all-legs

# All 6 legs, GPU search
bash scripts/verify.sh ./build/ff_sieve --all-legs --gpu-search

# Single leg, one device
bash scripts/verify.sh ./build/ff_sieve 1048576 --devices amd
```

Usage: `verify.sh [<gpu_bin>] [<limit>] [--all-legs] [--gpu-search] [--devices <backend>] [--self-test]`. It checks solution counts, byte-identical stdout against `goldens/out_ff_seg_<leg>.txt`, and (with `--dump-map`) the prime-map sha256 against the CPU reference.

### Regenerating goldens (`scripts/regenerate_goldens.sh`)

Rebuilds the 18 golden files from the vendored, untouched reference sources:

```bash
cmake --build --preset dev --target ff_seg pen pen2   # ensure reference binaries
bash scripts/regenerate_goldens.sh
```

Only run this when the *reference* behavior legitimately changes — regenerated goldens are the new contract.

### Benchmarks

| Script | What it does | Output |
|--------|--------------|--------|
| `scripts/bench.sh` | Reference vs new binary across modes × legs | `scripts/bench_results.csv` |
| `scripts/bench_full.sh` | Comprehensive GPU-vs-CPU sweep | `bench_full_results.csv`, `bench_full_report.md` |
| `scripts/bench_per_device.sh` | Per-device: CPU vs each GPU, CPU-search vs GPU-search | `bench_per_device_results.csv`, `bench_per_device_report.md` |
| `scripts/bench_resume.sh` | Run only the remaining unfinished legs and append to `bench_results.csv` | `bench_results.csv` |
| `scripts/m0_bench.sh` | M0 memory-bandwidth benchmark (write/H2D/D2H per GPU) | `config/m0-benchmarks.json` |
| `scripts/check_overlap.sh` | Overlap-engine verification (host-tier spill overlap) | console diagnostics |
| `scripts/check_pull_balance.sh` | Weighted-pull balance checker | console diagnostics |

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
- **GPU All**: GPU sieve + GPU search (fully functional; correctness verified)
- **AMD limitation**: Cannot handle the 2M leg (VRAM insufficient)
- **AMD note**: AMD GPU may require troubleshooting on systems with deep-idle power management (see Known Issues)

## Project Structure

```
ff-gpu/
├── source/                # Source code
│   ├── main.cpp           # Entry point, CLI parsing
│   ├── config.cpp         # Configuration parsing & validation
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
├── CMakePresets.json      # Shared presets (dev-mode, CI)
├── CMakeUserPresets.json  # Local dev preset (Ninja, -O2 -Wall, build/)
└── README.md
```

Build outputs land in `build/` (fully gitignored); reference binaries stay in `reference/`.

## Known Issues

### 1. `slab_cmp` — all cases pass

The `slab_cmp` test previously failed 5/10 cases due to a test-side indexing mismatch (the GPU kernel correctly used segLo-relative indexing matching the production slab engine, but the test compared as global-indexed). The test has been corrected and **all 10 cases pass** with byte-identical output.

### 3. AMD RX 9070 XT Deep-Idle Behavior (INVESTIGATING)

AMD GPU initialization may hang or time out on systems where the GPU enters deep idle power state. Investigation ongoing — may be a ROCm/KFD driver issue, a hardware-specific issue, or a configuration problem.

**Workaround (if needed)**: Set `FF_DISABLE_DEVICE=amd` at runtime to bypass AMD entirely. This disables the AMD GPU but allows NVIDIA to function normally.

**Status**: Active investigation. Not confirmed as a universal AMD bug.

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
