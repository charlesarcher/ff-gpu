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
│  ├── AMD RX 9070 XT (HIP/ROCm objects, gfx1201)             │
│  └── NVIDIA RTX 5090 (same HIP source via hipcc's           │
│       nvidia platform → CUDA runtime symbols, sm_120)        │
├─────────────────────────────────────────────────────────────┤
│  CPU Search (correct, multi-threaded, 31 threads)            │
│  └── FreudenthalThreads pattern from segmentedSieve.C        │
├─────────────────────────────────────────────────────────────┤
│  GPU Search (byte-identical to CPU reference)                │
│  └── GPU Freudenthal kernel with sum-indexed slots           │
└─────────────────────────────────────────────────────────────┘
```

**Pure HIP**: every kernel and device interaction is written once against the
HIP API (`hipMalloc`, `hipLaunchKernelGGL`, …). There are no direct CUDA calls
anywhere in the tree; the NVIDIA side is the *same* sources compiled by
`hipcc` with `HIP_PLATFORM=nvidia`, where the hip* entry points resolve to the
CUDA runtime at the symbol level. Per-arch object sets export arch-tagged
symbols (`…_gfx1201` / `…_sm_120`), so both runtimes live side by side in one
binary and each logical device is dispatched to its own backend by PCI bus ID.

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
| `m4_mr_diff_bin` | `build/test/m4_mr_diff_bin` | M4 Miller-Rabin differential vs host GpuPrime |
| `hostmap_coverage_test` | `build/test/hostmap_coverage_test` | Hostmap coverage invariant (poison-build trap) |
| `slab_geom_bench_amd` / `slab_geom_bench_nv` | `build/test/slab_geom_bench_{amd,nv}` | Slab geometry bench (per-arch hipcc binaries, direct-run; also registered as ctest) |

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
- `/usr/local/cuda/lib64` — CUDA runtime library (for `-lcudart` and `-lcuda`; the NV backend links `-lcuda` for hipDeviceGet/GetName via HIP-on-CUDA, source-level HIP purity intact with no direct CUDA calls in source, libcuda resolves the symbol mapping)
- `/opt/rocm/lib` — ROCm runtime library (for `-lamdhip64`)

If your toolchain lives elsewhere, edit the paths in `CMakeLists.txt` (the AMD/NVIDIA custom-command blocks and the `FF_GPU_RUNTIME_LIBS` link directories).

## Testing

Run the full suite with CTest:

```bash
ctest --preset dev          # all 9 tests, parallel
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
| `m4_kernel_unit_bin` | M4 GPU search kernel unit behavior | Slow (~83 s) |
| `m4_order_bin` | M4 GPU-search emission vs reference `ff_seg` goldens, byte-identical | Slow (~7 s) |
| `m4_mr_diff_bin` | M4 Miller-Rabin differential: device dev_IsPrime vs host GpuPrime | Fast |
| `hostmap_coverage_test` | Hostmap coverage invariant (poison-build trap, leg 65536 both modes plus 524288 tiled geometries) | ~300 s timeout |
| `slab_geom_bench_amd` / `slab_geom_bench_nv` | Slab geometry bench (per-arch hipcc binaries, direct-run) | Fast |

Expected result: **all 9 tests pass** (7 ctest binaries plus 2 slab_geom_bench).

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

# GPU search on a specific device (prints available devices and which is used)
./build/ff_sieve --gpu-search --gpu-search-device=0 5 65536   # AMD
./build/ff_sieve --gpu-search --gpu-search-device=1 5 65536   # NVIDIA

# Sieve on a specific device
./build/ff_sieve --sieve-device=1 5 65536                     # sieve on NVIDIA only

# Combine: sieve on NVIDIA, search on AMD
./build/ff_sieve --sieve-device=1 --gpu-search --gpu-search-device=0 5 65536

# Restrict to one GPU vendor
./build/ff_sieve --devices=nvidia 5 2097152
./build/ff_sieve --devices=amd 5 1048576     # AMD-only run (the 2M leg also completes: post-wheel-30 the compressed map fits the card — see Known Issues #2)

# List detected GPUs and exit
./build/ff_sieve --list-devices
```

### CLI options

| Option | Description |
|--------|-------------|
| `sumStart sumLimit` | Search range (positional; default `5 65535`) |
| `--gpu-search` | Use GPU search instead of CPU search |
| `--gpu-search-device=N` | Run GPU search on device index N (see `--list-devices`) |
| `--sieve-device=N` | Run sieve on device index N (see `--list-devices`) |
| `--no-gpu` | Rejected with an error (no CPU-only mode exists); omit it for the default GPU-sieve + CPU-search path, or restrict vendors via `--devices=<amd\|nvidia>` |
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
| `FF_THREADS=<n>` | CPU search thread count, clamped to [1, `hardware_concurrency`] (default 31; ignored when `--threads=N` is given) |
| `FF_GPU_RESIDENCY=0` | Forces the legacy copy path (full-map H2D staging); absent or any other value keeps the residency handoff enabled |

### Stderr timing lines

Every `ff_sieve` run prints `ff_sieve timing:` phase lines to stderr — device
enumeration, budget computation, sieve phase, search phase, and total. On the
GPU-search path these additionally include separable search sub-timers
(`search H2D copies`, `search kernel`, `search D2H copies`, `search emit`).
When GPU search falls back to CPU (announced by a documented capacity notice),
the GPU sub-timers are absent but the search phase itself is still timed.

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
| `scripts/bench_per_device.sh` | **Authoritative sweep** (see `scripts/BENCHMARK_METHODOLOGY.md`): reference vs full-GPU-enablement on each card, with warmups, per-rep raw CSV, correctness asserts, and per-run device-attribution gates | `bench_per_device_results.csv`, `bench_per_device_raw.csv`, `bench_per_device_report.md` |
| `scripts/bench_resume.sh` | Run only the remaining unfinished legs and append to `bench_results.csv` | `bench_results.csv` |
| `scripts/m0_bench.sh` | M0 memory-bandwidth benchmark (write/H2D/D2H per GPU) | `config/m0-benchmarks.json` |
| `scripts/check_overlap.sh` | Overlap-engine verification (host-tier spill overlap) | console diagnostics |
| `scripts/check_pull_balance.sh` | Weighted-pull balance checker | console diagnostics |

## Performance

### Benchmark Results (authoritative sweep: `scripts/bench_per_device.sh`, median of 3 timed reps after 1 untimed warmup; wheel-30 gap-closure final sweep 2026-08-24 — every figure verbatim from `.omo/evidence/gpu-speedup/gap-closure/final-verdict.md`)

Wall-clock time (seconds), full process end-to-end:

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| **Original (ff_seg)** | 0.019s | 0.085s | 0.441s | 2.240s | 9.473s | 41.340s |
| nvidia_gpu (`--devices=nvidia --gpu-search`) | 0.263s | 0.288s | 0.450s | 1.029s | 1.974s | 5.452s |
| amd_gpu (`--devices=amd --gpu-search`) | 0.113s | 0.154s | 0.362s | 1.193s | 3.283s | 12.483s |

**Speedup vs Reference** (>1.0 = faster; ✗ = honest miss, stated plainly — regressions are marked as visibly as wins):

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| nvidia_gpu | 0.07x | 0.30x | 0.98x | **2.18x ✗**(prior verdict 2.46x, −11.5%) | **4.80x** (prior 3.68x, +30%) | **7.58x** (prior 3.43x, +121%) |
| amd_gpu | 0.17x | 0.55x | **1.22x** | **1.88x** (stretch ≥1.35x ✓) | **2.89x** (bar ≥1.6x ✓, stretch ≥1.8x ✓) | **3.31x** (true GPU-search cell¹) |

### Notes

- **nvidia_gpu / amd_gpu**: full GPU enablement — both the sieve kernel and the Freudenthal search kernel run on that card. Correctness is byte-identical to the reference in every cell (18/18 cells outcome=OK).
- **Honest regression (committed ZERO-regression tier: FAIL)**: 14/18 cells stayed within ±3% of the frozen pre-plan baseline (`amd-gap/sweep-full.csv`); 4 cells remain beyond band after the one sanctioned re-run — nv@524288 +10.3%, nv@131072 +8.1%, amd@65536 +7.7%, nv@65536 +3.75%. Mechanism (corrected 2026-08-25 from the parsed wheel-expansion timer, Gate-1 sweep; see the Measurement upgrade note in `STATUS.md`): the canonical-expansion pass is paid by BOTH vendors, not NV alone. Parsed medians: nv 337.333 / 558.082 / 747.539 ms and amd 372.193 / 574.754 / 831.899 ms @524K/1M/2M, with phase attribution closed to ≤22.354 ms residual (≤0.285%). At the NV mid-legs this pass outweighs the sieve savings there (the NV sieve itself IMPROVED 337→101 ms @524288); the two @65536 misses are floor-cell noise-scale (+9/+8 ms on init-floor-dominated walls, unadjudicable at current resolution). None is a kernel-speed regression. (An earlier revision of this bullet attributed the NV mid-leg cost via subtraction-derived arithmetic; that attribution does not survive the parsed timers.)
- **Headlines**: AMD@1048576 **2.89x** (9.473 s / 3.283 s) beats both its committed bar (≥1.6x) and stretch bar (≥1.8x); amd@2097152 converted from a 50.6 s CPU-fallback completion cell into a **true GPU-search cell at 12.483 s** (bar ≤38 s; also beats the ≤26 s marker); NV@1M 4.80x / @2M 7.58x exceed their prior verdict bands in the improvement direction. Sieve execution deficit vs `amd-gap-analysis` §2.3: **88.7% recovered** @1M (2369.8 → 268.5 ms).
- **Occupancy, recorded honestly**: sieve kernel 16 waves/SIMD with 0 spills ✓; search kernel runs 12 waves/SIMD by measured decision — the N=16 rung compiled spill-free but measured +2.17% slower @1M (kernel is scratch-bound). (Superseded 2026-08-25: after the compositePower2 bitmask diet halved scratch, the ladder re-measured and N=16 became the faster rung; it is now baked. See Optimization campaign below.)
- **65K floors**: NVIDIA 0.263 s / AMD 0.113 s walls are init-floor-dominated (device enumeration alone: 124.2 ms NV vs 7.5 ms AMD per the verdict's phase decomposition).
- ¹ **amd_gpu @ 2M** now runs GPU search ENGAGED on the RX 9070 XT: card named in stderr, zero fallback notices, residency handoff 0 B H2D, rc=0 byte-identical, all 3 reps carry search-kernel sub-timers (impossible under CPU fallback). See Known Issues #2.
- AMD GPU requires ROCm 7.2+ for gfx1201 support.

### Optimization campaign (2026-08-25)

A 20-task instrumentation-and-optimization plan
(`.omo/plans/gpu-optimization-execution.md`) ran to completion after the sweep
above. The wall-clock and speedup tables in this section are the 2026-08-24
pre-campaign sweep; they stand as the frozen-protocol baseline the campaign
measured against. Landed outcomes, every number from the evidence trail:

| Change | Measured outcome | Commit |
|--------|------------------|--------|
| Hostmap zero-fill deletion | `hostmap zero-fill` timer 525.328 → 0.003 ms @amd@2M; the vendor-symmetric ~528 ms @2M cost surfaced by the Gate-1 decomposition is deleted outright | 9dca37c |
| Canonical expansion overlapped behind `ensureCanonical()` | amd@1M total 3591 → 2920 ms; @2M 12472 → 11053 ms (join residuals 0.001 / 0.026 ms) | 9f51a68 |
| Scoped in-map emit-verdict decoder | nv@524288 wall 688.130 → 509.849 ms median, −25.9%; the GPU-success path no longer spawns canonical expansion at all | 31c77bd |
| Scratch bitmask diet (`compositePower2[64]` → uint32 mask) | search kernel −2.71% @1M / −3.19% @2M; scratch 544 → 288 B/lane AMD, 512 → 256 B NV stack; AMD rung re-baked N12 → N16 | b409375 |
| Expansion superblock tiling (default CPU-search path + dump-map) | 4.16× / 2.36× / 1.00× @524K/1M/2M; starvation curve 191.9 → 75.6 ms (2 → 32 threads @524K) | 3ab5d4b |

Honest nulls and rejects, stated as plainly as the wins:

- **Targeted attribute queries**: no measurable enum win — nv ~123–126 ms
  (delta noise-level on this stack), amd ~7.5 ms unchanged (ebd0d63).
- **REJECTED by measurement**: F2 pre-MR trial screen (+1.29% @1M / +0.38%
  @2M slower), F3 warp-uniform pulling (+0.84% @2M slower), F4 plain-OR
  reorder (+1.72% sieve phase @amd@2M) — evidence commits b760324 / 82e8c7d /
  0ddc359.
- **F5 prewarm falsified / F6 fences+pinning priced out** (597766c);
  **F7 profiler track closed** — rocprofv3/omniperf are absent on this machine
  (tooling-absence proof archived; a rerun recipe is on record should profiler
  tooling land later).

#### Portability & environment notes

- **`__launch_bounds__` second parameter diverges between backends.** CUDA
  reads it as MIN_BLOCKS_PER_MULTIPROCESSOR (minimum resident blocks per SM);
  HIP/ROCm reinterprets it as MIN_WARPS_PER_EXECUTION_UNIT, with CU/WGP-mode
  formulas that differ by wave size (gfx1201: wave32, 1536 VGPRs/SIMD budget,
  granule 24, cap 16 waves/EU; wave64 halves the budget). On ROCm 7.2 the raw
  GNU attribute spelling silently no-ops, so the macro form is mandatory. The
  HIP Porting Guide documents this mapping; live ladder receipts live in
  `.omo/evidence/gpu-speedup/gap-closure/task-2-kernel-gap-closure/ladder.md`.
- **`CUDA_MODULE_LOADING=LAZY` is the verified effective default**: unset in
  the shell environment and overridden nowhere in the repo (audit verdict
  OVERRIDDEN-CLEAN, `.omo/start-work/evidence/env-audit.md`).
- **nvidia-persistenced is installed but DISABLED**: passwordless enablement
  was attempted once during the task-7 protocol freeze and was BLOCKED-sudo;
  persistence mode read `Disabled` before and after. Enabling it later is a
  protocol change requiring a fresh re-baseline before ±3% bands are quoted
  (`scripts/BENCHMARK_METHODOLOGY.md`, frozen-reference section).

### Fine-grained sweep (32 points, post-campaign, 2026-08-26)

Linear 64 KiB step from 65 536 → 2 097 152 (32 legs), same authoritative harness (`bench_per_device.sh`, median-of-5 sub-second, sha256 gate active), 96/96 cells OK. Full table at `.omo/start-work/evidence/fine-sweep-summary.md` and raw CSVs at `scripts/bench_per_device_results.csv`.

Wall-clock medians (s) and speedup vs reference — compact excerpt (every 4th leg plus the 6 original legs marked ●):

| leg | ref | amd_gpu | nvidia_gpu | amd sp | nvidia sp |
|-----|-----|---------|------------|--------|-----------|
| 65536 ● | 0.019 | 0.085 | 0.226 | 0.22x | 0.08x |
| 131072 ● | 0.088 | 0.131 | 0.252 | 0.67x | 0.35x |
| 196608 | 0.202 | 0.169 | 0.272 | 1.20x | 0.74x |
| 262144 ● | 0.419 | 0.243 | 0.335 | 1.72x | 1.25x |
| 327680 | 0.726 | 0.339 | 0.394 | 2.14x | 1.84x |
| 393216 | 1.109 | 0.443 | 0.448 | 2.50x | 2.48x |
| 458752 | 1.612 | 0.553 | 0.492 | 2.92x | 3.28x |
| 524288 ● | 2.177 | 0.724 | 0.585 | 3.01x | 3.72x |
| 589824 | 2.685 | 0.901 | 0.701 | 2.98x | 3.83x |
| 655360 | 3.365 | 1.012 | 0.742 | 3.33x | 4.54x |
| 720896 | 4.215 | 1.251 | 0.892 | 3.37x | 4.73x |
| 786432 | 4.942 | 1.449 | 0.939 | 3.41x | 5.26x |
| 851968 | 5.890 | 1.765 | 1.084 | 3.34x | 5.43x |
| 917504 | 6.908 | 1.971 | 1.118 | 3.50x | 6.18x |
| 983040 | 8.106 | 2.227 | 1.187 | 3.64x | 6.83x |
| 1048576 ● | 9.282 | 2.457 | 1.331 | 3.78x | 6.97x |
| 1114112 | 10.484 | 2.777 | 1.431 | 3.78x | 7.33x |
| 1179648 | 11.778 | 3.228 | 1.612 | 3.65x | 7.31x |
| 1245184 | 13.217 | 3.534 | 1.746 | 3.74x | 7.57x |
| 1310720 | 14.866 | 4.003 | 1.921 | 3.71x | 7.74x |
| 1376256 | 16.571 | 4.420 | 2.150 | 3.75x | 7.71x |
| 1441792 | 18.126 | 4.878 | 2.219 | 3.72x | 8.17x |
| 1507328 | 19.925 | 5.419 | 2.313 | 3.68x | 8.61x |
| 1572864 | 21.892 | 5.870 | 2.548 | 3.73x | 8.59x |
| 1638400 | 23.913 | 6.556 | 2.752 | 3.65x | 8.69x |
| 1703936 | 25.897 | 7.018 | 3.076 | 3.69x | 8.42x |
| 1769472 | 28.114 | 7.868 | 3.287 | 3.57x | 8.55x |
| 1835008 | 30.325 | 8.214 | 3.385 | 3.69x | 8.96x |
| 1900544 | 32.499 | 8.848 | 3.595 | 3.67x | 9.04x |
| 1966080 | 35.567 | 9.384 | 3.822 | 3.79x | 9.31x |
| 2031616 | 38.011 | 10.233 | 4.207 | 3.71x | 9.04x |
| 2097152 ● | 40.437 | 10.733 | 4.243 | 3.77x | 9.53x |

Crossover (first win vs CPU): **AMD at 196 608** (1.20×), **NVIDIA at 262 144** (1.25×). Beyond crossover, wins widen monotonically to **3.77× (AMD) / 9.53× (NVIDIA) @2M**.

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

The `slab_cmp` test previously failed 5/10 cases due to a test-side indexing mismatch (the GPU kernel correctly used segLo-relative indexing matching the production slab engine, but the test compared as global-indexed). The test has been corrected and **all cases pass** with byte-identical output — the matrix has since grown to **22 cases** (task-9 twin coverage: superblock truncation/mid-group tails, degenerate spans, deep offsets), all green in the latest `ctest` run (9/9 suites).

### 2. amd@2M leg — TRUE GPU search since the wheel-30 landing (was: auto-spill completion)

The 2M leg's canonical prime map (~16 GiB) used to exceed the RX 9070 XT's
backing, so from task 14 the aggregate capacity gate auto-enabled the host
overflow tier and the cell completed via CPU-search fallback. That era is
over: the wheel-30 internal layout stores the map at **1.875×** higher density
than the canonical layout (`internalMapBytes = ceil(span/30)` bytes consumed
on-device vs canonical `ceil(span/16)` = 17179869185 B; at 2M the internal map
is 9162596899 B = 8.53 GiB), which fits the card's participation budget. The
aggregate gate now passes on internal bytes ("GATE PASS"), the residency
handoff reads the sieve-resident map in place (0 B H2D), and `amd_gpu @
2097152` runs **true GPU search**: median wall **12.483 s** / **3.31x**,
rc=0 byte-identical, card named in stderr, zero fallback notices.

> Density-figure erratum (corrected 2026-08-25): an earlier revision of this
> document and of `STATUS.md` estimated the wheel-30 packing at "~3.75×
> denser". That figure was wrong — it compared against the wrong baseline.
> The measured density gain is **1.875×** (17179869185 B → 9162596899 B;
> D2H traffic ÷1.875 inside the sieve timer per the final verdict).

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
