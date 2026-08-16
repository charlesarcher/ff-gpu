# GPU Implementation Plan — `segmentedSieve.C`

## Status / scope

Plan for porting the sieve (and, later, the Freudenthal search) of `segmentedSieve.C` to a
heterogeneous multi-GPU system. The sieve becomes a **GPU-backed working-set engine**: the
full prime map lives in a distributed **backing store** across all GPUs' VRAM (and a host
overflow tier), while **working sets** (slabs) are actively sieved on the devices, with data
movement overlapped against computation. The search stays on CPU in phase 1 (guaranteed
identical output), with an optional GPU phase later.

---

## 1. Hardware environment (measured)

| Card | Bus | VRAM | 90% budget | Bandwidth | Arch / toolchain |
|---|---|---|---|---|---|
| RTX 5090 | `01:00.0` | 32 GB | ≈ 26.8 GiB | ~1.79 TB/s | sm_120 (Blackwell) · CUDA 13.3 at `/usr/local/cuda` |
| RX 9070 XT | `05:00.0` | 17.1 GB | ≈ 14.4 GiB | ~644 GB/s | gfx1201 (RDNA4) · ROCm 7.2.4 + HIP at `/opt/rocm` |

- CPU: Ryzen 9 9950X3D (16 cores / 32 threads) — runs the phase-1 search.
- PCIe 5.0 x16, ~64 GB/s per direction. **No NVLink; no inter-vendor P2P.**
- Note: no AMD OpenCL ICD is registered (`clinfo` sees only the NVIDIA platform) — the AMD
  card's device language is **HIP**, not OpenCL.

---

## 2. What we're porting (geometry from the code)

- `sumStart`, `sumLimit` are the CLI args (`argv[1]`, `argv[2]`).
- Map geometry: `map_bytes ≈ sumLimit²/256`.

| sumLimit | map size |
|---|---:|
| 2M | 14.6 GiB |
| 4M | 58 GiB |
| 8M | 233 GiB |
| 16M | 932 GiB |

- Sieve (the easy half): small sieve to `sqrt(maxPrimeMapValue) ≈ sumLimit/4`, then
  `SegmentFill` sweeps the map in sub-blocks of 2^19 values (32 KB). Each sub-block needs only
  primes `p < sqrt(bHi)` (early break), marks only odd multiples `p² ≤ i < bHi`, with exclusive
  byte ownership and plain stores.
- Search (the hard half): reads primality for values up to `(sum/2)²` scattered across the
  whole map — the "one big random-access table" problem.

---

## 3. Architecture overview

```
            HOST (CPU)                       GPU side
 ┌─────────────────────────┐    ┌───────────────────────────────────────┐
 │ Config (budget, slabs)  │    │  GPU 0 (5090)   backing [ 0 .. R0 )   │
 │ WorkQueue (slab list)   │    │  GPU 1 (9070XT) backing [R0 .. R0+R1) │
 │ Residency map           │───▶│  ... (any # / any vendor)             │
 │ Owner threads (1/GPU)   │    │  WORKING SETS: in-place when resident;│
 │  stream pools, events   │    │  double-buffered staging otherwise    │
 │ Sieve done → CPU search │    └───────────────────────────────────────┘
 └─────────────────────────┘
```

- **Backing store** = the logical 1-D, bit-packed `primeMap` (same layout as CPU), distributed
  as per-device contiguous regions sized by the configurable budget.
- **Working set** = one slab (byte-aligned logical window) actively sieved by a kernel, resident
  on a GPU. In-place when the slab's home GPU computes it; staged through a scratch buffer when
  it must run elsewhere or spill from the host tier.
- **Overlap engine** = per-device stream pools; copy-in of slab *i+1* runs concurrently with the
  kernel on slab *i* (double/triple buffering).

---

## 4. Configurable memory budget spec

### 4.1 Configuration knobs

**CLI flags** (precedence: CLI > env > default):

| Flag | Default | Meaning |
|---|---|---|
| `--vram-fraction <f>` | `0.90` | Global VRAM utilization fraction of *free* memory per device |
| `--device-vram-fraction <spec>` | — | Per-device override, e.g. `nvidia=0.90,amd=0.80` (by index or vendor) |
| `--vram-budget <bytes>\|GiB` | — | Absolute per-device cap; replaces fraction if set (beats fraction) |
| `--scratch <GiB>` | `auto` | Per-device scratch reservation (staging buffers, prime list, kernel scratch), out of budget |
| `--slab-size <GiB>` | `1` | Working-set slab size (fixed across devices) |
| `--host-tier-cap <GiB>` | `0` | Max host pinned memory for overflow tier; `0` = disabled, `auto` = host RAM − 4 GiB |
| `--no-host-tier` | — | Force-disable the host overflow tier |

**Env vars** (reproducible runs): `FF_VRAM_FRACTION`, `FF_VRAM_BUDGET`,
`FF_DEVICE_VRAM_FRACTION`, `FF_SCRATCH`, `FF_SLAB_SIZE`, `FF_HOST_TIER_CAP`.

### 4.2 Budget formula (per device, at init)

```
free        = cudaMemGetInfo / hipMemGetInfo free bytes (re-queried at init)
f           = device fraction (default 0.90)
cap         = --vram-budget if set, else +∞
budget      = min(f × free, cap)
scratch     = --scratch if set, else min(0.15 × budget, 1 GiB)
headroom    = 64 MiB (driver/allocator/display safety margin, always reserved)
backing     = budget − scratch − headroom          # this device's share of the map
slabs       = floor(backing / slabSize); remainder unused
```

Each device gets its own independent budget — heterogeneous cards sized by their own free VRAM.

### 4.3 Validation & behavior

- Reject `f ∉ [0.10, 1.0]`; reject negative/zero caps and `slab-size > backing`.
- Clamp budgets to allocation granularity; `slab-size` must keep slabs on 16-value boundaries
  (correctness invariant).
- If a `cudaMalloc`/`hipMalloc` for a backing region fails (fragmentation, another process
  grabbed VRAM): **fall back** — log a warning, shrink `f` for that device only (binary search
  down to a floor of 0.50), re-run sizing. Never silently oversubscribe.
- The budget, free/total per device, backing sizes, and active config are printed at init
  (one line per device) so a run is auditable.

---

## 5. Kernel design — `SieveSlabKernel`

Mirror `SegmentFill` at GPU granularity (this is why we port *this* design, not the wheel one):

- **Byte `i>>4`, bit `i>>1&7`** — identical bit layout to CPU.
- **Slab boundaries byte-aligned** → exclusive byte ownership → **no atomics**, no cache-line
  ping-pong. Last slab truncated to `maxPrimeMapValue+1` (the old wheel overrun bug is
  structurally impossible: slab bounds are clamped).
- **Block per sub-block** (32 KB = 512 cache lines); uniform `bHi` per block → clean early break
  `if (p*p >= bHi) break;`.
- **Line ownership:** each thread owns 64-byte cache lines within the sub-block; loops over the
  prime list, loading its lines into registers, clearing bits, storing back. Whole-line stores,
  no RMW conflicts, vectorizable load/mark/store. This is the GPU analog of the CPU
  cache-blocking insight.
- First multiple: `first = max(ceil(bLo/p)*p, p*p)`, odd-adjusted, then step `2p`.
- Primes list: full small-prime list (`≤ sqrt(maxPrimeMapValue)`, ~43 K entries at 2M) copied to
  each device once into `__constant__` memory (a few hundred KB).
- Launch: e.g., 256 threads/block × 1 block/sub-block; grid = slabs × sub-blocks. Integer/bit
  ops only (Blackwell/RDNA4 FP64 rates irrelevant).

### 5.1 Device language

Single kernel source in **HIP** (CUDA-compatible syntax), compiled twice:

- `hipcc --offload-arch=gfx1201` → the 9070 XT,
- HIP CUDA backend (`--cuda-path=/usr/local/cuda --offload-arch=sm_120`) → the 5090.

Both device objects link into one host binary running the CUDA *and* HIP runtimes side by side.
**Fallback** if the HIP→sm_120 backend misbehaves: two kernel bodies sharing one math header
(identical arithmetic either way).

### 5.2 Device abstraction layer

`DevAlloc` / `DevCopy` / `DevStream` / `DevEvent` / `DevLaunch`, with two implementations
(`cuda_*.cpp`, `hip_*.cpp`), selected per device at runtime. Host code (scheduler, overlap
engine, search) is vendor-neutral — the design works for N GPUs of any vendor mix.

---

## 6. Multi-device distribution & scheduling

- **Discovery:** enumerate devices via each runtime; per-device `free` via the budget config.
- **Backing pool:** device *i* owns region `[base_i, base_i + size_i)`, sized by budget; each slab
  wholly inside one device's region when possible.
- **Dynamic pull queue:** a global atomic work counter; each device's owner thread pulls the next
  unassigned slab. Balances automatically (low slabs cost far more than high slabs).
- **Weighted pulls:** pull with weight ∝ measured device write bandwidth from M0
  (~2.8 : 1 NVIDIA:AMD). Sieve is bandwidth-bound, so this yields near-linear aggregate
  throughput.
- **Fast path:** slab assigned to its home device → kernel writes backing in place, no copies.
  Resident problems (≤ aggregate backing) are pure distributed compute.

---

## 7. Overlap engine (spill path, problems > aggregate backing)

- Slabs that don't fit page out to a **host pinned tier** (`cudaMallocHost` / `hipHostMalloc`,
  `cudaMemcpyAsync` / `hipMemcpyAsync`).
- Per-device pipeline, depth 2–3: while `SieveSlabKernel` marks slab *i* from staging buffer *A*,
  async H2D of slab *i+1* into buffer *B* runs on a separate copy stream; events order
  dependencies. Prefetch in map order → sequential host reads.
- Cross-vendor staging (slab must run on a non-home device) routes through host pinned memory
  (no inter-vendor P2P). Within-vendor P2P used when available. The scheduler prefers home-device
  execution, keeping cross-vendor movement rare.
- **No copy-out for the sieve** — the marked map stays in the backing store for the search. The
  engine only hides copy-*in*, and in the resident case it is a no-op.

---

## 8. Two-phase pipeline & search handoff

- **Phase 1 (sieve) = GPU working-set engine** (sections 3–7).
- **Phase 2 (search), phase-1 approach:** after the sieve, stream the finished map from backing
  to host pinned memory (14.6 GiB at 2M ≈ 0.25 s at ~64 GB/s, overlapped with the first sums)
  and run the **existing CPU search unchanged**. `GpuPrime` keeps `Prime`'s public API
  (`IsPrime`, `operator[]`, `MaxPrimeMapValue`) so `Sammy2Loopy` / `FreudenthalTools` / `RunIt`
  compile untouched. Output byte-identical.
- **Ceiling:** this handoff needs host RAM ≥ map size. Beyond that (very large limits) the search
  must move to GPU too — the known-hard part (sum-sliced, replicated resident map; sort output by
  sum to preserve order). Flagged as phase-2 work (M4), not required for the working-set engine.

---

## 9. Validation — exact-output equivalence

**Target:** the GPU version must produce *exactly* the same output as the reference CPU program
`segmentedSieve.C` (`ff_seg`). Two guarantees back this:

1. **Byte-identical prime map** (GPU sieve output == CPU `SegmentFill` output). This is the
   structural guarantee: identical bit layout + identical arithmetic + exclusive byte ownership.
   Since the search is a deterministic function of the map, a byte-identical map ⇒ identical
   solutions, identical `primes[]`, identical `maxPrimeMapValue` boundary.
2. **Unchanged search phase** (phase-1 handoff, §8) with the same merge order ⇒ identical output
   *order* (globally ascending by sum).

### 9.1 Reference goldens

- **Authoritative reference:** current patched `ff_seg` build; re-run to regenerate goldens before
  GPU work begins (M0).
- Existing goldens in `/tmp/opencode/` are deterministic (byte-identical across runs):
  - `out_ff_seg_{65536,131072,262144,524288,1048576,2097152}.txt` — full program stdout.
  - `out_pen_*.txt`, `out_pen2_*.txt` — full stdout from the wheel versions. Their *header*
    lines differ from `ff_seg` by a few bytes; their **solution blocks must still match** (they
    do today). GPU solution blocks must match all three.

### 9.2 What "exactly the same output" means (byte-for-byte, no whitespace normalization)

The golden file shape is:

```
maxPrimeMapValue 274877906959 numPrimesRequested 82601 totalBytes 17180199589   ← must match exactly (same geometry/sizing)
maxPrimeMapValue 274877906959 numPrimesRequested 82601 sqrtLimit 524288          ← must match exactly
      1) sum =       17, product =              52,  low               4 (2^2),   high term =       13 (prime)   ← solution block
  71424) sum =  2097103, product =    257691615232,  low   13107  131072 (2^17),  high term =  1966031 (prime)
Prime time: 11811114 μs                                                          ← PRESENT but value-normalized
Freudenthal time: 34681668 μs                                                    ← PRESENT but value-normalized
```

Equivalence rules:
- **Exact match:** the two `maxPrimeMapValue … totalBytes …` / `… sqrtLimit …` header lines and
  every solution line (`") sum ="`), including whitespace, widths, tag fields
  (`(prime)`, `(2^n)`, blank-tag column), and ascending-by-sum order. No sorting or whitespace
  normalization in the test.
- **Value-normalized:** `Prime time:` / `Freudenthal time:` lines must be present, their μs
  values regex-replaced (only non-deterministic lines).
- **Exit code:** must be 0.
- **Solution count:** asserted explicitly (byte-exactness implies it, but assert for a clear
  failure message). Reference counts per leg: 2357 / 4776 / 9163 / 18408 / 35556 / 71424.

### 9.3 Test matrix

| Level | Test | Assertion |
|---|---|---|
| Unit | GPU `SieveSlabKernel` vs CPU `SegmentFill` on identical ranges: first slab, last (truncated) slab, boundary slabs, N random slabs | byte-exact `cmp` per slab |
| Unit | Slab spanning 16-value/byte-alignment edges | byte-exact |
| Map | Full-map checksum, GPU vs CPU reference (`--dump-map` test option producing identical map files) | equal `sha256`; optional full `cmp` |
| Map | `primes[]` list derived from GPU map vs CPU reference (also implies `maxPrimeMapValue`/`sqrtLimit` geometry) | identical |
| E2E | Run GPU binary on all 6 legs; normalize timing lines per §9.2 | stdout **byte-identical** to `out_ff_seg_*.txt`, rc=0 |
| E2E | Solution blocks vs `out_pen_*.txt` and `out_pen2_*.txt` | solution block identical to all three originals |
| Order | GPU M4 search path (if enabled) | output still ascending by sum — enforced by §9.2 exact diff (no sort) |

### 9.4 Verification tooling

- `verify.sh <limit> <gpu_bin>`: runs both binaries, extracts the solution block
  (`grep ') sum'`), normalizes timing lines (`sed -E 's/Prime time: [0-9]+/Prime time: N/'` etc.),
  diffs stdout, and asserts rc + solution count. Exits non-zero with a clear message on any
  mismatch; prints the first differing line for diagnosis.
- Map comparison: `--dump-map <file>` test option in both the CPU reference and GPU binary;
  `cmp` (byte-exact) or `sha256sum` (fast) per §9.3.
- CI-able: all assertions return non-zero on failure; the goldens + reference binaries are part of
  M0 deliverable.

### 9.5 Failure taxonomy (what a mismatch means)

| Symptom | Likely cause |
|---|---|
| Header lines differ | Sizing formula diverged (maxPrimeMapValue/numPrimesRequested/geometry) |
| Slab `cmp` mismatch | Bit layout, first-multiple math, odd-only, or early-break (`p*p>=bHi`) wrong in kernel |
| Full-map hash differs but slabs pass | A non-sieved region differs (small map, priming bytes, last-slab clamp) |
| Solution count/order differs | Map or search-phase merge-order regression |
| Only timing lines differ | OK — expected (normalized) |

---

## 10. Milestones

| # | Scope | Exit criteria |
|---|---|---|
| M0 | Recon on both cards: PCIe/H2D/D2D bandwidth, memset/benchmark, VRAM accounting incl. all budget override paths, hipcc sanity on gfx1201 *and* sm_120, dual-runtime (CUDA+ROCm) coexistence smoke test | numbers + goldens regenerated (`out_ff_seg_*.txt`); `--dump-map` option added to reference; `verify.sh` passes vs current reference |
| M1 | Single-device working-set sieve on the 5090 | §9.3 slab + map + E2E tests pass at 2M vs reference; sieve ≈ 0.5–1.5 s |
| M1b | Same engine against the 9070 XT via HIP | validates the abstraction layer |
| M2 | Heterogeneous backing pool + weighted dynamic scheduling + host-staged cross-vendor copies | near-linear aggregate sieve throughput |
| M3 | Host pinned overflow tier + double-buffered overlap | 4M+ runs (spill) without idle DMA; overlap verified |
| M4 | (optional) GPU search | output identical to CPU search, sorted by sum |

---

## 11. Expected numbers (defaults)

- Sieve at 2M: **~12 s CPU → ~0.4–0.7 s** across both cards (NVIDIA-limited; AMD adds
  ~25–30%).
- Whole 2M run with CPU search: dominated by the ~35 s search until M4; with M4, low single
  digits.
- Capacity: combined 90% backing ≈ **41 GiB → fully resident up to sumLimit ≈ 3.2M**;
  past that the host tier keeps going up to host RAM while working sets stay VRAM-sized.

---

## 12. Risks & mitigations

- **Two vendor runtimes in one process** (CUDA + ROCm): test allocator/VMM interaction in M0.
- **HIP-CUDA backend for sm_120:** verify a trivial kernel compiles/runs early; keep the
  dual-body fallback.
- **RDNA4 gfx1201 support** in ROCm 7.2.4: confirm a HIP kernel runs on the 9070 XT before M1.
- **No inter-vendor P2P / no NVLink:** cross-vendor staging via host; scheduler avoids it.
- **PCIe is the ceiling for spill** (~64 GB/s one-way): a 233 GiB map streams ~2× per pass → a
  few seconds of movement, fully overlapped with compute.
- **Aggressive VRAM budget:** reserve scratch + headroom; alloc-failure fallback path; 90% is the
  default, always configurable (§4).
- **Search scales worst** (whole-map random access) — mitigated by the phase-1 CPU-search
  handoff; M4 GPU search is the only genuinely hard part.
- **Output ordering** if the search ever runs on GPU: sort solutions by sum before printing.
