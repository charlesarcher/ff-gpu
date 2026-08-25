# F3 warp-uniform work pulling — rejected by interleaved A/B

Task 17 (F3) of `.omo/plans/gpu-optimization-execution.md`. Scope:
`source/m4/gpu_search_kernel.h` work-pull loop only (post-F1-bitmask,
post-N16-rung state). No emission, launcher, scheduler, or main.cpp change;
isolated single-kernel A/B.

## Mechanism tested (the fine-grained variant only)

Per-thread `atomicAdd(counter, 1)` replaced by warp-uniform pulling:

- lane 0 performs ONE `atomicAdd(counter, warpSize)` per round — stride is the
  RUNTIME `warpSize`, never a hardcoded 32/64 (RDNA4 wave-variable, ROCm
  issue #6111);
- chunk base broadcast to all lanes via full-mask `__shfl_sync`
  (`unsigned` mask on CUDA; 64-bit mask on HIP AMD per its static_assert,
  auto-trimmed for wave32 by `__hip_adjust_mask_for_wave32`);
- lane i owns odd-sum index `base + i`;
- exit test `base >= numOddSums` reads only the broadcast scalar ⇒
  warp-uniform exit by construction; straddle-tail lanes
  (`base + lane >= numOddSums`) idle one round and rejoin the next pull.

Tail-exit proof sketch (P1–P4) was recorded in the treatment source comment:
counter hands out disjoint consecutive k·W chunks (P1); the only break
predicate is a function of the broadcast base alone (P2); a straddling round
has no exiting lane and its idle lanes claim nothing beyond numOddSums (P3);
divergence never spans atomicAdd→shuffle→break, so every shuffle sees all
lanes active (P4). Work can be neither dropped nor duplicated at any
numOddSums/W alignment.

This is deliberately NOT the previously REJECTED block-chunk variant (~15%
loss on sm_120 @2097152, task-11 matrix): chunks are one wave wide with zero
`__syncthreads` coupling.

## Gates (treatment binary) — all green before measurement

- flock builds both archs: ninja clean; stale-state probe — header mtime
  16:45:55 → `gpu_search_kernel_gfx1201.o` / `_sm120.o` both 16:46:01.
- `verify.sh --all-legs --gpu-search --devices amd`: ALL LEGS PASS
  (byte-identical).
- `verify.sh --all-legs --gpu-search --devices nvidia`: ALL LEGS PASS.
- `ctest -R m4_order`: PASS (byte-exact vs goldens).
- `ctest -R m4_kernel_unit`: PASS (75.3 s).

Grep-clean proof over the full treatment diff: stride `(uint32_t)warpSize`,
lane `threadIdx.x % warpSize`; the only `32`/`64` tokens are a full-wave MASK
literal (`0xffffffffu`, annotated), and two comments stating the rule /
HIP's 64-bit mask TYPE. No bare 32/64 as lane count or stride anywhere.

## Interleaved A/B — amd gpu-search, ≥6 v 6 under ff-gpu.lock

Baseline = HEAD e6711a7 built in isolated worktree `/tmp/opencode/f3-base`
(only diff vs treatment: the kernel header edit). Treatment = working-tree
build. One untimed warmup per side per leg, then strict alternation B,T×6.
Raw stderr: `/tmp/opencode/f3_ab_{base,treat}_{leg}_{1..6}.err`. Metric:
parsed `ff_sieve timing: search kernel` sub-timer (ms).

@2097152:

| rep | baseline | treatment | Δ |
|-----|----------|-----------|---|
| 1   | 8019.023 | 7838.586  | −180.437 |
| 2   | 8240.907 | 8003.767  | −237.140 |
| 3   | 7902.561 | 8228.625  | +326.064 |
| 4   | 8039.530 | 8035.675  | −3.855 |
| 5   | 7912.215 | 8158.555  | +246.340 |
| 6   | 8290.563 | 8280.137  | −10.426 |
| **median** | **8029.276** | **8097.115** | **+67.839 (+0.84% slower)** |
| spread | 7902.6–8290.6 | 7838.6–8280.1 | fully OVERLAPPING |

@1048576:

| rep | baseline | treatment | Δ |
|-----|----------|-----------|---|
| 1   | 1894.989 | 1935.700  | +40.711 |
| 2   | 1863.717 | 1879.507  | +15.790 |
| 3   | 1817.714 | 1811.489  | −6.225 |
| 4   | 1856.681 | 1874.267  | +17.586 |
| 5   | 1812.766 | 1810.229  | −2.537 |
| 6   | 1931.078 | 1860.213  | −70.865 |
| **median** | **1860.199** | **1867.240** | **+7.041 (+0.38% slower)** |
| spread | 1812.8–1931.1 | 1810.2–1935.7 | fully OVERLAPPING |

Pairwise deltas flip sign rep-to-rep on both legs (2M: −180,−237,+326,−4,+246,−10)
— pure noise-scale interleave, no distributional separation in either
direction.

## VERDICT

**REJECT** — adopt bar was ≥2% median win @2M amd gpu-search; measured
median delta @2M = **+0.84% slower** (8097.115 vs 8029.276 ms), @1M +0.38%
slower, distributions fully overlapping. The coin flip landed flat-to-
slightly-negative: ÷32/÷64 fewer global atomics did not convert (kernel is
latency/occupancy-bound post-F1, consistent with F2's finding), while the
added shuffle + tail predicate + lane-0 pull serialization cost at least the
same scale. Kernel change REVERTED to HEAD content; build re-synced and
re-verified byte-identical (@1048576 amd gpu-search PASS). Treatment diff
archived at `/tmp/opencode/f3_treatment.diff` (95 lines) for any future
retry; a retry would need a mechanism that avoids the per-round lane-0
serialization (e.g., multi-chunk strided pulls) — none tested here.
