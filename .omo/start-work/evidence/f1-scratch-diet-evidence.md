# F1 — Scratch bitmask diet: scratch-size evidence + occupancy re-ladder + interleaved A/B

Date: 2026-08-25 · Base HEAD: `9f51a68` · Product change: `source/m4/gpu_search_kernel.h` ONLY.
Correctness companion: `f1-replay-order-proof.md` (replay-order induction).
All GPU runs and builds wrapped in `flock /tmp/opencode/ff-build.lock` / `flock /tmp/opencode/ff-gpu.lock`.
Kernel identity: `FFSearchKernel_gfx1201` / `FFSearchKernel_sm_120`.

## Verdict

**ADOPT.** Compile-time scratch halved on BOTH archs with zero spills; byte-exact everywhere;
interleaved same-session A/B measures the landed configuration (diet + re-baked N=16 rung) at
**−2.71% @1M / −3.19% @2M** kernel time vs the pre-task binary — inside the plan's expected 2–6% band,
5/6 pairs favoring the new binary on each leg.

## 1. Scratch-size evidence (compile-time, both archs)

Method per arch (proven in task-2 ladder; substitutes for the unavailable F7 profiler):
AMD — `-Rpass-analysis=kernel-resource-usage` remark line `ScratchSize [bytes/lane]`;
NV — `--ptxas-options=-v` stack-frame line.

```
# AMD before (HEAD 9f51a68, N=12 baked):
    TotalSGPRs: 101   VGPRs: 100   ScratchSize [bytes/lane]: 544   Occupancy [waves/SIMD]: 12   Spill: 0/0
# AMD after diet (N=12):        ScratchSize [bytes/lane]: 272   VGPRs: 100  waves 12  Spill: 0/0
# AMD after diet (N=16 baked):  ScratchSize [bytes/lane]: 288   VGPRs: 88   waves 16  Spill: 0/0

# NV before (ptxas -v):
    Used 78 registers ... 512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# NV after diet:
    Used 78 registers ... 256 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

Δ = **−272 B/lane AMD (544→288 at the baked rung), −256 B/thread NV (512→256)** — matches the
predicted ~256 B/lane compositePower2[64] removal. Zero-spill gate intact on both sides.

Reproduction:

```bash
HIP_PLATFORM=amd /opt/rocm/bin/hipcc -O3 -Wall --offload-arch=gfx1201 \
  -I/opt/rocm/include -Isource -DSIEVE_KERNEL_ARCH=gfx1201 \
  [-DFF_SEARCH_MIN_BLOCKS_PER_SM=<R>] -Rpass-analysis=kernel-resource-usage \
  -c source/m4/gpu_search_kernel.cpp -o /tmp/out.o 2> remarks.log
grep -A9 FFSearchKernel_gfx1201 remarks.log

HIP_PLATFORM=nvidia /opt/rocm/bin/hipcc -O3 -I/opt/rocm/include -Isource \
  -DFF_BACKEND_NV=1 -DSIEVE_KERNEL_ARCH=sm_120 -x cu -arch=sm_120 \
  --ptxas-options=-v -c source/m4/gpu_search_kernel.cpp -o /tmp/out.o 2> ptxas.log
```

Stale-state guard: rebuild excerpt shows the kernel TU recompiled on BOTH archs after each header
edit (`[13/14] hipcc (AMD, gfx1201): source/m4/gpu_search_kernel.cpp`,
`[14/14] hipcc (NVIDIA, sm_120): source/m4/gpu_search_kernel.cpp`, then ff_sieve relink).

## 2. Occupancy re-ladder (post-diet, AMD gfx1201, ROCm 7.2 remarks)

| Rung | VGPRs | waves/SIMD | Scratch B/lane | Spills | Timed? |
|---|---|---|---|---|---|
| N=8 | 100 | 12 | 272 | 0/0 | no — codegen-identical to N=12 |
| N=10 | 100 | 12 | 272 | 0/0 | no — codegen-identical to N=12 |
| N=12 (pre-rebake default) | 100 | 12 | 272 | 0/0 | yes (landed binary of that moment) |
| N=14 | 88 | 16 | 288 | 0/0 | no — resource-identical to N=16 (degenerate-rung receipt, same as pre-diet) |
| N=16 | 88 | 16 | 288 | 0/0 | yes |

The granule-24 two-point structure survives the diet; only the tradeoff between the points flipped.

### Rung A/B #1 — diet@N=12 vs diet@N=16 (same session, interleaved ×6, `search kernel` ms)

| Leg | N=12 median | N=16 median | Δ median | pairs → N=16 |
|---|---|---|---|---|
| 1048576 | 1948.086 | 1868.084 | **−4.11%** | 5/6 |
| 2097152 | 8274.262 | 8005.768 | **−3.25%** | 5/6 |

**N=12 does NOT remain optimal post-diet** — the pre-diet verdict ("scratch-bound, squeeze loses")
inverted once scratch halved. AMD baked rung re-baked 12 → 16 in the same commit (one-line macro +
rationale comment update; NVIDIA stays unclamped).

## 3. Interleaved A/B protocol and results

Baseline binary: HEAD `9f51a68` build captured BEFORE any edit (`/tmp/opencode/f1/ff_sieve_base`;
`ninja -n` showed the tree up-to-date first). Protocol: alternate base,new reps under one
`ff-gpu.lock` hold, parse stderr `search kernel = <ms>`, ≥5 reps/side/leg (10 for the diet-only
comparison), report medians + paired-median diffs.

### Phase A — diet alone (base N=12/544 B vs diet N=12/272 B), 10 reps/side

| Leg | base median | new median | Δ median | paired-median Δ | pairs → new |
|---|---|---|---|---|---|
| 1M | 1933.782 | 1899.374 | −1.78% | −1.98% | 8/10 |
| 2M | 8791.178 | 8510.327 | −3.19%* | −0.21% | 5/10 |

\* @2M unpaired median is outlier-position-confounded (base absorbed 11342/10063 ms outliers);
paired read ≈ noise. Diet alone ≈ sub-band win @1M, ~nothing @2M.

### Phase B — full effect (base N=12/544 B vs landed diet N=16/288 B), same-session interleaved ×6

| Leg | base median | new median | Δ median | paired-median Δ | pairs → new |
|---|---|---|---|---|---|
| 1M | 1921.520 | 1869.387 | **−2.71%** | −2.83% | 5/6 |
| 2M | 8411.953 | 8143.459 | **−3.19%** | −4.28% | 5/6 |

Both legs inside the expected 2–6% band, consistent direction → adoption criterion met by the LANDED
configuration. Raw rep tables preserved in session log; per-rep stderr captures under
`/tmp/opencode/f1/ab_*`, `lad_*`, `fin_*` (session-scratch, not committed).

## 4. Gates (final source state, both archs rebuilt)

| Gate | Result |
|---|---|
| flock build both archs | rc=0 (excerpt §1) |
| verify.sh --all-legs --gpu-search --devices amd | rc=0 ALL LEGS PASS byte-identical |
| verify.sh --all-legs --gpu-search --devices nvidia | rc=0 ALL LEGS PASS byte-identical |
| ctest --preset dev (full, incl. m4_order_bin byte-exact) | 9/9 PASS |
| Manual golden diff probe (leg 1M amd, normalization `s/(Prime\|Freudenthal) time: [0-9]+/\1 time: N/`) | CLEAN — all 35558 solution lines identical; only normalized timing footer differs |
| MAX_FACTORS | untouched (=32); 32→24 trim stays withdrawn per research bundle §H |
| main.cpp / pull_scheduler | untouched |

## 5. Risks / notes

- N=16 squeezes VGPRs 100→88. If future work regrows register pressure past the 96-VGPR 16-wave
  budget, the clamp forces loud spills (remarks-visible) rather than silent wave loss — same
  property task 2 valued, now at the higher rung.
- The uint32 mask's domain bound (power2 ≤ 31 ⇔ sum < 2³²+2) is strictly wider than the domain where
  GPU search was ever byte-correct (`GpuRecord.sum` is uint32); proof doc Lemma 2.
- Cross-session comparisons are not attributable (task-2 drift note reconfirmed: base medians moved
  ~2–4% between sessions today); only the same-session Phase-B table supports the verdict.
