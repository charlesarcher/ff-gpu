# Task 19 — F5+F6 Opportunists: Falsified / Priced Out (evidence)

Date: 2026-08-25. HEAD at measurement: `b409375` (perf(kernel): compositePower2
bitmask diet). Device under test: AMD RX 9070 XT (gfx1201), ROCm 7.2.4.
All timed runs: `FF_F5_PROBE=1 timeout 300 ...` inside `flock /tmp/opencode/ff-gpu.lock`;
builds inside `flock /tmp/opencode/ff-build.lock`. Watchdog: every GPU invocation
wrapped in `timeout 300` (longest actual run: 2M leg, ~13 s wall; zero hangs).

Method: temporary env-gated probe (`FF_F5_PROBE`) added std::chrono steady-clock
ns timestamps around (a) the `t_sieve` window in main.cpp, (b) the scheduler's
internal `t0`, (c) pool-setup sub-blocks, and (d) every `hipLaunchKernelGGL`
site in sieve_slab_engine.cpp (first-launch entry/exit, host-side). Probe was
REVERTED after measurement (`git checkout --` on the three touched files);
product code in this commit is unchanged.

## VERDICTS (verbatim)

- F5 prewarm: **DROP** — first `hipLaunchKernelGGL` IS inside t_sieve but its
  host-side cost is 0.258-0.362 ms across 3 runs; no ~25-30 ms module-load
  constant exists at launch time on this stack. Adoption bar (>=15 ms median
  sieve-timer improvement) missed by >40x. Nothing for a dummy launch to amortize.
- F6a fence demotion (setup regions): **DROP** — sync pool ops confirmed outside
  the scheduler wall timer (before internal t0) but INSIDE the "sieve phase"
  timer; measured total per-op fence cost is 84 us @65K and <=2.9 ms @2M, of
  which 2.84 ms is ONE genuine dependency wait that batching cannot remove
  (all ops share one stream). Max theoretical saving ~tens of us vs >=15 ms bar.
- F6b hipHostRegister: **DROP** — pin cost priced CHEAP (~1.5 ms/GiB one-time,
  RSS delta 0 KB, runtime attr gate passes) but the benefit surface on current
  architecture is ~zero: staging bounce buffers are already pinned, async D2H
  drains already target hostMap directly, and large-leg GPU-search runs use the
  residency handoff (0 B H2D, no full-map D2H). No mechanism to a >=15 ms win.

---

## F5 falsification transcript (probe output, pasted inline)

Run 1 — amd@65536 --gpu-search:

    [f5probe] t_sieve_start_ns=428980217467876
    [f5probe] sched_t0_ns=428980252325909
    [f5probe] first_launch site=poolSlabComputeAsync enter_ns=428980252543428 exit_ns=428980252905396 host_dur_ms=0.362
    [ff_sieve] pull scheduler wall time: 4421 us
    ff_sieve timing: sieve phase = 39.290 ms
    [f5probe] t_sieve_end_ns=428980256758976

Run 2 — amd@131072 --gpu-search (rep 1):

    [f5probe] t_sieve_start_ns=429018759334492
    [f5probe] sched_t0_ns=429018794335072
    [f5probe] first_launch site=poolSlabComputeAsync enter_ns=429018794563611 exit_ns=429018794925329 host_dur_ms=0.362
    ff_sieve timing: sieve phase = 45.439 ms
    [f5probe] t_sieve_end_ns=429018804775366

Run 3 — amd@131072 --gpu-search (rep 2):

    [f5probe] t_sieve_start_ns=429018902020840
    [f5probe] sched_t0_ns=429018931478151
    [f5probe] first_launch site=poolSlabComputeAsync enter_ns=429018931624540 exit_ns=429018931882738 host_dur_ms=0.258
    ff_sieve timing: sieve phase = 39.660 ms
    [f5probe] t_sieve_end_ns=429018941681825

Reading: first-launch enter_ns lies strictly between t_sieve_start_ns and
t_sieve_end_ns in ALL runs -> the first launch IS inside the timed window.
But its HOST duration is 0.258/0.362/0.362 ms. A lazy ROCm code-object load
would appear exactly there (the load is synchronous host-side when triggered
by launch). Conclusion: the executable is loaded before the window (eager load
at registration/context init during the ~390 ms device enumeration) or its
launch-time cost is sub-ms. The hypothesized 25-30 ms recoverable constant
does not exist at this call site. F5 precondition falsified -> DROP.

Side observation (recorded, NOT implemented - m4 lane owns that region):
`ff_sieve timing: search kernel = 25.284 ms` at amd@65536 suggests a ~25 ms
per-process first-launch constant sits in the SEARCH-phase timer, not the
sieve timer. Out of task-19 scope (F5 claims only t_sieve); flagged for the
kernel A/B queue owners / closeout docs.

## F6a: setup-region sync sites — location + pricing transcripts

Static finding (current code): per-op `hipDeviceSynchronize` live in
poolMemset / poolCopyH2D / poolCopyD2H / poolSlabCompute
(source/sieve_slab_engine.cpp). In the pull-scheduler flow the home-slab init
loop (memset + 1-byte copyH2D x2, each followed by a synced pool op) runs on
the main thread BEFORE the scheduler's internal t0 (pull_scheduler.cpp:560)
and before worker threads spawn -> OUTSIDE the scheduler wall timer (Gate-1-era
finding HOLDS for that timer), but INSIDE the outer "sieve phase" timer
(t_sieve spans engine.prepare() + all scheduler setup). Correction to the
Gate-1 note recorded here so nobody infers "untouchable" from "outside t0".

Decomposition inside t_sieve (probe, amd-only):

@65536 (sieve phase = 37.10 ms):

    [f5probe] t_sieve_start_ns=429175522768722
    [f5probe] prepare_end_ns=429175522825172          (prepare = 56.5 us)
    [f5probe] pool_setup_start_ns=429175522827632
    [f5probe] home_init_start_ns=429175523148320      (allocs before loop = 0.32 ms)
    [f5probe] home_init_end_ns=429175539017852        (home-init loop = 15.87 ms)
    [f5probe] pool_setup_end_ns=429175555614389       (staging+pinned+streams+events = 16.60 ms)
    [f5probe] sched_t0_ns=429175555618689             (total setup inside window = 32.79 ms)
    [f5probe] first_launch site=poolSlabComputeAsync enter_ns=429175555869947 exit_ns=429175556187526 host_dur_ms=0.318
    ff_sieve timing: sieve phase = 37.099 ms

Per-op fence price @65536 (op = transfer call, sync = following hipDeviceSynchronize):

    [f5probe] pool_memset bytes=8947850 op_us=15858.8 sync_us=53.4
    [f5probe] pool_copyH2D bytes=1 op_us=27.8 sync_us=16.5
    [f5probe] pool_copyH2D bytes=1 op_us=18.1 sync_us=14.3

Total fence cost @65K = 53.4 + 16.5 + 14.3 = 84.2 us (+45.2 us from two small
synced ops in the staging block: sync_us=25.9 and 19.3). NOTE the trap: the
home-init LOOP looks like 15.87 ms of "sync overhead" but the memset itself is
15.86 ms (first-touch VRAM page commitment of the freshly hipMalloc'd region);
the fence is 53 us. Demoting fences would move ~0.1 ms.

@2097152 (sieve phase = 2818.9 ms):

    [f5probe] prepare_end_ns=429175624466534          (prepare = 1.78 ms)
    [f5probe] home_init_start_ns=429175716020143
    [f5probe] home_init_end_ns=429175745329379        (home-init loop = 29.31 ms)
    [f5probe] pool_setup_end_ns=429175794209696       (total setup = 169.74 ms)
    ff_sieve timing: sieve phase = 2818.924 ms

Per-op fence price @2M:

    [f5probe] pool_memset bytes=1073741824 op_us=12870.6 sync_us=2842.3   <- first slab: page-commit wait
    [f5probe] pool_copyH2D bytes=1 op_us=35.8 sync_us=16.9
    [f5probe] pool_memset bytes=1073741824 op_us=2012.4 sync_us=0.2
    [f5probe] pool_memset bytes=1073741824 op_us=1795.3 sync_us=0.1
    [f5probe] pool_memset bytes=1073741824 op_us=1789.2 sync_us=0.1
    [f5probe] pool_memset bytes=1073741824 op_us=1801.4 sync_us=0.1
    [f5probe] pool_memset bytes=1073741824 op_us=1800.8 sync_us=0.1
    [f5probe] pool_memset bytes=1073741824 op_us=1788.7 sync_us=0.1
    [f5probe] pool_memset bytes=1073741824 op_us=1792.3 sync_us=0.1
    [f5probe] pool_memset bytes=572662307 op_us=954.8 sync_us=0.1
    [f5probe] pool_copyH2D bytes=1 op_us=16.7 sync_us=14.7

Total fence cost @2M ~= 2.91 ms, of which 2.842 ms is the SINGLE first fence
waiting for real device work (page-commitment of the first 1 GiB memset).
Every subsequent idle fence costs 0.1-17 us. All ops enqueue on the same
device stream, so batching N syncs into one final sync cannot overlap anything:
the 2.84 ms dependency wait would simply reappear at the final fence. Net
theoretical saving = sum of idle-sync host overhead ~= tens of microseconds
per leg. Verdict: F6-no-timed-window-benefit CONFIRMED BY MEASUREMENT -> skip
implementation. (Worker-path sync copies for homeless slabs are explicitly
OUT of scope per task wording "ONLY in provably single-threaded setup
regions"; not priced, not touched.)

## F6b: hipHostRegister pricing transcript (standalone probe, no product code)

Program: /tmp/opencode/f6pin/f6pin_price.cpp (malloc 1 GiB, touch pages,
hipHostRegister + unregister x5, RSS via getrusage, rlimit + attr checks).

    device0: AMD Radeon RX 9070 XT
    attr HostRegisterSupported: rc=0(hipSuccess) val=1
    RLIMIT_MEMLOCK soft=8388608 hard=8388608 (8 MiB soft)
    rep0 register rc=0(hipSuccess) dur_ms=1.526 rss_delta_kb=0
         unregister rc=0(hipSuccess) dur_ms=0.012
    rep1 register rc=0(hipSuccess) dur_ms=1.568 rss_delta_kb=0
         unregister rc=0(hipSuccess) dur_ms=0.005
    rep2 register rc=0(hipSuccess) dur_ms=1.477 rss_delta_kb=0
         unregister rc=0(hipSuccess) dur_ms=0.007
    rep3 register rc=0(hipSuccess) dur_ms=1.423 rss_delta_kb=0
         unregister rc=0(hipSuccess) dur_ms=0.004
    rep4 register rc=0(hipSuccess) dur_ms=1.424 rss_delta_kb=0
         unregister rc=0(hipSuccess) dur_ms=0.003

Pricing read: runtime gate PASSES (val=1); pin cost ~1.42-1.57 ms/GiB one-time;
RSS impact 0 KB; unregister free. Notable: registration SUCCEEDS despite the
8 MiB RLIMIT_MEMLOCK on this ROCm stack (the driver does not enforce it on
this path) - if adoption ever happened, the mandated ulimit-failure fallback
would still be required as defense for stacks that DO enforce it.

Benefit-side analysis (why DROP despite cheap pinning):
1. Staging bounce buffers are ALREADY pinned (allocPinned at scheduler setup).
2. Async D2H drains already write DIRECTLY into hostMap
   (pull_scheduler.cpp: dest = sh.hostMap + ..., ops->copyD2HAsync(...)).
3. Large-leg GPU-search cells use the residency handoff: map stays
   device-resident, 0 B H2D, no full-map D2H at all (README Known Issues #2).
   Small legs move KB-MB (sub-ms). Legacy-copy-path D2H volume where pinning
   could matter therefore has no remaining >=15 ms surface on any benchmarked
   cell. Cost known and small; benefit ~zero -> priced out -> DROP.

## Adversarial probes

1. misleading_success_output: the relevant contract is the plan's binding
   constraint "all gains quoted against parsed timers only" plus byte-exact
   stdout (verify.sh goldens), NOT a spawn-to-exit wall contract. For F5 we
   therefore distinguish "timer moved" from "work moved outside the timer":
   had the first launch carried a real load cost, a prewarm would have moved
   it out of t_sieve without shrinking the process - reportable only against
   the parsed-timer contract with the wall effect stated alongside. Moot: the
   measured launch-time cost is 0.26-0.36 ms, so there is nothing to move.
   For F6a the same distinction is quantified: fences are 84 us (@65K) /
   <=2.9 ms mostly-non-removable (@2M) inside a 37 ms / 2819 ms window.
2. hung_commands: every GPU run executed as `timeout 300 <cmd>` while holding
   flock /tmp/opencode/ff-gpu.lock. Longest run ~13 s (amd@2M). No watchdog
   firings, no hangs.

## Verification (pristine HEAD worktree; shared tree carries concurrent lane WIP)

During measurement another lane (kernel A/B queue, tasks 16/17) modified
source/m4/gpu_search_kernel.{cpp,h} + CMakeLists.txt in the shared tree.
Per lane discipline those files were NOT touched, NOT reverted, NOT built-on
for verdicts (all F5/F6a numbers come from sieve-window timestamps unaffected
by m4 sources). Final verification ran in a clean `git worktree` at HEAD:

- Build: cmake --preset dev + build -> rc=0 (56/56 targets, both archs).
- verify.sh (under flock): 65536 default amd PASS byte-identical;
  65536 --gpu-search amd PASS; 2097152 --gpu-search amd PASS (ALL LEGS PASS x3).
- ctest fast suites (-E m4_): 6/6 passed (slab_cmp, ff_budget_selftest,
  slab_geom_bench_amd, slab_geom_bench_nv, abstraction_smoke,
  hostmap_coverage_test).

Environment interference note: one mis-targeted cmake invocation (preset
ignores -B) briefly rebuilt the SHARED tree's build/ from HEAD+WIP-m4 sources
(rc=0). No shared SOURCE files were modified by this task; ninja will treat
that dir as warm cache for the owning lane.

## Changed files / cleanup / risks

- Product code changed: NONE (probe fully reverted; `git diff` clean on
  source/main.cpp, source/pull_scheduler.cpp, source/sieve_slab_engine.cpp).
- This evidence file is the sole artifact of task 19.
- Risks: none carried forward. The ~25 ms search-kernel first-launch constant
  is recorded above for the m4 lane; the memlock non-enforcement quirk is
  recorded for any future pinning proposal.
