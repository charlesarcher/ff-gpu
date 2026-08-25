# F4 slice-disjointness argument — cut-prime plain-OR after large-prime barrier

Task 14 (F4) of `.omo/plans/gpu-optimization-execution.md`.
Scope: `source/sieve_slab_kernel.h` marking phase only. No map-layout change,
no flush-word boundary change, no init-protocol change.

## Reorder under analysis

Per LDS stage phase, the marking order changes from

    zero-init → barrier A → CUT pass (byte-sliced, atomicOr)
              → LARGE pass (prime-strided, atomicOr) → barrier B → flush → barrier C

to

    zero-init → barrier A → LARGE pass (prime-strided, atomicOr)
              → barrier D (NEW) → CUT pass (byte-sliced, PLAIN OR)
              → barrier B → flush → barrier C

Barrier count per phase: **3 → 4**. Justification: the plain-OR RMW in the
cut pass is only race-free if no other thread can touch the same stage word
concurrently. The large pass is prime-strided (many threads hit the same
word), so it must fully drain before any non-atomic writer starts; that drain
point IS barrier D and cannot be merged with barrier B (which must follow the
cut pass to protect the cooperative flush). The +1 is the minimum cost of
de-atomicizing any pass that shares words with another pass.

## Claims

**C1 — slice exclusivity of the cut pass.** Thread t's window is
[phValLo + t·S·30, phValLo + (t+1)·S·30) clamped to phValHi, S = kSieveSliceBytes.
`WheelMarkAP_` marks only v with winLo ≤ v < wHi ⇒ off = v − phLo ∈
[t·S·30, (t+1)·S·30) ∩ [0, phSpan) ⇒ rb = off/30 ∈ [t·S, min((t+1)·S,
ceil(phSpan/30))). Every mark byte lies inside the owner's slice.

**C2 — word exclusivity across slices.** S % 4 == 0 on both archs
(AMD: 512·256/4/256 = 128; NV: 256·256/2/256 = 128). Slices are whole numbers
of 4-byte stage words ⇒ no stage word straddles two slices; during the cut
pass each word has AT MOST ONE writer. Pinned by a new
`static_assert(kSieveSliceBytes % 4 == 0)` so geometry overrides cannot
silently break it (slab_geom_bench builds with -D overrides).

**C3 — no concurrent writers during the cut pass.** `__syncthreads()` (barrier
D) orders all large-pass shared-memory atomicOr writes before every thread's
post-barrier operations. During the cut pass the writer set is exactly the
sliced writers (C1), one per word (C2); within one thread its own read-
modify-writes are program-ordered. No concurrent access to any word ⇒ plain
`stage[w] |= mask` is semantically identical to `atomicOr`.

**C4 — value preservation.** Marking never clears bits; OR into a word already
holding large-pass marks preserves them (OR idempotent/commutative). The
flush (`&= ~mask` into the global map) is unchanged and still runs behind
barrier B. Map bytes exact by the same commutative-OR algebra as task 1.

**C5 — truncated slabs / mid-group tails.**
- Blocks with blockBase ≥ bufMinBytes exit WHOLLY before any barrier
  (block-uniform; invariant (3) untouched).
- Active blocks clamp blockEnd to bufMinBytes; the phase loop breaks when
  phValLo ≥ blkValHi (block-uniform condition); phValHi clamps to blkValHi.
- The sliced window clamps wHi = min(winHi, phValHi): truncation SHRINKS
  windows, it never moves a mark outside [winLo, wHi), hence never outside
  the owner's slice (C1 still applies verbatim).
- Threads whose window starts at/after phValHi skip entirely
  (`if (winLo < phValHi)`): their slices simply have no writer — exclusivity
  trivially holds.
- Mid-group tails: the last block of a sub-block group may be partially
  truncated; same mechanisms apply per-block. Slicing partitions the FULL
  stage word range regardless of truncation; empty-window threads write
  nothing.

**C6 — degenerate spans.**
- phSpan < 30 (sub-byte spans): possibly only thread 0's window intersects;
  single-writer ⇒ exclusive. If even more threads act, C1/C2 bound them to
  disjoint words anyway.
- Empty large pass (numList ≤ smallCut): barrier D is executed by ALL threads
  of active blocks (it sits at phase-body top level, outside every divergent
  branch; the only exit is the pre-loop whole-block return). Cut pass then
  runs on the zeroed stage: plain OR into zeros ≡ original semantics.

**C7 — flush-word ownership invariant untouched.** Flush reads the stage only
after barrier B; ownership math (wordBase + ph·kSieveStageWords + w, tail
clamp b < bufMinBytes, fullWords split) is byte-for-byte unchanged. The
reorder does not change which thread flushes which global word, nor any
global-word straddling argument (block bases remain multiples of
blockSpanBytes, B % 4 == 0).

**C8 — marking-set invariance under reorder.** Cut primes are list indices
[0, smallCut), large primes [smallCut, numList): disjoint index sets. Both
loops break on p² ≥ blkValHi — a condition on blkValHi only, independent of
pass order. The marked-bit set is the union over primes of AP marks; union of
commutative ORs is order-independent.

## slab_cmp twin-matrix checklist (22 cases → claims)

| Case(s) | Description | Covered by |
|---|---|---|
| 1–3, 7, 10 | first/multi sub-blocks, larger range | C1–C4, C7 |
| 21, 22 | stress 20M, deep offset (3rd sub-block + window) | C1–C4, C5 (block-base arithmetic unchanged) |
| 4, 11, 20 | truncated last sub-block / superblock truncation / mid-superblock truncated | C5 |
| 12 | tail mid-group | C5 |
| 13–15 | minimum span (1 byte) / sub-byte span / degenerate span | C6 |
| 17 | byte-boundary end (span == exact multiple) | C5 boundary (wHi == phValHi exactly) |
| 5, 16, 18, 19 | non-aligned starts, sub-block straddle window | segLo ≡ 0 (mod 30) contract + block bases unchanged; C1–C4 per block |
| 6, 8, 9 | tiny ranges (< superblock) | C5 + C6 |

Every case reduces to C1–C8. **Verdict: SOUND** — the disjointness argument is
watertight for the full twin matrix; no case requires atomics in the cut pass
once barrier D exists.

## Honest costs / risks

- +1 `__syncthreads()` per phase (AMD 4 phases, NV 2 phases per block span).
  On AMD this raises barriers/block from 12 to 16. If the A/B shows the
  barrier cost eats the atomic savings, REJECT by numbers.
- Plain OR removes LDS atomic contention from the cut pass ({7..31} dominate
  total mark count, so this is where the atoms are).

## Implementation + gates (change WAS built and verified before A/B)

Reorder implemented exactly as specified (large atomicOr pass → drain
barrier → sliced plain-OR pass; `WheelMarkAP_<bool kAtomic>` with `if
constexpr`; new `static_assert(kSieveSliceBytes % 4 == 0)`). Gates, all green
on the treatment binary:

- flock build both archs: ninja 29/29; fresh per-arch objects verified by
  mtime (sieve_slab_kernel_gfx1201.o / _sm120.o and sieve_slab_engine_*.o all
  rebuilt after the header edit) — stale-state probe covered.
- `verify.sh --all-legs --gpu-search --devices amd`: 6/6 PASS byte-identical.
- `verify.sh --all-legs --gpu-search --devices nvidia`: 6/6 PASS.
- `ctest -R slab_cmp`: **22/22 PASS** (`=== slab_cmp: 22/22 tests passed ===`)
  — twin matrix incl. truncated slabs (4/11/20), tail mid-group (12),
  degenerate/sub-byte spans (13–15), deep offset (22).
- `ctest -R m4_order_bin`: PASS.

## Interleaved A/B — @amd@2097152 --gpu-search, 6 v 6 reps under ff-gpu.lock

Baseline = HEAD 9f51a68 built in an isolated worktree (only diff vs treatment:
the kernel header edit); treatment = working-tree build. One untimed warmup
per side, then strict alternation B,T,B,T,… Raw stderr captured to
/tmp/opencode/f4_ab_{base,treat}_{1..6}.err.

Sieve-phase parsed timer (ms):

| rep | baseline | treatment |
|-----|----------|-----------|
| 1   | 2670.365 | 2725.542  |
| 2   | 2672.336 | 2722.774  |
| 3   | 2668.867 | 2710.927  |
| 4   | 2677.599 | 2723.701  |
| 5   | 2683.091 | 2714.978  |
| 6   | 2684.144 | 2719.051  |
| **median** | **2674.968** | **2720.913** |
| spread | 2668.9–2684.1 | 2710.9–2725.5 |

Δ = **+45.95 ms (+1.72% slower)**; distributions fully DISJOINT (baseline max
2684.144 < treatment min 2710.927) — a real kernel regression, not noise.

### Contamination audit (concurrent workers in the shared tree)

While the A/B ran, other lanes edited search-side files in the shared tree
(m4/gpu_search_kernel.h @13:10:52, emission/gpu_prime later). Audit:

- Sieve-phase metric is UNCONTAMINATED in every scenario: the sieve objects
  compile `source/sieve_slab_kernel.h`, whose on-disk content throughout the
  A/B window was exactly the F4 version (revert happened only afterwards);
  the baseline worktree was clean HEAD; and none of the co-workers' edits
  (search kernel / emission / host search) execute inside the sieve-phase
  timer. The +1.72% disjoint regression is a valid F4-vs-HEAD sieve result.
- E2E walls are ADVISORY ONLY: if a co-worker relinked the main binary with
  their in-flight search-kernel changes mid-window, treatment e2e could carry
  their delta too. The REJECT verdict therefore rests on the sieve-phase
  metric alone (which independently fails the "only if free" bar); the e2e
  direction merely agrees.

E2E process wall (s):

| rep | baseline | treatment |
|-----|----------|-----------|
| 1   | 11.032 | 11.107 |
| 2   | 10.929 | 12.051 |
| 3   | 11.803 | 12.149 |
| 4   | 12.006 | 11.945 |
| 5   | 11.930 | 12.005 |
| 6   | 12.015 | 12.936 |
| **median** | **11.867** | **12.028** |

Δ = **+0.161 s (+1.36% slower e2e)** at the top-of-range cell where research
predicted ≤3.5% upside.

## VERDICT

**REJECT** — cut-prime plain-OR reorder after large-prime barrier: interleaved
A/B @amd@2097152 --gpu-search (6 v 6 reps under ff-gpu.lock) shows sieve-phase
median 2720.913 ms (treatment) vs 2674.968 ms (baseline) = **+1.72% slower**,
distributions disjoint; e2e wall median 12.028 s vs 11.867 s = **+1.36%
slower**. The +1 barrier per phase (intrinsic: the plain-OR RMW is only
race-free after the atomic pass drains, and that drain point cannot merge with
the pre-flush barrier) costs more than the removed LDS atomics on gfx1201.
Kernel change REVERTED; build re-synced to source. The C1–C8 disjointness
proof above is retained on record — correctness was never in question; the
mechanism loses on measured cost. Any future retry needs a design without the
extra barrier (none exists that keeps plain OR safe against a concurrent
atomicOr writer).
