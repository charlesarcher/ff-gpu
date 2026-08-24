// SieveSlabKernel — GPU implementation of Prime::SegmentFill
// (segmentedSieve.C:253-268), compiled TWICE per arch (AMD gfx1201 + NVIDIA
// sm_120) into one binary, following the EXACT same per-arch symbol-rename
// pattern as smoke/smoke_kernel.h.
//
// Byte-exact mirror of the CPU SegmentFill:
//   byte i>>4,    bit i>>1&7,    same bit-clear operations
//
// Architecture — shared-staged commutative marking (landed from the task-1
// spike winner "arm B"; evidence and margins:
// .omo/evidence/gpu-speedup/gap-closure/task-1-kernel-gap-closure/spike-report.md):
//   Each active block stages its segment into a 32 KiB static __shared__
//   bitmap in FF_SIEVE_STAGES phases (4 on AMD, 2 on NVIDIA — 2^19 values per
//   phase on both), marks commutatively via shared-memory atomicOr, and
//   flushes cooperatively through the UNCHANGED SieveSlabFlushWord_ word/tail
//   machinery once per phase. This replaces the previous barrier-per-prime
//   design (lane-0 entry staging + two __syncthreads() per prime + direct
//   global word RMW), which serialized every block twice per prime
//   (~2×23k barriers × thousands of blocks/slab) and ran 82–93% below the
//   execution floor on both cards (amd-gap-analysis §2.3). Measured
//   end-to-end through the production pipeline (median-of-3 interleaved,
//   same-session): sieve phase −80.3% AMD / −68.1% NV @1048576;
//   −85.1% / −83.2% @2097152.
//
//   Marking split at FF_SIEVE_SMALL_PRIME_CUT (= 32):
//     - primes {2..31}: cooperative VALUE-SLICED pass — each thread owns one
//       contiguous slice of the phase span and walks ALL small primes inside
//       it (small primes have short strides; slicing balances their work);
//     - larger primes: thread-strided — each thread owns every
//       FF_SIEVE_THREADS_PER_BLOCK-th prime and walks its multiples across
//       the whole phase span (long strides make per-prime ownership cheap).
//   Every mark is an atomicOr (commutative). Slices partition WORK, never
//   memory: any thread may mark any stage word. The stage is zeroed
//   cooperatively ONCE per phase behind a single __syncthreads(); three
//   barriers per phase total (post-init, post-marking/pre-flush,
//   post-flush/pre-restage) vs two per PRIME previously.
//
//   Invariants carried over from the previous design (all preserved):
//     (1) word-exclusive GLOBAL ownership — restored at flush time: each
//         segLo-relative uint32 word is handed to exactly one thread (block
//         bases are multiples of vpt·tpb from segLo and the vpt % 64 == 0
//         assert stands, so flush words never straddle threads); inside the
//         stage everything is commutative atomicOr.
//     (2) tail policy — flush goes through SieveSlabFlushWord_ verbatim:
//         full-word stores only strictly below fullWords = bufMinBytes>>2,
//         partial trailing word via the byte path clamped to bufMinBytes;
//         marks never land at bytes >= bufMinBytes and the last-slab buffer
//         equals copyBytes == bufMinBytes.
//     (3) block-uniform barrier participation — only whole blocks exit
//         (bLo >= segHi). The previous design's per-thread chunk early-return
//         is GONE: threads whose legacy chunk would be empty still slice,
//         strand, and hit every barrier (spike bring-up bug #2 — keeping the
//         return left slices unmarked and made barrier participation
//         divergent). Phase boundaries (phLo >= bHi) are block-uniform.
//     (4) init protocol stays caller-owned (global 0xff memset + byte0 0x7f
//         iff segLo==0); only the LDS stage is zeroed per phase in-kernel.
//     (5) odd-only marking algebra byte-exact: first multiple ≥
//         max(ceil-to-p(base), p²), odd adjust (+p when even), step 2p, bit
//         ~(0x80>>((off>>1)&7)) at byte off>>4. Stage-local offsets are bit-
//         identical to segLo-relative ones because every block/phase base is
//         ≡ segLo (mod 64) — statically guaranteed below — so the staged
//         bitmap produces IDENTICAL bytes through the existing flush path.
//
// Primes live in DEVICE GLOBAL read-only memory (const __restrict__).
//
// Geometry (compile-time table below; per-arch defaults measured by the
// task-13 sweep — evidence under .omo/evidence/gpu-speedup/task-13-gpu-kernel-speedup/):
//   subBlockSize     = 1 << FF_SIEVE_SUB_BLOCK_LOG2   (values)
//   threads/block    = FF_SIEVE_THREADS_PER_BLOCK
//   values/thread    = FF_SIEVE_VALUES_PER_THREAD
//   blocks/sub-block = FF_SIEVE_BLOCKS_PER_SUB_BLOCK
//   grid = numSubBlocks * blocksPerSubBlock per slab, one kernel launch per slab
//
// Per-arch kernel name: SieveSlab_<arch> via two-level macro pasting.

#ifndef FF_SIEVE_SLAB_KERNEL_H
#define FF_SIEVE_SLAB_KERNEL_H

#define FF_KERN_CAT2(a, b) a##b
#define FF_KERN_CAT(a, b)  FF_KERN_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

#define SIEVE_SLAB_KERNEL FF_KERN_CAT(SieveSlab_, SIEVE_KERNEL_ARCH)

// ---- Launch-geometry table (compile-time) -----------------------------------
//
// Four knobs, resolved at compile time so the kernel body, the engine launch
// loop, and the test driver can never disagree:
//   FF_SIEVE_SUB_BLOCK_LOG2        log2 of the sub-block span, in values
//   FF_SIEVE_VALUES_PER_THREAD     values covered by one thread chunk
//   FF_SIEVE_BLOCKS_PER_SUB_BLOCK  blocks sharing one sub-block
//   FF_SIEVE_THREADS_PER_BLOCK     threads launched per block
//
// Invariants (statically enforced below):
//   exact cover — VALUES_PER_THREAD * THREADS_PER_BLOCK * BLOCKS_PER_SUB_BLOCK
//                 == sub-block span. Any other combination leaves values
//                 permanently unmarked (coverage holes); there is no runtime
//                 recovery for it.
//   word alignment — VALUES_PER_THREAD % 64 == 0. Block bases are multiples of
//                 VALUES_PER_THREAD*THREADS_PER_BLOCK from segLo, so no 64-value
//                 word ever straddles two flush owners and global word
//                 ownership stays exclusive.
//
// Defaults are per backend platform, chosen by measurement (task 13):
// median-of-5 kernel time over a fixed 1-GiB region, byte-exactness gated
// per combo on both cards, then finalists re-ranked by end-to-end
// `--devices <vendor> 5 1048576` sieve-phase A/B through the real pipeline.
// A compile may override any subset via -D. Evidence:
// .omo/evidence/gpu-speedup/task-13-gpu-kernel-speedup/sweep.csv
//
//   sm_120:  sub=1<<22 vpt=4096 bps=4 tpb=256
//            sweep median 179.9 ms (old default 382.7); pipeline 1M
//            sieve phase 1440-1454 ms vs 1876-1885 (-23%). Baked winner.
//   gfx1201: sub=1<<22 vpt=8192 bps=2 tpb=256 (task-9 baseline KEPT).
//            The sweep preferred vpt=4096 there (640.7 vs 1069.1 ms), but
//            EVERY vpt=4096 combo REGRESSED the real pipeline (+12..26%);
//            no swept candidate beat the incumbent (3737-3778 ms vs next
//            best 3787+). Kept by measurement, not assumption.
//
// Lesson recorded for tasks 15/16: the isolated cold-launch harness
// overrates small-chunk geometries on AMD vs the copy-overlapped pipeline;
// confirm sweep finalists end-to-end before baking.

#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2        22
#define FF_SIEVE_GEOM_DEFAULT_VALUES_PER_THREAD   4096
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 4
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK  256
#elif defined(__HIP_PLATFORM_AMD__)
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2        22
#define FF_SIEVE_GEOM_DEFAULT_VALUES_PER_THREAD   8192
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 2
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK  256
#else
// Host-only inclusion (no vendor platform): baseline values.
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2        22
#define FF_SIEVE_GEOM_DEFAULT_VALUES_PER_THREAD   8192
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 2
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK  256
#endif

#ifndef FF_SIEVE_SUB_BLOCK_LOG2
#define FF_SIEVE_SUB_BLOCK_LOG2 FF_SIEVE_GEOM_DEFAULT_SUB_LOG2
#endif
#ifndef FF_SIEVE_VALUES_PER_THREAD
#define FF_SIEVE_VALUES_PER_THREAD FF_SIEVE_GEOM_DEFAULT_VALUES_PER_THREAD
#endif
#ifndef FF_SIEVE_BLOCKS_PER_SUB_BLOCK
#define FF_SIEVE_BLOCKS_PER_SUB_BLOCK FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK
#endif
#ifndef FF_SIEVE_THREADS_PER_BLOCK
#define FF_SIEVE_THREADS_PER_BLOCK FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK
#endif

inline constexpr uint64_t kSieveSubBlockSize =
    1ull << FF_SIEVE_SUB_BLOCK_LOG2;
inline constexpr uint64_t kSieveValuesPerThread = FF_SIEVE_VALUES_PER_THREAD;
inline constexpr uint32_t kSieveBlocksPerSubBlock =
    FF_SIEVE_BLOCKS_PER_SUB_BLOCK;
inline constexpr uint32_t kSieveThreadsPerBlock = FF_SIEVE_THREADS_PER_BLOCK;

static_assert((kSieveValuesPerThread & 63ull) == 0,
              "thread chunks must start on 64-value word boundaries");
static_assert(kSieveValuesPerThread <= kSieveSubBlockSize,
              "a thread chunk cannot exceed the sub-block span");
static_assert(kSieveValuesPerThread * kSieveThreadsPerBlock *
                      kSieveBlocksPerSubBlock ==
                  kSieveSubBlockSize,
              "geometry must exactly cover the sub-block span");

// ---- Shared-staging table (compile-time) ------------------------------------
//
// Two knobs on top of the launch geometry, resolved at compile time so the
// kernel body can never disagree with itself:
//   FF_SIEVE_STAGES           phases the block span is split into; the LDS
//                             stage holds blockSpan/STAGES values per phase
//   FF_SIEVE_SMALL_PRIME_CUT  primes strictly below this cut are marked by
//                             the cooperative value-sliced pass; primes at or
//                             above it by thread-owned prime strides
//
// Defaults bake the spike-measured shape: a 32 KiB static stage on BOTH
// platforms (8192 uint32 words = 2^19 values per phase). Literal two-phase
// staging (64 KiB/block) is impossible as portable static LDS — above the
// 48 KiB static-shared ceiling on sm_120 and equal to the whole ROCm-reported
// 64 KiB block limit on gfx1201 with zero headroom — so STAGES=4 (AMD,
// 2^21-value block span) / STAGES=2 (NV, 2^20) lands both archs at 32 KiB.
// The occupancy cost of 32 KiB LDS is sanctioned by measurement: the staged
// design wins end-to-end by 68–85% despite dropping waves/SM (spike §4/§9).
//
// Evidence: .omo/evidence/gpu-speedup/gap-closure/task-1-kernel-gap-closure/
// spike-report.md (shape decision + sensitivity context).

#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_STAGES_DEFAULT 2
#elif defined(__HIP_PLATFORM_AMD__)
#define FF_SIEVE_STAGES_DEFAULT 4
#else
// Host-only inclusion (no vendor platform): AMD-shaped default.
#define FF_SIEVE_STAGES_DEFAULT 4
#endif

#ifndef FF_SIEVE_STAGES
#define FF_SIEVE_STAGES FF_SIEVE_STAGES_DEFAULT
#endif
#ifndef FF_SIEVE_SMALL_PRIME_CUT
#define FF_SIEVE_SMALL_PRIME_CUT 32u
#endif

inline constexpr uint32_t kSieveBlockSpanValues =
    kSieveValuesPerThread * kSieveThreadsPerBlock;
inline constexpr uint32_t kSieveStageWords =
    kSieveBlockSpanValues / (64u * FF_SIEVE_STAGES);
inline constexpr uint32_t kSieveSliceWords =
    kSieveStageWords / kSieveThreadsPerBlock;

// Staging cover asserts (the staged analogues of the geometry asserts):
// the phases must exactly tile the block span, and the per-thread value
// slices must exactly tile one stage — any other combination leaves values
// unmarked or double-sliced with no runtime recovery.
static_assert((uint64_t)kSieveStageWords * 64u * FF_SIEVE_STAGES ==
                  (uint64_t)kSieveBlockSpanValues,
              "staging phases must exactly cover the block span");
static_assert((uint64_t)kSieveSliceWords * kSieveThreadsPerBlock ==
                  (uint64_t)kSieveStageWords,
              "value slices must exactly cover the stage");
// Stage-local bit positions must coincide with segLo-relative ones (invariant
// 5): block bases sit at multiples of subBlockSize/blocksPerSubBlock from
// segLo and phase bases add multiples of kSieveStageWords*64 values, so both
// must stay 64-value aligned for the staged bytes to match the CPU layout.
static_assert((kSieveSubBlockSize / kSieveBlocksPerSubBlock) % 64ull == 0 &&
                  ((uint64_t)kSieveStageWords * 64ull) % 64ull == 0,
              "block/phase bases must stay 64-value aligned so stage-local "
              "bits match the segLo-relative layout");

// ---- Read-only load hint (task 15) ------------------------------------------
// primeList is uploaded once per device (pool static cache) and never written
// by any kernel in this tree, so an explicit read-only load is legal. NVIDIA
// backend: __ldg -> ld.global.nc. AMD backend: plain dereference (HIP porting
// guide documents __ldg as a no-op there; RDNA L1/L2 are read-allocate).
// Guarded so we never depend on AMD intrinsic availability. The hint never
// changes loaded VALUES.
#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_LD_RO_U32(p) __ldg(p)
#else
#define FF_SIEVE_LD_RO_U32(p) (*(p))
#endif

// ---- Word flush helper ------------------------------------------------------
//
// Flushes one accumulated 64-value word mask at word index `w` (bytes
// [4w, 4w+4) of the segLo-relative map).
//
// OUT-OF-BOUNDS TAIL POLICY (mandatory): the kernel receives no buffer-size
// parameter, but every launch site allocates AT LEAST bufMinBytes =
// ceil((segHi-segLo)/16) bytes for the slab under launch:
//   - test harness (sieve_slab_kernel.cpp): hipMalloc((segHi+15)>>4) >= bufMin;
//   - engine / pool slabs: hipMalloc(slabBytes) >= bufMin since
//     segHi - segLo <= slabBytes*16;
//   - right-sized final slab allocations equal copyBytes which equals bufMin
//     exactly (segHi is clamped to totalMapBytes*16 there).
// Only FULL words strictly below bufMinBytes are written as words; any mark
// landing in the partial trailing word (only possible in the last chunk of
// the last block) decomposes into byte RMWs clamped to bufMinBytes. Marks at
// bytes >= bufMinBytes cannot exist (values < segHi), the clamp is defense.
__device__ __forceinline__ void SieveSlabFlushWord_(
    uint8_t* __restrict__ primeMap, uint64_t w, uint32_t mask,
    uint64_t fullWords, uint64_t bufMinBytes)
{
    if (w < fullWords)
    {
        // Full word strictly below the buffer end: single uint32 RMW.
        *reinterpret_cast<uint32_t*>(primeMap + (w << 2)) &= ~mask;
    }
    else
    {
        // Partial trailing word: byte-path fallback, clamped to bufMinBytes.
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
        {
            const uint32_t byteMask = (mask >> (j << 3)) & 0xffu;
            if (byteMask)
            {
                const uint64_t b = (w << 2) | (uint64_t)j;
                if (b < bufMinBytes)
                    primeMap[b] &= (uint8_t)(~byteMask);
            }
        }
    }
}

// ---- Kernel body (shared-staged commutative marking; byte-exact mirror of
//      the reference SegmentFill) ----

__global__ void SIEVE_SLAB_KERNEL(
    const uint32_t* __restrict__ primeList,   // small primes, device global RO
    uint32_t numList,
    uint64_t segLo,
    uint64_t segHi,
    uint8_t* __restrict__ primeMap)            // slab map, word-owned at flush
{
    // Sub-block geometry from the compile-time table above.
    const uint64_t subBlockSize = kSieveSubBlockSize;

    // Each sub-block is split across kSieveBlocksPerSubBlock blocks.
    const uint32_t blocksPerSubBlock = kSieveBlocksPerSubBlock;
    uint64_t subBlockIdx = blockIdx.x / blocksPerSubBlock;
    uint64_t bLo = segLo + subBlockIdx * subBlockSize
                 + (blockIdx.x % blocksPerSubBlock) * (subBlockSize / blocksPerSubBlock);
    uint64_t bHi = bLo + subBlockSize / blocksPerSubBlock;
    if (bHi > segHi) bHi = segHi;
    // ONLY exit in the kernel: whole-block and taken before any barrier, so
    // barrier participation stays block-uniform. There is deliberately NO
    // per-thread chunk early-return here — every thread of an active block
    // participates in stage init, value slices, prime strands, and barriers
    // regardless of whether its legacy chunk would have been empty (spike
    // bring-up bug #2: removing those threads left slices unmarked and made
    // __syncthreads() divergent).
    if (bLo >= segHi) return;

    // Tightest allocation lower bound derivable from the launch parameters
    // (see tail-policy note above SieveSlabFlushWord_).
    const uint64_t span        = segHi - segLo;
    const uint64_t bufMinBytes = (span + 15u) >> 4;
    const uint64_t fullWords   = bufMinBytes >> 2;   // words fully below end

    // Word stores need a 4-byte aligned base. Every allocation site uses
    // hipMalloc (>=256B alignment); this check is pure defense — if it ever
    // fails, the cooperative flush falls back to its byte path below.
    const bool wordOk = ((((uint64_t)(uintptr_t)primeMap) & 3ull) == 0ull);

    const uint32_t tpb = kSieveThreadsPerBlock;
    const uint64_t stageVals = (uint64_t)kSieveStageWords * 64ull;
    const uint64_t sliceVals = (uint64_t)kSieveSliceWords * 64ull;

    // First stage word of this block within the segLo-relative map.
    const uint64_t wordBase = (bLo - segLo) >> 6;

    // Leading run of primes below the cut ({2..31} for the default cut of
    // 32); identical in every thread (same reads) — computed uniformly.
    uint32_t smallCut = 0;
    while (smallCut < numList &&
           (uint64_t)FF_SIEVE_LD_RO_U32(primeList + smallCut) <
               FF_SIEVE_SMALL_PRIME_CUT)
        ++smallCut;

    // 32 KiB static stage bitmap (0 = untouched) reused across phases.
    __shared__ uint32_t stage[kSieveStageWords];

    for (uint32_t ph = 0; ph < FF_SIEVE_STAGES; ++ph)
    {
        // Phase boundaries are block-uniform (phLo/bHi uniform), so the
        // truncated-tail break keeps barrier counts identical per thread.
        const uint64_t phLo = bLo + (uint64_t)ph * stageVals;
        if (phLo >= bHi) break;
        const uint64_t phHi = (phLo + stageVals < bHi) ? (phLo + stageVals)
                                                       : bHi;
        const uint64_t phSpan = phHi - phLo;

        // Cooperative zero-init; ONE barrier before any marking touches the
        // stage (spike bring-up bug #1: init crossing slice ownership plus
        // non-commutative marking silently wiped marks — zero everything
        // first, then mark exclusively through atomicOr).
        for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            stage[w] = 0u;
        __syncthreads();

        // ---- small primes: cooperative value-sliced pass -------------------
        // Thread t owns stage-local value range [t·sliceVals,
        // (t+1)·sliceVals) ∩ [0, phSpan) and walks EVERY prime below the
        // cut inside it. Slices partition WORK only — marks go through
        // commutative atomicOr and may land in any stage word.
        const uint64_t slStart = (uint64_t)threadIdx.x * sliceVals;
        const uint64_t slStop = ((slStart + sliceVals) < phSpan)
                                    ? (slStart + sliceVals)
                                    : phSpan;      // stage-local end offset
        if (slStart < phSpan)
        for (uint32_t k = 0; k < smallCut; ++k)
        {
            const uint64_t p = (uint64_t)FF_SIEVE_LD_RO_U32(primeList + k);
            if (p * p >= bHi) break;              // own slice only — no barrier impact
            const uint64_t slLo = phLo + slStart;
            // First multiple of p >= slLo, floored at p², odd-only adjusted
            // (identical algebra to the reference SegmentFill).
            uint64_t r = slLo % p;
            uint64_t first = slLo + (r ? p - r : 0);
            const uint64_t pp = p * p;
            if (first < pp) first = pp;
            if (!(first & 1)) first += p;
            uint64_t off = first - phLo;          // stage-local offset
            if (off >= slStop) continue;
            uint64_t w = off >> 6;
            uint32_t mask = 0;
            while (off < slStop)
            {
                const uint64_t cw = off >> 6;
                if (cw != w)
                {
                    if (mask) atomicOr(&stage[w], mask);
                    w = cw; mask = 0;
                }
                // bit-in-byte 0x80>>((off>>1)&7) at byte (off>>4)&3 of the
                // word → byte shift ((off & 48) >> 1) ∈ {0,8,16,24}.
                mask |= (0x80u >> ((uint32_t)(off >> 1) & 7u))
                        << (uint32_t)((off & 48u) >> 1);
                off += p << 1;
            }
            if (mask) atomicOr(&stage[w], mask);
        }

        // ---- large primes: thread-owned prime strides (atomicOr) -----------
        // Thread t owns primes smallCut+t, smallCut+t+tpb, ... and walks
        // each one's odd multiples across the WHOLE phase span.
        for (uint32_t k = smallCut + threadIdx.x; k < numList; k += tpb)
        {
            const uint64_t p = (uint64_t)FF_SIEVE_LD_RO_U32(primeList + k);
            if (p * p >= bHi) break;              // strands ascend
            uint64_t r = phLo % p;
            uint64_t first = phLo + (r ? p - r : 0);
            const uint64_t pp = p * p;
            if (first < pp) first = pp;
            if (!(first & 1)) first += p;
            uint64_t off = first - phLo;
            uint64_t w = off >> 6;
            uint32_t mask = 0;
            while (off < phSpan)
            {
                const uint64_t cw = off >> 6;
                if (cw != w)
                {
                    if (mask) atomicOr(&stage[w], mask);
                    w = cw; mask = 0;
                }
                mask |= (0x80u >> ((uint32_t)(off >> 1) & 7u))
                        << (uint32_t)((off & 48u) >> 1);
                off += p << 1;
            }
            if (mask) atomicOr(&stage[w], mask);
        }

        __syncthreads();   // marking complete before cooperative flush

        // ---- cooperative flush through the EXISTING word/tail machinery ---
        // Word-exclusive GLOBAL ownership is restored here: stage word w maps
        // to segLo-relative word wordBase + ph*kSieveStageWords + w, and the
        // strided loop hands each word to exactly one thread. Tail policy is
        // SieveSlabFlushWord_'s, verbatim.
        if (wordOk)
        {
            for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            {
                const uint32_t marks = stage[w];
                if (marks)
                    SieveSlabFlushWord_(primeMap,
                                        wordBase + (uint64_t)ph * kSieveStageWords +
                                            (uint64_t)w,
                                        marks, fullWords, bufMinBytes);
            }
        }
        else
        {
            // Misalignment fallback (never taken with hipMalloc pools): same
            // byte decomposition as SieveSlabFlushWord_'s tail path, minus
            // the full-word shortcut. Byte ownership still exclusive (bytes
            // inherit the word's single owner).
            for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            {
                const uint32_t marks = stage[w];
                if (!marks) continue;
                const uint64_t gw = wordBase +
                                    (uint64_t)ph * kSieveStageWords + (uint64_t)w;
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++j)
                {
                    const uint32_t byteMask = (marks >> (j << 3)) & 0xffu;
                    if (byteMask)
                    {
                        const uint64_t b = (gw << 2) | (uint64_t)j;
                        if (b < bufMinBytes)
                            primeMap[b] &= (uint8_t)(~byteMask);
                    }
                }
            }
        }
        __syncthreads();   // stage consumed before next-phase restage
    }
}

#endif  // FF_SIEVE_SLAB_KERNEL_H
