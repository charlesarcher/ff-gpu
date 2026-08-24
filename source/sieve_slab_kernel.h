// SieveSlabKernel — GPU implementation of Prime::SegmentFill
// (segmentedSieve.C:253-268), compiled TWICE per arch (AMD gfx1201 + NVIDIA
// sm_120) into one binary, following the EXACT same per-arch symbol-rename
// pattern as smoke/smoke_kernel.h.
//
// Byte-exact mirror of the CPU SegmentFill:
//   byte i>>4,    bit i>>1&7,    same bit-clear operations
// Slab boundaries are byte-aligned → exclusive byte ownership → NO atomics.
//
// Primes live in DEVICE GLOBAL read-only memory (const __restrict__).
//
// Optimizations vs. baseline (task 9 redesign):
//   1. Block-shared first-multiple math: bLo % p is identical for all threads
//      of a block, so lane 0 computes it ONCE per prime into __shared__
//      (first shared-memory use in this tree) together with a hoisted
//      round-up reciprocal of step = 2p; every other thread replaces its two
//      emulated 64-bit divisions with ONE multiply-high + shift (exact for
//      the guarded operand ranges, see proof below), falling back to plain
//      division outside those ranges.
//   2. Word-granular marking: marks accumulate into a uint32 mask and flush
//      as one aligned word load/mask/store per touched 64-value word instead
//      of one byte RMW per mark. Word ownership stays exclusive because
//      thread chunks start at multiples of 8192 values (= 128 words) from
//      segLo, so no 64-value word ever straddles two threads.
//   3. Inner loop: running offset avoids recomputing (i - segLo).
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
//   word alignment — VALUES_PER_THREAD % 64 == 0. Chunk starts are multiples
//                 of VALUES_PER_THREAD from segLo, so no 64-value word ever
//                 straddles two threads and word ownership stays exclusive.
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

// ---- Kernel body (optimized, byte-exact mirror of reference SegmentFill) ----

__global__ void SIEVE_SLAB_KERNEL(
    const uint32_t* __restrict__ primeList,   // small primes, device global RO
    uint32_t numList,
    uint64_t segLo,
    uint64_t segHi,
    uint8_t* __restrict__ primeMap)            // slab map, word-owned per thread
{
    // Sub-block geometry from the compile-time table above.
    const uint64_t subBlockSize = kSieveSubBlockSize;
    const uint64_t valuesPerThread = kSieveValuesPerThread;

    // Each sub-block is split across kSieveBlocksPerSubBlock blocks.
    const uint32_t blocksPerSubBlock = kSieveBlocksPerSubBlock;
    uint64_t subBlockIdx = blockIdx.x / blocksPerSubBlock;
    uint64_t bLo = segLo + subBlockIdx * subBlockSize
                 + (blockIdx.x % blocksPerSubBlock) * (subBlockSize / blocksPerSubBlock);
    uint64_t bHi = bLo + subBlockSize / blocksPerSubBlock;
    if (bHi > segHi) bHi = segHi;
    if (bLo >= segHi) return;

    // Each threadIdx.x covers one 8192-value (512-byte) chunk; chunk starts
    // are multiples of 8192 values from segLo, i.e. multiples of 128 uint32
    // words → word-exclusive ownership across the whole grid.
    uint64_t myStart = bLo + (uint64_t)threadIdx.x * valuesPerThread;
    if (myStart >= bHi) return;
    uint64_t myEnd = myStart + valuesPerThread;
    if (myEnd > bHi) myEnd = bHi;

    // Tightest allocation lower bound derivable from the launch parameters
    // (see tail-policy note above SieveSlabFlushWord_).
    const uint64_t span        = segHi - segLo;
    const uint64_t bufMinBytes = (span + 15u) >> 4;
    const uint64_t fullWords   = bufMinBytes >> 2;   // words fully below end

    // Word stores need a 4-byte aligned base. Every allocation site uses
    // hipMalloc (>=256B alignment); this check is pure defense — if it ever
    // fails, the whole launch falls back to the original byte path.
    const bool wordOk = ((((uint64_t)(uintptr_t)primeMap) & 3ull) == 0ull);

    // Per-thread loop-invariant offset form of the chunk end.
    const uint64_t myEndOff = myEnd - segLo;

    // Block-shared per-prime segment-entry info (written by lane 0 only):
    //   shFirst — first candidate multiple of p within [bLo, ...) after the
    //             p² floor and odd-only adjustment (identical semantics to
    //             the baseline scalar computation);
    //   shMul   — round-up reciprocal magic for step = 2p, valid when
    //             step < 2^30 and step not a power of two (0 otherwise);
    //   shShift — floor(log2(step)).
    // ~24 bytes of static shared memory, reused by every prime iteration.
    __shared__ uint64_t shFirst;
    __shared__ uint32_t shMul;
    __shared__ uint32_t shShift;

    // Sieve: identical logic to Prime::SegmentFill, with optimizations.
    for (uint32_t k = 0; k < numList; ++k)
    {
        uint64_t p = (uint64_t)primeList[k];
        if (p * p >= bHi) break;                  // early break (block-uniform)

        if (threadIdx.x == 0)
        {
            // First multiple of p >= bLo. (p - r) % p is r==0 ? 0 : p-r,
            // i.e. branchless-free conditional add — no second division.
            uint64_t r = bLo % p;
            uint64_t first = bLo + (r ? p - r : 0);
            const uint64_t pp = p * p;
            if (first < pp) first = pp;           // no smaller than p²
            if (!(first & 1)) first += p;         // odd multiples only
            shFirst = first;

            const uint64_t step = p << 1;
            shShift = 63u - (uint32_t)__clzll((unsigned long long)step);
            shMul = 0u;
            if (step < (1ull << 30) && (step & (step - 1)) != 0)
            {
                // Round-up reciprocal (Granlund–Montgomery): M = ⌈2^(32+s)/d⌉.
                // Exactness condition n·e < 2^(32+s) with e = M·d − 2^(32+s)
                // ≤ d−1 ≤ 2^(s+1)−1 holds for all n < 2^31 (n·e <
                // 2^31·2^(s+1) = 2^(32+s), strict). Guarded use below keeps
                // n = diff+step−1 < 2^31.
                shMul = (uint32_t)((((uint64_t)1 << (32 + shShift)) + step - 1) / step);
            }
        }
        __syncthreads();   // publish shFirst/shMul/shShift

        // Advance to myStart if first falls before our chunk: at most ONE
        // division-shaped op per prime (multiply-high on the fast path).
        uint64_t first = shFirst;
        if (first < myStart)
        {
            const uint64_t step = p << 1;
            const uint64_t diff = myStart - first;      // < chunk span (< 2^22 here)
            const uint64_t n    = diff + step - 1;      // ceil(diff/step) numerator
            uint64_t kk;
            if (step < (1ull << 30) && diff < (1ull << 31))
            {
                // Fast path: n < 2^31 fits the reciprocal's validity domain.
                // q = floor(n/step) = (n·M) >> (32+s). Power-of-two steps
                // (only p==2 → step==4) take the plain shift.
                kk = shMul ? (((uint64_t)(uint32_t)n * shMul) >> (32 + shShift))
                           : (n >> shShift);
            }
            else
            {
                kk = n / step;   // original division, exact for any geometry
            }
            first += kk * step;
        }

        if (first < myEnd)
        {
            const uint64_t step = p << 1;
            uint64_t off = first - segLo;   // running segLo-relative offset
            if (wordOk)
            {
                // Word-granular marking: accumulate cleared bits per 64-value
                // word, one uint32 load/mask/store per touched word. Marks
                // are strictly increasing → word indices never decrease.
                uint64_t w = off >> 6;
                uint32_t mask = 0;
                do
                {
                    const uint64_t cw = off >> 6;
                    if (cw != w)
                    {
                        SieveSlabFlushWord_(primeMap, w, mask, fullWords, bufMinBytes);
                        w = cw;
                        mask = 0;
                    }
                    // bit-in-byte 0x80>>((off>>1)&7) at byte (off>>4)&3 of the
                    // word → byte shift ((off & 48) >> 1) ∈ {0,8,16,24}.
                    mask |= (0x80u >> ((uint32_t)(off >> 1) & 7u))
                            << (uint32_t)((off & 48u) >> 1);
                    off += step;
                } while (off < myEndOff);
                SieveSlabFlushWord_(primeMap, w, mask, fullWords, bufMinBytes);
            }
            else
            {
                // Original byte-granular RMW path (misalignment fallback).
                for (; off < myEndOff; off += step)
                {
                    primeMap[off >> 4] &= ~(uint8_t)(0x80u >> ((off >> 1) & 7));
                }
            }
        }

        __syncthreads();   // all readers done before lane 0 overwrites next prime
    }
}

#endif  // FF_SIEVE_SLAB_KERNEL_H
