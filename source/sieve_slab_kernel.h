// SieveSlabKernel — GPU implementation of Prime::SegmentFill
// (segmentedSieve.C:253-268), compiled TWICE per arch (AMD gfx1201 + NVIDIA
// sm_120) into one binary, following the EXACT same per-arch symbol-rename
// pattern as smoke/smoke_kernel.h.
//
// ============================ WHEEL-30 LAYOUT ============================
// Since the tasks-6+7 pair, this kernel marks the INTERNAL wheel-30 map
// (source/geometry.{h,cpp}): byte k covers values [30k, 30k+30); bit i
// (mask 1u<<i) is value 30k + ff::kWheelResidues[i], residues {1,7,11,13,17,19,
// 23,29}. Primes {2,3,5} are STRUCTURAL (the modulus covers them) and are
// stripped from the marking list by SieveEngine::kernelPrimes(); the D2H
// boundary conversion back to the canonical --dump-map format lives in the
// pull scheduler / main (expandSieveMapToCanonical), so stdout and the
// sha256 map contract are unchanged.
//
// Marking algebra (per prime p > 5, gcd(p,30)=1): marked values are the
// multiples of p coprime to 30, i.e. v = p*t with gcd(t,30)=1. They form
// EIGHT arithmetic progressions of common difference 30p, one per residue
// class: v ≡ ρ (mod 30) ∧ v ≡ 0 (mod p) ⟺ v ≡ p·t₀ (mod 30p) with
// t₀ = (p⁻¹ · ρ) mod 30 (CRT; worked example from the plan draft: p=7,
// ρ=1 → t₀=13 → first member 91). CHOICE DOCUMENTED (plan 7a): eight
// uniform 30p APs, NOT the single-chain G=[6,4,2,4,2,4,6,2] phase stepping.
// Rationale: the chain saves 7 prologs per prime but adds per-mark phase
// bookkeeping; small primes dominate total marks (billions of marks for
// p=7..31 on a 1-GiB-internal slab) so per-mark cost wins over per-prime
// prolog cost, and the uniform-AP body keeps the exact loop shape of the
// proven task-4 strand/slice walkers (one fixed bit per AP, byte offset
// advancing by exactly p per mark).
//
// Per-mark monotonicity (flush relies on it): consecutive members of one AP
// differ by exactly 30p in value, hence by exactly p ≥ 1 in internal-byte
// index — rb is STRICTLY increasing along the walk, so the word index
// w = rb>>2 is nondecreasing and the accumulate-then-flush loop terminates
// with every mark landing at rb < ceil(phSpan/30) ≤ stage capacity.
//
// ====================== GEOMETRY CURRENCY: BYTES =========================
// All launch geometry is expressed in INTERNAL-MAP BYTES (Oracle O3): a
// thread owns B = FF_SIEVE_BYTES_PER_THREAD consecutive internal bytes, its
// value chunk is B·30 values (automatically ≡0 mod 30 and mod 120 — the old
// lcm(64,30)=960 chunk-alignment problem dissolves). Exact cover is
// restated as B·tpb·bps == sub-block span in internal bytes. GUARDRAIL
// DRIFT (plan-sanitized): the old FF_SIEVE_VALUES_PER_THREAD override was
// value-denominated; its numeric defaults (8192/4096) cannot map cleanly
// onto byte-denominated B (8192/30 is not an integer), so defining it now
// is a LOUD #error — use FF_SIEVE_BYTES_PER_THREAD. Defaults below were
// re-derived to preserve the task-4/task-13 measured VALUE-space shapes:
//   block span   = B·tpb bytes: AMD 512·256 = 128 KiB (= 3,932,160 values,
//                  was 2^21 values), NV 256·256 = 64 KiB (= 1,966,080
//                  values, was 2^20);
//   LDS stage    = blockSpan/STAGES bytes = 32 KiB on BOTH archs (8192
//                  words, same as before — one stage word now covers 4
//                  internal bytes = 120 values);
//   sub-block    = 2^18 internal bytes (= 7,864,320 values ≈ old 2^22);
//   grid/slab    = 2^30 B slab / 2^18 B sub-block = 4096 sub-blocks, same
//                  block count per slab as the canonical layout.
//
// RUNTIME CONTRACT (guaranteed by every launch site, checked by the pull
// scheduler): segLo ≡ 0 (mod 30) — slabs start on internal-byte boundaries,
// so segLo-relative byte math coincides with global byte math. Block/phase/
// chunk bases add multiples of 30·(byte quantities), preserving it.
//
// Architecture — shared-staged commutative marking (landed from the task-1
// spike winner; synchronization skeleton UNCHANGED by the wheel re-lay):
//   Each active block stages its segment into a 32 KiB static __shared__
//   bitmap in FF_SIEVE_STAGES phases (4 on AMD, 2 on NVIDIA), marks
//   commutatively via shared-memory atomicOr, and flushes cooperatively
//   through the UNCHANGED SieveSlabFlushWord_ word/tail machinery once per
//   phase. Three barriers per phase; slices/strands partition WORK, never
//   memory; only whole blocks exit before any barrier.
//
//   Marking split at FF_SIEVE_SMALL_PRIME_CUT (= 32; the stripped list makes
//   this {7,11,13,17,19,23,29,31}):
//     - those: cooperative VALUE-SLICED pass — each thread owns one
//       contiguous byte-slice of the phase and walks ALL cut primes inside
//       it;
//     - larger primes: thread-strided — each thread owns every
//       FF_SIEVE_THREADS_PER_BLOCK-th prime across the whole phase span.
//
//   Invariants (restated for the wheel layout):
//     (1) word-exclusive GLOBAL ownership at flush: a stage word covers 4
//         internal bytes; B % 4 == 0 and block bases are multiples of
//         B·tpb bytes from the slab base, so no 4-byte word straddles two
//         flush owners. The plan's sanctioned straddling-byte global-
//         atomicAnd exception is NOT NEEDED: ownership partitions whole
//         bytes by construction (chunks are whole numbers of words).
//     (2) tail policy — SieveSlabFlushWord_ verbatim: full-word stores
//         strictly below fullWords = bufMinBytes>>2, partial trailing word
//         via the byte path clamped to bufMinBytes = ceil((segHi-segLo)/30)
//         (the tightest allocation lower bound derivable from the launch
//         parameters under the 30-values-per-byte layout). Marks at bytes
//         ≥ bufMinBytes cannot exist (values < segHi ⇒ byte < ceil(span/30));
//         the clamp is defense.
//     (3) block-uniform barrier participation — only whole blocks exit
//         (byte base ≥ bufMinBytes). No per-thread early-return anywhere.
//     (4) init protocol stays caller-owned: global 0xff memset + byte 0
//         = 0xfe iff segLo==0 (clears residue-slot 0 = value 1); only the
//         LDS stage is zeroed per phase in-kernel.
//
// Primes live in DEVICE GLOBAL read-only memory (const __restrict__).
//
// Per-arch kernel name: SieveSlab_<arch> via two-level macro pasting.

#ifndef FF_SIEVE_SLAB_KERNEL_H
#define FF_SIEVE_SLAB_KERNEL_H

#include "geometry.h"   // kWheelResidues (constexpr, host/device-safe)

#define FF_KERN_CAT2(a, b) a##b
#define FF_KERN_CAT(a, b)  FF_KERN_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

#define SIEVE_SLAB_KERNEL FF_KERN_CAT(SieveSlab_, SIEVE_KERNEL_ARCH)

// ---- Launch-geometry table (compile-time; INTERNAL-BYTE currency) ----------
//
//   FF_SIEVE_SUB_BLOCK_LOG2          log2 of the sub-block span, internal bytes
//   FF_SIEVE_BYTES_PER_THREAD        internal bytes covered by one thread
//                                    (chunk = B·30 values)
//   FF_SIEVE_BLOCKS_PER_SUB_BLOCK    blocks sharing one sub-block
//   FF_SIEVE_THREADS_PER_BLOCK       threads launched per block
//
// Invariants (statically enforced below):
//   exact cover — BYTES_PER_THREAD * THREADS_PER_BLOCK * BLOCKS_PER_SUB_BLOCK
//                 == sub-block span (internal bytes);
//   word alignment — BYTES_PER_THREAD % 4 == 0 (chunks never split a
//                 4-byte stage/flush word).
//
// Defaults preserve the measured task-13 value-space shapes (see the wheel
// block above); both archs land on a 32 KiB LDS stage.

#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2            18
#define FF_SIEVE_GEOM_DEFAULT_BYTES_PER_THREAD    256
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 4
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK   256
#elif defined(__HIP_PLATFORM_AMD__)
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2            18
#define FF_SIEVE_GEOM_DEFAULT_BYTES_PER_THREAD    512
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 2
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK   256
#else
// Host-only inclusion (no vendor platform): AMD-shaped default.
#define FF_SIEVE_GEOM_DEFAULT_SUB_LOG2            18
#define FF_SIEVE_GEOM_DEFAULT_BYTES_PER_THREAD    512
#define FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK 2
#define FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK   256
#endif

#ifndef FF_SIEVE_SUB_BLOCK_LOG2
#define FF_SIEVE_SUB_BLOCK_LOG2 FF_SIEVE_GEOM_DEFAULT_SUB_LOG2
#endif
// Guardrail drift (documented at the top): the value-denominated override is
// gone; byte-denominated B replaced it. Fail loudly rather than silently
// reinterpreting a value-space number as a byte-space one.
#ifdef FF_SIEVE_VALUES_PER_THREAD
#error "FF_SIEVE_VALUES_PER_THREAD was value-based; the wheel-30 kernel is \
byte-based — override FF_SIEVE_BYTES_PER_THREAD instead"
#endif
#ifndef FF_SIEVE_BYTES_PER_THREAD
#define FF_SIEVE_BYTES_PER_THREAD FF_SIEVE_GEOM_DEFAULT_BYTES_PER_THREAD
#endif
#ifndef FF_SIEVE_BLOCKS_PER_SUB_BLOCK
#define FF_SIEVE_BLOCKS_PER_SUB_BLOCK FF_SIEVE_GEOM_DEFAULT_BLOCKS_PER_SUB_BLOCK
#endif
#ifndef FF_SIEVE_THREADS_PER_BLOCK
#define FF_SIEVE_THREADS_PER_BLOCK FF_SIEVE_GEOM_DEFAULT_THREADS_PER_BLOCK
#endif

inline constexpr uint64_t kSieveSubBlockSize =
    1ull << FF_SIEVE_SUB_BLOCK_LOG2;                     // internal bytes
inline constexpr uint64_t kSieveBytesPerThread = FF_SIEVE_BYTES_PER_THREAD;
inline constexpr uint32_t kSieveBlocksPerSubBlock =
    FF_SIEVE_BLOCKS_PER_SUB_BLOCK;
inline constexpr uint32_t kSieveThreadsPerBlock = FF_SIEVE_THREADS_PER_BLOCK;

static_assert((kSieveBytesPerThread & 3ull) == 0,
              "thread chunks must start on 4-byte word boundaries");
static_assert(kSieveBytesPerThread <= kSieveSubBlockSize,
              "a thread chunk cannot exceed the sub-block span");
static_assert(kSieveBytesPerThread * kSieveThreadsPerBlock *
                      kSieveBlocksPerSubBlock ==
                  kSieveSubBlockSize,
              "geometry must exactly cover the sub-block span (internal bytes)");

// ---- Shared-staging table (compile-time) ------------------------------------

#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_STAGES_DEFAULT 2
#elif defined(__HIP_PLATFORM_AMD__)
#define FF_SIEVE_STAGES_DEFAULT 4
#else
#define FF_SIEVE_STAGES_DEFAULT 4
#endif

#ifndef FF_SIEVE_STAGES
#define FF_SIEVE_STAGES FF_SIEVE_STAGES_DEFAULT
#endif
#ifndef FF_SIEVE_SMALL_PRIME_CUT
#define FF_SIEVE_SMALL_PRIME_CUT 32u
#endif

inline constexpr uint64_t kSieveBlockSpanBytes =
    kSieveBytesPerThread * kSieveThreadsPerBlock;
inline constexpr uint32_t kSieveStageWords =
    static_cast<uint32_t>(kSieveBlockSpanBytes / (4ull * FF_SIEVE_STAGES));
inline constexpr uint32_t kSieveSliceBytes =
    static_cast<uint32_t>(kSieveBlockSpanBytes / FF_SIEVE_STAGES /
                          kSieveThreadsPerBlock);

static_assert((uint64_t)kSieveStageWords * 4ull * FF_SIEVE_STAGES ==
                  (uint64_t)kSieveBlockSpanBytes,
              "staging phases must exactly cover the block span (bytes)");
static_assert((uint64_t)kSieveSliceBytes * kSieveThreadsPerBlock ==
                  kSieveBlockSpanBytes / FF_SIEVE_STAGES,
              "byte slices must exactly cover the stage");

// ---- Read-only load hint (task 15, unchanged) --------------------------------
#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SIEVE_LD_RO_U32(p) __ldg(p)
#else
#define FF_SIEVE_LD_RO_U32(p) (*(p))
#endif

// ---- Word flush helper (UNCHANGED from task 4; semantics now over internal
//      bytes: word w covers bytes [4w, 4w+4) of the segLo-relative map) ------
__device__ __forceinline__ void SieveSlabFlushWord_(
    uint8_t* __restrict__ primeMap, uint64_t w, uint32_t mask,
    uint64_t fullWords, uint64_t bufMinBytes)
{
    if (w < fullWords)
    {
        *reinterpret_cast<uint32_t*>(primeMap + (w << 2)) &= ~mask;
    }
    else
    {
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

// ---- Wheel-30 marking helpers ------------------------------------------------

// Residue by slot / p⁻¹ mod 30. Deliberate LITERAL tables in device code
// (host constexpr arrays are not indexable from __device__ on this toolchain)
// — do NOT derive them from ff::kWheelResidues (task-5 oracle lesson); the
// static_assert below pins the two tables to each other.
__device__ __forceinline__ uint32_t WheelResidue_(uint32_t slot)
{
    switch (slot)
    {
        case 0: return 1;
        case 1: return 7;
        case 2: return 11;
        case 3: return 13;
        case 4: return 17;
        case 5: return 19;
        case 6: return 23;
        case 7: return 29;
    }
    return 0;
}

__device__ __forceinline__ uint32_t WheelInv30_(uint32_t x)
{
    switch (x)
    {
        case 1:  return 1;
        case 7:  return 13;
        case 11: return 11;
        case 13: return 7;
        case 17: return 23;
        case 19: return 19;
        case 23: return 17;
        case 29: return 29;
    }
    return 0;
}

namespace
{
struct WheelTablePin_
{
    // Host-side pin: the device literal tables must mirror ff::kWheelResidues
    // and its modular inverses (compile-time drift guard).
    static constexpr bool ok =
        []() constexpr {
            const uint8_t res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
            const uint32_t inv[8] = {1, 13, 11, 7, 23, 19, 17, 29};
            for (unsigned i = 0; i < 8; ++i)
                if (res[i] != ff::kWheelResidues[i] ||
                    (res[i] * inv[i]) % 30u != 1u)
                    return false;
            return true;
        }();
};
static_assert(WheelTablePin_::ok, "device wheel tables drifted from geometry");
}  // namespace

// Walk ONE residue class of prime p across the window [winLo, winHi) (values,
// ⊆ [phLo, phLo+phSpan)), accumulating commutative marks into the LDS stage.
// Class member: v ≡ p·t₀ (mod 30p), t₀ = (u·ρ) mod 30, u = p⁻¹ mod 30 —
// see the CRT block in the file header. Stage-local byte offset rb advances
// by exactly p per mark (strictly monotone; header proof).
__device__ __forceinline__ void WheelMarkAP_(
    uint32_t* __restrict__ stage,
    uint64_t phLo, uint64_t phSpan,
    uint64_t p, uint32_t u, uint32_t rhoSlot,
    uint64_t winLo, uint64_t winHi)
{
    const uint64_t Mp = 30ull * p;
    const uint64_t pp = p * p;
    const uint64_t S  = winLo > pp ? winLo : pp;   // ternaries: nvcc min() gotcha
    const uint32_t t0 = (u * WheelResidue_(rhoSlot)) % 30u;
    const uint64_t c  = p * t0;
    uint64_t v = c;
    if (S > c) v = c + Mp * ((S - c + Mp - 1) / Mp);
    if (v >= winHi) return;

    const uint32_t bit = 1u << rhoSlot;            // LSB-first slot == bit i
    const uint64_t spanLim = winHi - phLo;         // ≤ phSpan: window bound
    uint64_t off = v - phLo;                       // ≡ ρ (mod 30)
    uint64_t rb  = off / 30;                       // stage-local byte index
    uint64_t w   = rb >> 2;
    uint32_t mask = 0;
    while (off < spanLim)
    {
        const uint64_t cw = rb >> 2;
        if (cw != w)
        {
            if (mask) atomicOr(&stage[w], mask);
            w = cw; mask = 0;
        }
        mask |= bit << ((uint32_t)(rb & 3u) << 3);
        off += Mp;
        rb  += p;
    }
    if (mask) atomicOr(&stage[w], mask);
}

// ---- Kernel body (shared-staged commutative marking; wheel-30 layout) ----

__global__ void SIEVE_SLAB_KERNEL(
    const uint32_t* __restrict__ primeList,   // marking primes (>5), device RO
    uint32_t numList,
    uint64_t segLo,                            // ≡ 0 (mod 30) — runtime contract
    uint64_t segHi,
    uint8_t* __restrict__ primeMap)            // slab map, internal bytes,
                                               // word-owned at flush
{
    const uint64_t bufMinBytes = ((segHi - segLo) + 29ull) / 30ull;
    const uint64_t fullWords   = bufMinBytes >> 2;

    const bool wordOk = ((((uint64_t)(uintptr_t)primeMap) & 3ull) == 0ull);

    const uint32_t tpb = kSieveThreadsPerBlock;
    const uint64_t stageBytes = (uint64_t)kSieveStageWords * 4ull;
    const uint64_t sliceBytes = kSieveSliceBytes;

    // Leading run of primes below the cut ({7..31} on the stripped list),
    // identical in every thread — computed uniformly.
    uint32_t smallCut = 0;
    while (smallCut < numList &&
           (uint64_t)FF_SIEVE_LD_RO_U32(primeList + smallCut) <
               FF_SIEVE_SMALL_PRIME_CUT)
        ++smallCut;

    // 32 KiB static stage bitmap reused across phases.
    __shared__ uint32_t stage[kSieveStageWords];

    // Sub-block/block decomposition in INTERNAL BYTES.
    const uint32_t blocksPerSubBlock = kSieveBlocksPerSubBlock;
    const uint64_t blockSpanBytes = kSieveSubBlockSize / blocksPerSubBlock;
    const uint64_t blockBase =
        (blockIdx.x / blocksPerSubBlock) * kSieveSubBlockSize +
        (blockIdx.x % blocksPerSubBlock) * blockSpanBytes;   // rel to segLo
    // ONLY exit in the kernel: whole-block and taken before any barrier, so
    // barrier participation stays block-uniform.
    if (blockBase >= bufMinBytes) return;
    uint64_t blockEnd = blockBase + blockSpanBytes;
    if (blockEnd > bufMinBytes) blockEnd = bufMinBytes;

    // Block value window (bases are segLo + multiples of 30·bytes ⇒ ≡0 mod 30).
    const uint64_t blkValLo = segLo + blockBase * 30ull;
    const uint64_t blkValHi = segLo + blockEnd * 30ull;
    const uint64_t wordBase = blockBase >> 2;

    for (uint32_t ph = 0; ph < FF_SIEVE_STAGES; ++ph)
    {
        const uint64_t phValLo = blkValLo + (uint64_t)ph * stageBytes * 30ull;
        if (phValLo >= blkValHi) break;                // block-uniform
        const uint64_t phValHi = (phValLo + stageBytes * 30ull < blkValHi)
                                     ? (phValLo + stageBytes * 30ull)
                                     : blkValHi;
        const uint64_t phSpan = phValHi - phValLo;

        // Cooperative zero-init; ONE barrier before any marking touches the
        // stage (spike bring-up bug #1), then exclusive atomicOr marking.
        for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            stage[w] = 0u;
        __syncthreads();

        // ---- cut primes: cooperative byte-sliced pass ----------------------
        // Thread t owns stage-local bytes [t·sliceBytes, (t+1)·sliceBytes)
        // and walks EVERY cut prime inside the corresponding value window.
        const uint64_t slLoB = (uint64_t)threadIdx.x * sliceBytes;
        const uint64_t winLo = phValLo + slLoB * 30ull;
        const uint64_t winHi = phValLo + (slLoB + sliceBytes) * 30ull;
        if (winLo < phValHi)
        {
            const uint64_t wHi = winHi < phValHi ? winHi : phValHi;
            for (uint32_t k = 0; k < smallCut; ++k)
            {
                const uint64_t p = (uint64_t)FF_SIEVE_LD_RO_U32(primeList + k);
                if (p * p >= blkValHi) break;          // own slice only
                const uint32_t u = WheelInv30_((uint32_t)(p % 30ull));
                for (uint32_t rho = 0; rho < ff::kWheelResidueCount; ++rho)
                    WheelMarkAP_(stage, phValLo, phSpan, p, u, rho,
                                 winLo, wHi);
            }
        }

        // ---- large primes: thread-owned prime strides (atomicOr) -----------
        for (uint32_t k = smallCut + threadIdx.x; k < numList; k += tpb)
        {
            const uint64_t p = (uint64_t)FF_SIEVE_LD_RO_U32(primeList + k);
            if (p * p >= blkValHi) break;              // strands ascend
            const uint32_t u = WheelInv30_((uint32_t)(p % 30ull));
            for (uint32_t rho = 0; rho < ff::kWheelResidueCount; ++rho)
                WheelMarkAP_(stage, phValLo, phSpan, p, u, rho,
                             phValLo, phValHi);
        }

        __syncthreads();   // marking complete before cooperative flush

        // ---- cooperative flush through the UNCHANGED word/tail machinery ---
        // Word-exclusive GLOBAL ownership restored here: stage word w maps to
        // the segLo-relative internal word wordBase + ph*kSieveStageWords + w;
        // the strided loop hands each word to exactly one thread. Tail policy
        // is SieveSlabFlushWord_'s, verbatim.
        if (wordOk)
        {
            for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            {
                const uint32_t marks = stage[w];
                if (marks)
                    SieveSlabFlushWord_(primeMap,
                                        wordBase +
                                            (uint64_t)ph * kSieveStageWords +
                                            (uint64_t)w,
                                        marks, fullWords, bufMinBytes);
            }
        }
        else
        {
            // Misalignment fallback (never taken with hipMalloc pools): same
            // byte decomposition as the tail path, minus the word shortcut.
            for (uint32_t w = threadIdx.x; w < kSieveStageWords; w += tpb)
            {
                const uint32_t marks = stage[w];
                if (!marks) continue;
                const uint64_t gw = wordBase +
                                    (uint64_t)ph * kSieveStageWords +
                                    (uint64_t)w;
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
