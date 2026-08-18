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
// Optimizations vs. baseline:
//   1. Division elimination: bLo_mod_p = bLo % p, then first = bLo + (p - bLo_mod_p) % p
//   2. Inner loop: running offset accumulator avoids recomputing (i - segLo)
//   3. Larger sub-blocks: 4x (2M values) reduces kernel launches from 17 → ~5
//
// Geometry (optimized):
//   subBlockSize = 1<<21 = 2097152 values (4x baseline)
//   block size   = 256 threads
//   values/thread = 4096 (256 map bytes = 4 cache lines)
//   blocks/sub-block = 2   (512 threads ÷ 256)
//   grid = numSubBlocks per slab, one kernel launch per slab
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

// ---- Kernel body (optimized, byte-exact mirror of reference SegmentFill) ----

__global__ void SIEVE_SLAB_KERNEL(
    const uint32_t* __restrict__ primeList,   // small primes, device global RO
    uint32_t numList,
    uint64_t segLo,
    uint64_t segHi,
    uint8_t* __restrict__ primeMap)            // slab map, byte-owned per thread
{
    // Optimized: 8x larger sub-blocks to reduce kernel launch overhead.
    // 4M values per sub-block = 256 KB map bytes.
    const uint64_t subBlockSize = 1ull << 22;  // 4194304 values (8x baseline)
    // 8x values per thread: 8192 values = 512 map bytes = 8 cache lines
    const uint64_t valuesPerThread = 8192;

    // Each sub-block is split across kBlocksPerSubBlock=2 blocks,
    // each block owning subBlockSize/2 values (1048576).
    const uint32_t blocksPerSubBlock = 2;
    uint64_t subBlockIdx = blockIdx.x / blocksPerSubBlock;
    uint64_t bLo = segLo + subBlockIdx * subBlockSize
                 + (blockIdx.x % blocksPerSubBlock) * (subBlockSize / blocksPerSubBlock);
    uint64_t bHi = bLo + subBlockSize / blocksPerSubBlock;
    if (bHi > segHi) bHi = segHi;
    if (bLo >= segHi) return;

    // Each threadIdx.x covers one 4096-value (256-byte) chunk.
    uint64_t myStart = bLo + (uint64_t)threadIdx.x * valuesPerThread;
    if (myStart >= bHi) return;
    uint64_t myEnd = myStart + valuesPerThread;
    if (myEnd > bHi) myEnd = bHi;

    // Sieve: identical logic to Prime::SegmentFill, with optimizations.
    for (uint32_t k = 0; k < numList; ++k)
    {
        uint64_t p = (uint64_t)primeList[k];
        if (p * p >= bHi) break;                  // early break

        // Optimization 1: Division elimination.
        // Original: first = ((bLo + p - 1) / p) * p;
        // Equivalent: first = bLo + (p - bLo % p) % p;
        // This avoids the multiplication after division.
        uint64_t bLo_mod_p = bLo % p;
        uint64_t first = bLo + ((p - bLo_mod_p) % p);

        if (first < p * p) first = p * p;          // no smaller than p²
        if (!(first & 1)) first += p;              // odd multiples only

        // Advance to myStart if first falls before our chunk.
        uint64_t step = p << 1;
        if (first < myStart)
        {
            // Compute offset from myStart to next valid position.
            uint64_t diff = myStart - first;
            uint64_t k = (diff + step - 1) / step;  // still need this division
            first += k * step;
        }

        // Optimization 2: Inner loop with running offset.
        // Avoid recomputing (i - segLo) for each iteration.
        uint64_t offset = first - segLo;
        for (uint64_t i = first; i < myEnd; i += step)
        {
            // Compute map index and bit offset from running offset.
            uint64_t mapIdx = offset >> 4;
            uint32_t bitOff = (uint32_t)((offset >> 1) & 7u);
            primeMap[mapIdx] &= ~(uint8_t)(0x80u >> bitOff);
            offset += step;
        }
    }
}

#endif  // FF_SIEVE_SLAB_KERNEL_H
