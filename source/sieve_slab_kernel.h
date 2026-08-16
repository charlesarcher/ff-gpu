// SieveSlabKernel — GPU implementation of Prime::SegmentFill
// (segmentedSieve.C:253-268), compiled TWICE per arch (AMD gfx1201 + NVIDIA
// sm_120) into one binary, following the EXACT same per-arch symbol-rename
// pattern as smoke/smoke_kernel.h.
//
// Byte-exact mirror of the CPU SegmentFill:
//   byte i>>4,    bit i>>1&7,    same bit-clear operations
// Slab boundaries are byte-aligned → exclusive byte ownership → NO atomics.
//
// Primes live in DEVICE GLOBAL read-only memory (const __restrict__), NOT
// __constant__: the ~43K primes × 4 bytes = ~172 KB far exceeds the 64 KB
// __constant__ memory limit at 2M.
//
// Geometry:
//   subBlockSize = 1<<19 = 524288 values = 32 KB map bytes = 512 cache lines
//   block size   = 256 threads
//   values/thread = 1024 (one full 64-byte cache line of map data)
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

// ---- Kernel body (byte-exact mirror of reference SegmentFill) ----

__global__ void SIEVE_SLAB_KERNEL(
    const uint32_t* __restrict__ primeList,   // small primes, device global RO
    uint32_t numList,
    uint64_t segLo,
    uint64_t segHi,
    uint8_t* __restrict__ primeMap)            // slab map, byte-owned per thread
{
    const uint64_t subBlockSize = 1ull << 19;  // 524288 values
    const uint64_t valuesPerThread = 1024;     // 64 map bytes = 1 cache line

    // Each sub-block is split across kBlocksPerSubBlock=2 blocks,
    // each block owning subBlockSize/2 values (262144).
    // blockIdx.x is a FLAT block index.  Derive sub-block and
    // intra-sub-block offset from it.
    const uint32_t blocksPerSubBlock = 2;
    uint64_t subBlockIdx = blockIdx.x / blocksPerSubBlock;
    uint64_t bLo = segLo + subBlockIdx * subBlockSize
                 + (blockIdx.x % blocksPerSubBlock) * (subBlockSize / blocksPerSubBlock);
    uint64_t bHi = bLo + subBlockSize / blocksPerSubBlock;
    if (bHi > segHi) bHi = segHi;
    if (bLo >= segHi) return;

    // Each threadIdx.x covers one 1024-value (64-byte) chunk.
    uint64_t myStart = bLo + (uint64_t)threadIdx.x * valuesPerThread;
    if (myStart >= bHi) return;
    uint64_t myEnd = myStart + valuesPerThread;
    if (myEnd > bHi) myEnd = bHi;

    // Sieve: identical logic to Prime::SegmentFill.
    for (uint32_t k = 0; k < numList; ++k)
    {
        uint64_t p = (uint64_t)primeList[k];
        if (p * p >= bHi) break;                  // early break

        uint64_t first = ((bLo + p - 1) / p) * p; // first multiple of p in [bLo, …)
        if (first < p * p) first = p * p;          // no smaller than p²
        if (!(first & 1)) first += p;              // odd multiples only

        // Advance to myStart if first falls before our chunk.
        uint64_t step = p << 1;
        if (first < myStart)
        {
            uint64_t diff = myStart - first;
            first += ((diff + step - 1) / step) * step;
        }

        for (uint64_t i = first; i < myEnd; i += step)
            primeMap[(i - segLo) >> 4] &= ~(0x80 >> ((i - segLo) >> 1 & 7));  // byte-exact bit clear
    }
}

#endif  // FF_SIEVE_SLAB_KERNEL_H