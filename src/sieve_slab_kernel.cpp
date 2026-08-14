// SieveSlabKernel host helper (sieve_slab_kernel.cpp)
//
// Compiled with hipcc — ONCE per arch (AMD / NVIDIA) — so this TU can use
// hip runtime calls directly. The function name is arch-tagged via the
// SIEVE_KERNEL_ARCH macro (SieveSlabRun_gfx1201 / SieveSlabRun_sm120) so
// the two objects never collide at link time.

#include <hip/hip_runtime.h>

#include "sieve_slab_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint64_t kSubBlockSize = 1ull << 19;   // 524288 values
constexpr uint64_t kValuesPerThread = 1024;      // 64 map bytes = 1 cache line
constexpr uint32_t kThreadsPerSubBlock = 512;    // kSubBlockSize / kValuesPerThread
constexpr uint32_t kBlocksPerSubBlock = 2;       // kThreadsPerSubBlock / kBlockSize

#define SLAB_HIP_CHECK(call)                                                 \
    do {                                                                     \
        hipError_t err_ = (call);                                            \
        if (err_ != hipSuccess) {                                            \
            std::fprintf(stderr, "slab_hip error at %s:%d: %s (%s)\n",       \
                         __FILE__, __LINE__,                                 \
                         hipGetErrorName(err_), hipGetErrorString(err_));    \
            return -1;                                                       \
        }                                                                    \
    } while (0)

// Arch-tagged host function name (same two-level paste as the kernel).
#define FF_HOST_CAT2(a, b) a##b
#define FF_HOST_CAT(a, b)  FF_HOST_CAT2(a, b)
#define SIEVE_SLAB_RUN_NAME FF_HOST_CAT(SieveSlabRun_, SIEVE_KERNEL_ARCH)

using SieveSlabRunFn = int (*)(int, const uint32_t*, uint32_t,
                               uint64_t, uint64_t, uint8_t*);

// Forward declaration so the dispatch helper (below) can return &SieveSlabRun_.
extern "C" int SIEVE_SLAB_RUN_NAME(int deviceIndex,
                                   const uint32_t* h_primes, uint32_t numPrimes,
                                   uint64_t segLo, uint64_t segHi,
                                   uint8_t* h_out);

}  // namespace

// Run the SieveSlab kernel for slab [segLo, segHi) on device `deviceIndex`.
//
// h_primes:     host pointer to small-prime list (uint32_t, numPrimes entries)
//               Must already contain primes up to sqrt(segHi).
// segLo/segHi:  value-range [segLo, segHi).  segLo is assumed byte-aligned
//               (multiple of 16) for the test harness.
// h_out:        host buffer the caller pre-allocates to ((segHi+15)>>4) bytes.
//               On return, bytes [segLo>>4 .. (segHi-1)>>4] contain the
//               sieve result (all other bytes are 0xff except [0]=0x7f if
//               segLo==0).
//
// Returns 0 on success, -1 on any HIP error.
extern "C" int SIEVE_SLAB_RUN_NAME(int deviceIndex,
                                   const uint32_t* h_primes, uint32_t numPrimes,
                                   uint64_t segLo, uint64_t segHi,
                                   uint8_t* h_out)
{
    // Map size in bytes from global value 0 up to segHi.
    const uint64_t mapBytes = (segHi + 15u) >> 4;

    // ---- 1. Host-side slab init: memset 0xff, clear bit-0 if range starts at 0 ----
    std::vector<uint8_t> h_init(mapBytes, 0xff);
    if (segLo == 0) {
        h_init[0] ^= 0x80u;   // 1 is not prime
    }

    // ---- 2. Allocate device slab buffer ----
    uint8_t* d_map = nullptr;
    SLAB_HIP_CHECK(hipMalloc(&d_map, mapBytes));

    // ---- 3. Copy init buffer to device ----
    SLAB_HIP_CHECK(hipMemcpy(d_map, h_init.data(), mapBytes,
                             hipMemcpyHostToDevice));

    // ---- 4. Copy small-prime list to device (const __restrict__ global RO) ----
    const size_t primeBytes = static_cast<size_t>(numPrimes) * sizeof(uint32_t);
    uint32_t* d_primes = nullptr;
    SLAB_HIP_CHECK(hipMalloc(&d_primes, primeBytes));
    SLAB_HIP_CHECK(hipMemcpy(d_primes, h_primes, primeBytes,
                             hipMemcpyHostToDevice));

    // ---- 5. Launch kernel ----
    // Flat block grid: each sub-block uses kBlocksPerSubBlock blocks.
    const uint64_t numValues = segHi - segLo;
    const uint32_t numSubBlocks = static_cast<uint32_t>(
        (numValues + kSubBlockSize - 1) / kSubBlockSize);
    const uint32_t totalBlocks = numSubBlocks * kBlocksPerSubBlock;

    hipLaunchKernelGGL(SIEVE_SLAB_KERNEL,
                       dim3(totalBlocks), dim3(kBlockSize), 0, 0,
                       d_primes, numPrimes, segLo, segHi, d_map);
    SLAB_HIP_CHECK(hipGetLastError());
    SLAB_HIP_CHECK(hipDeviceSynchronize());

    // ---- 6. Copy full map back to host ----
    SLAB_HIP_CHECK(hipMemcpy(h_out, d_map, mapBytes,
                             hipMemcpyDeviceToHost));

    // ---- 7. Cleanup ----
    (void)hipFree(d_primes);
    (void)hipFree(d_map);
    return 0;
}

// Per-arch dispatch helper: queries the HIP runtime for device count.
// On the matching arch runtime the device is visible (count > 0); on the
// non-matching arch runtime it is not (count == 0).  Returns the arch-
// specific SieveSlabRun entry point, or nullptr if the device is not
// visible on this runtime.  The g++ test binary calls BOTH helpers and
// uses whichever returns non-null.
extern "C" SieveSlabRunFn FF_HOST_CAT(SieveSlabGetLaunchFn_, SIEVE_KERNEL_ARCH)(
    int deviceIndex)
{
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess) return nullptr;
    if (deviceIndex < 0 || deviceIndex >= devCount) return nullptr;
    return &SIEVE_SLAB_RUN_NAME;
}