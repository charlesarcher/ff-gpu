// Per-arch full-map SieveSlab engine run (sieve_slab_engine.cpp)
//
// Compiled TWICE per arch (AMD gfx1201 + NVIDIA sm_120) into one binary,
// following the EXACT same per-arch symbol-rename pattern as sieve_slab_kernel.cpp.
//
// Slab configuration:
//   SLAB_SIZE_BYTES  = 1 GiB  (1 << 30)
//   values per slab  = SLAB_SIZE_BYTES * 16 = 1 << 34
//   slab k covers    [k * (1<<34), (k+1) * (1<<34))
//
// At 2M the full map is ~16 GiB → 16 full slabs + 1 truncated slab (16 values)
// = 17 launches of SieveSlabKernel.
//
// The kernel operates on the FULL device map (primeMap parameter) — each
// launch only writes to the sub-range [segLo, segHi) by byte-offset indexing.
// The init buffer (all 0xff, bit 0 cleared for value 1) is copied to the
// device ONCE before the slab loop; subsequent kernel launches only mark
// composites within their slab's range.
//
// Arch-tagged via SIEVE_KERNEL_ARCH two-level paste:
//   SieveSlabEngineRun_gfx1201  /  SieveSlabEngineRun_sm_120

#include <hip/hip_runtime.h>

#include "sieve_slab_kernel.h"
#include "sieve_slab_engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint64_t SLAB_SIZE_BYTES = 1ull << 30;   // 1 GiB
constexpr uint32_t kBlockSize = 256;
constexpr uint64_t kSubBlockSize = 1ull << 22;     // 4194304 values (8x baseline)
constexpr uint32_t kBlocksPerSubBlock = 2;

#define ENGINE_HIP_CHECK(call)                                                 \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "engine_hip error at %s:%d: %s (%s)\n",       \
                         __FILE__, __LINE__,                                   \
                         hipGetErrorName(err_), hipGetErrorString(err_));      \
            return -1;                                                         \
        }                                                                      \
    } while (0)

// Arch-tagged host function name (same two-level paste as the kernel and the
// slab_run helper).
#define FF_HOST_CAT2(a, b) a##b
#define FF_HOST_CAT(a, b)  FF_HOST_CAT2(a, b)
#define SIEVE_ENGINE_RUN_NAME  FF_HOST_CAT(SieveSlabEngineRun_,  SIEVE_KERNEL_ARCH)
#define SIEVE_ENGINE_GET_NAME  FF_HOST_CAT(SieveSlabEngineGetLaunchFn_, SIEVE_KERNEL_ARCH)

using SieveSlabEngineRunFn = int (*)(int, const uint32_t*, uint32_t, uint64_t, uint8_t*);

// Forward declaration.
extern "C" int SIEVE_ENGINE_RUN_NAME(int deviceIndex,
                                      const uint32_t* h_primes, uint32_t numPrimes,
                                      uint64_t maxPrimeMapValue, uint8_t* h_out);

// ---- M2 backing pool ops (plan todo 10) ----
#define SIEVE_ENGINE_GET_POOL_NAME FF_HOST_CAT(SievePoolGet_, SIEVE_KERNEL_ARCH)

void* poolAlloc(int vendorIndex, uint64_t bytes);
int poolMemset(int vendorIndex, void* d_ptr, int v, uint64_t bytes);
int poolFree(int vendorIndex, void* d_ptr);
int poolCopyH2D(int vendorIndex, void* d_dst, const void* h_src, uint64_t bytes);
int poolCopyD2H(int vendorIndex, void* h_dst, const void* d_src, uint64_t bytes);
int poolSlabCompute(int vendorIndex, const uint32_t* h_primes, uint32_t numPrimes,
                    uint64_t segLo, uint64_t segHi, void* d_slab,
                    uint64_t slabBytes, int primeSlab);
void* poolCreateStream(int vendorIndex);
int poolDestroyStream(int vendorIndex, void* stream);
void* poolCreateEvent(int vendorIndex);
int poolDestroyEvent(int vendorIndex, void* ev);
int poolRecordEvent(int vendorIndex, void* ev, void* stream);
int poolWaitEvent(int vendorIndex, void* ev, void* stream);
int poolSyncEvent(int vendorIndex, void* ev);
int poolElapsedEvents(int vendorIndex, void* startEv, void* endEv, double* msOut);
int poolAllocPinned(int vendorIndex, void** h, uint64_t bytes);
int poolFreePinned(int vendorIndex, void* h);
int poolCopyH2DAsync(int vendorIndex, void* d_dst, const void* h_src, uint64_t bytes, void* stream);
int poolCopyD2HAsync(int vendorIndex, void* h_dst, const void* d_src, uint64_t bytes, void* stream);
int poolSlabComputeAsync(int vendorIndex, const uint32_t* h_primes, uint32_t numPrimes,
                         uint64_t segLo, uint64_t segHi, void* d_slab, uint64_t slabBytes,
                         void* computeStream, void* waitEvent);

const SievePoolOps kPoolOps = {
    poolAlloc, poolMemset, poolFree, poolCopyH2D, poolCopyD2H, poolSlabCompute,
    poolCreateStream, poolDestroyStream, poolCreateEvent, poolDestroyEvent,
    poolRecordEvent, poolWaitEvent, poolSyncEvent, poolElapsedEvents,
    poolAllocPinned, poolFreePinned, poolCopyH2DAsync, poolCopyD2HAsync,
    poolSlabComputeAsync
};

}  // namespace

// Full-map engine run: allocates per-slab device buffers (each SLAB_SIZE_BYTES
// = 1 GiB), initialises them, loops over slabs launching SieveSlabKernel per
// slab, copies the result back to hostMap.  Splitting into per-slab allocs
// avoids the ~16 GiB single-buffer OOM that HIP/CUDA returns on some GPUs
// (e.g. the 5090) even when total free VRAM is sufficient.
//
// h_primes:     host pointer to small-prime list (uint32_t, numPrimes entries)
//               Must contain primes up to sqrt(maxPrimeMapValue).
// maxPrimeMapValue: the top value covered by the sieve map.
// h_out:        host buffer the caller pre-allocates to ((maxPrimeMapValue+1)+15)>>4
//               bytes.  On return it contains the full sieve map.
//
// Returns 0 on success, -1 on any HIP/CUDA error.
extern "C" int SIEVE_ENGINE_RUN_NAME(int deviceIndex,
                                      const uint32_t* h_primes, uint32_t numPrimes,
                                      uint64_t maxPrimeMapValue, uint8_t* h_out)
{
    // ---- 0. Select the target device (the dispatch helper validated the
    // index but never switched to it; we must do it here so all subsequent
    // hipMalloc / hipLaunch calls hit the right device).
    int prevDev = 0;
    (void)hipGetDevice(&prevDev);
    ENGINE_HIP_CHECK(hipSetDevice(deviceIndex));

    // ---- 0b. Total map size in bytes ----
    const uint64_t totalMapBytes = (maxPrimeMapValue + 1 + 15u) >> 4;

    // ---- 1. Host-side init buffer: all 0xff, clear bit-0 (value 1 not prime) ----
    std::vector<uint8_t> h_init(totalMapBytes, 0xff);
    h_init[0] ^= 0x80u;   // bit 0 of value 0 is index bit 0 → clear for value 1

    // ---- 1b. Per-slab device allocations ----
    // Each slab covers SLAB_SIZE_BYTES*16 values; bytes per slab = SLAB_SIZE_BYTES.
    constexpr uint64_t slabBytes = SLAB_SIZE_BYTES;
    uint64_t numSlabs = (totalMapBytes + slabBytes - 1) / slabBytes;
    if (numSlabs == 0) numSlabs = 1;

    std::vector<uint8_t*> d_slabs(numSlabs, nullptr);
    for (uint64_t i = 0; i < numSlabs; ++i) {
        uint64_t off = i * slabBytes;
        uint64_t copyBytes = ((i + 1) * slabBytes > totalMapBytes)
                                 ? totalMapBytes - off
                                 : slabBytes;
        ENGINE_HIP_CHECK(hipMalloc(&d_slabs[i], slabBytes));
        ENGINE_HIP_CHECK(hipMemcpy(d_slabs[i], h_init.data() + off, copyBytes,
                                   hipMemcpyHostToDevice));
    }

    // ---- 2. Copy small-prime list to device ----
    const size_t primeBytes = static_cast<size_t>(numPrimes) * sizeof(uint32_t);
    uint32_t* d_primes = nullptr;
    ENGINE_HIP_CHECK(hipMalloc(&d_primes, primeBytes));
    ENGINE_HIP_CHECK(hipMemcpy(d_primes, h_primes, primeBytes,
                               hipMemcpyHostToDevice));

    // ---- 3. Loop over slabs ----
    uint64_t segLo = 0;
    uint64_t segHi = SLAB_SIZE_BYTES * 16;   // values per slab
    uint64_t ceiling = maxPrimeMapValue + 1;
    uint64_t slabIdx = 0;

    while (segLo < ceiling) {
        if (segHi > ceiling) segHi = ceiling;

        // Launch SieveSlabKernel for this slab, passing this slab's device ptr.
        const uint64_t numValues = segHi - segLo;
        const uint32_t numSubBlocks = static_cast<uint32_t>(
            (numValues + kSubBlockSize - 1) / kSubBlockSize);
        const uint32_t totalBlocks = numSubBlocks * kBlocksPerSubBlock;

        hipLaunchKernelGGL(SIEVE_SLAB_KERNEL,
                           dim3(totalBlocks), dim3(kBlockSize), 0, 0,
                           d_primes, numPrimes, segLo, segHi, d_slabs[slabIdx]);
        ENGINE_HIP_CHECK(hipGetLastError());
        ENGINE_HIP_CHECK(hipDeviceSynchronize());

        // Advance to next slab.
        segLo = segHi;
        segHi += SLAB_SIZE_BYTES * 16;
        if (segHi > ceiling + SLAB_SIZE_BYTES * 16)
            segHi = ceiling;   // guard against overflow
        ++slabIdx;
    }

    // ---- 4. Copy full device map back to host ----
    for (uint64_t i = 0; i < numSlabs; ++i) {
        uint64_t off = i * slabBytes;
        uint64_t copyBytes = ((i + 1) * slabBytes > totalMapBytes)
                                 ? totalMapBytes - off
                                 : slabBytes;
        ENGINE_HIP_CHECK(hipMemcpy(h_out + off, d_slabs[i], copyBytes,
                                   hipMemcpyDeviceToHost));
    }

    // ---- 5. Cleanup ----
    for (uint64_t i = 0; i < numSlabs; ++i) {
        if (d_slabs[i]) (void)hipFree(d_slabs[i]);
    }
    (void)hipFree(d_primes);
    return 0;
}

// Per-arch dispatch helper: queries the HIP runtime for device count.
// On the matching arch runtime the device is visible (count > 0); on the
// non-matching arch runtime it is not (count == 0).  Returns the arch-
// specific SieveSlabEngineRun entry point, or nullptr if the device is not
// visible on this runtime.  The g++ host code calls BOTH helpers and uses
// whichever returns non-null.
extern "C" SieveSlabEngineRunFn SIEVE_ENGINE_GET_NAME(int deviceIndex)
{
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess) return nullptr;
    if (deviceIndex < 0 || deviceIndex >= devCount) return nullptr;
    return &SIEVE_ENGINE_RUN_NAME;
}

// ---- M2 backing pool op implementations (plan todo 10) ----
// Defined in the anonymous namespace (same unique namespace as the forward
// declarations above) so the symbols get internal linkage — the AMD and NVIDIA
// objects each carry their own pool ops without colliding at link time.
// Every op switches the calling thread's current device to vendorIndex at
// entry; the scheduler threads are per-device, so each call is self-contained.
// hipMemset/hipMemcpy complete on the device's default stream and are followed
// by hipDeviceSynchronize so cross-thread staging transfers never race a peer
// thread's kernel on the same device.
namespace {

#define POOL_HIP_CHECK(call)                                                   \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "pool_hip error at %s:%d: %s (%s)\n",         \
                         __FILE__, __LINE__,                                   \
                         hipGetErrorName(err_), hipGetErrorString(err_));      \
            return -1;                                                         \
        }                                                                      \
    } while (0)

void* poolAlloc(int vendorIndex, uint64_t bytes)
{
    if (hipSetDevice(vendorIndex) != hipSuccess) return nullptr;
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    return p;
}

int poolMemset(int vendorIndex, void* d_ptr, int v, uint64_t bytes)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipMemset(d_ptr, v, bytes));
    POOL_HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

int poolFree(int vendorIndex, void* d_ptr)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipFree(d_ptr));
    return 0;
}

int poolCopyH2D(int vendorIndex, void* d_dst, const void* h_src, uint64_t bytes)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipMemcpy(d_dst, h_src, bytes, hipMemcpyHostToDevice));
    POOL_HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

int poolCopyD2H(int vendorIndex, void* h_dst, const void* d_src, uint64_t bytes)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipMemcpy(h_dst, d_src, bytes, hipMemcpyDeviceToHost));
    POOL_HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

int poolSlabCompute(int vendorIndex, const uint32_t* h_primes, uint32_t numPrimes,
                    uint64_t segLo, uint64_t segHi, void* d_slab,
                    uint64_t slabBytes, int primeSlab)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));

    if (primeSlab) {
        POOL_HIP_CHECK(hipMemset(d_slab, 0xff, slabBytes));
        if (segLo == 0) {
            const uint8_t notOne = 0x7f;   // 0xff with bit-0 clear (value 1)
            POOL_HIP_CHECK(hipMemcpy(d_slab, &notOne, 1, hipMemcpyHostToDevice));
        }
    }

    const size_t primeBytes = static_cast<size_t>(numPrimes) * sizeof(uint32_t);
    uint32_t* d_primes = nullptr;
    POOL_HIP_CHECK(hipMalloc(&d_primes, primeBytes));
    POOL_HIP_CHECK(hipMemcpy(d_primes, h_primes, primeBytes,
                             hipMemcpyHostToDevice));

    const uint64_t numValues = segHi - segLo;
    const uint32_t numSubBlocks = static_cast<uint32_t>(
        (numValues + kSubBlockSize - 1) / kSubBlockSize);
    const uint32_t totalBlocks = numSubBlocks * kBlocksPerSubBlock;

    hipLaunchKernelGGL(SIEVE_SLAB_KERNEL,
                       dim3(totalBlocks), dim3(kBlockSize), 0, 0,
                       d_primes, numPrimes, segLo, segHi,
                       static_cast<uint8_t*>(d_slab));
    POOL_HIP_CHECK(hipGetLastError());
    POOL_HIP_CHECK(hipDeviceSynchronize());
    POOL_HIP_CHECK(hipFree(d_primes));
    return 0;
}

void* poolCreateStream(int vendorIndex)
{
    if (hipSetDevice(vendorIndex) != hipSuccess) return nullptr;
    hipStream_t s = nullptr;
    if (hipStreamCreate(&s) != hipSuccess) return nullptr;
    return reinterpret_cast<void*>(s);
}

int poolDestroyStream(int vendorIndex, void* stream)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipStreamDestroy(reinterpret_cast<hipStream_t>(stream)));
    return 0;
}

void* poolCreateEvent(int vendorIndex)
{
    if (hipSetDevice(vendorIndex) != hipSuccess) return nullptr;
    hipEvent_t e = nullptr;
    if (hipEventCreate(&e) != hipSuccess) return nullptr;
    return reinterpret_cast<void*>(e);
}

int poolDestroyEvent(int vendorIndex, void* ev)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipEventDestroy(reinterpret_cast<hipEvent_t>(ev)));
    return 0;
}

int poolRecordEvent(int vendorIndex, void* ev, void* stream)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipEventRecord(reinterpret_cast<hipEvent_t>(ev),
                                  reinterpret_cast<hipStream_t>(stream)));
    return 0;
}

int poolWaitEvent(int vendorIndex, void* ev, void* stream)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(stream),
                                      reinterpret_cast<hipEvent_t>(ev), 0));
    return 0;
}

int poolSyncEvent(int vendorIndex, void* ev)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipEventSynchronize(reinterpret_cast<hipEvent_t>(ev)));
    return 0;
}

int poolElapsedEvents(int vendorIndex, void* startEv, void* endEv, double* msOut)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    float ms = 0.0f;
    POOL_HIP_CHECK(hipEventElapsedTime(&ms, reinterpret_cast<hipEvent_t>(startEv),
                                       reinterpret_cast<hipEvent_t>(endEv)));
    *msOut = static_cast<double>(ms);
    return 0;
}

int poolAllocPinned(int vendorIndex, void** h, uint64_t bytes)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipHostMalloc(h, bytes, hipHostMallocDefault));
    return 0;
}

int poolFreePinned(int vendorIndex, void* h)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipHostFree(h));
    return 0;
}

int poolCopyH2DAsync(int vendorIndex, void* d_dst, const void* h_src, uint64_t bytes, void* stream)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipMemcpyAsync(d_dst, h_src, bytes, hipMemcpyHostToDevice,
                                  reinterpret_cast<hipStream_t>(stream)));
    return 0;
}

int poolCopyD2HAsync(int vendorIndex, void* h_dst, const void* d_src, uint64_t bytes, void* stream)
{
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    POOL_HIP_CHECK(hipMemcpyAsync(h_dst, d_src, bytes, hipMemcpyDeviceToHost,
                                  reinterpret_cast<hipStream_t>(stream)));
    return 0;
}

// Per-vendor small-prime cache: uploaded once per device, reused across slabs.
// One static per TU (the AMD TU and the NVIDIA TU each carry their own).
namespace {
uint32_t* g_dPrimes = nullptr;
uint32_t g_dPrimeCount = 0;
int g_primeVendor = -1;
}  // namespace

int poolSlabComputeAsync(int vendorIndex, const uint32_t* h_primes, uint32_t numPrimes,
                         uint64_t segLo, uint64_t segHi, void* d_slab, uint64_t slabBytes,
                         void* computeStream, void* waitEvent)
{
    (void)slabBytes;   // no priming here — caller pre-primed via copyH2DAsync
    POOL_HIP_CHECK(hipSetDevice(vendorIndex));
    hipStream_t cs = reinterpret_cast<hipStream_t>(computeStream);
    if (waitEvent) {
        POOL_HIP_CHECK(hipStreamWaitEvent(cs, reinterpret_cast<hipEvent_t>(waitEvent), 0));
    }
    if (g_dPrimes == nullptr || g_primeVendor != vendorIndex || g_dPrimeCount != numPrimes) {
        if (g_dPrimes) { (void)hipFree(g_dPrimes); g_dPrimes = nullptr; }
        const size_t primeBytes = static_cast<size_t>(numPrimes) * sizeof(uint32_t);
        POOL_HIP_CHECK(hipMalloc(&g_dPrimes, primeBytes));
        POOL_HIP_CHECK(hipMemcpy(g_dPrimes, h_primes, primeBytes, hipMemcpyHostToDevice));
        g_dPrimeCount = numPrimes;
        g_primeVendor = vendorIndex;
    }
    const uint64_t numValues = segHi - segLo;
    const uint32_t numSubBlocks = static_cast<uint32_t>(
        (numValues + kSubBlockSize - 1) / kSubBlockSize);
    const uint32_t totalBlocks = numSubBlocks * kBlocksPerSubBlock;
    hipLaunchKernelGGL(SIEVE_SLAB_KERNEL, dim3(totalBlocks), dim3(kBlockSize), 0, cs,
                       g_dPrimes, numPrimes, segLo, segHi,
                       static_cast<uint8_t*>(d_slab));
    POOL_HIP_CHECK(hipGetLastError());
    return 0;
}

#undef POOL_HIP_CHECK

}  // namespace

// Per-arch pool dispatch: same vendor-runtime visibility check as the engine
// getter. The g++ scheduler selects by VENDOR (never "try both in order" — the
// ROCm HIP runtime reports AMD+NVIDIA total device count, so the AMD getter
// would wrongly claim NVIDIA devices).
extern "C" const SievePoolOps* SIEVE_ENGINE_GET_POOL_NAME(int vendorIndex)
{
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess) return nullptr;
    if (vendorIndex < 0 || vendorIndex >= devCount) return nullptr;
    return &kPoolOps;
}