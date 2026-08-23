// gpu_search_launcher.cpp — Host-side wrapper that resolves and launches
// the per-arch Freudenthal search kernel on a given GPU device.
//
// Strategy:
//   1. Dispatch by the device's vendor string to the matching arch-tagged
//      launch function (gfx1201 = AMD HIP objects, sm_120 = NVIDIA
//      HIP-platform objects — both compiled from the same HIP source).
//   2. If the launch function is unavailable (no such device),
//      return failure — the caller degrades to CPU search.
//
// The per-arch SearchKernelRun_<arch> function handles the actual
// hipLaunchKernelGGL call (blockSize=256, auto-computed numBlocks),
// so this wrapper only needs to dispatch to the right symbol.

#include "m4/gpu_search_launcher.h"

#include <cstring>

extern "C" {
    typedef int (*SearchKernelRunFn)(int deviceIndex,
                                      const uint8_t* d_primeMap,
                                      uint64_t d_maxPrimeMapValue,
                                      uint64_t d_sumStart,
                                      uint64_t d_sumLimit,
                                      uint32_t* d_pAtomicCount,
                                      GpuRecord* d_pRecords,
                                      const uint32_t* d_smallPrimes,
                                      uint32_t d_smallPrimeCount);

    extern SearchKernelRunFn SearchKernelGetLaunchFn_gfx1201(int deviceIndex);
    extern SearchKernelRunFn SearchKernelGetLaunchFn_sm_120(int deviceIndex);

    extern int GpuSearchMemAlloc_gfx1201(int deviceIndex, uint64_t bytes, void** outPtr);
    extern int GpuSearchMemCopyH2D_gfx1201(int deviceIndex, void* dst, const void* src, uint64_t bytes);
    extern int GpuSearchMemCopyD2H_gfx1201(int deviceIndex, void* dst, const void* src, uint64_t bytes);
    extern int GpuSearchMemFree_gfx1201(void* ptr);

    extern int GpuSearchMemAlloc_sm_120(int deviceIndex, uint64_t bytes, void** outPtr);
    extern int GpuSearchMemCopyH2D_sm_120(int deviceIndex, void* dst, const void* src, uint64_t bytes);
    extern int GpuSearchMemCopyD2H_sm_120(int deviceIndex, void* dst, const void* src, uint64_t bytes);
    extern int GpuSearchMemFree_sm_120(void* ptr);
}

static bool isNvidia(const char* vendor)
{
    return vendor != nullptr && std::strcmp(vendor, "nvidia") == 0;
}

int GpuSearchLaunch(int deviceIndex,
                    const char* vendor,
                    const uint8_t* d_primeMap,
                    uint64_t d_maxPrimeMapValue,
                    uint64_t d_sumStart,
                    uint64_t d_sumLimit,
                    GpuRecord* d_pRecords,
                    uint32_t* d_pAtomicCount,
                    const uint32_t* d_smallPrimes,
                    uint32_t d_smallPrimeCount)
{
    SearchKernelRunFn runFn = isNvidia(vendor)
                                  ? SearchKernelGetLaunchFn_sm_120(deviceIndex)
                                  : SearchKernelGetLaunchFn_gfx1201(deviceIndex);
    if (runFn == nullptr) return -1;

    return runFn(deviceIndex,
                 d_primeMap, d_maxPrimeMapValue,
                 d_sumStart, d_sumLimit,
                 d_pAtomicCount, d_pRecords,
                 d_smallPrimes, d_smallPrimeCount);
}

// ---- Memory-management wrappers: dispatch to arch-tagged per-arch symbols ----

int GpuSearchAlloc(int deviceIndex, const char* vendor, uint64_t bytes, void** outPtr)
{
    if (outPtr == nullptr) return -1;
    *outPtr = nullptr;
    return isNvidia(vendor)
               ? GpuSearchMemAlloc_sm_120(deviceIndex, bytes, outPtr)
               : GpuSearchMemAlloc_gfx1201(deviceIndex, bytes, outPtr);
}

int GpuSearchCopyH2D(int deviceIndex, const char* vendor, void* dst, const void* src, uint64_t bytes)
{
    return isNvidia(vendor)
               ? GpuSearchMemCopyH2D_sm_120(deviceIndex, dst, src, bytes)
               : GpuSearchMemCopyH2D_gfx1201(deviceIndex, dst, src, bytes);
}

int GpuSearchCopyD2H(int deviceIndex, const char* vendor, void* dst, const void* src, uint64_t bytes)
{
    return isNvidia(vendor)
               ? GpuSearchMemCopyD2H_sm_120(deviceIndex, dst, src, bytes)
               : GpuSearchMemCopyD2H_gfx1201(deviceIndex, dst, src, bytes);
}

int GpuSearchFree(int deviceIndex, const char* vendor, void* ptr)
{
    (void)deviceIndex;
    if (ptr == nullptr) return 0;
    return isNvidia(vendor) ? GpuSearchMemFree_sm_120(ptr)
                            : GpuSearchMemFree_gfx1201(ptr);
}