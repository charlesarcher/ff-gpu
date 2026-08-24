// gpu_search_kernel.cpp — Per-arch host entry point for the GPU Freudenthal
// search kernel. Compiled TWICE (AMD gfx1201 + NVIDIA sm_120) with the same
// SIEVE_KERNEL_ARCH mechanism as sieve_slab_kernel.cpp.
//
// Provides extern "C" entry points with arch-tagged names:
//   SearchKernelRun_<arch>(deviceIndex, ...)  — launches SEARCH_KERNEL
//   SearchKernelGetLaunchFn_<arch>            — returns fn pointer or nullptr
//
// Memory management (allocate, copy, launch, free) is done by the test
// harness via DevAlloc/DevCopy from devabstraction.h. The test
// links both per-arch .o files and resolves the active one at runtime.
//
// This file only needs to exist to emit the SEARCH_KERNEL symbol per-arch.
// The kernel body itself is in gpu_search_kernel.h (included by both .cpp files).

#include <hip/hip_runtime.h>

#include <cstdio>

#include "gpu_search_kernel.h"

extern "C" {

// Arch-tagged launch function name (same two-level paste as the kernel).
#define FF_HOST_CAT2(a, b) a##b
#define FF_HOST_CAT(a, b)  FF_HOST_CAT2(a, b)
#define SEARCH_KERNEL_RUN_NAME FF_HOST_CAT(SearchKernelRun_, SIEVE_KERNEL_ARCH)
#define SEARCH_KERNEL_GET_LAUNCH_FN FF_HOST_CAT(SearchKernelGetLaunchFn_, SIEVE_KERNEL_ARCH)

// Launch SEARCH_KERNEL on the given device.
// All memory (primeMap, atomicCount, records, smallPrimes) is pre-allocated on device
// by the caller; this function just launches the kernel.

// Persistent-grid multiplier: blocks per SM (multiProcessorCount), chosen by
// measurement (task 11, median-of-3 `search kernel` ms over leg 1048576,
// sweep k e{1,2,4,8,16,32} on both cards; evidence
// .omo/evidence/gpu-speedup/task-11-gpu-kernel-speedup/happy-deltas.txt).
// k<=2 starves both cards; k>=8 plateaus. Baked winner k=16 on BOTH arches
// (amd 2348.0 ms median, -7% vs k=4; nvidia 851.2 ms, flat vs k=8/32).
// A compile may override via -DFF_SEARCH_BLOCKS_PER_SM.
#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SEARCH_DEFAULT_BLOCKS_PER_SM 16
#elif defined(__HIP_PLATFORM_AMD__)
#define FF_SEARCH_DEFAULT_BLOCKS_PER_SM 16
#else
#define FF_SEARCH_DEFAULT_BLOCKS_PER_SM 1
#endif

#ifndef FF_SEARCH_BLOCKS_PER_SM
#define FF_SEARCH_BLOCKS_PER_SM FF_SEARCH_DEFAULT_BLOCKS_PER_SM
#endif

int SEARCH_KERNEL_RUN_NAME(
    int deviceIndex,
    const uint8_t* d_primeMap, uint64_t d_maxPrimeMapValue,
    uint64_t d_sumStart, uint64_t d_sumLimit,
    uint32_t* d_pAtomicCount, GpuRecord* d_pRecords,
    const uint32_t* d_smallPrimes, uint32_t d_smallPrimeCount)
{
    int prevDevice = -1;
    hipGetDevice(&prevDevice);
    if (hipSetDevice(deviceIndex) != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] kernel: hipSetDevice(%d) failed\n", deviceIndex);
        if (prevDevice >= 0) hipSetDevice(prevDevice);
        return -1;
    }

    int smCount = 0;
    if (hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount,
                              deviceIndex) != hipSuccess || smCount <= 0) {
        std::fprintf(stderr, "  [ffdev] kernel: multiProcessorCount query failed\n");
        hipSetDevice(prevDevice);
        return -1;
    }

    // Zero the device-side work-stealing counter for this launch. Default
    // stream: ordered before the kernel below.
    uint32_t zero = 0;
    if (hipMemcpyToSymbol(HIP_SYMBOL(FF_KERN_CAT(ffSearchWork_, SIEVE_KERNEL_ARCH)),
                          &zero, sizeof(zero)) != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] kernel: work-counter reset failed\n");
        hipSetDevice(prevDevice);
        return -1;
    }

    const uint32_t blockSize = 256;
    const uint32_t numBlocks = (uint32_t)smCount * FF_SEARCH_BLOCKS_PER_SM;

    hipLaunchKernelGGL(SEARCH_KERNEL,
                       dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_primeMap, d_maxPrimeMapValue,
                       d_sumStart, d_sumLimit,
                       d_pAtomicCount, d_pRecords,
                       d_smallPrimes, d_smallPrimeCount);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] kernel launch failed: %s\n", hipGetErrorString(err));
        hipSetDevice(prevDevice);
        return -1;
    }

    hipError_t syncErr = hipDeviceSynchronize();
    if (syncErr != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] kernel sync failed: %s\n", hipGetErrorString(syncErr));
        hipSetDevice(prevDevice);
        return -1;
    }

    hipSetDevice(prevDevice);
    return 0;
}

// Returns a pointer to the arch-specific SearchKernelRun, or nullptr if
// this arch has no device.  The test binary calls BOTH per-arch helpers
// and uses whichever returns non-null.
typedef int (*SearchKernelRunFn)(int, const uint8_t*, uint64_t,
                                  uint64_t, uint64_t,
                                  uint32_t*, GpuRecord*,
                                  const uint32_t*, uint32_t);

SearchKernelRunFn SEARCH_KERNEL_GET_LAUNCH_FN(int deviceIndex)
{
    // Verify the device is usable by this arch before returning the fn.
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess) return nullptr;
    if (deviceIndex < 0 || deviceIndex >= devCount) return nullptr;
    return &SEARCH_KERNEL_RUN_NAME;
}

// ---- Test-only launcher for MRVerdictKernel_<arch> (m4_mr_diff) ----
// Same shape as SEARCH_KERNEL_RUN_NAME: caller pre-allocates all buffers
// through DevAlloc on the SAME logical device it passes here.

#define MR_DIFF_RUN_NAME      FF_HOST_CAT(MRDiffRun_, SIEVE_KERNEL_ARCH)
#define MR_DIFF_GET_LAUNCH_FN FF_HOST_CAT(MRDiffGetLaunchFn_, SIEVE_KERNEL_ARCH)

int MR_DIFF_RUN_NAME(
    int deviceIndex,
    const uint64_t* d_ns, uint8_t* d_verdicts, uint32_t count,
    const uint8_t* d_primeMap, uint64_t maxPrimeMapValue)
{
    int prevDevice = -1;
    hipGetDevice(&prevDevice);
    if (hipSetDevice(deviceIndex) != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] mrdiff: hipSetDevice(%d) failed\n", deviceIndex);
        if (prevDevice >= 0) hipSetDevice(prevDevice);
        return -1;
    }

    uint32_t blockSize = 256;
    uint32_t numBlocks = count ? (count + blockSize - 1) / blockSize : 1;

    hipLaunchKernelGGL(FF_HOST_CAT(MRVerdictKernel_, SIEVE_KERNEL_ARCH),
                       dim3(numBlocks), dim3(blockSize), 0, 0,
                       d_ns, d_verdicts, d_primeMap, maxPrimeMapValue, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] mrdiff launch failed: %s\n", hipGetErrorString(err));
        hipSetDevice(prevDevice);
        return -1;
    }

    hipError_t syncErr = hipDeviceSynchronize();
    if (syncErr != hipSuccess) {
        std::fprintf(stderr, "  [ffdev] mrdiff sync failed: %s\n", hipGetErrorString(syncErr));
        hipSetDevice(prevDevice);
        return -1;
    }

    hipSetDevice(prevDevice);
    return 0;
}

typedef int (*MRDiffRunFn)(int, const uint64_t*, uint8_t*, uint32_t,
                           const uint8_t*, uint64_t);

MRDiffRunFn MR_DIFF_GET_LAUNCH_FN(int deviceIndex)
{
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess) return nullptr;
    if (deviceIndex < 0 || deviceIndex >= devCount) return nullptr;
    return &MR_DIFF_RUN_NAME;
}

// ---- Arch-tagged HIP memory-management wrappers ----
// main.cpp is compiled with g++ and cannot include hip/hip_runtime.h, so
// these wrappers live here (compiled with hipcc per-arch) and are called
// through the vendor-dispatching helpers in gpu_search_launcher.cpp.

#define MEM_WRAPPER_CAT2(a, b) a##b
#define MEM_WRAPPER_CAT(a, b)  MEM_WRAPPER_CAT2(a, b)
#define GpuSearchMemAlloc   MEM_WRAPPER_CAT(GpuSearchMemAlloc_,   SIEVE_KERNEL_ARCH)
#define GpuSearchMemCopyH2D MEM_WRAPPER_CAT(GpuSearchMemCopyH2D_, SIEVE_KERNEL_ARCH)
#define GpuSearchMemCopyD2H MEM_WRAPPER_CAT(GpuSearchMemCopyD2H_, SIEVE_KERNEL_ARCH)
#define GpuSearchMemFree    MEM_WRAPPER_CAT(GpuSearchMemFree_,    SIEVE_KERNEL_ARCH)

int GpuSearchMemAlloc(int deviceIndex, uint64_t bytes, void** outPtr)
{
    int prevDev = -1;
    hipGetDevice(&prevDev);
    if (hipSetDevice(deviceIndex) != hipSuccess) return -1;
    void* p = nullptr;
    hipError_t rc = hipMalloc(&p, static_cast<size_t>(bytes));
    hipSetDevice(prevDev);
    if (rc != hipSuccess) return -1;
    *outPtr = p;
    return 0;
}

int GpuSearchMemCopyH2D(int deviceIndex, void* dst, const void* src, uint64_t bytes)
{
    int prevDev = -1;
    hipGetDevice(&prevDev);
    if (hipSetDevice(deviceIndex) != hipSuccess) return -1;
    hipError_t rc = hipMemcpy(dst, src, static_cast<size_t>(bytes), hipMemcpyHostToDevice);
    hipSetDevice(prevDev);
    return (rc == hipSuccess) ? 0 : -1;
}

int GpuSearchMemCopyD2H(int deviceIndex, void* dst, const void* src, uint64_t bytes)
{
    int prevDev = -1;
    hipGetDevice(&prevDev);
    if (hipSetDevice(deviceIndex) != hipSuccess) return -1;
    hipError_t rc = hipMemcpy(dst, src, static_cast<size_t>(bytes), hipMemcpyDeviceToHost);
    hipSetDevice(prevDev);
    return (rc == hipSuccess) ? 0 : -1;
}

int GpuSearchMemFree(void* ptr)
{
    if (ptr == nullptr) return 0;
    return (hipFree(ptr) == hipSuccess) ? 0 : -1;
}

}  // extern "C"