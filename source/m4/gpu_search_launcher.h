// gpu_search_launcher.h — Host-side wrapper that launches the GPU Freudenthal
// search kernel and copies results back to the host.
//
// This is the integration bridge between main.cpp (or any caller) and the
// per-architecture kernel (gpu_search_kernel.h).  The caller is responsible
// for:
//   - Allocating d_pRecords  and d_pAtomicCount on the device via DevAlloc
//   - Ensuring d_pRecords has enough slots (sumLimit - sumStart)/2 + 1
//   - Zero-initialising d_pAtomicCount before launch
//
// This function does NOT:
//   - Allocate device memory
//   - Sort results
//   - Format or print output
//   - Use FP64

#ifndef GPU_SEARCH_LAUNCHER_H
#define GPU_SEARCH_LAUNCHER_H

#include <cstdint>
#include "m4/gpu_search_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Launches the GPU Freudenthal search kernel on the specified device.
///
/// @param deviceIndex          runtime device ordinal within the vendor runtime
/// @param vendor               "amd" -> gfx1201 kernel, "nvidia" -> sm_120 kernel
///                             (both compiled from the same HIP source)
/// @param d_primeMap           Device pointer to the sieve/prime bitmap
/// @param d_maxPrimeMapValue   Maximum value the primeMap covers
/// @param d_sumStart           First odd sum to evaluate
/// @param d_sumLimit           First odd sum PAST the range (exclusive)
/// @param d_pRecords           Device pointer to pre-allocated GpuRecord array
/// @param d_pAtomicCount       Device pointer to pre-allocated atomic uint32
/// @param d_smallPrimes        Device pointer to pre-allocated small primes array
/// @param d_smallPrimeCount    Number of primes in d_smallPrimes
/// @return 0 on success, non-zero on failure
int GpuSearchLaunch(int deviceIndex,
                    const char* vendor,
                    const uint8_t* d_primeMap,
                    uint64_t d_maxPrimeMapValue,
                    uint64_t d_sumStart,
                    uint64_t d_sumLimit,
                    GpuRecord* d_pRecords,
                    uint32_t* d_pAtomicCount,
                    const uint32_t* d_smallPrimes,
                    uint32_t d_smallPrimeCount);

int GpuSearchAlloc(int deviceIndex, const char* vendor, uint64_t bytes, void** outPtr);
int GpuSearchCopyH2D(int deviceIndex, const char* vendor, void* dst, const void* src, uint64_t bytes);
int GpuSearchCopyD2H(int deviceIndex, const char* vendor, void* dst, const void* src, uint64_t bytes);
int GpuSearchFree(int deviceIndex, const char* vendor, void* ptr);

#ifdef __cplusplus
}
#endif

#endif // GPU_SEARCH_LAUNCHER_H