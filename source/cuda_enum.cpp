// NVIDIA enumeration TU (plan todo 3). Compiled with HIP_PLATFORM=nvidia
// /opt/rocm/bin/hipcc -x cu -arch=sm_120 (delegates to nvcc, CUDA 13.3) — the
// exact todo-4 NVIDIA path. Includes ONLY the CUDA headers — vendor headers
// are never mixed in one TU (ROCm/HIP#2703). Fills vendor-neutral
// ff::DeviceInfo records for ff_enum_cuda (extern "C", called from main).
//
// Per-device query set required by the plan: cudaGetDeviceCount (count),
// cudaGetDeviceProperties (name, PCI bus ID, compute capability, max threads,
// shared-mem limits) and cudaMemGetInfo (free/total VRAM — current-device
// scoped, so the device is made current before the query).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#include "device_info.h"

namespace {

void formatBusId(char* out, size_t n, const cudaDeviceProp& p)
{
    std::snprintf(out, n, "%04x:%02x:%02x", p.pciDomainID, p.pciBusID,
                  p.pciDeviceID);
}

}  // namespace

// Enumerates the NVIDIA devices into `out` (max `maxDevices`). Returns 0 on
// success (outCount = devices filled, possibly 0 = no NVIDIA devices — not an
// error), -1 on a runtime failure (message on stderr, outCount = 0).
extern "C" int ff_enum_cuda(ff::DeviceInfo* out, int maxDevices, int* outCount)
{
    *outCount = 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        std::fprintf(stderr, "NV  [cuda]: cudaGetDeviceCount failed: %s (%s)\n",
                     cudaGetErrorName(cudaGetLastError()),
                     cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    if (count < 1) return 0;   // no NVIDIA devices visible — not an error
    if (count > maxDevices) count = maxDevices;

    int prev = -1;
    cudaGetDevice(&prev);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) {
            std::fprintf(stderr, "NV  [cuda]: cudaGetDeviceProperties(%d) failed\n", i);
            if (prev >= 0) cudaSetDevice(prev);
            return -1;
        }
        if (cudaSetDevice(i) != cudaSuccess) {
            std::fprintf(stderr, "NV  [cuda]: cudaSetDevice(%d) failed\n", i);
            if (prev >= 0) cudaSetDevice(prev);
            return -1;
        }
        size_t freeBytes = 0, totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
            std::fprintf(stderr, "NV  [cuda]: cudaMemGetInfo failed on device %d\n", i);
            if (prev >= 0) cudaSetDevice(prev);
            return -1;
        }
        ff::DeviceInfo& d = out[*outCount];
        std::memset(&d, 0, sizeof d);
        std::snprintf(d.vendor, sizeof d.vendor, "%s", "nvidia");
        std::snprintf(d.name, sizeof d.name, "%s", p.name);
        formatBusId(d.busId, sizeof d.busId, p);
        d.pciDomain = p.pciDomainID;
        d.pciBus = p.pciBusID;
        d.pciDevice = p.pciDeviceID;
        d.freeBytes = freeBytes;
        d.totalBytes = totalBytes;
        d.computeMajor = p.major;
        d.computeMinor = p.minor;
        d.maxThreadsPerBlock = p.maxThreadsPerBlock;
        d.sharedMemPerBlock = p.sharedMemPerBlock;
        d.sharedMemPerMultiprocessor = p.sharedMemPerMultiprocessor;
        d.multiProcessorCount = p.multiProcessorCount;
        d.maxGridDimX = p.maxGridSize[0];
        d.maxGridDimY = p.maxGridSize[1];
        d.maxGridDimZ = p.maxGridSize[2];
        d.warpSize = p.warpSize;
        d.runtimeIndex = i;
        ++*outCount;
    }
    if (prev >= 0) cudaSetDevice(prev);
    return 0;
}
