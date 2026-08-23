// Per-platform enumeration TU (plan todo 3). Compiled TWICE from this one
// HIP-only source (no direct CUDA calls anywhere — the NVIDIA side is the
// same HIP API compiled with HIP_PLATFORM=nvidia, where the hip* entry
// points resolve to the CUDA runtime):
//   AMD: hipcc (default platform)            -DSIEVE_KERNEL_ARCH=gfx1201
//   NV:  HIP_PLATFORM=nvidia hipcc -x cu     -DSIEVE_KERNEL_ARCH=sm_120
// Each compile sees only its own vendor's devices and exports a uniquely
// arch-tagged symbol ff_enum_hip_<arch> so both objects can link into one
// binary without collisions. Includes ONLY the HIP headers — vendor headers
// are never mixed in one TU (ROCm/HIP#2703). Fills vendor-neutral
// ff::DeviceInfo records.
//
// Per-device query set required by the plan: hipGetDeviceCount (count),
// hipGetDeviceProperties (name, PCI bus ID, compute capability, max threads,
// shared-mem limits) and hipMemGetInfo (free/total VRAM — queried with the
// device made current, since memGetInfo is current-device scoped).

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#include "device_info.h"

namespace {

#if defined(FF_BACKEND_NV) || defined(__HIP_PLATFORM_NVIDIA__)
#define FF_ENUM_VENDOR "nvidia"
#define FF_ENUM_LABEL  "NV [hip-nv]"
#else
#define FF_ENUM_VENDOR "amd"
#define FF_ENUM_LABEL  "AMD [hip]"
#endif

#define FF_TAG_CAT2(a, b) a##b
#define FF_TAG_CAT(a, b)  FF_TAG_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

// Arch-tagged entry point: ff_enum_hip_gfx1201 / ff_enum_hip_sm_120.
#define FF_ENUM_FN FF_TAG_CAT(ff_enum_hip_, SIEVE_KERNEL_ARCH)

void formatBusId(char* out, size_t n, const hipDeviceProp_t& p)
{
    std::snprintf(out, n, "%04x:%02x:%02x", p.pciDomainID, p.pciBusID,
                  p.pciDeviceID);
}

}  // namespace

// Enumerates this compile's vendor devices into `out` (max `maxDevices`).
// Returns 0 on success (outCount = devices filled, possibly 0 = no devices
// visible to this platform — not an error), -1 on a runtime failure
// (message on stderr, outCount = 0).
extern "C" int FF_ENUM_FN(ff::DeviceInfo* out, int maxDevices, int* outCount)
{
    *outCount = 0;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        std::fprintf(stderr, FF_ENUM_LABEL ": hipGetDeviceCount failed: %s (%s)\n",
                     hipGetErrorName(hipGetLastError()),
                     hipGetErrorString(hipGetLastError()));
        return -1;
    }
    if (count < 1) return 0;   // no devices visible — not an error
    if (count > maxDevices) count = maxDevices;

    int prev = -1;
    (void)hipGetDevice(&prev);   // best-effort; the runtime is already up
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) != hipSuccess) {
            std::fprintf(stderr, FF_ENUM_LABEL ": hipGetDeviceProperties(%d) failed\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        if (hipSetDevice(i) != hipSuccess) {
            std::fprintf(stderr, FF_ENUM_LABEL ": hipSetDevice(%d) failed\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        size_t freeBytes = 0, totalBytes = 0;
        if (hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess) {
            std::fprintf(stderr, FF_ENUM_LABEL ": hipMemGetInfo failed on device %d\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        ff::DeviceInfo& d = out[*outCount];
        std::memset(&d, 0, sizeof d);
        std::snprintf(d.vendor, sizeof d.vendor, "%s", FF_ENUM_VENDOR);
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
    if (prev >= 0) (void)hipSetDevice(prev);   // restore previous current device
    return 0;
}
