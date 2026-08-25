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
// Per-device query set (plan todo 10 / F8): targeted hipDeviceGetAttribute
// calls instead of the monolithic hipGetDeviceProperties populate — only the
// fields with live downstream consumers are queried (name via hipDeviceGetName,
// PCI domain/bus/device, compute capability, max threads-per-block, shared-mem
// limits, SM count). Dead fields (maxGridDim*, warpSize) are no longer queried.
// Plus hipGetDeviceCount (count) and hipMemGetInfo (free/total VRAM — queried
// with the device made current, since memGetInfo is current-device scoped).

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

void formatBusId(char* out, size_t n, int domain, int bus, int device)
{
    std::snprintf(out, n, "%04x:%02x:%02x", domain, bus, device);
}

// One targeted attribute query; on failure reports and returns false.
bool queryAttr(int dev, hipDeviceAttribute_t attr, const char* label, int* out)
{
    if (hipDeviceGetAttribute(out, attr, dev) == hipSuccess) return true;
    std::fprintf(stderr, FF_ENUM_LABEL ": %s(%d) failed\n", label, dev);
    return false;
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
        // Targeted queries (F8): name + the 9 attributes with live consumers.
        hipDevice_t hdev = 0;
        char name[256] = {};
        if (hipDeviceGet(&hdev, i) != hipSuccess ||
            hipDeviceGetName(name, static_cast<int>(sizeof name), hdev) !=
                hipSuccess) {
            std::fprintf(stderr, FF_ENUM_LABEL ": device name query(%d) failed\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        int pciDomain = 0, pciBus = 0, pciDev = 0;
        int major = 0, minor = 0, maxThreads = 0;
        int smemBlock = 0, smemMp = 0, smCount = 0;
        if (!queryAttr(i, hipDeviceAttributePciDomainID, "PciDomainID", &pciDomain) ||
            !queryAttr(i, hipDeviceAttributePciBusId, "PciBusId", &pciBus) ||
            !queryAttr(i, hipDeviceAttributePciDeviceId, "PciDeviceId", &pciDev) ||
            !queryAttr(i, hipDeviceAttributeComputeCapabilityMajor,
                       "ComputeCapabilityMajor", &major) ||
            !queryAttr(i, hipDeviceAttributeComputeCapabilityMinor,
                       "ComputeCapabilityMinor", &minor) ||
            !queryAttr(i, hipDeviceAttributeMaxThreadsPerBlock,
                       "MaxThreadsPerBlock", &maxThreads) ||
            !queryAttr(i, hipDeviceAttributeMaxSharedMemoryPerBlock,
                       "SharedMemoryPerBlock", &smemBlock) ||
            !queryAttr(i, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                       "MaxSharedMemoryPerMultiprocessor", &smemMp) ||
            !queryAttr(i, hipDeviceAttributeMultiprocessorCount,
                       "MultiprocessorCount", &smCount)) {
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
        std::snprintf(d.name, sizeof d.name, "%s", name);
        formatBusId(d.busId, sizeof d.busId, pciDomain, pciBus, pciDev);
        d.pciDomain = pciDomain;
        d.pciBus = pciBus;
        d.pciDevice = pciDev;
        d.freeBytes = freeBytes;
        d.totalBytes = totalBytes;
        d.computeMajor = major;
        d.computeMinor = minor;
        d.maxThreadsPerBlock = maxThreads;
        d.sharedMemPerBlock = static_cast<unsigned long long>(smemBlock);
        d.sharedMemPerMultiprocessor = static_cast<unsigned long long>(smemMp);
        d.multiProcessorCount = smCount;
        d.runtimeIndex = i;
        ++*outCount;
    }
    if (prev >= 0) (void)hipSetDevice(prev);   // restore previous current device
    return 0;
}
