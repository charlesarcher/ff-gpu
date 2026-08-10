// AMD enumeration TU (plan todo 3). Compiled with the DEFAULT HIP platform
// (hipcc as-is, no HIP_PLATFORM) so hipGetDeviceCount sees the AMD devices
// only (this system: the RX 9070 XT). Includes ONLY the HIP headers — vendor
// headers are never mixed in one TU (ROCm/HIP#2703). Fills vendor-neutral
// ff::DeviceInfo records for ff_enum_hip (extern "C", called from main).
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

void formatBusId(char* out, size_t n, const hipDeviceProp_t& p)
{
    std::snprintf(out, n, "%04x:%02x:%02x", p.pciDomainID, p.pciBusID,
                  p.pciDeviceID);
}

}  // namespace

// Enumerates the AMD devices into `out` (max `maxDevices`). Returns 0 on
// success (outCount = devices filled, possibly 0 = no AMD devices — not an
// error), -1 on a runtime failure (message on stderr, outCount = 0).
extern "C" int ff_enum_hip(ff::DeviceInfo* out, int maxDevices, int* outCount)
{
    *outCount = 0;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        std::fprintf(stderr, "AMD [hip]: hipGetDeviceCount failed: %s (%s)\n",
                     hipGetErrorName(hipGetLastError()),
                     hipGetErrorString(hipGetLastError()));
        return -1;
    }
    if (count < 1) return 0;   // no AMD devices visible — not an error
    if (count > maxDevices) count = maxDevices;

    int prev = -1;
    (void)hipGetDevice(&prev);   // best-effort; the runtime is already up
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) != hipSuccess) {
            std::fprintf(stderr, "AMD [hip]: hipGetDeviceProperties(%d) failed\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        if (hipSetDevice(i) != hipSuccess) {
            std::fprintf(stderr, "AMD [hip]: hipSetDevice(%d) failed\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        size_t freeBytes = 0, totalBytes = 0;
        if (hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess) {
            std::fprintf(stderr, "AMD [hip]: hipMemGetInfo failed on device %d\n", i);
            if (prev >= 0) (void)hipSetDevice(prev);
            return -1;
        }
        ff::DeviceInfo& d = out[*outCount];
        std::memset(&d, 0, sizeof d);
        std::snprintf(d.vendor, sizeof d.vendor, "%s", "amd");
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
