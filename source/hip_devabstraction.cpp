// Per-arch backend TU of the device abstraction layer (GPU_PLAN §5.2, plan
// todo 6). Compiled TWICE from this one HIP-only source — no direct CUDA
// calls anywhere; the NVIDIA side is the SAME HIP API compiled with
// HIP_PLATFORM=nvidia:
//   AMD: hipcc (default platform)            -DSIEVE_KERNEL_ARCH=gfx1201
//   NV:  HIP_PLATFORM=nvidia hipcc -x cu     -DSIEVE_KERNEL_ARCH=sm_120
// Each compile sees only its own vendor's devices and exports uniquely
// arch-tagged extern "C" symbols ff_dev_hip_<arch>_* so both objects link
// into one binary without collisions. Includes ONLY the HIP headers — vendor
// headers are never mixed in one TU (ROCm/HIP#2703) — plus the shared trivial
// kernel (smoke/smoke_kernel.h, arch-renamed) and the POD-only ffdev
// interface. Every entry point takes the VENDOR runtime index, returns 0/-1.
//
// current-device discipline (todo-3 learning): memGetInfo and default-stream
// ops are current-device scoped — every entry point selects the device,
// does the work, and restores the previous current device.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#include "smoke/smoke_kernel.h"
#include "devabstraction.h"

namespace {

#if defined(FF_BACKEND_NV) || defined(__HIP_PLATFORM_NVIDIA__)
#define FF_BACKEND_VENDOR "nvidia"
#define FF_FAIL_LABEL     "NV [dev-hip]"
#else
#define FF_BACKEND_VENDOR "amd"
#define FF_FAIL_LABEL     "AMD [dev-hip]"
#endif

#define FF_TAG_CAT2(a, b) a##b
#define FF_TAG_CAT(a, b)  FF_TAG_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

#define FF_DEV_FN(name) FF_TAG_CAT(ff_dev_hip_##name##_, SIEVE_KERNEL_ARCH)

constexpr int kBlock = 256;

void hipFail(const char* op, int dev)
{
    std::fprintf(stderr, FF_FAIL_LABEL ": %s on device %d failed: %s (%s)\n",
                 op, dev, hipGetErrorName(hipGetLastError()),
                 hipGetErrorString(hipGetLastError()));
}

void restoreDevice(int prev)
{
    if (prev >= 0) (void)hipSetDevice(prev);   // [[nodiscard]]; best-effort
}

// Fills a DeviceInfo POD the same way src/hip_enum.cpp does, but for a
// single device with a LIVE hipMemGetInfo (current-device scoped) so the
// abstraction and todo 3's CLI report identical fields.
int fillDeviceInfo(int dev, ff::DeviceInfo* out)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) {
        hipFail("hipGetDeviceProperties", dev);
        restoreDevice(prev);
        return -1;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess) {
        hipFail("hipMemGetInfo", dev);
        restoreDevice(prev);
        return -1;
    }
    restoreDevice(prev);

    std::memset(out, 0, sizeof *out);
    std::snprintf(out->vendor, sizeof out->vendor, "%s", FF_BACKEND_VENDOR);
    std::snprintf(out->name, sizeof out->name, "%s", p.name);
    std::snprintf(out->busId, sizeof out->busId, "%04x:%02x:%02x",
                  p.pciDomainID, p.pciBusID, p.pciDeviceID);
    out->pciDomain = p.pciDomainID;
    out->pciBus = p.pciBusID;
    out->pciDevice = p.pciDeviceID;
    out->freeBytes = freeBytes;
    out->totalBytes = totalBytes;
    out->computeMajor = p.major;
    out->computeMinor = p.minor;
    out->maxThreadsPerBlock = p.maxThreadsPerBlock;
    out->sharedMemPerBlock = p.sharedMemPerBlock;
    out->sharedMemPerMultiprocessor = p.sharedMemPerMultiprocessor;
    out->multiProcessorCount = p.multiProcessorCount;
    out->maxGridDimX = p.maxGridSize[0];
    out->maxGridDimY = p.maxGridSize[1];
    out->maxGridDimZ = p.maxGridSize[2];
    out->warpSize = p.warpSize;
    out->runtimeIndex = dev;
    return 0;
}

}  // namespace

extern "C" int FF_DEV_FN(deviceprops)(int dev, ff::DeviceInfo* out)
{
    return fillDeviceInfo(dev, out);
}

extern "C" int FF_DEV_FN(meminfo)(int dev, size_t* freeBytes, size_t* totalBytes)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipError_t e = hipMemGetInfo(freeBytes, totalBytes);
    restoreDevice(prev);
    if (e != hipSuccess) {
        hipFail("hipMemGetInfo", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(alloc)(int dev, size_t bytes, void** out)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipError_t e = hipMalloc(out, bytes);
    restoreDevice(prev);
    if (e != hipSuccess) {
        hipFail("hipMalloc", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(free)(int dev, void* p)
{
    hipError_t e = hipFree(p);
    if (e != hipSuccess) {
        hipFail("hipFree", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(copy_h2d)(int dev, void* dst, const void* src,
                                   size_t bytes)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipError_t e = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
    restoreDevice(prev);
    if (e != hipSuccess) {
        hipFail("hipMemcpy H2D", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(copy_d2h)(int dev, void* dst, const void* src,
                                   size_t bytes)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipError_t e = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    restoreDevice(prev);
    if (e != hipSuccess) {
        hipFail("hipMemcpy D2H", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(stream_create)(int dev, void** out)
{
    hipStream_t s = nullptr;
    hipError_t e = hipStreamCreate(&s);
    if (e != hipSuccess) {
        hipFail("hipStreamCreate", dev);
        return -1;
    }
    *out = s;
    return 0;
}

extern "C" int FF_DEV_FN(stream_destroy)(int dev, void* s)
{
    hipError_t e = hipStreamDestroy(static_cast<hipStream_t>(s));
    if (e != hipSuccess) {
        hipFail("hipStreamDestroy", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(stream_sync)(int dev, void* s)
{
    hipError_t e = hipStreamSynchronize(static_cast<hipStream_t>(s));
    if (e != hipSuccess) {
        hipFail("hipStreamSynchronize", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(event_create)(int dev, void** out)
{
    hipEvent_t ev = nullptr;
    // Default flags: timing enabled (needed for hipEventElapsedTime).
    hipError_t e = hipEventCreate(&ev);
    if (e != hipSuccess) {
        hipFail("hipEventCreate", dev);
        return -1;
    }
    *out = ev;
    return 0;
}

extern "C" int FF_DEV_FN(event_destroy)(int dev, void* e)
{
    hipError_t err = hipEventDestroy(static_cast<hipEvent_t>(e));
    if (err != hipSuccess) {
        hipFail("hipEventDestroy", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(event_record)(int dev, void* e, void* s)
{
    hipError_t err = hipEventRecord(static_cast<hipEvent_t>(e),
                                    static_cast<hipStream_t>(s));
    if (err != hipSuccess) {
        hipFail("hipEventRecord", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(event_sync)(int dev, void* e)
{
    hipError_t err = hipEventSynchronize(static_cast<hipEvent_t>(e));
    if (err != hipSuccess) {
        hipFail("hipEventSynchronize", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(event_elapsed_ms)(int dev, void* e0, void* e1,
                                           float* ms)
{
    // hipEventElapsedTime is only defined for ordered, completed events —
    // a successful call therefore proves e1 was recorded after e0 and both
    // completed (DevEvent ordering).
    hipError_t err = hipEventElapsedTime(ms, static_cast<hipEvent_t>(e0),
                                         static_cast<hipEvent_t>(e1));
    if (err != hipSuccess) {
        hipFail("hipEventElapsedTime", dev);
        return -1;
    }
    return 0;
}

extern "C" int FF_DEV_FN(launch)(int dev, void* buf, int n, void* s)
{
    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(dev) != hipSuccess) {
        hipFail("hipSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    hipStream_t stream = static_cast<hipStream_t>(s);
    hipLaunchKernelGGL(SIEVE_SMOKE_KERNEL, dim3((n + kBlock - 1) / kBlock),
                       dim3(kBlock), 0, stream,
                       static_cast<unsigned int*>(buf), n);
    hipError_t e = hipGetLastError();
    if (e == hipSuccess)
        e = stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
    restoreDevice(prev);
    if (e != hipSuccess) {
        hipFail("hipLaunchKernelGGL", dev);
        return -1;
    }
    return 0;
}
