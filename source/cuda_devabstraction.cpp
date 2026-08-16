// NVIDIA backend TU of the device abstraction layer (GPU_PLAN §5.2, plan
// todo 6). Compiled with HIP_PLATFORM=nvidia /opt/rocm/bin/hipcc -x cu
// -arch=sm_120 (delegates to nvcc, CUDA 13.3) so this TU sees ONLY the
// NVIDIA devices (this system: the RTX 5090). Includes ONLY the CUDA
// headers — vendor headers are never mixed in one TU (ROCm/HIP#2703) —
// plus the shared trivial kernel (smoke/smoke_kernel.h, arch-renamed to
// SieveSlab_sm_120 here) and the POD-only ffdev interface. Every entry
// point is extern "C", takes the VENDOR runtime index, returns 0 / -1.
//
// current-device discipline (todo-3 learning): memGetInfo and default-stream
// ops are current-device scoped — every entry point selects the device,
// does the work, and restores the previous current device.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#include "smoke/smoke_kernel.h"
#include "devabstraction.h"

namespace {

constexpr int kBlock = 256;

void cudaFail(const char* op, int dev)
{
    std::fprintf(stderr, "NV [dev-cuda]: %s on device %d failed: %s (%s)\n",
                 op, dev, cudaGetErrorName(cudaGetLastError()),
                 cudaGetErrorString(cudaGetLastError()));
}

void restoreDevice(int prev)
{
    if (prev >= 0) cudaSetDevice(prev);
}

// Fills a DeviceInfo POD the same way src/cuda_enum.cpp does, but for a
// single device with a LIVE cudaMemGetInfo (current-device scoped) so the
// abstraction and todo 3's CLI report identical fields.
int fillDeviceInfo(int dev, ff::DeviceInfo* out)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cudaFail("cudaGetDeviceProperties", dev);
        restoreDevice(prev);
        return -1;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        cudaFail("cudaMemGetInfo", dev);
        restoreDevice(prev);
        return -1;
    }
    restoreDevice(prev);

    std::memset(out, 0, sizeof *out);
    std::snprintf(out->vendor, sizeof out->vendor, "%s", "nvidia");
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

extern "C" int ff_dev_cuda_deviceprops(int dev, ff::DeviceInfo* out)
{
    return fillDeviceInfo(dev, out);
}

extern "C" int ff_dev_cuda_meminfo(int dev, size_t* freeBytes,
                                   size_t* totalBytes)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaError_t e = cudaMemGetInfo(freeBytes, totalBytes);
    restoreDevice(prev);
    if (e != cudaSuccess) {
        cudaFail("cudaMemGetInfo", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_alloc(int dev, size_t bytes, void** out)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaError_t e = cudaMalloc(out, bytes);
    restoreDevice(prev);
    if (e != cudaSuccess) {
        cudaFail("cudaMalloc", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_free(int dev, void* p)
{
    cudaError_t e = cudaFree(p);
    if (e != cudaSuccess) {
        cudaFail("cudaFree", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_copy_h2d(int dev, void* dst, const void* src,
                                    size_t bytes)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    restoreDevice(prev);
    if (e != cudaSuccess) {
        cudaFail("cudaMemcpy H2D", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_copy_d2h(int dev, void* dst, const void* src,
                                    size_t bytes)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    restoreDevice(prev);
    if (e != cudaSuccess) {
        cudaFail("cudaMemcpy D2H", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_stream_create(int dev, void** out)
{
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreate(&s);
    if (e != cudaSuccess) {
        cudaFail("cudaStreamCreate", dev);
        return -1;
    }
    *out = s;
    return 0;
}

extern "C" int ff_dev_cuda_stream_destroy(int dev, void* s)
{
    cudaError_t e = cudaStreamDestroy(static_cast<cudaStream_t>(s));
    if (e != cudaSuccess) {
        cudaFail("cudaStreamDestroy", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_stream_sync(int dev, void* s)
{
    cudaError_t e = cudaStreamSynchronize(static_cast<cudaStream_t>(s));
    if (e != cudaSuccess) {
        cudaFail("cudaStreamSynchronize", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_event_create(int dev, void** out)
{
    cudaEvent_t ev = nullptr;
    // Default flags: timing enabled (needed for cudaEventElapsedTime).
    cudaError_t e = cudaEventCreate(&ev);
    if (e != cudaSuccess) {
        cudaFail("cudaEventCreate", dev);
        return -1;
    }
    *out = ev;
    return 0;
}

extern "C" int ff_dev_cuda_event_destroy(int dev, void* e)
{
    cudaError_t err = cudaEventDestroy(static_cast<cudaEvent_t>(e));
    if (err != cudaSuccess) {
        cudaFail("cudaEventDestroy", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_event_record(int dev, void* e, void* s)
{
    cudaError_t err = cudaEventRecord(static_cast<cudaEvent_t>(e),
                                      static_cast<cudaStream_t>(s));
    if (err != cudaSuccess) {
        cudaFail("cudaEventRecord", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_event_sync(int dev, void* e)
{
    cudaError_t err = cudaEventSynchronize(static_cast<cudaEvent_t>(e));
    if (err != cudaSuccess) {
        cudaFail("cudaEventSynchronize", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_event_elapsed_ms(int dev, void* e0, void* e1,
                                            float* ms)
{
    // cudaEventElapsedTime is only defined for ordered, completed events —
    // a successful call therefore proves e1 was recorded after e0 and both
    // completed (DevEvent ordering).
    cudaError_t err = cudaEventElapsedTime(ms, static_cast<cudaEvent_t>(e0),
                                           static_cast<cudaEvent_t>(e1));
    if (err != cudaSuccess) {
        cudaFail("cudaEventElapsedTime", dev);
        return -1;
    }
    return 0;
}

extern "C" int ff_dev_cuda_launch(int dev, void* buf, int n, void* s)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaFail("cudaSetDevice", dev);
        restoreDevice(prev);
        return -1;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(s);
    int nArgs = n;
    void* args[] = {&buf, &nArgs};
    cudaError_t e = cudaLaunchKernel(
        reinterpret_cast<const void*>(SIEVE_SMOKE_KERNEL),
        dim3((n + kBlock - 1) / kBlock), dim3(kBlock), args, 0, stream);
    if (e == cudaSuccess)
        e = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    restoreDevice(prev);
    if (e != cudaSuccess) {
        cudaFail("cudaLaunchKernel", dev);
        return -1;
    }
    return 0;
}
