// NVIDIA TU (RTX 5090, sm_120) of the dual-runtime smoke test.
//
// Compiled with HIP_PLATFORM=nvidia /opt/rocm/bin/hipcc -x cu -arch=sm_120,
// which delegates to /usr/local/cuda/bin/nvcc (CUDA 13.3). Includes ONLY the
// CUDA headers — vendor headers are never mixed in one TU (ROCm/HIP#2703).
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "smoke_kernel.h"

namespace {

constexpr int kSmokeElems = 1 << 20;  // 4 MiB of unsigned ints
constexpr int kBlock = 256;

#define CUDA_SMOKE_CHECK(call)                                                 \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                         cudaGetErrorName(err_), cudaGetErrorString(err_));    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

}  // namespace

// Runs the trivial kernel on the NVIDIA device: alloc, memset, launch, D2H
// verify, then prints the device name + alloc/launch OK. Returns 0 on success.
extern "C" int ff_smoke_cuda(void)
{
    int devCount = 0;
    CUDA_SMOKE_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1) {
        std::fprintf(stderr, "NV  [cuda]: no CUDA devices enumerated\n");
        return 1;
    }
    // The CUDA runtime enumerates only the NVIDIA GPU (the RTX 5090).
    CUDA_SMOKE_CHECK(cudaSetDevice(0));

    cudaDeviceProp props{};
    CUDA_SMOKE_CHECK(cudaGetDeviceProperties(&props, 0));

    unsigned int* devBuf = nullptr;
    CUDA_SMOKE_CHECK(cudaMalloc(&devBuf, kSmokeElems * sizeof(unsigned int)));
    CUDA_SMOKE_CHECK(cudaMemset(devBuf, 0, kSmokeElems * sizeof(unsigned int)));

    // cudaLaunchKernel with the arch-renamed kernel symbol
    // (SieveSlab_sm120 in this TU).
    int n = kSmokeElems;
    void* args[] = {&devBuf, &n};
    CUDA_SMOKE_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void*>(SIEVE_SMOKE_KERNEL),
        dim3(kSmokeElems / kBlock), dim3(kBlock), args, 0, nullptr));
    CUDA_SMOKE_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> hostBuf(kSmokeElems);
    CUDA_SMOKE_CHECK(cudaMemcpy(hostBuf.data(), devBuf,
                                kSmokeElems * sizeof(unsigned int),
                                cudaMemcpyDeviceToHost));
    CUDA_SMOKE_CHECK(cudaFree(devBuf));

    for (int i = 0; i < kSmokeElems; ++i) {
        if (hostBuf[i] != 1u) {
            std::fprintf(stderr, "NV  [cuda]: kernel verify FAILED at %d (got %u)\n",
                         i, hostBuf[i]);
            return 1;
        }
    }

    std::fprintf(stderr,
                 "NV   [cuda] : device %s : alloc OK launch OK verify OK (%d elems)\n",
                 props.name, kSmokeElems);
    return 0;
}
