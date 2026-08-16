// NVIDIA backend TU for M0 benchmark kernels (RTX 5090, sm_120).
//
// Compiled with HIP_PLATFORM=nvidia /opt/rocm/bin/hipcc -x cu -arch=sm_120,
// which delegates to nvcc. Includes ONLY the CUDA headers. Provides extern
// "C" entry points that launch the M0 kernels (smoke/m0_kernel.h) and
// synchronize before returning, matching the DevLaunch contract.

#include <cuda_runtime.h>

#include <cstdio>

#include "m0_kernel.h"

namespace {

constexpr int kBlock = 256;

void sync_stream(cudaStream_t s)
{
    if (s) {
        cudaStreamSynchronize(s);
    } else {
        cudaDeviceSynchronize();
    }
}

}  // namespace

// kernel_id: 0 = M0_MEMSET (constant fill), 1 = M0_BW_SEQ (sequential pattern).
extern "C" int ff_dev_cuda_m0_launch(int dev, int kernel_id, void* buf, int n, void* s)
{
    if (cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "[m0] cudaSetDevice(%d) failed\n", dev);
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    cudaError_t err;

    if (kernel_id == 0) {
        cudaLaunchKernel(
            reinterpret_cast<const void*>(M0_MEMSET_KERN),
            dim3((n + kBlock - 1) / kBlock), dim3(kBlock),
            (void*[]){&buf, &n}, 0, stream);
    } else {
        cudaLaunchKernel(
            reinterpret_cast<const void*>(M0_BW_SEQ_KERN),
            dim3((n + kBlock - 1) / kBlock), dim3(kBlock),
            (void*[]){&buf, &n}, 0, stream);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[m0] cuda m0_launch error: %s (%s)\n",
                     cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }
    sync_stream(stream);
    return 0;
}