// Per-arch backend TU for the M0 benchmark kernels, compiled TWICE from this
// one HIP-only source (AMD default platform + HIP_PLATFORM=nvidia), each
// exporting an arch-tagged ff_dev_hip_<arch>_m0_launch symbol. Includes ONLY
// the HIP headers. Launches the M0 kernels (smoke/m0_kernel.h) and
// synchronizes before returning, matching the DevLaunch contract.

#include <hip/hip_runtime.h>

#include <cstdio>

#include "m0_kernel.h"

#define FF_TAG_CAT2(a, b) a##b
#define FF_TAG_CAT(a, b)  FF_TAG_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

namespace {

constexpr int kBlock = 256;

void sync_stream(hipStream_t s)
{
    if (s) {
        hipStreamSynchronize(s);
    } else {
        hipDeviceSynchronize();
    }
}

}  // namespace

// kernel_id: 0 = M0_MEMSET (constant fill), 1 = M0_BW_SEQ (sequential pattern).
extern "C" int FF_TAG_CAT(ff_dev_hip_m0_launch_, SIEVE_KERNEL_ARCH)(
    int dev, int kernel_id, void* buf, int n, void* s)
{
    if (hipSetDevice(dev) != hipSuccess) {
        std::fprintf(stderr, "[m0] hipSetDevice(%d) failed\n", dev);
        return -1;
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(s);
    hipError_t err;

    if (kernel_id == 0) {
        hipLaunchKernelGGL(M0_MEMSET_KERN,
                           dim3((n + kBlock - 1) / kBlock), dim3(kBlock), 0, stream,
                           reinterpret_cast<unsigned int*>(buf), n);
    } else {
        hipLaunchKernelGGL(M0_BW_SEQ_KERN,
                           dim3((n + kBlock - 1) / kBlock), dim3(kBlock), 0, stream,
                           reinterpret_cast<unsigned int*>(buf), n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[m0] hip m0_launch error: %s (%s)\n",
                     hipGetErrorName(err), hipGetErrorString(err));
        return -1;
    }
    sync_stream(stream);
    return 0;
}