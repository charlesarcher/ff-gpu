// Per-platform TU of the dual-runtime smoke test, compiled TWICE from this
// one HIP-only source (AMD default platform + HIP_PLATFORM=nvidia). Each
// compile sees only its own vendor's devices and exports an arch-tagged
// ff_smoke_hip_<arch> symbol. Includes ONLY the HIP headers — vendor headers
// are never mixed in one TU (ROCm/HIP#2703).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#include "smoke_kernel.h"

namespace {

constexpr int kSmokeElems = 1 << 20;  // 4 MiB of unsigned ints
constexpr int kBlock = 256;

#define FF_TAG_CAT2(a, b) a##b
#define FF_TAG_CAT(a, b)  FF_TAG_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

// Arch-tagged entry point: ff_smoke_hip_gfx1201 / ff_smoke_hip_sm_120.
#define FF_TAG_NAME FF_TAG_CAT(ff_smoke_hip_, SIEVE_KERNEL_ARCH)

#if defined(FF_BACKEND_NV) || defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SMOKE_VENDOR "NV  [hip-nv]"
#else
#define FF_SMOKE_VENDOR "AMD [hip]"
#endif

#define HIP_SMOKE_CHECK(call)                                                  \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr,                                               \
                         "HIP error at %s:%d: %s (%s)\n", __FILE__, __LINE__,  \
                         hipGetErrorName(err_), hipGetErrorString(err_));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

}  // namespace

// Runs the trivial kernel on device 0 of this compile's platform: alloc,
// memset, launch, D2H verify, then prints the device name + alloc/launch OK.
// Returns 0 on success.
extern "C" int FF_TAG_NAME(void)
{
    int devCount = 0;
    HIP_SMOKE_CHECK(hipGetDeviceCount(&devCount));
    if (devCount < 1) {
        std::fprintf(stderr, FF_SMOKE_VENDOR ": no HIP devices enumerated\n");
        return 1;
    }
    // This platform's runtime enumerates only its own vendor's GPUs.
    HIP_SMOKE_CHECK(hipSetDevice(0));

    hipDeviceProp_t props{};
    HIP_SMOKE_CHECK(hipGetDeviceProperties(&props, 0));

    unsigned int* devBuf = nullptr;
    HIP_SMOKE_CHECK(hipMalloc(&devBuf, kSmokeElems * sizeof(unsigned int)));
    HIP_SMOKE_CHECK(hipMemset(devBuf, 0, kSmokeElems * sizeof(unsigned int)));

    // hipLaunchKernelGGL expands to the arch-renamed kernel symbol
    // (SieveSlab_gfx1201 in this TU).
    hipLaunchKernelGGL(SIEVE_SMOKE_KERNEL,
                       dim3(kSmokeElems / kBlock), dim3(kBlock), 0, 0,
                       devBuf, kSmokeElems);
    HIP_SMOKE_CHECK(hipGetLastError());
    HIP_SMOKE_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> hostBuf(kSmokeElems);
    HIP_SMOKE_CHECK(hipMemcpy(hostBuf.data(), devBuf,
                              kSmokeElems * sizeof(unsigned int),
                              hipMemcpyDeviceToHost));
    HIP_SMOKE_CHECK(hipFree(devBuf));

    for (int i = 0; i < kSmokeElems; ++i) {
        if (hostBuf[i] != 1u) {
            std::fprintf(stderr, FF_SMOKE_VENDOR ": kernel verify FAILED at %d (got %u)\n",
                         i, hostBuf[i]);
            return 1;
        }
    }

    std::fprintf(stderr,
                 FF_SMOKE_VENDOR " : device %s : alloc OK launch OK verify OK (%d elems)\n",
                 props.name, kSmokeElems);
    return 0;
}
