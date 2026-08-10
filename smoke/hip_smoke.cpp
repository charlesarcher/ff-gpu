// AMD TU (RX 9070 XT, gfx1201) of the dual-runtime smoke test.
//
// Compiled with the DEFAULT HIP platform (hipcc as-is, no HIP_PLATFORM), so
// hipGetDeviceCount sees the AMD devices only (this system: the 9070 XT).
// Includes ONLY the HIP headers — vendor headers are never mixed in one TU
// (ROCm/HIP#2703).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#include "smoke_kernel.h"

namespace {

constexpr int kSmokeElems = 1 << 20;  // 4 MiB of unsigned ints
constexpr int kBlock = 256;

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

// Runs the trivial kernel on the AMD device: alloc, memset, launch, D2H
// verify, then prints the device name + alloc/launch OK. Returns 0 on success.
extern "C" int ff_smoke_hip(void)
{
    int devCount = 0;
    HIP_SMOKE_CHECK(hipGetDeviceCount(&devCount));
    if (devCount < 1) {
        std::fprintf(stderr, "AMD [hip]: no HIP devices enumerated\n");
        return 1;
    }
    // Default-platform HIP runtime enumerates only the AMD GPU (the 9070 XT).
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
            std::fprintf(stderr, "AMD [hip]: kernel verify FAILED at %d (got %u)\n",
                         i, hostBuf[i]);
            return 1;
        }
    }

    std::fprintf(stderr,
                 "AMD  [hip] : device %s : alloc OK launch OK verify OK (%d elems)\n",
                 props.name, kSmokeElems);
    return 0;
}
