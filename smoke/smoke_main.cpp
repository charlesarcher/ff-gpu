// Dual-runtime smoke entry point (plan todo 4): ONE process brings up BOTH
// the ROCm (AMD) and CUDA (NVIDIA) runtimes and runs a trivial kernel on each
// GPU. This TU is vendor-neutral — it includes NO vendor headers and only
// calls the per-arch smoke entry points, so no translation unit ever mixes
// CUDA and HIP headers. The binary is pinned at the repo root as `ff_sieve`.
#include <cstdio>

extern "C" int ff_smoke_hip(void);
extern "C" int ff_smoke_cuda(void);

int main()
{
    std::fprintf(stderr,
                 "== ff_sieve dual-runtime smoke (one process, two GPUs) ==\n");
    int rcHip = ff_smoke_hip();
    int rcCuda = ff_smoke_cuda();
    if (rcHip != 0 || rcCuda != 0) {
        std::fprintf(stderr, "SMOKE FAILED (hip rc=%d, cuda rc=%d)\n",
                     rcHip, rcCuda);
        return 1;
    }
    std::fprintf(stderr, "SMOKE OK: both devices ran a kernel in one process\n");
    return 0;
}
