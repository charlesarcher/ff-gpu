// Dual-runtime smoke entry point (plan todo 4): ONE process brings up BOTH
// vendor runtimes — AMD via HIP/ROCm and NVIDIA via the same HIP source
// compiled with HIP_PLATFORM=nvidia — and runs a trivial kernel on each GPU.
// This TU is vendor-neutral: it includes NO vendor headers and only calls
// the two arch-tagged smoke entry points (ff_smoke_hip_gfx1201 /
// ff_smoke_hip_sm_120), so no translation unit ever mixes CUDA and HIP
// headers. The binary is pinned at the repo root as `ff_sieve`.
//
// Todo 3: `main` was renamed to `ff_smoke_main` so the todo-3 CLI main
// (src/main.cpp) owns main(); a no-args `./ff_sieve` invokes this smoke,
// which keeps the dual-runtime regression green.
#include <cstdio>

extern "C" int ff_smoke_hip_gfx1201(void);
extern "C" int ff_smoke_hip_sm_120(void);

extern "C" int ff_smoke_main(void)
{
    std::fprintf(stderr,
                 "== ff_sieve dual-runtime smoke (one process, two GPUs) ==\n");
    int rcAmd = ff_smoke_hip_gfx1201();
    int rcNv = ff_smoke_hip_sm_120();
    if (rcAmd != 0 || rcNv != 0) {
        std::fprintf(stderr, "SMOKE FAILED (amd rc=%d, nvidia rc=%d)\n",
                     rcAmd, rcNv);
        return 1;
    }
    std::fprintf(stderr, "SMOKE OK: both devices ran a kernel in one process\n");
    return 0;
}
