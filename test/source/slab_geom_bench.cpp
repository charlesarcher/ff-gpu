// slab_geom_bench.cpp — launch-geometry timing + byte-exactness harness for
// SieveSlabKernel (task 13).
//
// Compiled per arch and per geometry combo: the four FF_SIEVE_* geometry
// macros from sieve_slab_kernel.h are overridden via -D on the hipcc command
// line, so each binary measures exactly one compile-time configuration. The
// CMake target `slab_geom_bench` builds the same source at the baked defaults
// so the chosen configuration stays regression-checked under ctest.
//
// Protocol per invocation (fixed region: segLo=0, segHi=2^34 values, i.e. a
// 1-GiB map — one full production slab):
//   1. primes up to sqrt(segHi); CPU reference SegmentFill over the region,
//      cached to a file (the reference is geometry-independent, so every
//      combo on every arch reuses one computed reference);
//   2. device buffer init (0xff, bit-0 clear), correctness launch, D2H,
//      byte-compare vs reference -> CORRECT/FAIL gate;
//   3. N timed reps: reset buffer (device-side memset outside the timing
//      window), hipEvent-bracketed kernel launch, sync; CSV rows on stdout;
//   4. re-verify bytes after the timed reps (guards the reset path).
//
// Exit code 0 iff every byte-compare matched. Timings from a failing run are
// void by construction (rc != 0 lets the sweep script discard them).
//
// Output rows: arch,sub_log2,vpt,bps,tpb,kind,rep,ms  (kind: CORRECT|TIME|VERIFY)

#include <hip/hip_runtime.h>

#include "sieve_slab_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CPU wheel-30 TWIN reference (tasks 6+7 pair): same independent filler as
// slab_cmp.cpp — brute-force multiples with explicit coprimality filtering,
// no shared tables with the kernel. Task 9 productionizes this harness.
// ---------------------------------------------------------------------------

static bool coprime30(uint64_t x)
{
    return x % 2 != 0 && x % 3 != 0 && x % 5 != 0;
}

static unsigned slotOfResidue(uint64_t r)
{
    static const uint64_t res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    for (unsigned i = 0; i < 8; ++i)
        if (res[i] == r) return i;
    return 8;
}

static void REF_WHEEL_Fill(const uint64_t segLo, const uint64_t segHi,
                           const uint32_t* const primeList,
                           const uint32_t numList,
                           uint8_t* const internalMap)
{
    const uint64_t bufMin = ((segHi - segLo) + 29) / 30;
    for (uint32_t k = 0; k < numList; ++k) {
        const uint64_t p = primeList[k];
        if (p < 7) continue;
        if (p * p >= segHi) break;
        for (uint64_t t = 1;; ++t) {
            if (!coprime30(t)) continue;
            const uint64_t v = p * t;
            if (v >= segHi) break;
            if (v < p * p) continue;
            const uint64_t rel = v / 30 - segLo / 30;
            if (rel >= bufMin) continue;
            internalMap[rel] &=
                static_cast<uint8_t>(~(1u << slotOfResidue(v % 30)));
        }
    }
}

static uint64_t isqrt64(uint64_t n)
{
    if (n == 0) return 0;
    uint64_t x = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
    while ((x + 1) * (x + 1) <= n) ++x;
    while (x * x > n) --x;
    return x;
}

static std::vector<uint32_t> generateSmallPrimes(uint64_t limit)
{
    if (limit < 2) return {2u};
    uint64_t sqrtLimit = isqrt64(limit);
    if (sqrtLimit < 2) sqrtLimit = 2;

    uint64_t smallMapSize = (sqrtLimit + 15) >> 4;
    std::vector<uint8_t> smallMap(smallMapSize, 0xff);
    smallMap[0] ^= 0x80;

    for (uint64_t p = 3; p <= isqrt64(sqrtLimit); p += 2)
        if (smallMap[p >> 4] & (0x80 >> (p >> 1 & 7)))
            for (uint64_t i = p * p; i <= sqrtLimit; i += p << 1)
                smallMap[i >> 4] &= ~(0x80 >> (i >> 1 & 7));

    std::vector<uint32_t> primes;
    primes.push_back(2u);
    for (uint64_t p = 3; p <= sqrtLimit; p += 2)
        if (smallMap[p >> 4] & (0x80 >> (p >> 1 & 7)))
            primes.push_back(static_cast<uint32_t>(p));

    return primes;
}

static std::vector<uint32_t> stripStructural(const std::vector<uint32_t>& in)
{
    std::vector<uint32_t> out;
    for (uint32_t p : in)
        if (p >= 7) out.push_back(p);
    return out;
}

// ---------------------------------------------------------------------------
// Reference-map cache: the CPU fill over the fixed region costs seconds and
// does not depend on geometry or arch, so compute it once and reuse.
// ---------------------------------------------------------------------------

static bool loadRefCache(const std::string& path, std::vector<uint8_t>& ref)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const size_t want = ref.size();
    const size_t got = std::fread(ref.data(), 1, want, f);
    std::fclose(f);
    return got == want;
}

static void storeRefCache(const std::string& path, const std::vector<uint8_t>& ref)
{
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    if (std::fwrite(ref.data(), 1, ref.size(), f) == ref.size()) {
        std::fclose(f);
        std::rename(tmp.c_str(), path.c_str());
    } else {
        std::fclose(f);
        std::remove(tmp.c_str());
    }
}

// ---------------------------------------------------------------------------
// Device selection: pick the first device matching the platform this binary
// was compiled for (AMD builds require gfx1201; NVIDIA builds require a
// non-gfx device with compute capability 12.x = sm_120).
// ---------------------------------------------------------------------------

#define BENCH_ARCH_STR2(x) #x
#define BENCH_ARCH_STR(x)  BENCH_ARCH_STR2(x)
static const char* kArchName = BENCH_ARCH_STR(SIEVE_KERNEL_ARCH);

static int pickDevice(int forced, int& devOut, char* nameOut, size_t nameCap)
{
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) return -1;
    const bool wantAmd = (kArchName[0] == 'g');
    for (int d = 0; d < count; ++d) {
        if (forced >= 0 && d != forced) continue;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, d) != hipSuccess) continue;
        const bool isGfx = (std::strstr(prop.gcnArchName, "gfx") != nullptr);
        bool match = false;
        if (wantAmd) {
            match = isGfx && std::strstr(prop.gcnArchName, "gfx1201") != nullptr;
        } else {
            match = !isGfx && prop.major == 12;
        }
        if (match) {
            devOut = d;
            std::snprintf(nameOut, nameCap, "%s [%s]", prop.name, prop.gcnArchName);
            return 0;
        }
    }
    return -1;
}

#define BENCH_HIP_CHECK(call)                                                  \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            std::fprintf(stderr, "bench_hip error at %d: %s (%s)\n", __LINE__, \
                         hipGetErrorName(err_), hipGetErrorString(err_));      \
            return 2;                                                          \
        }                                                                      \
    } while (0)

static void emitRow(const char* kind, int rep, double ms)
{
    std::printf("%s,%d,%llu,%u,%u,%s,%d,%.3f\n",
                kArchName, FF_SIEVE_SUB_BLOCK_LOG2,
                (unsigned long long)kSieveBytesPerThread,
                kSieveBlocksPerSubBlock, kSieveThreadsPerBlock,
                kind, rep, ms);
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    int device = -1;
    int reps = 5;
    std::string cachePath;
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--device=", 9)) device = std::atoi(argv[i] + 9);
        else if (!std::strncmp(argv[i], "--reps=", 7)) reps = std::atoi(argv[i] + 7);
        else if (!std::strncmp(argv[i], "--ref-cache=", 12)) cachePath = argv[i] + 12;
    }
    if (reps < 1) reps = 1;

    // Fixed region: [0, 2^34) values -> wheel-30 internal map bytes.
    const uint64_t segLo = 0;
    const uint64_t segHi = 1ull << 34;
    const uint64_t mapBytes = (segHi + 29u) / 30u;

    char devName[256] = {0};
    int dev = -1;
    if (pickDevice(device, dev, devName, sizeof(devName)) != 0) {
        std::printf("# no matching GPU visible — skipping\n");
        return 0;
    }

    // ---- primes + CPU reference (cached; wheel layout gets its own cache
    //      file so stale canonical caches can never be loaded) ----
    std::vector<uint32_t> primes =
        stripStructural(generateSmallPrimes(segHi));
    std::vector<uint8_t> h_ref(mapBytes, 0xff);
    h_ref[0] = 0xfe;   // value 1 is not prime (residue-slot 0)
    {
        const uint64_t bufMin = (segHi + 29u) / 30u;
        const uint64_t base = segLo + (bufMin - 1) * 30ull;
        static const uint64_t res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
        for (unsigned r = 0; r < 8; ++r)
            if (base + res[r] >= segHi)
                h_ref[bufMin - 1] &= static_cast<uint8_t>(~(1u << r));
    }

    if (cachePath.empty()) {
        cachePath = "/tmp/ff_slab_geom_refcache_w30_" +
                    std::to_string(segHi) + ".bin";
    }
    if (!loadRefCache(cachePath, h_ref)) {
        REF_WHEEL_Fill(segLo, segHi, primes.data(),
                       static_cast<uint32_t>(primes.size()), h_ref.data());
        storeRefCache(cachePath, h_ref);
    }

    // ---- device buffers ----
    BENCH_HIP_CHECK(hipSetDevice(dev));
    std::vector<uint8_t> h_init(mapBytes, 0xff);
    h_init[0] = 0xfe;
    {
        const uint64_t bufMin = (segHi + 29u) / 30u;
        const uint64_t base = segLo + (bufMin - 1) * 30ull;
        static const uint64_t res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
        for (unsigned r = 0; r < 8; ++r)
            if (base + res[r] >= segHi)
                h_init[bufMin - 1] &= static_cast<uint8_t>(~(1u << r));
    }

    uint8_t* d_map = nullptr;
    BENCH_HIP_CHECK(hipMalloc(&d_map, mapBytes));
    BENCH_HIP_CHECK(hipMemcpy(d_map, h_init.data(), mapBytes, hipMemcpyHostToDevice));

    const size_t primeBytes = primes.size() * sizeof(uint32_t);
    uint32_t* d_primes = nullptr;
    BENCH_HIP_CHECK(hipMalloc(&d_primes, primeBytes));
    BENCH_HIP_CHECK(hipMemcpy(d_primes, primes.data(), primeBytes, hipMemcpyHostToDevice));

    // Grid math identical to the engine's launch loop, from the same table
    // (internal-byte currency: ceil(span/30) bytes covered).
    const uint64_t numValues = segHi - segLo;
    const uint64_t bufMinBytes = (numValues + 29u) / 30u;
    const uint32_t numSubBlocks = static_cast<uint32_t>(
        (bufMinBytes + kSieveSubBlockSize - 1) / kSieveSubBlockSize);
    const uint32_t totalBlocks = numSubBlocks * kSieveBlocksPerSubBlock;

    hipEvent_t ev0, ev1;
    BENCH_HIP_CHECK(hipEventCreate(&ev0));
    BENCH_HIP_CHECK(hipEventCreate(&ev1));

    std::vector<uint8_t> h_gpu(mapBytes);
    std::printf("# arch=%s dev=%d (%s) segHi=%llu mapBytes=%llu primes=%zu "
                "subLog2=%d bpt=%llu bps=%u tpb=%u blocks=%u\n",
                kArchName, dev, devName,
                (unsigned long long)segHi, (unsigned long long)mapBytes,
                primes.size(), FF_SIEVE_SUB_BLOCK_LOG2,
                (unsigned long long)kSieveBytesPerThread,
                kSieveBlocksPerSubBlock, kSieveThreadsPerBlock, totalBlocks);

    auto runAndCompare = [&](const char* kind, int rep) -> bool {
        hipLaunchKernelGGL(SIEVE_SLAB_KERNEL,
                           dim3(totalBlocks), dim3(kSieveThreadsPerBlock), 0, 0,
                           d_primes, static_cast<uint32_t>(primes.size()),
                           segLo, segHi, d_map);
        BENCH_HIP_CHECK(hipGetLastError());
        BENCH_HIP_CHECK(hipDeviceSynchronize());
        BENCH_HIP_CHECK(hipMemcpy(h_gpu.data(), d_map, mapBytes, hipMemcpyDeviceToHost));
        const bool ok = std::memcmp(h_gpu.data(), h_ref.data(), mapBytes) == 0;
        emitRow(kind, rep, ok ? 0.0 : 1.0);
        return ok;
    };

    // ---- correctness gate (also serves as warm-up) ----
    if (!runAndCompare("CORRECT", 0)) {
        std::fprintf(stderr, "BYTE MISMATCH: geometry %d/%llu/%u/%u is not byte-exact\n",
                     FF_SIEVE_SUB_BLOCK_LOG2,
                     (unsigned long long)kSieveBytesPerThread,
                     kSieveBlocksPerSubBlock, kSieveThreadsPerBlock);
        return 1;
    }

    // ---- timed reps ----
    for (int r = 1; r <= reps; ++r) {
        BENCH_HIP_CHECK(hipMemset(d_map, 0xff, mapBytes));
        BENCH_HIP_CHECK(hipMemcpy(d_map, h_init.data(), 1, hipMemcpyHostToDevice));
        // restore the padded trailing byte too (production init contract)
        BENCH_HIP_CHECK(hipMemcpy(d_map + (mapBytes - 1),
                                  h_init.data() + (mapBytes - 1), 1,
                                  hipMemcpyHostToDevice));
        BENCH_HIP_CHECK(hipDeviceSynchronize());

        BENCH_HIP_CHECK(hipEventRecord(ev0, 0));
        hipLaunchKernelGGL(SIEVE_SLAB_KERNEL,
                           dim3(totalBlocks), dim3(kSieveThreadsPerBlock), 0, 0,
                           d_primes, static_cast<uint32_t>(primes.size()),
                           segLo, segHi, d_map);
        BENCH_HIP_CHECK(hipGetLastError());
        BENCH_HIP_CHECK(hipEventRecord(ev1, 0));
        BENCH_HIP_CHECK(hipEventSynchronize(ev1));
        float ms = 0.0f;
        BENCH_HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        emitRow("TIME", r, static_cast<double>(ms));
    }

    // ---- post-timing byte verification ----
    BENCH_HIP_CHECK(hipMemcpy(h_gpu.data(), d_map, mapBytes, hipMemcpyDeviceToHost));
    if (std::memcmp(h_gpu.data(), h_ref.data(), mapBytes) != 0) {
        std::fprintf(stderr, "BYTE MISMATCH after timed reps\n");
        return 1;
    }
    emitRow("VERIFY", 0, 0.0);

    (void)hipFree(d_primes);
    (void)hipFree(d_map);
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);
    return 0;
}
