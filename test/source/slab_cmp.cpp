// slab_cmp.cpp — Byte-exact GPU slab-sieve unit test
//
// Compares the GPU SieveSlabKernel output against a CPU reference that is a
// BYTE-EXACT copy of Prime::SegmentFill from segmentedSieve.C:253-268.
// The reference function is brought in via a renaming #define so the original
// source is never modified.
//
// Build: g++ test main + per-arch sieve_slab_host_<arch>.o + HIP runtimes.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// CPU reference — BYTE-EXACT copy of segmentedSieve.C:253-268
// Renamed via macro so the original source is never edited.
// ---------------------------------------------------------------------------

// Renaming macro: the reference function Prime::SegmentFill is renamed to
// REF_CPU_SegmentFill via this #define.  The function body below is copied
// VERBATIM from segmentedSieve.C lines 253-268.
#define SEGMENT_FILL_RENAME REF_CPU_SegmentFill

// Extracted from segmentedSieve.C:253-268 (Prime::SegmentFill).
// Original signature: void Prime::SegmentFill(const uint64_t segLo, ...
// Renamed to a free function via the macro above.
void SEGMENT_FILL_RENAME(const uint64_t segLo, const uint64_t segHi,
                         const uint32_t* const primeList,
                         const uint32_t numList,
                         unsigned char* const primeMap)
{
    // ---- BEGIN EXTRACT: segmentedSieve.C:254-267 (body only) ----
    const uint64_t subBlockSize=1<<19; // 512K values, small enough to stay cache-resident
    for (uint64_t bLo=segLo; bLo<segHi; bLo+=subBlockSize)
    {uint64_t bHi=bLo+subBlockSize;
     if (bHi>segHi) bHi=segHi;
     for (uint32_t k=0; k<numList; ++k)
      {uint64_t p=primeList[k];
       if (p*p>=bHi) break; // Larger primes no longer influence this sub-block
       uint64_t first=((bLo+p-1)/p)*p; // First multiple of p within this sub-block
       if (first<p*p) first=p*p;       // ...but no smaller than p squared
       if (!(first&1)) first+=p;       // Only the odd multiples are marked
       for (uint64_t i=first; i<bHi; i+=p<<1)
         primeMap[i>>4]&=~(0x80>>(i>>1&7)); // Plain store: each chunk is owned exclusively
      }
    }
    // ---- END EXTRACT ----
}

// ---------------------------------------------------------------------------
// Small-prime generator — mirrors segmentedSieve.C:181-200 (single-threaded
// small sieve up to sqrt(limit), then collects primes).
// ---------------------------------------------------------------------------

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

    // Small sieve up to sqrtLimit.
    uint64_t smallMapSize = (sqrtLimit + 15) >> 4;
    std::vector<uint8_t> smallMap(smallMapSize, 0xff);
    smallMap[0] ^= 0x80;  // 1 is not prime

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

// ---------------------------------------------------------------------------
// GPU dispatch — per-arch entry points provided by sieve_slab_kernel.cpp.
//
// The host helper is compiled per-arch.  Each .o exports:
//   SieveSlabRun_<arch>      — the actual sieve launch
//   SieveSlabGetLaunchFn_<arch> — returns a fn-pointer to SieveSlabRun_<arch>,
//                                or nullptr if the HIP runtime cannot see any
//                                device (non-matching arch runtime).
//
// The g++ test binary links BOTH .o files and resolves the right one at
// runtime by calling both GetLaunchFn helpers and using whichever returns
// non-null.
// ---------------------------------------------------------------------------

using SieveSlabRunFn = int (*)(int, const uint32_t*, uint32_t,
                               uint64_t, uint64_t, uint8_t*);

extern "C" SieveSlabRunFn SieveSlabGetLaunchFn_gfx1201(int deviceIndex);
extern "C" SieveSlabRunFn SieveSlabGetLaunchFn_sm_120(int deviceIndex);

static SieveSlabRunFn resolveSieveSlab(int deviceIndex)
{
    SieveSlabRunFn fn = SieveSlabGetLaunchFn_gfx1201(deviceIndex);
    if (fn) return fn;
    fn = SieveSlabGetLaunchFn_sm_120(deviceIndex);
    return fn;
}

// ---------------------------------------------------------------------------
// Test infrastructure.
// ---------------------------------------------------------------------------

static int gTestsRun = 0;
static int gTestsPassed = 0;
static int gTestsFailed = 0;

struct TestCase {
    const char* name;
    uint64_t segLo;
    uint64_t segHi;
};

// Compare the CPU reference map (GLOBAL indexing: result bytes live at
// [segLo>>4 .. (segHi-1)>>4]) against the GPU kernel output (segLo-RELATIVE
// indexing: byte b holds values [segLo+16b, segLo+16b+16), matching the
// production slab engine's per-slab local buffers).  For segLo==0 the two
// coincide; for segLo!=0 the GPU byte for a value is gpu[b], not
// gpu[segLo>>4 + b].
static bool compareMaps(const uint8_t* cpu, const uint8_t* gpu,
                        uint64_t segLo, uint64_t segHi)
{
    uint64_t startByte = segLo >> 4;
    uint64_t endByte = (segHi > 0) ? ((segHi - 1) >> 4) + 1 : 0;
    uint64_t numBytes = endByte - startByte;

    for (uint64_t b = 0; b < numBytes; ++b) {
        if (cpu[startByte + b] != gpu[b]) {
            std::fprintf(stderr,
                "  BYTE MISMATCH at global byte %lu (value %lu): "
                "cpu=0x%02x gpu=0x%02x\n",
                (unsigned long)(startByte + b),
                (unsigned long)((startByte + b) << 4),
                cpu[startByte + b], gpu[b]);
            return false;
        }
    }
    return true;
}

// Run one test case: CPU reference + GPU kernel, then compare.
static bool runTest(const TestCase& tc)
{
    ++gTestsRun;
    std::printf("TEST %d: %s  [segLo=%lu, segHi=%lu]\n",
                gTestsRun, tc.name,
                (unsigned long)tc.segLo, (unsigned long)tc.segHi);

    // 1. Generate small primes up to sqrt(segHi).
    std::vector<uint32_t> primes = generateSmallPrimes(tc.segHi);
    std::printf("  small primes: %zu (max=%u)\n",
                primes.size(),
                primes.empty() ? 0u : primes.back());

    // 2. Allocate host buffers.
    uint64_t mapBytes = (tc.segHi + 15u) >> 4;
    std::vector<uint8_t> h_cpu(mapBytes, 0xff);
    std::vector<uint8_t> h_gpu(mapBytes, 0);

    // CPU init: memset 0xff, clear bit-0.
    if (tc.segLo == 0) h_cpu[0] ^= 0x80;

    // 3. Run CPU reference.
    REF_CPU_SegmentFill(tc.segLo, tc.segHi,
                        primes.data(), static_cast<uint32_t>(primes.size()),
                        h_cpu.data());

    // 4. Run GPU kernel.
    SieveSlabRunFn fn = resolveSieveSlab(0);
    if (!fn) {
        std::fprintf(stderr, "  GPU DISPATCH FAILED (no arch visible)\n");
        ++gTestsFailed;
        return false;
    }
    int rc = fn(0, primes.data(), static_cast<uint32_t>(primes.size()),
                tc.segLo, tc.segHi, h_gpu.data());
    if (rc != 0) {
        std::fprintf(stderr, "  GPU RUN FAILED (rc=%d)\n", rc);
        ++gTestsFailed;
        return false;
    }

    // 5. Compare.
    if (compareMaps(h_cpu.data(), h_gpu.data(), tc.segLo, tc.segHi)) {
        std::printf("  PASS: byte-exact match\n");
        ++gTestsPassed;
        return true;
    } else {
        std::fprintf(stderr, "  FAIL: GPU != CPU reference\n");
        ++gTestsFailed;
        return false;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/)
{
    std::printf("=== slab_cmp: byte-exact SieveSlabKernel unit tests ===\n");

    SieveSlabRunFn fn = resolveSieveSlab(0);
    if (!fn) {
        std::fprintf(stderr, "No GPU runtime visible — skipping tests.\n");
        std::printf("\n=== slab_cmp: 0 tests run (no GPU) ===\n");
        return 0;
    }
    std::printf("GPU dispatch: resolved (using SieveSlabRun_<arch>)\n\n");

    // ---- Test cases ----
    std::vector<TestCase> tests = {
        // 1. First slab (segLo=0): tests bit-0 clearing + initial sub-block
        {"first slab (segLo=0, segHi=65536)", 0, 65536},

        // 2. Single full sub-block
        {"single sub-block (segLo=0, segHi=524288)", 0, 524288},

        // 3. Two full sub-blocks
        {"multi sub-block (segLo=0, segHi=1048576)", 0, 1048576},

        // 4. Truncated last sub-block
        {"truncated last sub-block", 500000, 524288},

        // 5. Non-aligned start (segLo not on sub-block boundary)
        {"non-aligned start", 100000, 300000},

        // 6. Small range (< one cache line of 1024 values)
        {"tiny range (256 values)", 0, 256},

        // 7. Second sub-block (tests boundary transition)
        {"second sub-block", 524288, 1048576},

        // 8. Edge: segHi just past sub-block boundary
        //    (segLo must be 16-aligned: the kernel's segLo-relative bit layout
        //    only coincides with the CPU's global layout for 16-aligned starts,
        //    which the production engine always guarantees)
        {"past sub-block boundary", 524272, 524400},

        // 9. Range within one cache line (1024 values)
        {"within one cache line", 1200, 2000},

        // 10. Larger range for stress
        {"larger range (2M)", 0, 2000000},
    };

    for (const auto& tc : tests) {
        runTest(tc);
    }

    // ---- Summary ----
    std::printf("\n=== slab_cmp: %d/%d tests passed", gTestsPassed, gTestsRun);
    if (gTestsFailed > 0)
        std::printf(", %d FAILED", gTestsFailed);
    std::printf(" ===\n");

    return gTestsFailed > 0 ? 1 : 0;
}