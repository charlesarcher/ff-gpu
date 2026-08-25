// slab_cmp.cpp — Byte-exact GPU slab-sieve unit test (wheel-30 internal map)
//
// Compares the GPU SieveSlabKernel output (INTERNAL wheel-30 layout: byte k
// covers values [30k,30k+30), bit i = value 30k+kWheelResidues[i]) against a
// CPU twin filler that mirrors Prime::SegmentFill's SEMANTICS over the
// residue classes. The twin is deliberately INDEPENDENT of the kernel's
// helpers (no CRT inverse tables, no AP prologs): it walks every multiple of
// every marking prime directly and filters coprime-to-30 values, so a
// corrupted residue table / class-rep math on the GPU side cannot stay
// self-consistent with it.
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
// CPU wheel-30 TWIN reference (independent implementation).
//
// Marks, over [segLo,segHi) with segLo % 30 == 0, every value v = p*t with
// p > 5 prime, gcd(t,30)=1, v >= p*p — by brute-force iteration over t with
// explicit coprimality tests (no shared tables with the kernel or geometry).
// Buffer: segLo-relative internal bytes; init mirrors production (0xff,
// byte0 0xfe iff segLo==0, trailing padding slots zeroed).
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
    return 8;   // unreachable for coprime residues
}

static void REF_WHEEL_Fill(const uint64_t segLo, const uint64_t segHi,
                           const uint32_t* const primeList,
                           const uint32_t numList,
                           uint8_t* const internalMap)
{
    const uint64_t bufMin = ((segHi - segLo) + 29) / 30;
    for (uint32_t k = 0; k < numList; ++k) {
        const uint64_t p = primeList[k];
        if (p < 7) continue;                    // {2,3,5} structural strip
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

// ---------------------------------------------------------------------------
// GPU dispatch — per-arch entry points provided by sieve_slab_kernel.cpp.
// ---------------------------------------------------------------------------

using SieveSlabRunFn = int (*)(int, const uint32_t*, uint32_t,
                               uint64_t, uint64_t, uint8_t*);

extern "C" SieveSlabRunFn SieveSlabGetLaunchFn_gfx1201(int deviceIndex);

static SieveSlabRunFn resolveSieveSlab(int deviceIndex)
{
    return SieveSlabGetLaunchFn_gfx1201(deviceIndex);
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

// Compare the twin CPU map against the GPU kernel output; both are
// segLo-RELATIVE internal-byte buffers of bufMin = ceil((segHi-segLo)/30)
// bytes (the GPU harness zeroes nothing beyond that).
static bool compareMaps(const uint8_t* cpu, const uint8_t* gpu,
                        uint64_t segLo, uint64_t segHi)
{
    uint64_t numBytes = ((segHi - segLo) + 29) / 30;

    for (uint64_t b = 0; b < numBytes; ++b) {
        if (cpu[b] != gpu[b]) {
            std::fprintf(stderr,
                "  BYTE MISMATCH at rel byte %lu (value %lu): "
                "cpu=0x%02x gpu=0x%02x\n",
                (unsigned long)b,
                (unsigned long)(segLo + 30 * b),
                cpu[b], gpu[b]);
            return false;
        }
    }
    return true;
}

// Run one test case: CPU twin + GPU kernel, then compare.
static bool runTest(const TestCase& tc)
{
    ++gTestsRun;
    std::printf("TEST %d: %s  [segLo=%lu, segHi=%lu]\n",
                gTestsRun, tc.name,
                (unsigned long)tc.segLo, (unsigned long)tc.segHi);

    // 1. Generate small primes up to sqrt(segHi), strip {2,3,5} like the
    //    production kernelPrimes() marking list.
    std::vector<uint32_t> all = generateSmallPrimes(tc.segHi);
    std::vector<uint32_t> primes;
    for (uint32_t p : all)
        if (p >= 7) primes.push_back(p);
    std::printf("  marking primes: %zu (max=%u)\n",
                primes.size(),
                primes.empty() ? 0u : primes.back());

    // 2. Allocate host buffers (internal bytes).
    uint64_t bufMin = ((tc.segHi - tc.segLo) + 29) / 30;
    std::vector<uint8_t> h_cpu(bufMin, 0xff);
    std::vector<uint8_t> h_gpu(bufMin, 0);

    // CPU init mirrors production: byte0 0xfe iff segLo==0; padding slots of
    // the trailing byte zeroed.
    if (tc.segLo == 0) h_cpu[0] = 0xfe;
    {
        const uint64_t base = tc.segLo + (bufMin - 1) * 30ull;
        static const uint64_t res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
        for (unsigned r = 0; r < 8; ++r)
            if (base + res[r] >= tc.segHi)
                h_cpu[bufMin - 1] &= static_cast<uint8_t>(~(1u << r));
    }

    // 3. Run CPU twin.
    REF_WHEEL_Fill(tc.segLo, tc.segHi,
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
        std::fprintf(stderr, "  FAIL: GPU != CPU wheel-30 twin\n");
        ++gTestsFailed;
        return false;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/)
{
    std::printf("=== slab_cmp: byte-exact SieveSlabKernel unit tests "
                "(wheel-30 internal map) ===\n");

    SieveSlabRunFn fn = resolveSieveSlab(0);
    if (!fn) {
        std::fprintf(stderr, "No GPU runtime visible — skipping tests.\n");
        std::printf("\n=== slab_cmp: 0 tests run (no GPU) ===\n");
        return 0;
    }
    std::printf("GPU dispatch: resolved (using SieveSlabRun_<arch>)\n\n");

    // ---- Test cases (segLo must be a multiple of 30: slabs start on
    //      internal-byte boundaries; production guarantees it) ----
    std::vector<TestCase> tests = {
        // 1. First slab (segLo=0): tests byte-0 init + first superblocks
        {"first slab (segLo=0, segHi=65536)", 0, 65536},

        // 2. Single full sub-block span (2^18 internal bytes of values)
        {"single sub-block", 0, 7864320},

        // 3. Two full sub-blocks
        {"multi sub-block", 0, 15728640},

        // 4. Truncated last sub-block
        {"truncated last sub-block", 7500000, 7864320},

        // 5. Non-zero group-aligned start
        {"non-aligned start", 900000, 2700000},

        // 6. Small range (< one superblock)
        {"tiny range (240 values)", 0, 240},

        // 7. Second sub-block (boundary transition)
        {"second sub-block", 7864320, 15728640},

        // 8. Edge: range just past a superblock boundary
        {"past superblock boundary", 720, 1200},

        // 9. Range within a few superblocks
        {"within two superblocks", 480, 1080},

        // 10. Larger range for stress
        {"larger range (2M values)", 0, 2000000},

        // 11. Superblock-boundary truncation (span % 240 != 0)
        {"superblock truncation", 0, 961},

        // 12. Truncation mid-group at the tail
        {"tail mid-group", 30000, 31625},

        // ---- Task 9 full-matrix additions (30 values/byte geometry) ----

        // 13. Minimum in-contract span: exactly one internal byte
        {"minimum span (one byte)", 0, 30},

        // 14. Sub-byte span: only value 1 defined, all slots padding-cleared
        {"sub-byte span", 0, 7},

        // 15. Degenerate single-value span
        {"degenerate span", 0, 1},

        // 16. Second byte, non-zero byte-aligned start
        {"second byte", 30, 60},

        // 17. Full last-byte boundary (span == 32 bytes exactly)
        {"byte-boundary end", 0, 960},

        // 18. Mid-superblock start crossing a superblock edge
        {"mid-superblock start crossing", 210, 510},

        // 19. Window straddling a sub-block boundary (2^18 internal bytes)
        {"sub-block straddle window", 7864200, 7864500},

        // 20. Mid-superblock start, truncated non-aligned end
        {"mid-superblock truncated", 123450, 125000},

        // 21. Stress at 30 values/byte: 20M values (~667 KiB internal)
        {"stress 20M values", 0, 20000000},

        // 22. Deep offset stress: third sub-block + 2M-value window
        {"deep offset stress", 15728640, 17728640},
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
