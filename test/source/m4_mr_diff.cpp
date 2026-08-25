// m4_mr_diff.cpp — Differential test: DEVICE Miller-Rabin verdicts vs HOST
// GpuPrime::AskMillerRabin (source/gpu_prime.cpp, the proven CPU reference).
//
// The device side runs the REAL kernel code under change: MRVerdictKernel_
// <arch> (source/m4/gpu_search_kernel.h) calls dev_IsPrime exactly as the
// production search kernel does. Any verdict divergence between device and
// host is a FAILURE — perfect match is the pass condition.
//
// MAP FORMAT (kernel-gap-closure task 8): dev_IsPrime consumes the INTERNAL
// wheel-30 layout (byte k = values [30k,30k+30), bit i LSB-first = value
// 30k + kWheelResidues[i], source/geometry.h). The device buffers therefore
// carry generatePrimeMapInternal() output — the canonical sieve compacted by
// the landed task-5 helper ff::compactCanonicalToWheel30 — while the HOST
// oracle always runs on the CANONICAL map (GpuPrime's layout). The differential
// thus validates the full chain canonical-sieve -> compact -> H2D -> wheel
// decode against canonical truth; a residue-table or slot-order mismatch
// anywhere diverges loudly.
//
// Coverage (>= 200k sampled n across configurations):
//   A. contiguous [maxPrimeMapValue_65K+1 .. +100000]   (production 65K-leg
//      map bound, replicated from source/geometry.cpp:15-20)
//   B. 100k random n up to 2^44 (fixed seed, reproducible)
//   C. adversarial edges: 2^32+-1..3, Jaeschke strong-pseudoprime anchors
//      (3215031751 / 3474749660383 / 341550071728321), classic spsp,
//      2^k+1 maximal power2 chains, Mersenne values, m near 2^63
//   D. Carmichael numbers 561/1105/1729/2465/2821/6601 + small spsp under a
//      LOW map bound (500) so they take the MR branch
//   E. every odd n in [5..119] under a TINY map bound (4) — smallest possible
//      Montgomery moduli; also exercises the {2,3,5} fast path and the
//      3/5-multiple guard on nearly every wheel byte
//
// Runs on EVERY logical GPU (both vendors when present).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#include "devabstraction.h"
#include "gpu_prime.h"
#include "geometry.h"

extern "C" {
typedef int (*MRDiffRunFn)(int deviceIndex,
                           const uint64_t* d_ns, uint8_t* d_verdicts, uint32_t count,
                           const uint8_t* d_primeMap, uint64_t maxPrimeMapValue);
extern MRDiffRunFn MRDiffGetLaunchFn_gfx1201(int deviceIndex);
extern MRDiffRunFn MRDiffGetLaunchFn_sm_120(int deviceIndex);

}

#define CHECK(expr) \
    do { \
        int _rc = (expr); \
        if (_rc != 0) { \
            std::fprintf(stderr, "CHECK FAILED: %s (rc=%d) at %s:%d\n", \
                         #expr, _rc, __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

// ---- Prime map (same bit layout as production: byte per 16 values, bit per
// odd value; even values share their odd neighbor's bit) ----

static std::vector<uint8_t> generatePrimeMap(uint64_t limit) {
    std::vector<uint8_t> map((limit >> 4) + 1, 0xff);
    map[0] ^= 0x80; // 1 is not prime
    for (uint64_t p = 3; p * p <= limit; p += 2) {
        if (map[p >> 4] & (0x80 >> (p >> 1 & 7))) {
            for (uint64_t i = p * p; i <= limit; i += p << 1)
                map[i >> 4] &= ~(0x80 >> (i >> 1 & 7));
        }
    }
    return map;
}

// INTERNAL wheel-30 variant of the above (task 8): canonical sieve compacted
// by the landed task-5 helper. span = limit+1 matches the geometry.h contract
// (span == maxPrimeMapValue+1); generatePrimeMap(limit) already allocates
// exactly canonicalMapBytes(limit+1) readable bytes, so compact reads only
// defined bits. Padding slots come back zeroed per the helper contract.
static std::vector<uint8_t> generatePrimeMapInternal(uint64_t limit) {
    std::vector<uint8_t> canon = generatePrimeMap(limit);
    const uint64_t span = limit + 1;
    std::vector<uint8_t> internal(ff::internalMapBytes(span), 0);
    ff::compactCanonicalToWheel30(canon.data(), internal.data(), span);
    return internal;
}

// maxPrimeMapValue of the 65536 leg, replicated from source/geometry.cpp:15-20
// ((sumLimit+1)>>1)^2 >> 2 + 2 -> rounded up to a 16-value map boundary.
static uint64_t maxPrimeMapValue_65K() {
    const uint64_t sumLimit = 65536;
    uint64_t maxGeneratedPrime = (sumLimit + 1) >> 1;
    uint64_t productLimit = maxGeneratedPrime * maxGeneratedPrime;
    uint64_t primeLimit = (productLimit >> 2) + 2;
    uint64_t mapBytes = ((primeLimit > 1 ? primeLimit : 2) + 15) >> 4;
    return (mapBytes << 4) - 1;
}

// ---- One differential batch against one device ----

struct DiffStats {
    uint64_t tested = 0;
    uint64_t divergences = 0;
};

static bool runBatchOnDevice(MRDiffRunFn launchFn, int logicalIndex, int vendorIndex,
                             const std::vector<uint8_t>& h_map, uint64_t maxMapValue,
                             const char* label, const std::vector<uint64_t>& ns,
                             DiffStats& stats,
                             const std::vector<uint8_t>& oracleMap) {
    if (ns.empty()) {
        std::printf("  [%s] SKIP (empty bucket)\n", label);
        return true;
    }

    ffdev::DevHandle dNs, dVerdicts, dMap;
    CHECK(ffdev::DevAlloc(logicalIndex, ns.size() * sizeof(uint64_t), &dNs));
    CHECK(ffdev::DevAlloc(logicalIndex, ns.size(), &dVerdicts));
    CHECK(ffdev::DevAlloc(logicalIndex, h_map.size(), &dMap));
    CHECK(ffdev::DevCopy(&dNs, const_cast<uint64_t*>(ns.data()),
                         ns.size() * sizeof(uint64_t), ffdev::DevCopyDir::H2D));
    CHECK(ffdev::DevCopy(&dMap, const_cast<uint8_t*>(h_map.data()),
                         h_map.size(), ffdev::DevCopyDir::H2D));
    {   // poison: proves every verdict byte was written by THIS launch
        std::vector<uint8_t> poison(ns.size(), 0x5A);
        CHECK(ffdev::DevCopy(&dVerdicts, poison.data(), ns.size(),
                             ffdev::DevCopyDir::H2D));
    }

    int rc = launchFn(vendorIndex,
                      (const uint64_t*)dNs.ptr, (uint8_t*)dVerdicts.ptr,
                      (uint32_t)ns.size(),
                      (const uint8_t*)dMap.ptr, maxMapValue);

    std::vector<uint8_t> h_verdicts(ns.size(), 0xAA);
    std::vector<uint64_t> h_echo(ns.size(), 0xDEAD);
    if (rc == 0) {
        CHECK(ffdev::DevCopy(&dVerdicts, h_verdicts.data(), ns.size(),
                             ffdev::DevCopyDir::D2H));
        CHECK(ffdev::DevCopy(&dNs, h_echo.data(), ns.size() * sizeof(uint64_t),
                             ffdev::DevCopyDir::D2H));
    }

    ffdev::DevFree(&dMap);
    ffdev::DevFree(&dVerdicts);
    ffdev::DevFree(&dNs);

    if (rc != 0) {
        std::printf("  [%s] LAUNCH FAILED (rc=%d)\n", label, rc);
        return false;
    }

    uint32_t echoBad = 0;
    for (size_t i = 0; i < ns.size(); ++i) {
        if (h_echo[i] != ns[i]) {
            if (echoBad++ < 5)
                std::printf("  [%s] ECHO MISMATCH idx=%zu sent=%llu got=%llu\n",
                            label, i, (unsigned long long)ns[i],
                            (unsigned long long)h_echo[i]);
        }
    }
    if (echoBad) std::printf("  [%s] ECHO BAD=%u\n", label, echoBad);

    GpuPrime host(oracleMap.data(), maxMapValue);
    uint32_t printed = 0;
    for (size_t i = 0; i < ns.size(); ++i) {
        ++stats.tested;
        bool hostV = host.IsPrime(ns[i]) == GpuPrime::True;
        bool devV = h_verdicts[i] != 0;
        if (hostV != devV) {
            ++stats.divergences;
            if (printed++ < 20) {
                std::printf("  [%s] DIVERGENCE n=%llu host=%s device=%s\n",
                            label, (unsigned long long)ns[i],
                            hostV ? "PRIME" : "composite",
                            devV ? "PRIME" : "composite");
            }
        }
    }
    std::printf("  [%s] tested=%llu divergences=%llu\n",
                label, (unsigned long long)stats.tested,
                (unsigned long long)stats.divergences);
    return stats.divergences == 0;
}

static void appendRange(std::vector<uint64_t>& v, uint64_t lo, uint64_t hi) {
    for (uint64_t n = lo; n <= hi; ++n) v.push_back(n);
}

static void appendValues(std::vector<uint64_t>& v, std::initializer_list<uint64_t> xs) {
    for (uint64_t x : xs) v.push_back(x);
}

// ---- Per-device driver: all configurations and buckets ----

static bool runOnDevice(int logicalIndex, int vendorIndex, MRDiffRunFn launchFn,
                        const std::vector<uint8_t>& prodCanon, uint64_t prodThr) {
    bool ok = true;
    DiffStats total;

    // Device buffers carry the INTERNAL wheel-30 map; the host oracle keeps
    // the CANONICAL one (GpuPrime's layout). Same value span, both derived
    // from the same sieve — see the MAP FORMAT note in the header comment.
    const std::vector<uint8_t> prodInternal = generatePrimeMapInternal(prodThr);

    // -- Config P: production 65K-leg map bound --
    {
        DiffStats s;
        std::vector<uint64_t> contiguous;
        appendRange(contiguous, prodThr + 1, prodThr + 100000);
        ok &= runBatchOnDevice(launchFn, logicalIndex, vendorIndex, prodInternal, prodThr,
                               "P:A contiguous thr+1..thr+100000", contiguous, s,
                               prodCanon);

        std::mt19937_64 rng(0x853C49E6748FEA9BULL); // fixed seed: reproducible
        std::vector<uint64_t> random44;
        random44.reserve(100000);
        for (int i = 0; i < 100000; ++i)
            random44.push_back(rng() & ((1ull << 44) - 1));
        ok &= runBatchOnDevice(launchFn, logicalIndex, vendorIndex, prodInternal, prodThr,
                               "P:B random up to 2^44 (seed 0x853C49E6748FEA9B)", random44, s,
                               prodCanon);

        std::vector<uint64_t> adv;
        appendRange(adv, (1ull << 32) - 3, (1ull << 32) + 3);
        appendValues(adv, {
            3215031751ull, 3215031749ull, 3215031753ull, // spsp(2,3,5,7) anchor
            2152302898747ull, 2152302898745ull,
            3474749660383ull, 3474749660381ull,          // spsp(2,3,5,7,11) anchor
            341550071728321ull, 341550071728319ull,      // spsp(2,3,5,7,11,13) anchor
            2047ull, 3277ull, 4033ull, 25326001ull,      // classic small spsp
            (1ull << 61) - 1, (1ull << 43) - 1, (1ull << 31) - 1, // Mersennes
        });
        for (uint32_t k = 10; k <= 44; ++k)
            adv.push_back((1ull << k) + 1);              // maximal power2 chains
        for (int64_t k : {-31, -27, -25, -3, -1, 1, 3, 25})
            adv.push_back((1ull << 63) + k);             // m near 2^63
        ok &= runBatchOnDevice(launchFn, logicalIndex, vendorIndex, prodInternal, prodThr,
                               "P:C adversarial edges", adv, s,
                               prodCanon);
        total.tested += s.tested;
        total.divergences += s.divergences;
    }

    // -- Config L: low map bound (500) — Carmichaels + small spsp take MR --
    {
        DiffStats s;
        std::vector<uint8_t> lowCanon = generatePrimeMap(500);
        std::vector<uint8_t> lowInternal = generatePrimeMapInternal(500);
        std::vector<uint64_t> ns;
        appendValues(ns, {561ull, 1105ull, 1729ull, 2465ull, 2821ull, 6601ull});
        appendValues(ns, {2047ull, 3277ull, 4033ull});
        appendRange(ns, 501, 700); // contiguous MR sweep just above the bound
        ok &= runBatchOnDevice(launchFn, logicalIndex, vendorIndex, lowInternal, 500,
                               "L:D Carmichael+spsp+sweep @bound500", ns, s,
                               lowCanon);
        total.tested += s.tested;
        total.divergences += s.divergences;
    }

    // -- Config T: tiny map bound (4) — smallest Montgomery moduli (m >= 5) --
    {
        DiffStats s;
        std::vector<uint8_t> tinyCanon = generatePrimeMap(4);
        std::vector<uint8_t> tinyInternal = generatePrimeMapInternal(4);
        std::vector<uint64_t> ns;
        appendRange(ns, 5, 120);
        ok &= runBatchOnDevice(launchFn, logicalIndex, vendorIndex, tinyInternal, 4,
                               "T:E odd sweep 5..119 @bound4", ns, s,
                               tinyCanon);
        total.tested += s.tested;
        total.divergences += s.divergences;
    }

    std::printf("  device[%d] TOTAL tested=%llu divergences=%llu -> %s\n",
                logicalIndex, (unsigned long long)total.tested,
                (unsigned long long)total.divergences,
                total.divergences == 0 && ok ? "MATCH" : "MISMATCH");
    return ok && total.divergences == 0;
}

// ---- Main ----

int main() {
    std::printf("=== m4_mr_diff: device MR verdicts vs host GpuPrime ===\n");

    CHECK(ffdev::DevInit());
    int devCount = ffdev::DevGetDeviceCount();
    if (devCount == 0) {
        std::printf("  No GPU devices available — skipping.\n");
        std::printf("=== m4_mr_diff: PASS (vacuous, no devices) ===\n");
        return 0;
    }

    const uint64_t prodThr = maxPrimeMapValue_65K();
    std::printf("  maxPrimeMapValue_65K = %llu\n", (unsigned long long)prodThr);
    std::vector<uint8_t> prodCanon = generatePrimeMap(prodThr);
    std::printf("  canonical oracle map: %zu bytes; device wheel-30 map: %zu bytes\n",
                prodCanon.size(), ff::internalMapBytes(prodThr + 1));

    int devicesTested = 0;
    bool ok = true;
    for (int i = 0; i < devCount; ++i) {
        ff::DeviceInfo di;
        if (ffdev::DevGetDeviceProperties(i, &di) != 0) {
            std::printf("  device[%d]: properties unavailable, skipping\n", i);
            continue;
        }
        ffdev::DevBackend backend = ffdev::DevBackendOf(i);

        // vendorIndex: per-runtime device number used by the per-arch launcher
        // (logical index != vendor index under the dual-runtime abstraction).
        ffdev::DevHandle probe;
        if (ffdev::DevAlloc(i, 16, &probe) != 0) {
            std::printf("  device[%d]: allocation failed, skipping\n", i);
            continue;
        }
        int vendorIndex = probe.vendorIndex;
        ffdev::DevFree(&probe);

        MRDiffRunFn fn = nullptr;
        const char* archName = "?";
        if (backend == ffdev::DevBackend::HipAmd) {
            fn = MRDiffGetLaunchFn_gfx1201(vendorIndex);
            archName = "gfx1201";
        } else if (backend == ffdev::DevBackend::HipNv) {
            fn = MRDiffGetLaunchFn_sm_120(vendorIndex);
            archName = "sm_120";
        }
        if (!fn) {
            std::printf("  device[%d] %s (%s): no %s launcher, skipping\n",
                        i, di.name, ffdev::backendName(backend), archName);
            continue;
        }

        std::printf("  device[%d] %s (%s, vendorIndex=%d)\n",
                    i, di.name, ffdev::backendName(backend), vendorIndex);
        bool devOk = runOnDevice(i, vendorIndex, fn, prodCanon, prodThr);
        ok &= devOk;
        ++devicesTested;
    }

    if (devicesTested == 0) {
        std::printf("  No device with a usable launcher — skipping.\n");
        std::printf("=== m4_mr_diff: PASS (vacuous, no launchers) ===\n");
        return 0;
    }

    std::printf("=== m4_mr_diff: %s (%d device(s) tested) ===\n",
                ok ? "PASS" : "FAIL", devicesTested);
    return ok ? 0 : 1;
}
