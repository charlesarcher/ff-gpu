// Pure-logic self-test for the todo-3 modules (plan todo 3): leg geometry
// chain (mapBytes at 2M/1M/64K legs), searchWorkspaceBytes (M4 gate), size
// parser (GiB/MiB), fraction + device-vram-fraction spec parsing, §4.3
// config validation, PCI-bus-ID dedup (Metis MUST-COVER), §4.2 budget math,
// the §4.3 alloc-failure fraction shrink, and the task-5 wheel-30
// canonical<->internal sizing/layout round-trip. No vendor headers, no GPU
// needed — links with g++ only: `make selftest`.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "budget.h"
#include "config.h"
#include "device_info.h"
#include "device_registry.h"
#include "geometry.h"
#include "gpu_prime.h"
#include "m4/wheel_verdict.h"

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                           \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static void makeDev(ff::DeviceInfo& d, const char* vendor, const char* name,
                    const char* bus, unsigned long long freeBytes)
{
    std::memset(&d, 0, sizeof d);
    std::snprintf(d.vendor, sizeof d.vendor, "%s", vendor);
    std::snprintf(d.name, sizeof d.name, "%s", name);
    std::snprintf(d.busId, sizeof d.busId, "%s", bus);
    d.freeBytes = freeBytes;
    d.totalBytes = freeBytes;
}

int main()
{
    // ---- geometry chain (reference segmentedSieve.C:851-855 + :136) ----
    ff::LegGeometry g2m = ff::computeLegGeometry(5, 2097152);
    CHECK(g2m.sumStart == 5);
    CHECK(g2m.maxGeneratedPrime == 1048576ull);
    CHECK(g2m.productLimit == 1099511627776ull);
    CHECK(g2m.primeLimit == 274877906946ull);
    CHECK(g2m.mapBytes == 17179869185ull);          // 2^34+1 (16.0 GiB)
    CHECK(g2m.maxPrimeMapValue == 274877906959ull);

    ff::LegGeometry g1m = ff::computeLegGeometry(5, 1048576);
    CHECK(g1m.mapBytes == 4294967297ull);           // 2^32+1 (4.0 GiB)

    ff::LegGeometry g64k = ff::computeLegGeometry(5, 65536);
    CHECK(g64k.mapBytes == 16777217ull);            // 2^24+1 (16 MiB)

    CHECK(ff::computeLegGeometry(6, 100).sumStart == 7);   // sumStart forced odd

    // M4 SEARCH-participation workspace (todo 14): 24960 B at 2M
    CHECK(ff::searchWorkspaceBytes(2097152) == 24960ull);

    // ---- wheel-30 foundations (task 5): sizing split + expand/compact ----
    // Deliberately independent copy of the residue table: if geometry.h's
    // kWheelResidues is corrupted, these literals keep the oracles honest.
    static const unsigned kR[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    auto isCoprime30 = [](uint64_t v) {
        for (unsigned r : kR)
            if (v % 30 == r) return true;
        return false;
    };

    CHECK(ff::canonicalMapBytes(64) == 4);
    CHECK(ff::canonicalMapBytes(274877906960ull) == 17179869185ull);
    CHECK(ff::internalMapBytes(0) == 0);
    CHECK(ff::internalMapBytes(1) == 1);
    CHECK(ff::internalMapBytes(29) == 1);
    CHECK(ff::internalMapBytes(30) == 1);
    CHECK(ff::internalMapBytes(31) == 2);
    CHECK(ff::internalMapBytes(959) == 32);
    CHECK(ff::internalMapBytes(960) == 32);
    CHECK(ff::internalMapBytes(961) == 33);
    CHECK(ff::internalMapBytes(274877906960ull) == 9162596899ull);
    // Honest density: 30/16 = 1.875x vs canonical (NOT 3.75x).
    CHECK(g2m.internalMapBytes == 9162596899ull);
    CHECK(g1m.internalMapBytes == 2290649226ull);
    CHECK(g64k.internalMapBytes == 8947850ull);
    CHECK(g2m.internalMapBytes * 30 >= 274877906960ull);
    CHECK(g2m.mapBytes > g2m.internalMapBytes);

    uint32_t rng = 0x12345678u;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 8;
    };

    // Hardcoded layout oracles (mutation-discriminating: they pin value ->
    // canonical-bit placement independent of kWheelResidues).
    {
        std::vector<uint8_t> x(3, 0), y(4, 0xAA);
        x[0] = 1u << 2;  // block 0, slot 2 -> value 11
        ff::expandWheel30ToCanonical(x.data(), y.data(), 64);
        const uint8_t wantY[4] = {0x64, 0x00, 0x00, 0x00};  // 3,5 forced + 11
        CHECK(std::memcmp(y.data(), wantY, 4) == 0);
        std::vector<uint8_t> x2(3, 0xAA);
        ff::compactCanonicalToWheel30(y.data(), x2.data(), 64);
        const uint8_t wantX[3] = {0x04, 0x00, 0x00};
        CHECK(std::memcmp(x2.data(), wantX, 3) == 0);
    }
    {
        // Superblock boundary: span 240 = one full 8-byte -> 15-byte group;
        // block 7 slot 7 -> value 239 -> canonical byte 14, mask 0x01.
        std::vector<uint8_t> x(8, 0), y(15, 0);
        x[7] = 1u << 7;
        ff::expandWheel30ToCanonical(x.data(), y.data(), 240);
        CHECK(y[0] == 0x60 && y[14] == 0x01);
        bool restClear = true;
        for (unsigned b = 1; b < 14; ++b) restClear = restClear && y[b] == 0;
        CHECK(restClear);
    }
    {
        // Tail block: span 245, block 8 slot 0 -> value 241 -> byte 15, 0x80.
        std::vector<uint8_t> x(9, 0), y(16, 0);
        x[8] = 1u << 0;
        ff::expandWheel30ToCanonical(x.data(), y.data(), 245);
        CHECK(y[0] == 0x60 && y[15] == 0x80);
    }

    // Property round-trips: edge spans + random spans + one large sample.
    auto checkSpan = [&](uint64_t span) {
        const uint64_t nInt = ff::internalMapBytes(span);
        const uint64_t nCan = ff::canonicalMapBytes(span);
        std::vector<uint8_t> x(nInt), y1(nCan), x2(nInt), y(nCan), xp(nInt), y2(nCan);
        for (uint64_t i = 0; i < nInt; ++i) x[i] = static_cast<uint8_t>(next());
        for (uint64_t k = 0; k < nInt; ++k)          // zero padding slots
            for (unsigned i = 0; i < 8; ++i)
                if (30ull * k + kR[i] >= span) x[k] &= static_cast<uint8_t>(~(1u << i));

        ff::expandWheel30ToCanonical(x.data(), y1.data(), span);
        ff::compactCanonicalToWheel30(y1.data(), x2.data(), span);
        if (nInt) CHECK(std::memcmp(x2.data(), x.data(), nInt) == 0);

        // Residue oracle on y1 (independent of kWheelResidues).
        auto probe = [&](uint64_t v) {
            if (!(v & 1) || v >= span) return;
            const uint8_t mask = static_cast<uint8_t>(0x80u >> ((v >> 1) & 7));
            const bool set = (y1[v >> 4] & mask) != 0;
            if (!isCoprime30(v)) {
                CHECK(set == (v == 3 || v == 5));
            } else {
                unsigned slot = 8;
                for (unsigned i = 0; i < 8; ++i)
                    if (v % 30 == kR[i]) slot = i;
                CHECK(set == ((x[v / 30] >> slot) & 1u));
            }
        };
        if (span <= 20000) {
            for (uint64_t v = 1; v < span; v += 2) probe(v);
        } else {
            probe(1); probe(3); probe(5); probe(29); probe(31);
            probe(span - 1); probe(span - 2); probe(span - 31);
            for (unsigned t = 0; t < 128; ++t) probe(next() % span);
        }

        // Inverse direction: valid canonical y -> compact -> expand == y.
        for (uint64_t b = 0; b < nCan; ++b) y[b] = static_cast<uint8_t>(next());
        for (uint64_t v = 1; v < span; v += 2) {
            const uint8_t mask = static_cast<uint8_t>(0x80u >> ((v >> 1) & 7));
            if (!isCoprime30(v)) y[v >> 4] &= static_cast<uint8_t>(~mask);
            if (v == 3 && span > 3) y[0] |= 0x40u;
            if (v == 5 && span > 5) y[0] |= 0x20u;
        }
        for (uint64_t v = span; v < nCan * 16ull; ++v) {            // zero padding
            if (!(v & 1)) continue;
            const uint8_t mask = static_cast<uint8_t>(0x80u >> ((v >> 1) & 7));
            y[v >> 4] &= static_cast<uint8_t>(~mask);
        }
        ff::compactCanonicalToWheel30(y.data(), xp.data(), span);
        ff::expandWheel30ToCanonical(xp.data(), y2.data(), span);
        if (nCan) CHECK(std::memcmp(y2.data(), y.data(), nCan) == 0);
    };

    for (uint64_t s : {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 15ull,
                       16ull, 17ull, 29ull, 30ull, 31ull, 32ull, 45ull, 59ull,
                       60ull, 61ull, 89ull, 90ull, 91ull, 95ull, 239ull,
                       240ull, 241ull, 255ull, 256ull, 959ull, 960ull,
                       961ull, 1023ull, 1024ull, 4095ull, 4096ull})
        checkSpan(s);
    for (unsigned t = 0; t < 128; ++t) checkSpan(1 + next() % 60000);
    checkSpan(1048577ull);
    checkSpan(2400001ull);

    // ---- task 15 (D): Wheel30Verdict == GpuPrime::IsPrime ----
    // The emit-path decoder must be verdict-exact against the canonical
    // oracle: exhaustive n in [0, 2^22) over a random valid internal map
    // (zero padding slots) expanded by the task-5 helper, then 10^6 random
    // strata probes up to a full 2^28 span with boundary values, then a
    // multi-slab PACKED-layout config exercising the production address
    // translation ((s*slab/8)*15 placement shared with the H2D feed and
    // expandSieveMapToCanonical).
    {
        auto buildMaps = [&](uint64_t span, std::vector<uint8_t>& x,
                             std::vector<uint8_t>& y) {
            x.assign(ff::internalMapBytes(span), 0);
            y.assign(ff::canonicalMapBytes(span), 0);
            for (uint64_t i = 0; i < x.size(); ++i)
                x[i] = static_cast<uint8_t>(next());
            for (uint64_t k = 0; k < x.size(); ++k)          // zero padding slots
                for (unsigned i = 0; i < 8; ++i)
                    if (30ull * k + kR[i] >= span)
                        x[k] &= static_cast<uint8_t>(~(1u << i));
            ff::expandWheel30ToCanonical(x.data(), y.data(), span);
        };
        auto verdictEq = [](bool got, GpuPrime::Boolean want) {
            return got == (want == GpuPrime::True);
        };

        {   // exhaustive [0, 2^22): every residue class, 3/5 specials, bounds
            const uint64_t span = 1ull << 22;
            std::vector<uint8_t> x, y;
            buildMaps(span, x, y);
            GpuPrime oracle(y.data(), span - 1);
            // Contiguous internal image: one 8-aligned slab => identity
            // placement (single slab, s == 0 for every q).
            Wheel30Verdict dec(x.data(), span - 1, (x.size() + 7ull) & ~7ull);
            for (uint64_t n = 0; n < span; ++n)
                CHECK(verdictEq(dec.IsPrime(n), oracle.IsPrime(n)));
        }
        {   // 10^6 random strata up to the full 2^28 span + edges
            const uint64_t span = 1ull << 28;
            std::vector<uint8_t> x, y;
            buildMaps(span, x, y);
            GpuPrime oracle(y.data(), span - 1);
            Wheel30Verdict dec(x.data(), span - 1, (x.size() + 7ull) & ~7ull);
            const uint64_t edge[] = {0, 1, 2, 3, 5, 7, 29, 30, 31, 59, 61,
                                     (1ull << 20) - 1, 1ull << 20,
                                     span - 31, span - 30, span - 2, span - 1};
            for (uint64_t n : edge)
                CHECK(verdictEq(dec.IsPrime(n), oracle.IsPrime(n)));
            for (unsigned t = 0; t < 1000000; ++t) {
                const uint64_t n =
                    ((static_cast<uint64_t>(next()) << 24) | next()) % span;
                CHECK(verdictEq(dec.IsPrime(n), oracle.IsPrime(n)));
            }
        }
        {   // production PACKED layout: slabs of 512 internal bytes at
            // (s*slab/8)*15 — the exact placement main.cpp feeds H2D and
            // expandSlabRange widens. Odd sweep: layout-sensitive values.
            const uint64_t span = 30000;
            const uint64_t slab = 512;                       // 8-aligned
            std::vector<uint8_t> x, y;
            buildMaps(span, x, y);
            std::vector<uint8_t> packed((x.size() / 8) * 15 + 16, 0);
            for (uint64_t iOff = 0; iOff < x.size(); iOff += slab) {
                const uint64_t cBi = std::min(slab, x.size() - iOff);
                std::memcpy(packed.data() + (iOff / 8) * 15, x.data() + iOff,
                            cBi);
            }
            GpuPrime oracle(y.data(), span - 1);
            Wheel30Verdict dec(packed.data(), span - 1, slab);
            for (uint64_t n = 1; n < span; n += 2)
                CHECK(verdictEq(dec.IsPrime(n), oracle.IsPrime(n)));
        }
    }


    // ---- size parser ----
    uint64_t v = 0;
    CHECK(ff::parseSize("20GiB", &v) && v == 21474836480ull);
    CHECK(ff::parseSize("8GiB", &v) && v == 8589934592ull);
    CHECK(ff::parseSize("1GiB", &v) && v == 1073741824ull);
    CHECK(ff::parseSize("1024MiB", &v) && v == 1073741824ull);
    CHECK(ff::parseSize("64MiB", &v) && v == 67108864ull);
    CHECK(ff::parseSize("4096", &v) && v == 4096ull);
    CHECK(ff::parseSize("1.5GiB", &v) && v == 1610612736ull);
    CHECK(ff::parseSize("0", &v) && v == 0ull);
    CHECK(!ff::parseSize("auto", &v));
    CHECK(!ff::parseSize("", &v));
    CHECK(!ff::parseSize("abc", &v));
    CHECK(!ff::parseSize("-5GiB", &v));
    CHECK(!ff::parseSize("20GiBX", &v));

    // ---- fractions ----
    double f = 0.0;
    CHECK(ff::parseFraction("0.90", &f) && f > 0.899 && f < 0.901);
    CHECK(ff::parseFraction("2", &f) && f == 2.0);
    CHECK(ff::parseFraction("0.001", &f) && f == 0.001);
    CHECK(!ff::parseFraction("x", &f));

    // ---- device-vram-fraction spec ----
    std::map<std::string, double> m;
    CHECK(ff::parseDeviceFractionSpec("amd=0.001", &m) && m.size() == 1 &&
          m["amd"] == 0.001);
    CHECK(ff::parseDeviceFractionSpec("nvidia=0.90,amd=0.80", &m) &&
          m.size() == 2 && m["nvidia"] == 0.90 && m["amd"] == 0.80);
    CHECK(!ff::parseDeviceFractionSpec("amd", &m));
    CHECK(!ff::parseDeviceFractionSpec("amd=0.001,", &m));
    CHECK(!ff::parseDeviceFractionSpec("=0.5", &m));

    // ---- §4.3 config validation ----
    ff::Config cfg;
    CHECK(ff::validateConfig(&cfg) == 0);          // defaults valid
    cfg.globalFraction = 2.0;
    CHECK(ff::validateConfig(&cfg) != 0);          // f=2 rejected
    cfg = ff::Config();
    cfg.deviceFractions["amd"] = 0.001;
    CHECK(ff::validateConfig(&cfg) != 0);          // amd=0.001 rejected
    cfg = ff::Config();
    cfg.hasBudgetCap = true;
    cfg.budgetCapBytes = 0;
    CHECK(ff::validateConfig(&cfg) != 0);          // zero cap rejected
    cfg = ff::Config();
    cfg.slabSizeBytes = 3;
    CHECK(ff::validateConfig(&cfg) != 0);          // 3 B not 16-value aligned
    CHECK(!ff::slabSizeValueAligned(3));
    CHECK(ff::slabSizeValueAligned(1073741824ull));   // 1 GiB aligned

    // ---- PCI bus-ID dedup (Metis MUST-COVER) ----
    ff::DeviceInfo hip[2], cuda[2];
    makeDev(hip[0], "amd", "RX 9070 XT", "0000:05:00", 17095983104ull);
    makeDev(hip[1], "amd", "RX 9070 XT #2", "0000:06:00", 999ull);
    makeDev(cuda[0], "nvidia", "RTX 5090", "0000:01:00", 34182397952ull);
    makeDev(cuda[1], "nvidia", "RTX 5090 (dup)", "0000:01:00", 34182397952ull);
    int skipped = -1;
    std::vector<ff::DeviceInfo> merged =
        ff::mergeAndDedupe(hip, 2, cuda, 2, &skipped);
    CHECK(merged.size() == 3);                     // duplicate 5090 dropped
    CHECK(skipped == 1);
    CHECK(std::string(merged[0].busId) == "0000:05:00");
    CHECK(std::string(merged[1].busId) == "0000:06:00");
    CHECK(std::string(merged[2].busId) == "0000:01:00");
    makeDev(hip[0], "amd", "RX 9070 XT", "0000:05:00", 17095983104ull);
    makeDev(cuda[0], "nvidia", "RTX 5090", "0000:01:00", 34182397952ull);
    merged = ff::mergeAndDedupe(hip, 1, cuda, 1, &skipped);
    CHECK(merged.size() == 2 && skipped == 0);     // no overlap -> no dedup

    // ---- §4.2 budget math on a synthetic device (no hardware needed) ----
    ff::DeviceInfo dev;
    makeDev(dev, "nvidia", "RTX 5090", "0000:01:00", 34182397952ull);  // ~31.8 GiB free
    ff::Config bcfg;                                // defaults: f=0.90, slab 1 GiB
    ff::DeviceBudget b = ff::computeBudget(dev, bcfg, 0.90);
    CHECK(b.freeUsed == static_cast<unsigned long long>(0.90 * 34182397952.0));
    CHECK(b.budget == b.freeUsed);                  // uncapped
    CHECK(b.scratch == 1073741824ull);              // min(0.15 x budget, 1 GiB)
    CHECK(b.backing ==
          ((b.budget - b.scratch - ff::kHeadroomBytes) / 4096) * 4096);
    CHECK(b.slabCount == b.backing / 1073741824ull);

    bcfg.hasBudgetCap = true;
    bcfg.budgetCapBytes = 21474836480ull;           // 20 GiB cap
    b = ff::computeBudget(dev, bcfg, 0.90);
    CHECK(b.budget == 21474836480ull);
    CHECK(b.backing == 21474836480ull - 1073741824ull - 67108864ull);
    CHECK(b.slabCount == 18);                       // 19392 MiB / 1 GiB

    bcfg.budgetCapBytes = 8589934592ull;            // 8 GiB cap
    b = ff::computeBudget(dev, bcfg, 0.90);
    CHECK(b.budget == 8589934592ull);
    CHECK(b.backing == 8589934592ull - 1073741824ull - 67108864ull);
    CHECK(b.slabCount == 6);

    unsigned long long agg8 = 2 * b.backing;        // two 8 GiB-capped devices
    CHECK(agg8 < g2m.mapBytes);                     // ~13.9 GiB < 16.0 GiB -> FAIL
    unsigned long long agg20 =
        2 * (21474836480ull - 1073741824ull - 67108864ull);
    CHECK(agg20 >= g2m.mapBytes);                   // 20 GiB caps -> PASS

    // ---- amd@2M participation flip (tasks 6+7, BY DESIGN) ----
    // RX 9070 XT-class device: the CANONICAL 16-GiB map cannot fit its
    // backing, but the wheel-30 INTERNAL map (8.53 GiB) fits, so GPU search
    // participates at the 2M leg where the canonical era needed the host
    // overflow tier. Pinned on internal sizing so a future density change
    // re-visits this flip consciously.
    makeDev(dev, "amd", "RX 9070 XT", "0000:05:00", 17095983104ull);
    bcfg = ff::Config();
    b = ff::computeBudget(dev, bcfg, 0.90);
    CHECK(b.backing < g2m.mapBytes);                // canonical era: no fit -> host tier
    CHECK(b.backing >= g2m.internalMapBytes);       // wheel era: fits -> search participates

    // ---- §4.3 alloc-failure fallback: one midpoint step toward the 0.50 floor
    // ---- (caller loops on alloc failure; backing strictly shrinks each step) ----
    makeDev(dev, "nvidia", "RTX 5090", "0000:01:00", 21474836480ull);  // 20 GiB free
    bcfg = ff::Config();
    ff::DeviceBudget s1 = ff::shrinkFractionForAlloc(dev, bcfg, 0.90);
    CHECK(s1.fraction > 0.70 - 1e-9 && s1.fraction < 0.70 + 1e-9);
    CHECK(s1.backing < ff::computeBudget(dev, bcfg, 0.90).backing);
    ff::DeviceBudget s2 = ff::shrinkFractionForAlloc(dev, bcfg, s1.fraction);
    CHECK(s2.fraction > 0.60 - 1e-9 && s2.fraction < 0.60 + 1e-9);
    CHECK(s2.backing < s1.backing);
    ff::DeviceBudget s3 = ff::shrinkFractionForAlloc(dev, bcfg, s2.fraction);
    CHECK(s3.fraction > 0.55 - 1e-9 && s3.fraction < 0.55 + 1e-9);
    ff::DeviceBudget floorStep = ff::shrinkFractionForAlloc(dev, bcfg, 0.50);
    CHECK(floorStep.fraction == 0.50);   // at the floor: no further shrink
    CHECK(floorStep.backing == ff::computeBudget(dev, bcfg, 0.50).backing);

    // ---- humanize ----
    CHECK(ff::bytesToHuman(17179869185ull) == "16.00 GiB");
    CHECK(ff::bytesToHuman(1073741824ull) == "1.00 GiB");

    if (g_failures == 0) {
        std::printf("ff_budget_selftest: ALL PASS\n");
        return 0;
    }
    std::printf("ff_budget_selftest: %d FAILURE(S)\n", g_failures);
    return 1;
}
