// Pure-logic self-test for the todo-3 modules (plan todo 3): leg geometry
// chain (mapBytes at 2M/1M/64K legs), searchWorkspaceBytes (M4 gate), size
// parser (GiB/MiB), fraction + device-vram-fraction spec parsing, §4.3
// config validation, PCI-bus-ID dedup (Metis MUST-COVER), §4.2 budget math and
// the §4.3 alloc-failure fraction shrink. No vendor headers, no GPU needed —
// links with g++ only: `make selftest`.
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
