// ff_sieve main (plan todo 3): startup device introspection + GPU_PLAN §4
// budget CLI. Flow: env -> CLI parse -> §4.3 validation -> --list-devices
// (recon) -> no-args = todo-4 dual-runtime smoke (make smoke regression) ->
// leg path: geometry -> enumerate (HIP + CUDA, deduped by PCI bus ID) ->
// per-device budgets (§4.2) -> AGGREGATE sieve run-gate -> audit on STDERR.
//
// At this stage there is NO sieve: a passing gate prints the budget/audit
// lines and exits 0; a failing gate exits 1 with the aggregate capacity
// numbers + remediation. STDERR discipline (Metis BLOCKER #2): EVERY
// diagnostic goes to stderr — stdout stays byte-pure under every invocation.

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "budget.h"
#include "config.h"
#include "device_info.h"
#include "device_registry.h"
#include "geometry.h"

extern "C" int ff_enum_hip(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_enum_cuda(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_smoke_main(void);   // todo-4 dual-runtime smoke (smoke/smoke_main.cpp)

namespace {

constexpr int kMaxDevices = 64;

struct RunDevices {
    std::vector<ff::DeviceInfo> devs;
    int skippedDuplicates = 0;
    bool hipFailed = false;
    bool cudaFailed = false;
};

RunDevices enumerate()
{
    RunDevices r;
    ff::DeviceInfo hipBuf[kMaxDevices] = {};
    ff::DeviceInfo cudaBuf[kMaxDevices] = {};
    int hipCount = 0, cudaCount = 0;
    if (ff_enum_hip(hipBuf, kMaxDevices, &hipCount) != 0) {
        std::fprintf(stderr,
                     "[ff_sieve] warning: HIP (AMD) enumeration failed; no AMD "
                     "devices will participate\n");
        r.hipFailed = true;
    }
    if (ff_enum_cuda(cudaBuf, kMaxDevices, &cudaCount) != 0) {
        std::fprintf(stderr,
                     "[ff_sieve] warning: CUDA (NVIDIA) enumeration failed; no "
                     "NVIDIA devices will participate\n");
        r.cudaFailed = true;
    }
    r.devs = ff::mergeAndDedupe(hipBuf, hipCount, cudaBuf, cudaCount,
                                &r.skippedDuplicates);
    return r;
}

void printDeviceHeader(const ff::DeviceInfo& d, int idx)
{
    std::fprintf(stderr,
                 "[ff_sieve] device[%d] %s (vendor=%s bus=%s): free=%llu B (%s) "
                 "total=%llu B (%s) compute=%d.%d maxThreads=%d "
                 "sharedMem=%llu B smem/mp=%llu B smp=%d\n",
                 idx, d.name, d.vendor, d.busId,
                 static_cast<unsigned long long>(d.freeBytes),
                 ff::bytesToHuman(d.freeBytes).c_str(),
                 static_cast<unsigned long long>(d.totalBytes),
                 ff::bytesToHuman(d.totalBytes).c_str(), d.computeMajor,
                 d.computeMinor, d.maxThreadsPerBlock,
                 static_cast<unsigned long long>(d.sharedMemPerBlock),
                 static_cast<unsigned long long>(d.sharedMemPerMultiprocessor),
                 d.multiProcessorCount);
}

int runListDevices()
{
    RunDevices r = enumerate();
    if (r.devs.empty()) {
        std::fprintf(stderr, "[ff_sieve] error: no devices enumerated\n");
        return 1;
    }
    std::fprintf(stderr,
                 "== ff_sieve logical device list (deduped by PCI bus ID) ==\n");
    for (size_t i = 0; i < r.devs.size(); ++i)
        printDeviceHeader(r.devs[i], static_cast<int>(i));
    std::fprintf(stderr, "%zu logical device(s), %d bus-ID duplicate(s) skipped\n",
                 r.devs.size(), r.skippedDuplicates);
    return 0;
}

double fractionFor(const ff::Config& cfg, const ff::DeviceInfo& d, int idx)
{
    auto it = cfg.deviceFractions.find(d.vendor);
    if (it != cfg.deviceFractions.end()) return it->second;
    auto it2 = cfg.deviceFractions.find(std::to_string(idx));
    if (it2 != cfg.deviceFractions.end()) return it2->second;
    return cfg.globalFraction;
}

std::string renderDeviceFractions(const ff::Config& cfg)
{
    if (cfg.deviceFractions.empty()) return "(none)";
    std::string s;
    for (const auto& kv : cfg.deviceFractions) {
        if (!s.empty()) s += ",";
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.3g", kv.second);
        s += kv.first + "=" + buf;
    }
    return s;
}

int runLeg(const ff::Config& cfg, const std::vector<std::string>& positionals)
{
    // Positional semantics mirror the reference main (segmentedSieve.C:819-831).
    uint64_t sumStart = 0, sumLimit = 0;
    if (positionals.size() > 1) {
        sumStart = std::strtoull(positionals[0].c_str(), nullptr, 10) | 1;
        sumLimit = std::strtoull(positionals[1].c_str(), nullptr, 10);
    } else {
        sumStart = 5;
        sumLimit = positionals.size() == 1
                       ? std::strtoull(positionals[0].c_str(), nullptr, 10)
                       : 2627;   // reference default
    }
    ff::LegGeometry g = ff::computeLegGeometry(sumStart, sumLimit);

    std::fprintf(stderr,
                 "[ff_sieve] leg: sumStart=%llu sumLimit=%llu maxGeneratedPrime=%llu "
                 "productLimit=%llu primeLimit=%llu mapBytes=%llu B (%s) "
                 "maxPrimeMapValue=%llu\n",
                 static_cast<unsigned long long>(g.sumStart),
                 static_cast<unsigned long long>(g.sumLimit),
                 static_cast<unsigned long long>(g.maxGeneratedPrime),
                 static_cast<unsigned long long>(g.productLimit),
                 static_cast<unsigned long long>(g.primeLimit),
                 static_cast<unsigned long long>(g.mapBytes),
                 ff::bytesToHuman(g.mapBytes).c_str(),
                 static_cast<unsigned long long>(g.maxPrimeMapValue));
    std::fprintf(stderr,
                 "[ff_sieve] config: --vram-fraction=%.3g "
                 "--device-vram-fraction=%s --vram-budget=%s --scratch=%s "
                 "--slab-size=%llu B (%s) --host-tier-cap=%s%s\n",
                 cfg.globalFraction, renderDeviceFractions(cfg).c_str(),
                 cfg.hasBudgetCap ? ff::bytesToHuman(cfg.budgetCapBytes).c_str()
                                  : "uncapped",
                 cfg.hasScratch ? ff::bytesToHuman(cfg.scratchBytes).c_str()
                                : "auto",
                 static_cast<unsigned long long>(cfg.slabSizeBytes),
                 ff::bytesToHuman(cfg.slabSizeBytes).c_str(),
                 cfg.noHostTier ? "0 (forced by --no-host-tier)"
                                : ff::bytesToHuman(cfg.hostTierCapBytes).c_str(),
                 cfg.hostTierCapBytes == 0 && !cfg.noHostTier ? " (disabled)"
                                                              : "");

    RunDevices r = enumerate();
    if (r.devs.empty()) {
        std::fprintf(stderr,
                     "[ff_sieve] error: no devices enumerated (HIP%s, CUDA%s)\n",
                     r.hipFailed ? " failed" : " ok", r.cudaFailed ? " failed" : " ok");
        return 1;
    }
    if (r.skippedDuplicates > 0)
        std::fprintf(stderr, "[ff_sieve] note: %d bus-ID duplicate(s) skipped in "
                             "the device union\n", r.skippedDuplicates);

    // §4.2 per-device budgets.
    std::vector<ff::DeviceBudget> budgets;
    unsigned long long aggregateBacking = 0;
    for (size_t i = 0; i < r.devs.size(); ++i) {
        const ff::DeviceInfo& d = r.devs[i];
        printDeviceHeader(d, static_cast<int>(i));
        double f = fractionFor(cfg, d, static_cast<int>(i));
        ff::DeviceBudget b = ff::computeBudget(d, cfg, f);
        if (cfg.slabSizeBytes > b.backing) {
            std::fprintf(stderr,
                         "[ff_sieve] validation error: --slab-size %llu B (%s) > "
                         "backing %llu B (%s) on device[%zu] %s — no slab fits; "
                         "lower --slab-size or raise the budget\n",
                         static_cast<unsigned long long>(cfg.slabSizeBytes),
                         ff::bytesToHuman(cfg.slabSizeBytes).c_str(),
                         static_cast<unsigned long long>(b.backing),
                         ff::bytesToHuman(b.backing).c_str(), i, d.name);
            return 1;
        }
        budgets.push_back(b);
        std::fprintf(stderr,
                     "[ff_sieve] device[%zu] budget: f=%.3g freeUsed=%llu B (%s) "
                     "cap=%s budget=%llu B (%s) scratch=%llu B (%s) "
                     "headroom=%llu B (%s) backing=%llu B (%s) slabs=%llu\n",
                     i, b.fraction, static_cast<unsigned long long>(b.freeUsed),
                     ff::bytesToHuman(b.freeUsed).c_str(),
                     b.cap == std::numeric_limits<unsigned long long>::max()
                         ? "uncapped"
                         : ff::bytesToHuman(b.cap).c_str(),
                     static_cast<unsigned long long>(b.budget),
                     ff::bytesToHuman(b.budget).c_str(),
                     static_cast<unsigned long long>(b.scratch),
                     ff::bytesToHuman(b.scratch).c_str(),
                     static_cast<unsigned long long>(b.headroom),
                     ff::bytesToHuman(b.headroom).c_str(),
                     static_cast<unsigned long long>(b.backing),
                     ff::bytesToHuman(b.backing).c_str(),
                     static_cast<unsigned long long>(b.slabCount));
        aggregateBacking += b.backing;
    }

    // AGGREGATE sieve run-gate (Oracle round-3): sum(device backing) +
    // host-tier-cap >= mapBytes(leg); host-tier-cap defaults to 0. The
    // per-device predicate mapBytes+searchWorkspace <= budget remains the M4
    // SEARCH-participation gate (defined as searchWorkspaceBytes(), todo 14).
    unsigned long long capacity = aggregateBacking;
    if (cfg.hostTierCapBytes >
        std::numeric_limits<unsigned long long>::max() - capacity)
        capacity = std::numeric_limits<unsigned long long>::max();
    else
        capacity += cfg.hostTierCapBytes;

    std::fprintf(stderr,
                 "[ff_sieve] aggregate: deviceBacking=%llu B (%s) hostTier=%llu B "
                 "(%s) capacity=%llu B (%s) mapBytes=%llu B (%s) -> %s\n",
                 static_cast<unsigned long long>(aggregateBacking),
                 ff::bytesToHuman(aggregateBacking).c_str(),
                 static_cast<unsigned long long>(cfg.hostTierCapBytes),
                 ff::bytesToHuman(cfg.hostTierCapBytes).c_str(),
                 static_cast<unsigned long long>(capacity),
                 ff::bytesToHuman(capacity).c_str(),
                 static_cast<unsigned long long>(g.mapBytes),
                 ff::bytesToHuman(g.mapBytes).c_str(),
                 capacity >= g.mapBytes ? "GATE PASS" : "GATE FAIL");

    if (capacity < g.mapBytes) {
        std::fprintf(stderr,
                     "[ff_sieve] ERROR: no device fits leg %llu (map %llu B = %s): "
                     "aggregate capacity %llu B (%s) < map %llu B (%s)\n",
                     static_cast<unsigned long long>(g.sumLimit),
                     static_cast<unsigned long long>(g.mapBytes),
                     ff::bytesToHuman(g.mapBytes).c_str(),
                     static_cast<unsigned long long>(capacity),
                     ff::bytesToHuman(capacity).c_str(),
                     static_cast<unsigned long long>(g.mapBytes),
                     ff::bytesToHuman(g.mapBytes).c_str());
        std::fprintf(stderr,
                     "[ff_sieve] remediation: raise FF_VRAM_BUDGET and/or "
                     "--host-tier-cap\n");
        return 1;
    }

    std::fprintf(stderr,
                 "[ff_sieve] gate: PASS — no sieve in this stage; todo 7 adds "
                 "SieveSlabKernel\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    ff::Config cfg;
    if (ff::loadEnv(&cfg) != 0) return 1;
    std::vector<std::string> positionals;
    if (ff::parseArgs(argc, argv, &cfg, &positionals) != 0) return 1;
    if (ff::validateConfig(&cfg) != 0) return 1;
    if (cfg.listDevices) return runListDevices();
    if (positionals.empty()) return ff_smoke_main();   // no-args = todo-4 smoke
    return runLeg(cfg, positionals);
}
