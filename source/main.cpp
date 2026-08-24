// ff_sieve main (plan todo 3): startup device introspection + GPU_PLAN §4
// budget CLI. Flow: env -> CLI parse -> §4.3 validation -> --list-devices
// (recon) -> no-args = todo-4 dual-runtime smoke (make smoke regression) ->
// leg path: geometry -> enumerate (HIP + CUDA, deduped by PCI bus ID) ->
// per-device budgets (§4.2) -> AGGREGATE sieve run-gate -> SieveEngine::run
// -> GpuPrime -> RunIt Freudenthal search. STDOUT carries byte-exact headers
// and Freudenthal results; STDERR carries all budget/diagnostic/audit output.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "budget.h"
#include "config.h"
#include "device_info.h"
#include "device_registry.h"
#include "devabstraction.h"
#include "geometry.h"
#include "gpu_prime.h"
#include "sieve_engine.h"
#include "sieve_slab_engine.h"
#include "pull_scheduler.h"
#include "cpu_search.h"
#include "m4/gpu_search_launcher.h"
#include "m4/gpu_search_emission.cpp"
#include <chrono>
#include <iostream>
#include <unistd.h>   // sysconf: aggregate-gate auto host-tier resolution

extern "C" int ff_enum_hip_gfx1201(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_enum_hip_sm_120(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_smoke_main(void);   // todo-4 dual-runtime smoke (smoke/smoke_main.cpp)

namespace {

constexpr int kMaxDevices = 64;

struct RunDevices {
    std::vector<ff::DeviceInfo> devs;
    int skippedDuplicates = 0;
    bool hipFailed = false;
    bool nvFailed = false;
    bool hipSkipped = false;   // vendor-level pre-filter skipped the enum call
    bool nvSkipped = false;
};

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - start).count();
    return ms;
}

void dumpPhaseTimer(const char* phase, double ms) {
    std::fprintf(stderr, "ff_sieve timing: %s = %.3f ms\n", phase, ms);
}

// Task 14a filtered single-pass enumeration: consult ONLY the vendor-level
// filters (cfg.deviceFilter / disableVendor) BEFORE any enum call, so e.g.
// --devices=amd never initializes the NVIDIA runtime (~halves the startup
// floor). --sieve-device=N is an INDEX into enumeration results and is
// deliberately NOT consulted here — it resolves post-enumeration exactly as
// before. mergeAndDedupe (and its duplicate skip) stays for the unfiltered
// path; with one backend skipped there is simply nothing to dedupe against.
RunDevices enumerate(const ff::Config& cfg)
{
    RunDevices r;
    bool wantAmd = true, wantNv = true;
    if (!cfg.deviceFilter.empty()) {
        wantAmd = cfg.deviceFilter == "amd";
        wantNv = cfg.deviceFilter == "nvidia";
    }
    if (!cfg.disableVendor.empty()) {
        wantAmd = wantAmd && cfg.disableVendor != "amd";
        wantNv = wantNv && cfg.disableVendor != "nvidia";
    }
    ff::DeviceInfo hipBuf[kMaxDevices] = {};
    ff::DeviceInfo nvBuf[kMaxDevices] = {};
    int hipCount = 0, nvCount = 0;
    if (wantAmd) {
        if (ff_enum_hip_gfx1201(hipBuf, kMaxDevices, &hipCount) != 0) {
            std::fprintf(stderr,
                         "[ff_sieve] warning: HIP (AMD) enumeration failed; no AMD "
                         "devices will participate\n");
            r.hipFailed = true;
        }
    } else {
        r.hipSkipped = true;
        std::fprintf(stderr,
                     "[ff_sieve] device filter: skipping AMD-side enumeration "
                     "(vendor-level pre-filter)\n");
    }
    if (wantNv) {
        if (ff_enum_hip_sm_120(nvBuf, kMaxDevices, &nvCount) != 0) {
            std::fprintf(stderr,
                         "[ff_sieve] warning: HIP-NV (NVIDIA) enumeration failed; "
                         "no NVIDIA devices will participate\n");
            r.nvFailed = true;
        }
    } else {
        r.nvSkipped = true;
        std::fprintf(stderr,
                     "[ff_sieve] device filter: skipping NVIDIA-side enumeration "
                     "(vendor-level pre-filter)\n");
    }
    r.devs = ff::mergeAndDedupe(hipBuf, hipCount, nvBuf, nvCount,
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

int runListDevices(const ff::Config& cfg)
{
    auto t_list = std::chrono::steady_clock::now();
    RunDevices r = enumerate(cfg);
    double listMs = elapsedMs(t_list);
    dumpPhaseTimer("device enumeration (--list-devices)", listMs);
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

static uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t x = (uint64_t)std::sqrt((double)n);
    while ((x + 1) * (x + 1) <= n) ++x;
    while (x * x > n) --x;
    return x;
}

// Host-RAM-derived overflow-tier cap for the aggregate-gate auto-remediation
// (task 14b): same formula as validateConfig's "--host-tier-cap auto"
// resolution — physical RAM minus a 4 GiB reserve, 0 when RAM is smaller.
// Local copy because config.{h,cpp} are outside this task's file charter.
uint64_t hostTierAutoCapBytes() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz = sysconf(_SC_PAGE_SIZE);
    uint64_t ram = (pages > 0 && psz > 0) ? uint64_t(pages) * uint64_t(psz) : 0;
    const uint64_t kReserve = 4ull << 30;   // "auto" = host RAM - 4 GiB
    return ram > kReserve ? ram - kReserve : 0;
}

// FF_GPU_RESIDENCY=0 forces the legacy copy path (full-map H2D); absent or
// any other value keeps the residency handoff enabled.
bool residencyToggleOn() {
    const char* e = std::getenv("FF_GPU_RESIDENCY");
    return !(e != nullptr && std::strcmp(e, "0") == 0);
}

// Deterministic GPU-search device choice from budgets alone — no device calls,
// no output. The explicit --gpu-search-device=N when it fits, else the first
// device whose budget covers requiredBytes, else -1. Used both for pre-sieve
// residency planning and for the authoritative post-DevInit selection, so the
// two answers cannot diverge.
int selectSearchDevice(const ff::Config& cfg,
                       const std::vector<ff::DeviceInfo>& devs,
                       const std::vector<ff::DeviceBudget>& budgets,
                       unsigned long long requiredBytes) {
    if (cfg.gpuSearchDevice >= 0) {
        if (cfg.gpuSearchDevice >= static_cast<int>(devs.size())) return -1;
        if (budgets[cfg.gpuSearchDevice].budget < requiredBytes) return -1;
        return cfg.gpuSearchDevice;
    }
    for (int di = 0; di < static_cast<int>(devs.size()); ++di)
        if (budgets[di].budget >= requiredBytes) return di;
    return -1;
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

    auto t_enum = std::chrono::steady_clock::now();
    RunDevices r = enumerate(cfg);
    double enumMs = elapsedMs(t_enum);
    dumpPhaseTimer("device enumeration", enumMs);
    if (r.devs.empty()) {
        std::fprintf(stderr,
                     "[ff_sieve] error: no devices enumerated (HIP-AMD%s, "
                     "HIP-NV%s)\n",
                     r.hipFailed ? " failed"
                         : (r.hipSkipped ? " filtered-out" : " ok"),
                     r.nvFailed ? " failed"
                         : (r.nvSkipped ? " filtered-out" : " ok"));
        return 1;
    }
    if (r.skippedDuplicates > 0)
        std::fprintf(stderr, "[ff_sieve] note: %d bus-ID duplicate(s) skipped in "
                             "the device union\n", r.skippedDuplicates);

    // --devices <backend> restricts participation to one vendor; the leg-fit
    // gate below then evaluates against the filtered devices only.
    if (!cfg.deviceFilter.empty()) {
        std::vector<ff::DeviceInfo> kept;
        for (const ff::DeviceInfo& d : r.devs) {
            if (cfg.deviceFilter == d.vendor) kept.push_back(d);
        }
        std::fprintf(stderr, "[ff_sieve] device filter: --devices=%s kept %zu of "
                             "%zu logical device(s)\n",
                     cfg.deviceFilter.c_str(), kept.size(), r.devs.size());
        if (kept.empty()) {
            std::fprintf(stderr, "[ff_sieve] ERROR: no device matches "
                                 "--devices=%s (enumerated:",
                         cfg.deviceFilter.c_str());
            for (size_t i = 0; i < r.devs.size(); ++i)
                std::fprintf(stderr, " %s", r.devs[i].vendor);
            std::fprintf(stderr, ")\n");
            return 1;
        }
        r.devs = std::move(kept);
    }

    // --disable-vendor=<amd|nvidia>: removes a vendor's
    // devices before budgets/scheduling. An empty pool is a hard error —
    // never run an unscheduled fallback path silently.
    if (!cfg.disableVendor.empty()) {
        std::vector<ff::DeviceInfo> kept;
        for (const ff::DeviceInfo& d : r.devs) {
            if (cfg.disableVendor != d.vendor) kept.push_back(d);
        }
        std::fprintf(stderr,
                     "[ff_sieve] device filter: --disable-vendor=%s kept %zu of "
                     "%zu logical device(s)\n",
                     cfg.disableVendor.c_str(), kept.size(), r.devs.size());
        if (kept.empty()) {
            std::fprintf(stderr, "[ff_sieve] ERROR: --disable-vendor=%s leaves "
                                 "no devices (enumerated:",
                         cfg.disableVendor.c_str());
            for (size_t i = 0; i < r.devs.size(); ++i)
                std::fprintf(stderr, " %s", r.devs[i].vendor);
            std::fprintf(stderr, ")\n");
            return 1;
        }
        r.devs = std::move(kept);
    }

    // --dump-map <file> (todo 13): dump prime map after sieve, exit before search
    const std::string& dumpMapFile = cfg.dumpMapFile;

    // §4.2 per-device budgets.
    auto t_budget = std::chrono::steady_clock::now();
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

    dumpPhaseTimer("budget computation", elapsedMs(t_budget));

    // Print all available devices
    std::fprintf(stderr, "[ff_sieve] available devices:\n");
    for (size_t i = 0; i < r.devs.size(); ++i)
        printDeviceHeader(r.devs[i], static_cast<int>(i));

    // Save full device list; sieve may narrow it, search needs the original.
    auto allDevs = r.devs;
    auto allBudgets = budgets;
    if (cfg.sieveDevice >= 0) {
        if (cfg.sieveDevice >= static_cast<int>(r.devs.size())) {
            std::fprintf(stderr,
                         "[ff_sieve] ERROR: --sieve-device=%d but only %zu "
                         "device(s) available\n",
                         cfg.sieveDevice, r.devs.size());
            return 1;
        }
        std::fprintf(stderr, "[ff_sieve] sieve device: device[%d] %s "
                             "(--sieve-device=%d)\n",
                     cfg.sieveDevice, r.devs[cfg.sieveDevice].name,
                     cfg.sieveDevice);
        r.devs = {r.devs[cfg.sieveDevice]};
        budgets = {budgets[cfg.sieveDevice]};
    } else {
        std::fprintf(stderr, "[ff_sieve] sieve device: auto (all %zu devices)\n",
                     r.devs.size());
    }

    // AGGREGATE sieve run-gate (Oracle round-3): sum(device backing) +
    // host-tier-cap >= mapBytes(leg); host-tier-cap defaults to 0. The
    // per-device predicate mapBytes+searchWorkspace <= budget remains the M4
    // SEARCH-participation gate. Task 14b: when the gate fails and the user
    // gave NO explicit host-tier choice (--host-tier-cap / --no-host-tier),
    // the existing host overflow tier is auto-enabled ("--host-tier-cap
    // auto") with a stderr notice instead of refusing; an explicit choice is
    // always honored, including --no-host-tier's refusal.
    unsigned long long hostTierBytes = cfg.hostTierCapBytes;
    unsigned long long capacity = aggregateBacking;
    if (hostTierBytes >
        std::numeric_limits<unsigned long long>::max() - capacity)
        capacity = std::numeric_limits<unsigned long long>::max();
    else
        capacity += hostTierBytes;

    auto printAggregateLine = [&](unsigned long long ht,
                                  unsigned long long cap) {
        std::fprintf(stderr,
                     "[ff_sieve] aggregate: deviceBacking=%llu B (%s) "
                     "hostTier=%llu B (%s) capacity=%llu B (%s) mapBytes=%llu B "
                     "(%s) -> %s\n",
                     static_cast<unsigned long long>(aggregateBacking),
                     ff::bytesToHuman(aggregateBacking).c_str(),
                     static_cast<unsigned long long>(ht),
                     ff::bytesToHuman(ht).c_str(),
                     static_cast<unsigned long long>(cap),
                     ff::bytesToHuman(cap).c_str(),
                     static_cast<unsigned long long>(g.mapBytes),
                     ff::bytesToHuman(g.mapBytes).c_str(),
                     cap >= g.mapBytes ? "GATE PASS" : "GATE FAIL");
    };
    printAggregateLine(hostTierBytes, capacity);

    if (capacity < g.mapBytes) {
        const bool explicitTierChoice = cfg.hasHostTierCap || cfg.noHostTier;
        const unsigned long long autoCap =
            explicitTierChoice ? 0 : hostTierAutoCapBytes();
        unsigned long long remedied = aggregateBacking;
        if (autoCap >
            std::numeric_limits<unsigned long long>::max() - remedied)
            remedied = std::numeric_limits<unsigned long long>::max();
        else
            remedied += autoCap;
        if (remedied >= g.mapBytes) {
            std::fprintf(stderr,
                         "[ff_sieve] note: aggregate capacity %llu B (%s) < map "
                         "%llu B (%s); auto-enabling the host overflow tier "
                         "(--host-tier-cap auto => %llu B (%s)) to complete this "
                         "leg — pass --no-host-tier to refuse instead\n",
                         static_cast<unsigned long long>(capacity),
                         ff::bytesToHuman(capacity).c_str(),
                         static_cast<unsigned long long>(g.mapBytes),
                         ff::bytesToHuman(g.mapBytes).c_str(),
                         static_cast<unsigned long long>(autoCap),
                         ff::bytesToHuman(autoCap).c_str());
            hostTierBytes = autoCap;
            capacity = remedied;
            printAggregateLine(hostTierBytes, capacity);
        } else {
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
                         "[ff_sieve] remediation: raise --vram-budget and/or "
                         "--host-tier-cap\n");
            return 1;
        }
    }

    // Allocate host map
    uint64_t mapBytes = (g.maxPrimeMapValue + 1 + 15) >> 4;
    std::vector<uint8_t> hostMap(mapBytes, 0);

    // Residency planning, BEFORE the sieve: pick the GPU-search device now
    // (pure budget math via selectSearchDevice) and ask the scheduler to
    // leave the map resident on it. The scheduler re-validates feasibility
    // against its own budgets; an invalid report simply means legacy copies.
    ff::PullMapResidency residency;
    int plannedSearchDev = -1;
    if (cfg.gpuSearch && dumpMapFile.empty() && residencyToggleOn()) {
        const uint64_t oddSumsPlan = (g.sumLimit - g.sumStart) / 2 + 1;
        const unsigned long long reqBytesPlan =
            static_cast<unsigned long long>(mapBytes) +
            static_cast<unsigned long long>(oddSumsPlan * sizeof(GpuRecord)) +
            static_cast<unsigned long long>(
                ff::searchWorkspaceBytes(g.sumLimit));
        plannedSearchDev =
            selectSearchDevice(cfg, allDevs, allBudgets, reqBytesPlan);
    }
    int residencyDev = -1;   // scheduler-scope index (into r.devs)
    if (plannedSearchDev >= 0) {
        for (size_t i = 0; i < r.devs.size(); ++i) {
            if (r.devs[i].runtimeIndex ==
                    allDevs[plannedSearchDev].runtimeIndex &&
                std::string(r.devs[i].vendor) ==
                    std::string(allDevs[plannedSearchDev].vendor)) {
                residencyDev = static_cast<int>(i);
                break;
            }
        }
    }
    ff::PullMapResidency* residencyReq =
        (cfg.gpuSearch && residencyDev >= 0) ? &residency : nullptr;

    // M2 backing pool + weighted dynamic pulls (plan todo 10): device-
    // independent prep (geometry + host primes), then the pull scheduler
    // sieves the full map across ALL logical devices. SieveEngine's ctor
    // device/vendor are unused by prepare()/kernelPrimes() — keep them valid
    // for the CPU search's small-prime list below.
    auto t_sieve = std::chrono::steady_clock::now();
    SieveEngine engine(r.devs[0].runtimeIndex, r.devs[0].vendor);
    uint64_t maxPrimeMapValue = engine.prepare(g.sumLimit);
    if (maxPrimeMapValue == 0) {
        std::fprintf(stderr, "[ff_sieve] sieve prepare failed\n");
        return 1;
    }
    uint32_t kernelPrimeCount = 0;
    const uint32_t* kernelPrimes = engine.kernelPrimes(&kernelPrimeCount);
    uint64_t sieveUs = 0;
maxPrimeMapValue = ff::runPullScheduler(cfg, r.devs, budgets, g,
                                             kernelPrimes, kernelPrimeCount,
                                             hostMap.data(), &sieveUs,
                                             residencyDev, residencyReq);
    double sieveMs = elapsedMs(t_sieve);
    dumpPhaseTimer("sieve phase", sieveMs);
    if (maxPrimeMapValue == 0) {
        std::fprintf(stderr, "[ff_sieve] sieve failed\n");
        return 1;
    }

    // --dump-map: write prime map to file and exit before search
    if (!dumpMapFile.empty()) {
        FILE* f = std::fopen(dumpMapFile.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr,
                         "[ff_sieve] ERROR: cannot open %s for writing\n",
                         dumpMapFile.c_str());
            return 1;
        }
        std::fwrite(hostMap.data(), 1, mapBytes, f);
        std::fclose(f);
        std::fprintf(stderr,
                     "[ff_sieve] dump-map: wrote %s (%llu bytes)\n",
                     dumpMapFile.c_str(),
                     static_cast<unsigned long long>(mapBytes));
        return 0;
    }

    // Create GpuPrime
    GpuPrime prime(hostMap.data(), maxPrimeMapValue);

    // Compute header values (matching reference Prime constructor)
    uint64_t maxGeneratedPrime = g.maxGeneratedPrime;
    double xlogx = (double)maxGeneratedPrime / std::log((double)maxGeneratedPrime);
    uint32_t numPrimesRequested = (uint32_t)(xlogx + 1.2762 * xlogx / std::log((double)maxGeneratedPrime));
    uint64_t estimatedMaxPrime = (uint64_t)numPrimesRequested * ((uint64_t)(std::log((double)numPrimesRequested)) + (uint64_t)(std::log((double)std::log((double)numPrimesRequested))));
    if (maxPrimeMapValue < estimatedMaxPrime) maxPrimeMapValue = estimatedMaxPrime;
    uint64_t primeMapSize = (maxPrimeMapValue + 15) >> 4;
    uint64_t requestedPrimesSize = sizeof(uint32_t) * numPrimesRequested;
    uint64_t totalBytes = requestedPrimesSize + primeMapSize;
    uint64_t sqrtLimit = isqrt64(maxPrimeMapValue);

    // Print headers (stdout, byte-exact)
    std::cout << "maxPrimeMapValue " << maxPrimeMapValue
              << " numPrimesRequested " << numPrimesRequested
              << " totalBytes " << totalBytes << std::endl;
    std::cout << "maxPrimeMapValue " << maxPrimeMapValue
              << " numPrimesRequested " << numPrimesRequested
              << " sqrtLimit " << sqrtLimit << std::endl;

    // ---- GPU search path (budget-gated) ----
    auto t_search = std::chrono::steady_clock::now();
    {
        uint64_t numOddSums = (g.sumLimit - g.sumStart) / 2 + 1;
        uint64_t recordBytes = numOddSums * sizeof(GpuRecord);
        uint64_t searchWorkspace = ff::searchWorkspaceBytes(g.sumLimit);

        // GPU search path: when --gpu-search is requested, use the GPU Freudenthal
        // kernel instead of CPU (31 threads). Falls back to CPU if VRAM or
        // device allocation fails.
        bool useGpu = cfg.gpuSearch;
        int gpuDeviceIndex = -1;
        ffdev::DevHandle dPrimeMap, dRecords, dAtomic;

        if (useGpu) {
            // Task 14a: seed the abstraction from the ALREADY-ENUMERATED
            // filtered list (allDevs) — zero re-enumeration, where legacy
            // DevInit() re-enumerated BOTH backends here. Seeding verbatim
            // also keeps g_devices ordering identical to every logical index
            // used below (with a vendor filter the two lists used to
            // diverge). allDevs is non-empty: runLeg returned earlier
            // otherwise, and DevInitFromDevices refuses empty lists.
            if (ffdev::DevInitFromDevices(allDevs) == 0) {
                uint64_t requiredBytes =
                    mapBytes + recordBytes + searchWorkspace;

                // Restore full device list (sieve may have narrowed it).
                r.devs = allDevs;
                budgets = allBudgets;

                // Same deterministic choice the residency planner made; the
                // branches below only render its outcome.
                gpuDeviceIndex = selectSearchDevice(cfg, r.devs, budgets,
                                                    requiredBytes);
                if (cfg.gpuSearchDevice >= 0) {
                    if (cfg.gpuSearchDevice >= static_cast<int>(r.devs.size())) {
                        std::fprintf(stderr,
                            "[ff_sieve] GPU search: ERROR: --gpu-search-device=%d "
                            "but only %zu device(s) available\n",
                            cfg.gpuSearchDevice, r.devs.size());
                        useGpu = false;
                    } else if (gpuDeviceIndex < 0) {
                        std::fprintf(stderr,
                            "[ff_sieve] GPU search: device[%d] %s skipped — "
                            "need %llu B, budget %llu B\n",
                            cfg.gpuSearchDevice, r.devs[cfg.gpuSearchDevice].name,
                            static_cast<unsigned long long>(requiredBytes),
                            static_cast<unsigned long long>(budgets[cfg.gpuSearchDevice].budget));
                        useGpu = false;
                    } else {
                        gpuDeviceIndex = cfg.gpuSearchDevice;
                        std::fprintf(stderr,
                            "[ff_sieve] GPU search: device[%d] %s "
                            "(--gpu-search-device=%d)\n",
                            gpuDeviceIndex, r.devs[gpuDeviceIndex].name,
                            cfg.gpuSearchDevice);
                    }
                } else {
                    if (gpuDeviceIndex >= 0) {
                        std::fprintf(stderr,
                            "[ff_sieve] GPU search: device[%d] %s "
                            "(auto, budget=%llu B)\n",
                            gpuDeviceIndex, r.devs[gpuDeviceIndex].name,
                            static_cast<unsigned long long>(budgets[gpuDeviceIndex].budget));
                    } else {
                        std::fprintf(stderr,
                            "[ff_sieve] GPU search: no device fits "
                            "(need %llu B). Falling back to CPU.\n",
                            static_cast<unsigned long long>(requiredBytes));
                        useGpu = false;
                    }
                }
            } else {
                // Only reachable when GPU search was actually requested
                // (useGpu is false on the default CPU path, which skips this
                // whole block and prints nothing here).
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: DevInit failed, using CPU fallback\n");
                useGpu = false;
            }
        }

        // GPU search: allocate via the same per-arch HIP objects that launch
        // the kernel (GpuSearchAlloc dispatches by vendor), so the allocator
        // and the kernel launch share one runtime context per device.
        bool useResidentMap = false;
        bool primeMapOwned = false;
        if (useGpu) {
            // Residency handoff: when the scheduler left the map resident on
            // THIS device, borrow its contiguous buffer and skip both the
            // dPrimeMap allocation and the full-map H2D. Lifetime is safe:
            // scheduler teardown is deferred until releasePullScheduler()
            // below, after the internally synchronizing launch returns.
            useResidentMap = residency.valid &&
                             plannedSearchDev == gpuDeviceIndex;
            if (useResidentMap) {
                dPrimeMap.ptr = residency.devPtr;
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: residency handoff — reading the "
                    "sieve-resident map in place on device[%d] %s (%llu B, "
                    "no map H2D)\n",
                    gpuDeviceIndex, r.devs[gpuDeviceIndex].name,
                    static_cast<unsigned long long>(residency.mapBytes));
            } else if (GpuSearchAlloc(r.devs[gpuDeviceIndex].runtimeIndex,
                                      r.devs[gpuDeviceIndex].vendor,
                                      mapBytes, &dPrimeMap.ptr) != 0) {
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: device allocation failed, using CPU fallback\n");
                useGpu = false;
            } else {
                primeMapOwned = true;
            }
        }
        if (useGpu &&
            (GpuSearchAlloc(r.devs[gpuDeviceIndex].runtimeIndex,
                            r.devs[gpuDeviceIndex].vendor,
                            recordBytes, &dRecords.ptr) != 0 ||
             GpuSearchAlloc(r.devs[gpuDeviceIndex].runtimeIndex,
                            r.devs[gpuDeviceIndex].vendor,
                            sizeof(uint32_t), &dAtomic.ptr) != 0)) {
            std::fprintf(stderr,
                "[ff_sieve] GPU search: device allocation failed, using CPU fallback\n");
            useGpu = false;
        }

        // Get smallPrimes from engine (needed for GPU search; skip p=2 since kernel sieve already handles it).
        uint32_t smallPrimeCount = 0;
        const uint32_t* smallPrimesRaw = engine.getSmallPrimes(&smallPrimeCount);
        // GPU kernel skips p=2 (sieve already handles it); CPU AllButOne expects leading 2.
        const uint32_t* smallPrimesGpu = smallPrimesRaw;
        uint32_t smallPrimeCountGpu = smallPrimeCount;
        if (smallPrimeCountGpu > 1 && smallPrimesGpu[0] == 2) {
            smallPrimesGpu++;
            smallPrimeCountGpu--;
        }
        // Device handle for smallPrimes — zero-initialized, freed in cleanup below.
        ffdev::DevHandle dSmallPrimes;
        dSmallPrimes.ptr = nullptr;

        // Host buffers used by GPU path; declared here so they're visible in both if (useGpu) blocks.
        uint32_t hAtomicCount = 0;
        std::vector<GpuRecord> hRecords(numOddSums);

        // Outer scope: summed into the "search phase" total printed before
        // the GPU path's early return 0 below.
        double gpuH2dMs = 0.0;

        if (useGpu) {
            auto tH2d = std::chrono::steady_clock::now();
            int ri = r.devs[gpuDeviceIndex].runtimeIndex;
            const char* v = r.devs[gpuDeviceIndex].vendor;

            if (useResidentMap) {
                // Resident path: no map upload. Zero the record/atomic slots
                // device-side (same bytes the legacy host-filled upload
                // produces) and fill only slabs the sieve did NOT leave on
                // this device (foreign-owned or homeless).
                const SievePoolOps* pops = SievePoolGetForVendor(v, ri);
                if (pops != nullptr &&
                    pops->memset(ri, dRecords.ptr, 0, recordBytes) == 0 &&
                    pops->memset(ri, dAtomic.ptr, 0, sizeof(uint32_t)) == 0) {
                    hAtomicCount = 0;
                } else {
                    hAtomicCount = 0;
                    std::fill(hRecords.begin(), hRecords.end(), GpuRecord{});
                    GpuSearchCopyH2D(ri, v, dAtomic.ptr, &hAtomicCount,
                                     sizeof(uint32_t));
                    GpuSearchCopyH2D(ri, v, dRecords.ptr, hRecords.data(),
                                     recordBytes);
                }
                uint64_t fillSlabs = 0, fillBytes = 0;
                const uint64_t slabB = cfg.slabSizeBytes;
                for (uint64_t s = 0; s < residency.ownerOf.size(); ++s) {
                    if (residency.ownerOf[s] == residency.deviceIndex) continue;
                    const uint64_t off = s * slabB;
                    if (off >= mapBytes) break;
                    const uint64_t cb = std::min(slabB, mapBytes - off);
                    GpuSearchCopyH2D(ri, v, residency.devPtr + off,
                                     hostMap.data() + off, cb);
                    ++fillSlabs;
                    fillBytes += cb;
                }
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: residency fill on device[%d] %s: "
                    "%llu foreign/homeless slab(s), %llu B H2D\n",
                    gpuDeviceIndex, r.devs[gpuDeviceIndex].name,
                    static_cast<unsigned long long>(fillSlabs),
                    static_cast<unsigned long long>(fillBytes));
            } else {
                hAtomicCount = 0;
                GpuSearchCopyH2D(ri, v, dAtomic.ptr, &hAtomicCount, sizeof(uint32_t));

                std::fill(hRecords.begin(), hRecords.end(), GpuRecord{});
                GpuSearchCopyH2D(ri, v, dRecords.ptr, hRecords.data(), recordBytes);

                GpuSearchCopyH2D(ri, v, dPrimeMap.ptr, hostMap.data(), mapBytes);
            }

            if (GpuSearchAlloc(ri, v, smallPrimeCountGpu * sizeof(uint32_t), &dSmallPrimes.ptr) != 0) {
                std::fprintf(stderr, "[ff_sieve] GPU search: smallPrimes allocation failed\n");
                useGpu = false;
            } else {
                GpuSearchCopyH2D(ri, v, dSmallPrimes.ptr, smallPrimesGpu,
                                 smallPrimeCountGpu * sizeof(uint32_t));
            }
            gpuH2dMs = elapsedMs(tH2d);
            dumpPhaseTimer("search H2D copies", gpuH2dMs);
        }

        if (useGpu) {
            // Launch GPU search.  Dispatch by vendor: both runtimes number
            // their devices from 0, so an index-only launcher would pick the
            // wrong arch (page fault when the selected device != allocator).
            auto tKernel = std::chrono::steady_clock::now();
            int launchRet = GpuSearchLaunch(
                r.devs[gpuDeviceIndex].runtimeIndex,
                r.devs[gpuDeviceIndex].vendor,
                static_cast<const uint8_t*>(dPrimeMap.ptr),
                maxPrimeMapValue,
                g.sumStart, g.sumLimit,
                static_cast<GpuRecord*>(dRecords.ptr),
                static_cast<uint32_t*>(dAtomic.ptr),
                static_cast<const uint32_t*>(dSmallPrimes.ptr),
                smallPrimeCountGpu);
            double kernelMs = elapsedMs(tKernel);

            if (launchRet == 0) {
                dumpPhaseTimer("search kernel", kernelMs);
                int ri = r.devs[gpuDeviceIndex].runtimeIndex;
                const char* v = r.devs[gpuDeviceIndex].vendor;

                auto tD2h = std::chrono::steady_clock::now();
                GpuSearchCopyD2H(ri, v, &hAtomicCount, dAtomic.ptr, sizeof(uint32_t));

                GpuSearchCopyD2H(ri, v, hRecords.data(), dRecords.ptr, recordBytes);
                double d2hMs = elapsedMs(tD2h);
                dumpPhaseTimer("search D2H copies", d2hMs);

                // The launch synchronized internally, so the resident map has
                // been fully consumed — end the deferred scheduler lifetime.
                if (residency.valid) ff::releasePullScheduler(&residency);

                std::cout.flush();
                auto t1 = std::chrono::high_resolution_clock::now();
                GpuSearchEmit(prime, hRecords.data(),
                              static_cast<uint32_t>(numOddSums));
                auto t2 = std::chrono::high_resolution_clock::now();
                double emitMs =
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                dumpPhaseTimer("search emit", emitMs);

                GpuSearchFree(gpuDeviceIndex, r.devs[gpuDeviceIndex].vendor, dSmallPrimes.ptr);
                GpuSearchFree(gpuDeviceIndex, r.devs[gpuDeviceIndex].vendor, dRecords.ptr);
                GpuSearchFree(gpuDeviceIndex, r.devs[gpuDeviceIndex].vendor, dAtomic.ptr);
                if (primeMapOwned)
                    GpuSearchFree(gpuDeviceIndex, r.devs[gpuDeviceIndex].vendor, dPrimeMap.ptr);

                // Print timing (stdout, byte-exact)
                uint64_t searchUs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                std::cout << "Prime time: " << sieveUs << " \u03bcs" << std::endl;
                std::cout << "Freudenthal time: " << searchUs << " \u03bcs" << std::endl;

                // The GPU path returns here, skipping the CPU-path
                // dumpPhaseTimer("search phase", ...) tail; report the total
                // as the sum of the per-stage timers.
                dumpPhaseTimer("search phase",
                               gpuH2dMs + kernelMs + d2hMs + emitMs);

                return 0;
            } else {
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: launch failed (ret=%d), using CPU fallback\n",
                    launchRet);
                useGpu = false;
            }
        }

        // Free device memory on any failure path (gpuDeviceIndex >= 0
        // whenever any of these pointers was allocated).
        const char* fv = gpuDeviceIndex >= 0 ? r.devs[gpuDeviceIndex].vendor : nullptr;
        if (dSmallPrimes.ptr) GpuSearchFree(gpuDeviceIndex, fv, dSmallPrimes.ptr);
        if (dPrimeMap.ptr && primeMapOwned) GpuSearchFree(gpuDeviceIndex, fv, dPrimeMap.ptr);
        if (dRecords.ptr)     GpuSearchFree(gpuDeviceIndex, fv, dRecords.ptr);
        if (dAtomic.ptr)      GpuSearchFree(gpuDeviceIndex, fv, dAtomic.ptr);
        // Any non-success flow (DevInit, alloc, launch failures) still owes
        // the scheduler its deferred teardown.
        if (residency.valid) ff::releasePullScheduler(&residency);

        // CPU fallback (reuses smallPrimesRaw/smallPrimeCount from outer scope).
        auto t1 = std::chrono::high_resolution_clock::now();
        int count = 0;
        RunIt(prime, g.sumStart, g.sumLimit, g.productLimit, smallPrimesRaw, smallPrimeCount, cfg.threads, count);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Print timing (stdout, byte-exact) - μ is UTF-8 micro sign.
        // sieveUs is the M2 pull scheduler's wall time (thread spawn through
        // assembly); verify.sh normalizes any digit run to N.
        uint64_t searchUs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "Prime time: " << sieveUs << " \u03bcs" << std::endl;
        std::cout << "Freudenthal time: " << searchUs << " \u03bcs" << std::endl;
    }
    double searchMs = elapsedMs(t_search);
    dumpPhaseTimer("search phase", searchMs);

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    auto t0 = std::chrono::steady_clock::now();
    int ret = 0;
    {
        ff::Config cfg;
        if (ff::loadPullWeights(&cfg) != 0) { ret = 1; goto done; }
        std::vector<std::string> positionals;
        if (ff::parseArgs(argc, argv, &cfg, &positionals) != 0) { ret = 1; goto done; }
        if (ff::validateConfig(&cfg) != 0) { ret = 1; goto done; }
        if (cfg.hasHostTierCap && cfg.noHostTier) {
            std::fprintf(stderr,
                         "[ff_sieve] validation error: --host-tier-cap and "
                         "--no-host-tier are contradictory; give at most one "
                         "of them (omit both for the default: tier disabled, "
                         "auto-enabled only when the aggregate gate needs "
                         "it)\n");
            ret = 1; goto done;
        }
        if (cfg.listDevices) { ret = runListDevices(cfg); goto done; }
        if (positionals.empty()) { ret = ff_smoke_main(); goto done; }
        ret = runLeg(cfg, positionals);
    }
done:
    {
        double totalMs = elapsedMs(t0);
        std::fprintf(stderr, "ff_sieve timing: total = %.3f ms\n", totalMs);
    }
    return ret;
}
