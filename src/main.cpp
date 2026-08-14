// ff_sieve main (plan todo 3): startup device introspection + GPU_PLAN §4
// budget CLI. Flow: env -> CLI parse -> §4.3 validation -> --list-devices
// (recon) -> no-args = todo-4 dual-runtime smoke (make smoke regression) ->
// leg path: geometry -> enumerate (HIP + CUDA, deduped by PCI bus ID) ->
// per-device budgets (§4.2) -> AGGREGATE sieve run-gate -> SieveEngine::run
// -> GpuPrime -> RunIt Freudenthal search. STDOUT carries byte-exact headers
// and Freudenthal results; STDERR carries all budget/diagnostic/audit output.

#include <cstdio>
#include <cstdlib>
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
#include "pull_scheduler.h"
#include "cpu_search.h"
#include "m4/gpu_search_launcher.h"
#include "m4/gpu_search_emission.cpp"
#include <chrono>
#include <iostream>

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

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - start).count();
    return ms;
}

void dumpPhaseTimer(const char* phase, double ms) {
    std::fprintf(stderr, "ff_sieve timing: %s = %.3f ms\n", phase, ms);
}

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
    auto t_list = std::chrono::steady_clock::now();
    RunDevices r = enumerate();
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
    RunDevices r = enumerate();
    double enumMs = elapsedMs(t_enum);
    dumpPhaseTimer("device enumeration", enumMs);
    if (r.devs.empty()) {
        std::fprintf(stderr,
                     "[ff_sieve] error: no devices enumerated (HIP%s, CUDA%s)\n",
                     r.hipFailed ? " failed" : " ok", r.cudaFailed ? " failed" : " ok");
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

    // FF_DISABLE_DEVICE=<amd|nvidia> (plan todo 10): removes a vendor's
    // devices before budgets/scheduling (case-insensitive, enforced in
    // loadEnv). An empty pool is a hard error — never run an unscheduled
    // fallback path silently.
    if (!cfg.disableVendors.empty()) {
        std::vector<ff::DeviceInfo> kept;
        for (const ff::DeviceInfo& d : r.devs) {
            if (cfg.disableVendors != d.vendor) kept.push_back(d);
        }
        std::fprintf(stderr,
                     "[ff_sieve] device filter: FF_DISABLE_DEVICE=%s kept %zu of "
                     "%zu logical device(s)\n",
                     cfg.disableVendors.c_str(), kept.size(), r.devs.size());
        if (kept.empty()) {
            std::fprintf(stderr, "[ff_sieve] ERROR: FF_DISABLE_DEVICE=%s leaves "
                                 "no devices (enumerated:",
                         cfg.disableVendors.c_str());
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

    // Allocate host map
    uint64_t mapBytes = (g.maxPrimeMapValue + 1 + 15) >> 4;
    std::vector<uint8_t> hostMap(mapBytes, 0);

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
                                             hostMap.data(), &sieveUs);
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

        // GPU only activated with --gpu-search and not suppressed by --no-gpu.
        // Default path (no flags) uses CPU search.
        bool useGpu = (!cfg.noGpu && cfg.gpuSearch);
        int gpuDeviceIndex = -1;
        ffdev::DevHandle dPrimeMap, dRecords, dAtomic;

        if (useGpu && ffdev::DevInit() == 0) {
            for (int di = 0; di < (int)r.devs.size(); ++di) {
                if (r.devs[di].freeBytes >= mapBytes + recordBytes) {
                    gpuDeviceIndex = di;
                    std::fprintf(stderr,
                        "[ff_sieve] GPU search: device[%d] %s — "
                        "budget OK: map %llu B + records %llu B <= free %llu B "
                        "(host workspace: %llu B)\n",
                        di, r.devs[di].name,
                        static_cast<unsigned long long>(mapBytes),
                        static_cast<unsigned long long>(recordBytes),
                        static_cast<unsigned long long>(r.devs[di].freeBytes),
                        static_cast<unsigned long long>(searchWorkspace));
                    break;
                }
            }
            if (!useGpu) {
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: no device has enough free VRAM "
                    "for map(%llu B) + records(%llu B) = %llu B\n",
                    static_cast<unsigned long long>(mapBytes),
                    static_cast<unsigned long long>(recordBytes),
                    static_cast<unsigned long long>(mapBytes + recordBytes));
            }
        } else {
            std::fprintf(stderr,
                "[ff_sieve] GPU search: DevInit failed, using CPU fallback\n");
        }

        if (useGpu) {
            if (ffdev::DevAlloc(gpuDeviceIndex, mapBytes, &dPrimeMap) != 0 ||
                ffdev::DevAlloc(gpuDeviceIndex, recordBytes, &dRecords) != 0 ||
                ffdev::DevAlloc(gpuDeviceIndex, sizeof(uint32_t), &dAtomic) != 0) {
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: device allocation failed, using CPU fallback\n");
                useGpu = false;
            }
        }

        if (useGpu) {
            // Zero-init atomic counter
            uint32_t hAtomicCount = 0;
            ffdev::DevCopy(&dAtomic, &hAtomicCount, sizeof(uint32_t),
                           ffdev::DevCopyDir::H2D);

            // Copy host prime map to device
            ffdev::DevCopy(&dPrimeMap, hostMap.data(), mapBytes,
                           ffdev::DevCopyDir::H2D);

            // Launch GPU search
            int launchRet = GpuSearchLaunch(
                r.devs[gpuDeviceIndex].runtimeIndex,
                static_cast<const uint8_t*>(dPrimeMap.ptr),
                maxPrimeMapValue,
                g.sumStart, g.sumLimit,
                static_cast<GpuRecord*>(dRecords.ptr),
                static_cast<uint32_t*>(dAtomic.ptr));

            if (launchRet == 0) {
                // Copy atomic count back
                ffdev::DevCopy(&dAtomic, &hAtomicCount, sizeof(uint32_t),
                               ffdev::DevCopyDir::D2H);

                // Copy records back
                std::vector<GpuRecord> hRecords(hAtomicCount);
                if (hAtomicCount > 0) {
                    ffdev::DevCopy(&dRecords, hRecords.data(),
                                   hAtomicCount * sizeof(GpuRecord),
                                   ffdev::DevCopyDir::D2H);
                }

                // Format output (stdout, byte-exact)
                std::cout.flush();
                auto t1 = std::chrono::high_resolution_clock::now();
                GpuSearchEmit(prime, hRecords.data(), hAtomicCount);
                auto t2 = std::chrono::high_resolution_clock::now();

                // Free device memory
                ffdev::DevFree(&dRecords);
                ffdev::DevFree(&dAtomic);
                ffdev::DevFree(&dPrimeMap);

                // Print timing (stdout, byte-exact)
                uint64_t searchUs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                std::cout << "Prime time: " << sieveUs << " \u03bcs" << std::endl;
                std::cout << "Freudenthal time: " << searchUs << " \u03bcs" << std::endl;

                return 0;
            } else {
                std::fprintf(stderr,
                    "[ff_sieve] GPU search: launch failed (ret=%d), using CPU fallback\n",
                    launchRet);
                useGpu = false;
            }
        }

        // Free device memory on any failure path
        if (dPrimeMap.ptr) ffdev::DevFree(&dPrimeMap);
        if (dRecords.ptr)  ffdev::DevFree(&dRecords);
        if (dAtomic.ptr)   ffdev::DevFree(&dAtomic);

        // CPU fallback
        auto t1 = std::chrono::high_resolution_clock::now();
        int count = 0;
        uint32_t smallPrimeCount = 0;
        const uint32_t* smallPrimes = engine.getSmallPrimes(&smallPrimeCount);
        RunIt(prime, g.sumStart, g.sumLimit, g.productLimit, smallPrimes, smallPrimeCount, count);
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
        if (ff::loadEnv(&cfg) != 0) { ret = 1; goto done; }
        if (ff::loadPullWeights(&cfg) != 0) { ret = 1; goto done; }
        std::vector<std::string> positionals;
        if (ff::parseArgs(argc, argv, &cfg, &positionals) != 0) { ret = 1; goto done; }
        if (ff::validateConfig(&cfg) != 0) { ret = 1; goto done; }
        if (cfg.listDevices) { ret = runListDevices(); goto done; }
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
