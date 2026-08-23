// M2 backing pool + weighted dynamic pulls (plan todo 10) implementation.
//
// Pure g++ TU: talks to the per-arch pool ops (sieve_slab_engine.cpp, compiled
// with hipcc) through the extern "C" SievePoolGet_* getters, never through
// vendor headers. One owner thread per logical device; a global atomic work
// counter feeds slabs to the threads; per-device caps (largest-remainder
// weight share of the slab count) bias the pull rate without oversubscribing.

#include "pull_scheduler.h"

#include "sieve_slab_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

const SievePoolOps* SievePoolGetForVendor(const char* vendor, int vendorIndex)
{
    if (vendor != nullptr && std::strcmp(vendor, "nvidia") == 0)
        return SievePoolGet_sm_120(vendorIndex);
    return SievePoolGet_gfx1201(vendorIndex);
}

namespace ff {
namespace {

constexpr int kMaxDevices = 64;

struct PerDevice {
    const DeviceInfo* dev = nullptr;
    const DeviceBudget* budget = nullptr;
    const SievePoolOps* ops = nullptr;
    double weight = 1.0;
    uint64_t homeBase = 0;            // first slab index in this device's region
    uint64_t homeCount = 0;           // number of home slabs
    uint64_t cap = 0;                 // max slabs this device may pull
    std::vector<void*> region;        // one device buffer per home slab
    void* stagingDev = nullptr;       // 1-slab device scratch (non-home pulls)
    std::vector<uint8_t> stagingHost; // 1-slab host landing buffer
    uint64_t pulled = 0;
    uint64_t staged = 0;
    // M3 overlap engine (todo 12)
    void* computeStream = nullptr;
    void* copyStream = nullptr;
    void* evEndA = nullptr;
    void* evEndB = nullptr;
    double overlapComputeMs = 0;
    double overlapCopyMs = 0;
    double overlapTotalMs = 0;
};

struct Shared {
    uint64_t numSlabs = 0;
    uint64_t slabBytes = 0;
    uint64_t totalMapBytes = 0;
    const uint32_t* kernelPrimes = nullptr;
    uint32_t kernelPrimeCount = 0;
    uint8_t* hostMap = nullptr;
    std::atomic<uint64_t> nextSlab{0};
    std::atomic<uint64_t> slots[kMaxDevices];
    std::atomic<int> failed{0};
    std::vector<int> ownerOf;         // per slab: device index or -1 (homeless)
    std::vector<PerDevice> devs;
};

double weightFor(const Config& cfg, const DeviceInfo& dev)
{
    auto it = cfg.pullWeights.find(std::string(dev.vendor));
    if (it == cfg.pullWeights.end()) return 1.0;
    return it->second;
}

// Backing-proportional largest-remainder home assignment, clamped so every
// device keeps one slab of backing free for its staging scratch. Slabs beyond
// the aggregate home capacity are homeless (staged straight into hostMap).
void assignHome(Shared& sh)
{
    const size_t n = sh.devs.size();
    std::vector<uint64_t> maxHome(n);
    double totalBacking = 0.0;
    for (size_t i = 0; i < n; ++i) {
        maxHome[i] = sh.devs[i].budget->slabCount;
        totalBacking += static_cast<double>(sh.devs[i].budget->backing);
    }
    std::vector<uint64_t> home(n, 0);
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double share = sh.numSlabs * static_cast<double>(sh.devs[i].budget->backing) / totalBacking;
        uint64_t capHome = maxHome[i] > 1 ? maxHome[i] - 1 : 0;
        home[i] = std::min(static_cast<uint64_t>(share), capHome);
        sum += home[i];
    }
    while (sum < sh.numSlabs) {
        size_t best = n;
        double bestRem = -1.0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t capHome = maxHome[i] > 1 ? maxHome[i] - 1 : 0;
            if (home[i] >= capHome) continue;
            double share = sh.numSlabs * static_cast<double>(sh.devs[i].budget->backing) / totalBacking;
            double rem = share - static_cast<double>(home[i]);
            if (rem > bestRem) {
                bestRem = rem;
                best = i;
            }
        }
        if (best == n) break;   // no device has room; remaining slabs are homeless
        ++home[best];
        ++sum;
    }
    uint64_t base = 0;
    for (size_t i = 0; i < n; ++i) {
        sh.devs[i].homeBase = base;
        sh.devs[i].homeCount = home[i];
        base += home[i];
    }
}

// Weight-proportional largest-remainder pull caps. The floors always sum to
// <= numSlabs and the remainder loop tops the total back up to exactly
// numSlabs, so the caps never oversubscribe and the pull loop always drains
// the whole slab counter.
void assignCaps(Shared& sh)
{
    const size_t n = sh.devs.size();
    double totalWeight = 0.0;
    for (size_t i = 0; i < n; ++i) totalWeight += sh.devs[i].weight;
    if (totalWeight <= 0.0) totalWeight = 1.0;
    std::vector<uint64_t> cap(n, 0);
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cap[i] = static_cast<uint64_t>(sh.numSlabs * (sh.devs[i].weight / totalWeight));
        sum += cap[i];
    }
    while (sum < sh.numSlabs) {
        size_t best = n;
        double bestRem = -1.0;
        for (size_t i = 0; i < n; ++i) {
            double rem = sh.numSlabs * (sh.devs[i].weight / totalWeight) - static_cast<double>(cap[i]);
            if (rem > bestRem) {
                bestRem = rem;
                best = i;
            }
        }
        if (best == n) break;
        ++cap[best];
        ++sum;
    }
    for (size_t i = 0; i < n; ++i) sh.devs[i].cap = cap[i];
}

void deviceWorker(Shared& sh, size_t i)
{
    PerDevice& d = sh.devs[i];
    const int ri = d.dev->runtimeIndex;
    while (true) {
        uint64_t slot = sh.slots[i].fetch_add(1, std::memory_order_relaxed);
        if (slot >= d.cap) break;
        uint64_t s = sh.nextSlab.fetch_add(1, std::memory_order_relaxed);
        if (s >= sh.numSlabs) break;
        ++d.pulled;
        const uint64_t off = s * sh.slabBytes;
        const uint64_t copyBytes = std::min(sh.slabBytes, sh.totalMapBytes - off);
        const uint64_t segLo = s * sh.slabBytes * 16;
        const uint64_t segHi = std::min(segLo + sh.slabBytes * 16, sh.totalMapBytes * 16);
        const int owner = sh.ownerOf[s];
        int rc = 0;
        if (owner == static_cast<int>(i)) {
            void* buf = d.region[s - d.homeBase];
            // M3: async compute on computeStream with double-buffer events
            rc = d.ops->slabComputeAsync(ri, sh.kernelPrimes, sh.kernelPrimeCount,
                                          segLo, segHi, buf, sh.slabBytes,
                                          d.computeStream, nullptr);
            // Record end of compute for overlap tracking
            d.ops->recordEvent(ri, d.evEndA, d.computeStream);
            // Async copy back to hostMap on copyStream
            rc = d.ops->copyD2HAsync(ri, sh.hostMap + off, buf, copyBytes, d.copyStream);
            // Copy stream waits for compute to finish
            d.ops->waitEvent(ri, d.evEndA, d.copyStream);
        } else {
            ++d.staged;
            rc = d.ops->slabCompute(ri, sh.kernelPrimes, sh.kernelPrimeCount,
                                    segLo, segHi, d.stagingDev, sh.slabBytes, 1);
            if (rc == 0) {
                if (owner >= 0) {
                    const size_t oi = static_cast<size_t>(owner);
                    const PerDevice& od = sh.devs[oi];
                    void* obuf = od.region[s - od.homeBase];
                    rc = d.ops->copyD2H(ri, d.stagingHost.data(), d.stagingDev,
                                        copyBytes);
                    if (rc == 0)
                        rc = od.ops->copyH2D(od.dev->runtimeIndex, obuf,
                                             d.stagingHost.data(), copyBytes);
                } else {
                    rc = d.ops->copyD2H(ri, sh.hostMap + off, d.stagingDev,
                                        copyBytes);
                }
            }
        }
        if (rc != 0) {
            sh.failed.store(1, std::memory_order_relaxed);
            return;
        }
    }
}

void assemble(Shared& sh)
{
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        const int ri = d.dev->runtimeIndex;
        for (uint64_t k = 0; k < d.homeCount; ++k) {
            const uint64_t s = d.homeBase + k;
            const uint64_t off = s * sh.slabBytes;
            const uint64_t copyBytes = std::min(sh.slabBytes, sh.totalMapBytes - off);
            if (d.ops->copyD2H(ri, sh.hostMap + off, d.region[k], copyBytes) != 0) {
                sh.failed.store(1, std::memory_order_relaxed);
                return;
            }
        }
    }
}

void teardown(Shared& sh)
{
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        const int ri = d.dev->runtimeIndex;
        for (void* buf : d.region) (void)d.ops->free(ri, buf);
        d.region.clear();
        if (d.stagingDev) {
            (void)d.ops->free(ri, d.stagingDev);
            d.stagingDev = nullptr;
        }
        // M3 overlap: free streams and events
        if (d.computeStream) {
            (void)d.ops->destroyStream(d.dev->runtimeIndex, d.computeStream);
            d.computeStream = nullptr;
        }
        if (d.copyStream) {
            (void)d.ops->destroyStream(d.dev->runtimeIndex, d.copyStream);
            d.copyStream = nullptr;
        }
        if (d.evEndA) {
            (void)d.ops->destroyEvent(d.dev->runtimeIndex, d.evEndA);
            d.evEndA = nullptr;
        }
        if (d.evEndB) {
            (void)d.ops->destroyEvent(d.dev->runtimeIndex, d.evEndB);
            d.evEndB = nullptr;
        }
    }
}

void printStats(Shared& sh)
{
    double minW = 0.0, maxW = 0.0;
    std::string dist;
    unsigned long long total = 0;
    double obsMin = 0.0, obsMax = 0.0;
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        if (minW == 0.0 || d.weight < minW) minW = d.weight;
        if (d.weight > maxW) maxW = d.weight;
        total += d.pulled;
        if (!dist.empty()) dist += " ";
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s=%llu", d.dev->vendor,
                      static_cast<unsigned long long>(d.pulled));
        dist += buf;
        if (d.pulled > 0) {
            if (obsMin == 0.0 || static_cast<double>(d.pulled) < obsMin)
                obsMin = static_cast<double>(d.pulled);
            if (static_cast<double>(d.pulled) > obsMax)
                obsMax = static_cast<double>(d.pulled);
        }
        std::fprintf(stderr,
                     "[ff_sieve] pull-count device[%zu]=%llu vendor=%s home=%llu "
                     "staged=%llu\n",
                     i, static_cast<unsigned long long>(d.pulled), d.dev->vendor,
                     static_cast<unsigned long long>(d.homeCount),
                     static_cast<unsigned long long>(d.staged));
    }
    const double expected = minW > 0.0 ? maxW / minW : 1.0;
    const double observed = obsMin > 0.0 ? obsMax / obsMin : 1.0;
    std::fprintf(stderr, "[ff_sieve] pull weights: ");
    for (size_t i = 0; i < sh.devs.size(); ++i)
        std::fprintf(stderr, "%s=%.3g ", sh.devs[i].dev->vendor, sh.devs[i].weight);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr,
                 "[ff_sieve] pull distribution: %s total=%llu ratio=%.3f "
                 "expected=%.3f\n",
                 dist.c_str(), total, observed, expected);
}

}  // namespace

uint64_t runPullScheduler(const Config& cfg,
                          const std::vector<DeviceInfo>& devs,
                          const std::vector<DeviceBudget>& budgets,
                          const LegGeometry& g,
                          const uint32_t* kernelPrimes,
                          uint32_t kernelPrimeCount,
                          uint8_t* hostMap,
                          uint64_t* wallUs)
{
    if (devs.size() > static_cast<size_t>(kMaxDevices) ||
        devs.size() != budgets.size()) {
        std::fprintf(stderr,
                     "[ff_sieve] error: scheduler needs 1..%d index-aligned "
                     "devices/budgets (got %zu/%zu)\n",
                     kMaxDevices, devs.size(), budgets.size());
        return 0;
    }

    Shared sh;
    sh.slabBytes = cfg.slabSizeBytes;
    sh.totalMapBytes = g.mapBytes;
    sh.numSlabs = (sh.totalMapBytes + sh.slabBytes - 1) / sh.slabBytes;
    if (sh.numSlabs == 0) sh.numSlabs = 1;
    sh.kernelPrimes = kernelPrimes;
    sh.kernelPrimeCount = kernelPrimeCount;
    sh.hostMap = hostMap;

    sh.devs.reserve(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        sh.slots[i].store(0);
        PerDevice pd;
        pd.dev = &devs[i];
        pd.budget = &budgets[i];
        pd.weight = weightFor(cfg, devs[i]);
        const SievePoolOps* ops = SievePoolGetForVendor(devs[i].vendor,
                                                        devs[i].runtimeIndex);
        if (!ops) {
            std::fprintf(stderr,
                         "[ff_sieve] error: no pool ops for device[%zu] %s "
                         "(vendor=%s runtimeIndex=%d)\n",
                         i, devs[i].name, devs[i].vendor, devs[i].runtimeIndex);
            return 0;
        }
        pd.ops = ops;
        sh.devs.push_back(std::move(pd));
    }

    assignHome(sh);
    assignCaps(sh);

    sh.ownerOf.assign(sh.numSlabs, -1);
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        for (uint64_t k = 0; k < sh.devs[i].homeCount; ++k)
            sh.ownerOf[sh.devs[i].homeBase + k] = static_cast<int>(i);
    }

    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        const int ri = d.dev->runtimeIndex;
        d.region.reserve(d.homeCount);
        for (uint64_t k = 0; k < d.homeCount; ++k) {
            void* buf = d.ops->alloc(ri, sh.slabBytes);
            if (!buf) {
                std::fprintf(stderr,
                             "[ff_sieve] error: backing alloc failed on "
                             "device[%zu] %s (home slab %llu of %llu)\n",
                             i, d.dev->name, static_cast<unsigned long long>(k),
                             static_cast<unsigned long long>(d.homeCount));
                teardown(sh);
                return 0;
            }
            d.region.push_back(buf);
            if (d.ops->memset(ri, buf, 0xff, sh.slabBytes) != 0) {
                teardown(sh);
                return 0;
            }
            if (d.homeBase + k == 0) {
                const uint8_t notOne = 0x7f;
                if (d.ops->copyH2D(ri, buf, &notOne, 1) != 0) {
                    teardown(sh);
                    return 0;
                }
            }
        }
        if (d.cap > 0) {
            d.stagingDev = d.ops->alloc(ri, sh.slabBytes);
            if (!d.stagingDev) {
                std::fprintf(stderr,
                             "[ff_sieve] error: staging alloc failed on "
                             "device[%zu] %s\n",
                             i, d.dev->name);
                teardown(sh);
                return 0;
            }
            d.stagingHost.resize(sh.slabBytes);
        }
    }
    // M3 overlap: create per-device streams and double-buffer events
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        d.computeStream = d.ops->createStream(d.dev->runtimeIndex);
        d.copyStream = d.ops->createStream(d.dev->runtimeIndex);
        d.evEndA = d.ops->createEvent(d.dev->runtimeIndex);
        d.evEndB = d.ops->createEvent(d.dev->runtimeIndex);
    }

    std::fprintf(stderr, "[ff_sieve] pull scheduler: %llu slab(s) of %llu B "
                         "across %zu device(s), home=[",
                 static_cast<unsigned long long>(sh.numSlabs),
                 static_cast<unsigned long long>(sh.slabBytes), sh.devs.size());
    for (size_t i = 0; i < sh.devs.size(); ++i)
        std::fprintf(stderr, "%s%llu",
                     i > 0 ? " " : "",
                     static_cast<unsigned long long>(sh.devs[i].homeCount));
    std::fprintf(stderr, "] caps=[");
    for (size_t i = 0; i < sh.devs.size(); ++i)
        std::fprintf(stderr, "%s%llu",
                     i > 0 ? " " : "",
                     static_cast<unsigned long long>(sh.devs[i].cap));
    std::fprintf(stderr, "]\n");

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(sh.devs.size());
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        if (sh.devs[i].cap > 0)
            threads.emplace_back(deviceWorker, std::ref(sh), i);
    }
    for (auto& th : threads) th.join();
    if (sh.failed.load() == 0) assemble(sh);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    if (wallUs) *wallUs = us;
    if (!sh.devs.empty()) sh.devs[0].overlapTotalMs = us / 1000.0;
    std::fprintf(stderr, "[ff_sieve] pull scheduler wall time: %llu us\n",
                 static_cast<unsigned long long>(us));

    // M3 overlap stats
    if (sh.devs.size() > 0) {
        std::fprintf(stderr,
                     "[ff_sieve] overlap engine: wall=%.1fms\n",
                     sh.devs[0].overlapTotalMs);
    }

    printStats(sh);
    teardown(sh);
    if (sh.failed.load() != 0) {
        std::fprintf(stderr, "[ff_sieve] error: pull scheduler device op failed\n");
        return 0;
    }
    return g.maxPrimeMapValue;
}

}  // namespace ff
