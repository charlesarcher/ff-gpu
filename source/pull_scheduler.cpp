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
#include <memory>
#include <mutex>
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
    void* contigBase = nullptr;       // residency: single map-sized buffer;
                                      // region[] entries are interior offsets
    void* stagingDev = nullptr;       // 1-slab device scratch (non-home pulls)
    void* stagingHost = nullptr;      // pinned host landing buffer (allocPinned)
    uint64_t pulled = 0;
    uint64_t staged = 0;
    // M3 overlap engine (todo 12)
    void* computeStream = nullptr;
    void* copyStream = nullptr;
    void* evEndA = nullptr;           // compute-end marker; reused post-join to
                                      // fence copyStream before hostMap reads
    double overlapComputeMs = 0;
    double overlapCopyMs = 0;
    double overlapTotalMs = 0;
};

struct Shared {
    uint64_t numSlabs = 0;
    uint64_t slabBytes = 0;           // INTERNAL bytes per slab (device-side)
    uint64_t totalMapBytes = 0;       // INTERNAL map bytes (= g.internalMapBytes)
    uint64_t spanValues = 0;          // value span covered by the map
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
        // Wheel-30 internal currency: slab s covers internal bytes
        // [s*slabBytes, ...) == values [30*s*slabBytes, ...). Device buffers
        // hold internal bytes; hostMap receives them LEFT-ALIGNED inside the
        // slab's canonical region (canonical byte of value segLo is
        // segLo/16 == 15*off/8; slabSizeBytes % 8 == 0 keeps it integral).
        const uint64_t off = s * sh.slabBytes;                  // internal B
        const uint64_t copyBytes = std::min(sh.slabBytes,
                                            sh.totalMapBytes - off);
        const uint64_t segLo = off * 30ull;
        uint64_t segHi = segLo + sh.slabBytes * 30ull;
        if (segHi > sh.spanValues) segHi = sh.spanValues;
        uint8_t* dest = sh.hostMap + (off / 8ull) * 15ull;      // canonical base
        const int owner = sh.ownerOf[s];
        int rc = 0;
        if (owner == static_cast<int>(i)) {
            void* buf = d.region[s - d.homeBase];
            // M3: async compute on computeStream, then async D2H into hostMap
            // on copyStream — this copy is the SOLE producer of hostMap for
            // home slabs; runPullScheduler fences every copyStream after the
            // join before anyone reads.
            rc = d.ops->slabComputeAsync(ri, sh.kernelPrimes, sh.kernelPrimeCount,
                                          segLo, segHi, buf, copyBytes,
                                          d.computeStream, nullptr);
            if (rc == 0) {
                // Record end of compute, then gate the copy stream on it
                // BEFORE enqueueing the D2H — a wait enqueued after the copy
                // on the same stream would gate nothing (stream ops run in
                // enqueue order).
                d.ops->recordEvent(ri, d.evEndA, d.computeStream);
                rc = d.ops->waitEvent(ri, d.evEndA, d.copyStream);
                if (rc == 0)
                    rc = d.ops->copyD2HAsync(ri, dest, buf,
                                             copyBytes, d.copyStream);
                if (rc != 0) {
                    // Async path failed: synchronous fallback (error path
                    // only — poolCopyD2H device-syncs, so ordering holds).
                    rc = d.ops->copyD2H(ri, dest, buf, copyBytes);
                }
            }
        } else {
            ++d.staged;
            // Prime only the bytes this slab actually spans (the kernel never
            // touches beyond ceil((segHi-segLo)/30) = copyBytes).
            rc = d.ops->slabCompute(ri, sh.kernelPrimes, sh.kernelPrimeCount,
                                    segLo, segHi, d.stagingDev, copyBytes, 1);
            if (rc == 0 && owner < 0) {
                // Homeless slab: straight into hostMap (sync copy
                // self-completes before the API returns).
                rc = d.ops->copyD2H(ri, dest, d.stagingDev, copyBytes);
            } else if (rc == 0) {
                // Foreign-owned slab: relay through the pinned host landing
                // into the owner's resident buffer (kept for the task-12
                // residency handoff), then enqueue the final D2H into hostMap
                // on the OWNER's copyStream — the post-join fence drains it.
                const size_t oi = static_cast<size_t>(owner);
                PerDevice& od = sh.devs[oi];
                const int ori = od.dev->runtimeIndex;
                void* obuf = od.region[s - od.homeBase];
                rc = d.ops->copyD2H(ri, d.stagingHost, d.stagingDev,
                                    copyBytes);
                if (rc == 0)
                    rc = od.ops->copyH2D(ori, obuf, d.stagingHost, copyBytes);
                if (rc == 0 && od.copyStream) {
                    rc = od.ops->copyD2HAsync(ori, dest, obuf,
                                              copyBytes, od.copyStream);
                    if (rc != 0)   // enqueue failed: synchronous fallback
                        rc = od.ops->copyD2H(ori, dest, obuf, copyBytes);
                } else if (rc == 0) {
                    rc = od.ops->copyD2H(ori, dest, obuf, copyBytes);
                }
            }
        }
        if (rc != 0) {
            sh.failed.store(1, std::memory_order_relaxed);
            return;
        }
    }
}

void teardown(Shared& sh)
{
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        const int ri = d.dev->runtimeIndex;
        if (d.contigBase) {
            // region[] are interior offsets of contigBase — free the base only
            (void)d.ops->free(ri, d.contigBase);
            d.contigBase = nullptr;
            d.region.clear();
        } else {
            for (void* buf : d.region) (void)d.ops->free(ri, buf);
            d.region.clear();
        }
        if (d.stagingDev) {
            (void)d.ops->free(ri, d.stagingDev);
            d.stagingDev = nullptr;
        }
        if (d.stagingHost) {
            (void)d.ops->freePinned(ri, d.stagingHost);
            d.stagingHost = nullptr;
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
                          uint64_t* wallUs,
                          int residencyDev,
                          PullMapResidency* residencyOut)
{
    if (devs.size() > static_cast<size_t>(kMaxDevices) ||
        devs.size() != budgets.size()) {
        std::fprintf(stderr,
                     "[ff_sieve] error: scheduler needs 1..%d index-aligned "
                     "devices/budgets (got %zu/%zu)\n",
                     kMaxDevices, devs.size(), budgets.size());
        return 0;
    }

    // Heap-anchored so a deferred-teardown run can hand the whole state to
    // the residency consumer (releasePullScheduler deletes it); every other
    // path tears down internally and lets the unique_ptr free the shell.
    std::unique_ptr<Shared> shp = std::make_unique<Shared>();
    Shared& sh = *shp;
    if (cfg.slabSizeBytes % 8ull != 0ull) {
        // Wheel-30 requirement: slab edges must land on 8-internal-byte
        // superblock groups so per-slab canonical region starts (15*off/8)
        // stay integral and the backward in-place expansion stays safe.
        std::fprintf(stderr,
                     "[ff_sieve] error: --slab-size %llu B is not a multiple "
                     "of 8 B (wheel-30 superblock group)\n",
                     static_cast<unsigned long long>(cfg.slabSizeBytes));
        return 0;
    }
    sh.slabBytes = cfg.slabSizeBytes;      // INTERNAL bytes per slab
    sh.totalMapBytes = g.internalMapBytes;
    sh.spanValues = g.mapBytes << 4;
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

    // Residency feasibility: the candidate device must back the contiguous
    // map plus its staging scratch within its own budget envelope; otherwise
    // the run proceeds on the legacy per-slab shape and reports no handoff.
    bool residencyActive = false;
    if (residencyOut != nullptr && residencyDev >= 0 &&
        residencyDev < static_cast<int>(sh.devs.size())) {
        PerDevice& rd = sh.devs[static_cast<size_t>(residencyDev)];
        const uint64_t need =
            sh.totalMapBytes +
            (rd.budget->slabCount > 0 ? sh.slabBytes : 0);
        if (need <= rd.budget->backing) {
            residencyActive = true;
        } else {
            std::fprintf(stderr,
                         "[ff_sieve] residency: device[%d] %s cannot back "
                         "%llu B contiguously (backing %llu B) — legacy copy "
                         "path\n",
                         residencyDev, rd.dev->name,
                         static_cast<unsigned long long>(need),
                         static_cast<unsigned long long>(rd.budget->backing));
        }
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
        if (residencyActive && static_cast<int>(i) == residencyDev) {
            // One contiguous map-sized allocation; home slabs become interior
            // offsets so the consumer sees [0, totalMapBytes) behind a single
            // device pointer. Same aggregate footprint as the per-slab shape
            // (its right-sized caps summed to totalMapBytes).
            d.contigBase = d.ops->alloc(ri, sh.totalMapBytes);
            if (d.contigBase != nullptr) {
                uint8_t* base = static_cast<uint8_t*>(d.contigBase);
                for (uint64_t k = 0; k < d.homeCount; ++k)
                    d.region.push_back(base + (d.homeBase + k) * sh.slabBytes);
            } else {
                std::fprintf(stderr,
                             "[ff_sieve] residency: contiguous %llu B alloc "
                             "failed on device[%zu] %s — legacy copy path\n",
                             static_cast<unsigned long long>(sh.totalMapBytes),
                             i, d.dev->name);
                residencyActive = false;
            }
        }
        if (d.contigBase == nullptr) {
            for (uint64_t k = 0; k < d.homeCount; ++k) {
                const uint64_t s = d.homeBase + k;
                // Right-size to this slab's actual byte extent: every slab except
                // the last is a full slabBytes, the final one may be tiny. This
                // equals the kernel tail-policy floor ceil((segHi-segLo)/16)
                // exactly (sieve_slab_kernel.h), so word/byte marking stays in
                // bounds and buffer geometry per launch site is unchanged.
                const uint64_t off = s * sh.slabBytes;
                const uint64_t slabCap = std::min(sh.slabBytes,
                                                  sh.totalMapBytes - off);
                void* buf = d.ops->alloc(ri, slabCap);
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
            }
        }
        for (uint64_t k = 0; k < d.homeCount; ++k) {
            const uint64_t s = d.homeBase + k;
            const uint64_t off = s * sh.slabBytes;                 // internal B
            const uint64_t slabCap = std::min(sh.slabBytes,
                                              sh.totalMapBytes - off);
            if (d.ops->memset(ri, d.region[k], 0xff, slabCap) != 0) {
                teardown(sh);
                return 0;
            }
            if (s == 0) {
                // Value 1 is not prime: clear residue-slot 0 of byte 0
                // (internal init 0xfe; was 0x7f in the canonical layout).
                const uint8_t notOne = 0xfe;
                if (d.ops->copyH2D(ri, d.region[k], &notOne, 1) != 0) {
                    teardown(sh);
                    return 0;
                }
            }
            if (s + 1 == sh.numSlabs && slabCap > 0) {
                // Zero the padding slots past the value span (task-5 internal
                // contract: expansion and decode read them as "no residue").
                uint8_t lastByte = 0xff;
                const uint64_t base = (sh.totalMapBytes - 1) * 30ull;
                for (unsigned r = 0; r < kWheelResidueCount; ++r)
                    if (base + kWheelResidues[r] >= sh.spanValues)
                        lastByte &= static_cast<uint8_t>(~(1u << r));
                if (lastByte != 0xff &&
                    d.ops->copyH2D(ri,
                                   static_cast<uint8_t*>(d.region[k]) +
                                       (slabCap - 1),
                                   &lastByte, 1) != 0) {
                    teardown(sh);
                    return 0;
                }
            }
        }
        if (d.cap > 0) {
            // Staging scratch must hold ANY pulled slab, so it stays full
            // slabBytes; only the host landing buffer is bounded by the map.
            d.stagingDev = d.ops->alloc(ri, sh.slabBytes);
            if (!d.stagingDev) {
                std::fprintf(stderr,
                             "[ff_sieve] error: staging alloc failed on "
                             "device[%zu] %s\n",
                             i, d.dev->name);
                teardown(sh);
                return 0;
            }
            void* ph = nullptr;
            if (d.ops->allocPinned(ri, &ph,
                                   std::min(sh.slabBytes, sh.totalMapBytes)) != 0 ||
                !ph) {
                std::fprintf(stderr,
                             "[ff_sieve] error: pinned staging alloc failed "
                             "on device[%zu] %s\n",
                             i, d.dev->name);
                teardown(sh);
                return 0;
            }
            d.stagingHost = ph;
        }
    }
    // M3 overlap: create per-device streams and the compute-end event
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        d.computeStream = d.ops->createStream(d.dev->runtimeIndex);
        d.copyStream = d.ops->createStream(d.dev->runtimeIndex);
        d.evEndA = d.ops->createEvent(d.dev->runtimeIndex);
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
    // SYNC MANDATE: with assemble() gone, hostMap is produced exclusively by
    // async copyStream D2Hs (home slabs on the owning device's stream, relayed
    // slabs on the owner's stream) plus self-completing sync copies for
    // homeless slabs. Fence every copy stream AFTER the join (all copies
    // enqueued) and BEFORE hostMap is consumed or teardown destroys the
    // streams/events: record on the copyStream, then host-wait. Runs even on
    // the failure path so teardown never frees buffers under in-flight copies.
    for (size_t i = 0; i < sh.devs.size(); ++i) {
        PerDevice& d = sh.devs[i];
        if (!d.copyStream || !d.evEndA) continue;
        if (d.ops->recordEvent(d.dev->runtimeIndex, d.evEndA, d.copyStream) != 0 ||
            d.ops->syncEvent(d.dev->runtimeIndex, d.evEndA) != 0) {
            sh.failed.store(1, std::memory_order_relaxed);
        }
    }
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

    // Either hand the scheduler state to the residency consumer (deferred
    // teardown; released via releasePullScheduler after search) or tear down
    // internally as before. The post-join fence above already drained every
    // copy stream, so hostMap AND the contiguous residency buffer are final
    // at this point — including the producing kernels, since each copy was
    // enqueued behind its compute-end event.
    const bool deferTeardown = residencyOut != nullptr && residencyActive &&
                               sh.failed.load(std::memory_order_relaxed) == 0;
    if (deferTeardown) {
        PerDevice& rd = sh.devs[static_cast<size_t>(residencyDev)];
        uint64_t foreign = 0;
        for (size_t s = 0; s < sh.ownerOf.size(); ++s)
            if (sh.ownerOf[s] != residencyDev) ++foreign;
        residencyOut->valid = true;
        residencyOut->deviceIndex = residencyDev;
        residencyOut->devPtr = static_cast<uint8_t*>(rd.contigBase);
        residencyOut->mapBytes = sh.totalMapBytes;
        residencyOut->ownerOf = sh.ownerOf;
        residencyOut->handle = shp.release();
        std::fprintf(stderr,
                     "[ff_sieve] residency: map handed off on device[%d] %s "
                     "(%llu B contiguous, %llu/%llu slab(s) resident, %llu "
                     "foreign/homeless to fill); teardown deferred\n",
                     residencyDev, rd.dev->name,
                     static_cast<unsigned long long>(sh.totalMapBytes),
                     static_cast<unsigned long long>(sh.numSlabs - foreign),
                     static_cast<unsigned long long>(sh.numSlabs),
                     static_cast<unsigned long long>(foreign));
    } else {
        if (residencyOut != nullptr) residencyOut->valid = false;
        teardown(sh);
    }
    if (sh.failed.load() != 0) {
        std::fprintf(stderr, "[ff_sieve] error: pull scheduler device op failed\n");
        return 0;
    }
    return g.maxPrimeMapValue;
}

void releasePullScheduler(PullMapResidency* residency)
{
    if (residency == nullptr || residency->handle == nullptr) {
        if (residency != nullptr) residency->valid = false;
        return;
    }
    Shared* sh = static_cast<Shared*>(residency->handle);
    teardown(*sh);
    delete sh;
    residency->handle = nullptr;
    residency->valid = false;
    residency->devPtr = nullptr;
    residency->ownerOf.clear();
}

// ---- Wheel-30 -> canonical boundary conversion (tasks 6+7 pair) ------------

namespace {

// One superblock = 8 internal bytes <-> 15 canonical bytes (lcm(30,16)=240
// values). Split into four 16-bit quarters: quarter h holds internal bytes
// 2h,2h+1, whose bits map onto canonical bit positions 30h..30h+29 (odd
// value lv sits at MSB-first canonical bit (lv-1)/2; packed little-endian
// position = (m & ~7) | (7-(m&7))). Each table entry is PRE-SHIFTED into its
// final window position, so one superblock expands as 4 lookups + 4 ORs.
struct ExpandTables {
    uint64_t lo[4][65536];
    uint64_t hi[4][65536];
};

ExpandTables* g_expandTables = nullptr;
std::once_flag g_expandTablesOnce;

void buildExpandTables()
{
    g_expandTables = new ExpandTables();
    for (unsigned h = 0; h < 4; ++h) {
        for (unsigned x = 0; x < 65536; ++x) {
            uint64_t lo = 0, hi = 0;
            if (x != 0) {
                for (unsigned j = 0; j < 16; ++j) {
                    if (!((x >> j) & 1u)) continue;
                    const unsigned d = j >> 3, i = j & 7;
                    const uint64_t lv =
                        60ull * h + 30ull * d + kWheelResidues[i];
                    const uint64_t m = (lv - 1) / 2;
                    const uint64_t pos = (m & ~7ull) | (7ull - (m & 7ull));
                    if (pos < 64) lo |= 1ull << pos;
                    else          hi |= 1ull << (pos - 64);
                }
            }
            g_expandTables->lo[h][x] = lo;
            g_expandTables->hi[h][x] = hi;
        }
    }
}

// Backward IN-PLACE widening of one slab region: internal bytes occupy
// [0, cBi) of `region`, canonical bytes [0, cBc) are produced over them.
// Superblocks run back-to-front; group s writes [15s,15s+15) while every
// still-unread source lies below 8s+8 <= 15s for s >= 2, so nothing is
// clobbered; groups 0 and 1 stage their 16 source bytes first (their write
// ranges reach down to 15 < 8+8). The trailing partial group (last slab of
// the map only) is scalar-expanded FIRST, before any full-group write can
// reach its source.
void expandSlabInPlace(uint8_t* region, uint64_t cBc, uint64_t cBi,
                       uint64_t spanSlab)
{
    std::call_once(g_expandTablesOnce, buildExpandTables);
    const ExpandTables& t = *g_expandTables;

    const uint64_t nG = cBi / 8;
    const uint64_t tail = cBi % 8;
    if (tail != 0) {
        // Partial superblock nG: values [240*nG, spanSlab) -> canonical bytes
        // [15*nG, cBc). Runs before the backward loop (its source would be
        // overwritten otherwise: full-group writes reach 15*nG > 8*nG).
        uint8_t win[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (uint64_t q = 0; q < tail; ++q) {
            const uint8_t b = region[8 * nG + q];
            if (!b) continue;
            for (unsigned i = 0; i < kWheelResidueCount; ++i) {
                if (!((b >> i) & 1u)) continue;
                const uint64_t lv = 240ull * nG + 30ull * q + kWheelResidues[i];
                if (lv >= spanSlab) break;   // ascending residues: padding
                const uint64_t m = (lv - 1) / 2;
                win[(m >> 3) - 15 * nG] |=   // window-relative byte index
                    static_cast<uint8_t>(0x80u >> (m & 7u));
            }
        }
        const uint64_t outBytes = cBc - 15 * nG;
        for (uint64_t j = 0; j < outBytes && j < 15; ++j)
            region[15 * nG + j] = win[j];
    }

    if (nG == 0) return;
    // Groups 0 and 1 read below their own write bases — stage them first.
    uint8_t head[16];
    const uint64_t headBytes = nG >= 2 ? 16 : (nG == 1 ? 8 : 0);
    for (uint64_t j = 0; j < headBytes; ++j) head[j] = region[j];

    for (uint64_t s = nG; s-- > 0;) {
        const uint8_t* src;
        uint8_t single[8];
        if (s >= 2) {
            src = region + 8 * s;
        } else {
            for (uint64_t j = 0; j < 8; ++j) single[j] = head[8 * s + j];
            src = single;
        }
        uint64_t Q;
        std::memcpy(&Q, src, 8);
        const uint64_t lo = t.lo[0][Q & 0xffffu] |
                            t.lo[1][(Q >> 16) & 0xffffu] |
                            t.lo[2][(Q >> 32) & 0xffffu] |
                            t.lo[3][(Q >> 48) & 0xffffu];
        const uint64_t hi = t.hi[0][Q & 0xffffu] |
                            t.hi[1][(Q >> 16) & 0xffffu] |
                            t.hi[2][(Q >> 32) & 0xffffu] |
                            t.hi[3][(Q >> 48) & 0xffffu];
        std::memcpy(region + 15 * s, &lo, 8);
        std::memcpy(region + 15 * s + 8, &hi, 7);
    }
}

void expandSlabRange(uint8_t* hostMap, const LegGeometry& g,
                     uint64_t slabSizeBytes, uint64_t sBegin, uint64_t sEnd)
{
    const uint64_t internalTotal = g.internalMapBytes;
    const uint64_t span = g.mapBytes << 4;
    const uint64_t numSlabs =
        (internalTotal + slabSizeBytes - 1) / slabSizeBytes;
    for (uint64_t s = sBegin; s < sEnd && s < numSlabs; ++s) {
        const uint64_t iOff = s * slabSizeBytes;
        const uint64_t cBi = std::min(slabSizeBytes, internalTotal - iOff);
        const uint64_t segLo = iOff * 30ull;
        uint64_t segHi = segLo + slabSizeBytes * 30ull;
        if (segHi > span) segHi = span;
        const uint64_t cBc = (segHi - segLo + 15ull) / 16ull;
        expandSlabInPlace(hostMap + (iOff / 8ull) * 15ull, cBc, cBi,
                          segHi - segLo);
    }
}

// Task-12 debug trap: the expansion is destructive and non-idempotent
// (expandSlabInPlace widens backward IN PLACE), so a second entry within one
// process means an exactly-once contract violation in the caller — abort
// loudly instead of silently corrupting hostMap.
std::atomic<unsigned> g_expandEntries{0};

}  // namespace

void expandSieveMapToCanonical(uint8_t* hostMap, const LegGeometry& g,
                               uint64_t slabSizeBytes)
{
    if (hostMap == nullptr || g.internalMapBytes == 0) return;
    if (g_expandEntries.fetch_add(1, std::memory_order_relaxed) != 0) {
        std::fprintf(stderr,
                     "[ff_sieve] FATAL: expandSieveMapToCanonical entered "
                     "twice — exactly-once contract violated\n");
        std::abort();
    }
    const uint64_t numSlabs =
        (g.internalMapBytes + slabSizeBytes - 1) / slabSizeBytes;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const unsigned nThreads =
        static_cast<unsigned>(std::min<uint64_t>(hw, numSlabs));
    std::vector<std::thread> pool;
    pool.reserve(nThreads);
    for (unsigned t = 0; t < nThreads; ++t) {
        const uint64_t sBegin = numSlabs * t / nThreads;
        const uint64_t sEnd = numSlabs * (t + 1) / nThreads;
        if (sEnd > sBegin)
            pool.emplace_back(expandSlabRange, hostMap, std::cref(g),
                              slabSizeBytes, sBegin, sEnd);
    }
    for (auto& th : pool) th.join();
    // Structural primes 3 and 5 survive wheel-30 unrepresented; the expansion
    // rebuilds every byte from scratch, so re-insert them (byte 0, bits
    // 0x40|0x20). Value 1 arrives cleared through the internal 0xfe init.
    hostMap[0] |= static_cast<uint8_t>(0x60u);
}

}  // namespace ff
