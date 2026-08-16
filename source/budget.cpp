#include "budget.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace ff {
namespace {

// §4.3 "Clamp budgets to allocation granularity" — floor backing to 4 KiB.
constexpr unsigned long long kGranularity = 4096;

}  // namespace

DeviceBudget computeBudget(const DeviceInfo& dev, const Config& cfg,
                           double fraction)
{
    DeviceBudget b;
    b.fraction = fraction;
    b.freeQueried = dev.freeBytes;
    b.freeUsed = static_cast<unsigned long long>(fraction * double(dev.freeBytes));
    b.cap = cfg.hasBudgetCap ? cfg.budgetCapBytes
                             : std::numeric_limits<unsigned long long>::max();
    b.budget = std::min(b.freeUsed, b.cap);
    if (cfg.hasScratch) {
        b.scratch = cfg.scratchBytes;
    } else {
        unsigned long long autoScratch =
            static_cast<unsigned long long>(0.15 * double(b.budget));
        b.scratch = std::min(autoScratch, kAutoScratchCapBytes);
    }
    b.headroom = kHeadroomBytes;
    unsigned long long backing = 0;
    if (b.budget > b.scratch) {
        unsigned long long afterScratch = b.budget - b.scratch;
        if (afterScratch > b.headroom) backing = afterScratch - b.headroom;
    }
    backing -= backing % kGranularity;   // clamp to allocation granularity
    b.backing = backing;
    b.slabCount = cfg.slabSizeBytes ? backing / cfg.slabSizeBytes : 0;
    return b;
}

DeviceBudget shrinkFractionForAlloc(const DeviceInfo& dev, const Config& cfg,
                                    double fraction)
{
    constexpr double kFloor = 0.50;   // §4.3 binary-search floor
    if (fraction <= kFloor) return computeBudget(dev, cfg, kFloor);
    double next = 0.5 * (fraction + kFloor);
    return computeBudget(dev, cfg, next);
}

std::string bytesToHuman(unsigned long long bytes)
{
    char buf[64];
    const double GiB = double(1ull << 30), MiB = double(1ull << 20),
                 KiB = double(1ull << 10);
    if (bytes >= (1ull << 30))
        std::snprintf(buf, sizeof buf, "%.2f GiB", double(bytes) / GiB);
    else if (bytes >= (1ull << 20))
        std::snprintf(buf, sizeof buf, "%.1f MiB", double(bytes) / MiB);
    else if (bytes >= (1ull << 10))
        std::snprintf(buf, sizeof buf, "%.1f KiB", double(bytes) / KiB);
    else
        std::snprintf(buf, sizeof buf, "%llu B",
                      static_cast<unsigned long long>(bytes));
    return std::string(buf);
}

}  // namespace ff
