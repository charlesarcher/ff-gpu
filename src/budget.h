// GPU_PLAN §4.2/§4.3 budget math (plan todo 3). Vendor-neutral — operates on
// ff::DeviceInfo records filled by the vendor TUs.

#ifndef FF_BUDGET_H
#define FF_BUDGET_H

#include <string>

#include "config.h"
#include "device_info.h"

namespace ff {

// Documented spec constant (GPU_PLAN §4.2): driver/allocator/display safety
// margin, always reserved, NOT env-overridable.
constexpr unsigned long long kHeadroomBytes = 64ull * 1024 * 1024;
// Default working-set slab size (GPU_PLAN §4.1): 1 GiB.
constexpr unsigned long long kDefaultSlabBytes = 1ull << 30;
// Scratch auto-default cap (GPU_PLAN §4.2): min(0.15 x budget, 1 GiB).
constexpr unsigned long long kAutoScratchCapBytes = 1ull << 30;

struct DeviceBudget {
    double fraction = 0.90;                        // f actually used (after overrides/fallback)
    unsigned long long freeQueried = 0;            // free VRAM from the runtime
    unsigned long long freeUsed = 0;               // floor(f x freeQueried)
    unsigned long long cap = 0;                    // UINT64_MAX when uncapped
    unsigned long long budget = 0;                 // min(freeUsed, cap)
    unsigned long long scratch = 0;                // fixed, else min(0.15 x budget, 1 GiB)
    unsigned long long headroom = 0;               // kHeadroomBytes
    unsigned long long backing = 0;                // budget - scratch - headroom (floor 4 KiB)
    unsigned long long slabCount = 0;              // floor(backing / slabSize)
};

// §4.2 budget formula for one logical device.
DeviceBudget computeBudget(const DeviceInfo& dev, const Config& cfg,
                           double fraction);

// §4.3 alloc-failure fallback — ONE binary-search step: returns the budget
// re-sized at the midpoint between `fraction` and the 0.50 floor (backing is
// monotonic in f, so a midpoint step strictly shrinks backing). The caller
// retries the backing allocation with the returned budget and repeats until
// the alloc succeeds or fraction reaches the 0.50 floor — then it must fail
// loudly; never silently oversubscribe. Triggered by the device layers from
// todo 6 when a backing-region malloc fails; no allocations exist at todo 3,
// so the selftest drives the step sequence synthetically.
DeviceBudget shrinkFractionForAlloc(const DeviceInfo& dev, const Config& cfg,
                                    double fraction);

// "20.00 GiB" / "13.88 GiB" / "512.0 MiB" / "4096 B"
std::string bytesToHuman(unsigned long long bytes);

}  // namespace ff

#endif  // FF_BUDGET_H
