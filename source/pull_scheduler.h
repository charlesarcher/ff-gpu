// M2 backing pool + weighted dynamic pulls (plan todo 10). g++ host header —
// the scheduler drives the per-arch pool operations declared in
// sieve_slab_engine.h; device memory never crosses this TU boundary.

#ifndef FF_PULL_SCHEDULER_H
#define FF_PULL_SCHEDULER_H

#include <cstdint>
#include <vector>

#include "budget.h"
#include "config.h"
#include "device_info.h"
#include "geometry.h"

namespace ff {

// Runs the full-map sieve across ALL logical devices via the M2 backing pool:
//
//   - home assignment: each device owns a budget-sized backing region (slabs
//     [homeBase, homeBase+homeCount)), allocated as one device buffer per slab
//     and primed 0xff at setup;
//   - weighted pulls: per-device caps are the largest-remainder weight share
//     of the slab count (Config::pullWeights), so the caps always sum to
//     numSlabs — a global atomic work counter feeds slabs to the per-device
//     owner threads, and every slab is pulled exactly once (no deadlock, no
//     double pull, no oversubscription);
//   - fast path: a device computing its own home slab writes in place (zero
//     copies); cross-vendor slabs are computed into the pulling device's
//     staging scratch, landed host-side, and pushed into the owning device's
//     region buffer (or straight into hostMap when the slab is homeless). No
//     inter-vendor P2P is ever used;
//   - assembly: after all threads join, each device copies its region buffers
//     back into hostMap.
//
// devs/budgets must be index-aligned (both size N, the logical device list
// after --devices / --disable-vendor filtering). kernelPrimes is the small-
// prime list WITHOUT the leading 2 (SieveEngine::kernelPrimes()). hostMap is
// a pre-allocated g.mapBytes buffer.
//
// Returns maxPrimeMapValue on success, 0 on error. When wallUs is non-null it
// receives the sieve wall time (thread spawn through assembly) in microseconds.
uint64_t runPullScheduler(const Config& cfg,
                          const std::vector<DeviceInfo>& devs,
                          const std::vector<DeviceBudget>& budgets,
                          const LegGeometry& g,
                          const uint32_t* kernelPrimes,
                          uint32_t kernelPrimeCount,
                          uint8_t* hostMap,
                          uint64_t* wallUs = nullptr);

}  // namespace ff

#endif  // FF_PULL_SCHEDULER_H
