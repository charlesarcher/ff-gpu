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

// Task-12 residency handoff report. When runPullScheduler is given a non-null
// residencyOut AND a feasible residencyDev, the scheduler leaves the whole
// prime map resident on that device in ONE contiguous buffer, keeps the
// scheduler machinery alive after returning (DEFERRED TEARDOWN), and describes
// the layout here. The consumer MUST hand the report back to
// releasePullScheduler() once it no longer reads the device map — that call
// frees the device buffers/streams/events and ends the scheduler's lifetime.
//
// Since the tasks-6+7 wheel pair the DEVICE map is the INTERNAL wheel-30
// layout: mapBytes reports g.internalMapBytes (30 values per byte) and every
// slab offset below is an INTERNAL-byte offset (slab s starts at
// s*slabSizeBytes internal bytes = value 30*s*slabSizeBytes). The consumer
// (GPU search) must treat devPtr contents accordingly.
//
// Until release, for every slab s in [0, ownerOf.size()):
//   ownerOf[s] == deviceIndex  -> the slab's final bytes ALREADY sit on the
//                                 device at devPtr + s*slabSizeBytes: the
//                                 producing kernels finished before the
//                                 scheduler's post-join copy-stream fence
//                                 drained;
//   ownerOf[s] != deviceIndex  -> foreign-owned or homeless: the consumer must
//                                 fill that slab's range from the host-side
//                                 internal image (H2D) before reading the map
//                                 on the device.
// hostMap is produced complete on return of runPullScheduler, but in the
// WHEEL shape: each slab's internal bytes land LEFT-ALIGNED at that slab's
// canonical region start (hostMap + 15*(s*slabSizeBytes)/8). Call
// expandSieveMapToCanonical() to convert the whole buffer back into the
// canonical --dump-map format before any canonical consumer reads it.
struct PullMapResidency {
    bool valid = false;          // true iff a handoff was granted (handle set)
    int deviceIndex = -1;        // logical index into the scheduler's devs
    uint8_t* devPtr = nullptr;   // contiguous map base on that device
    uint64_t mapBytes = 0;       // usable bytes at devPtr (= g.internalMapBytes,
                                 // wheel-30 internal layout)
    std::vector<int> ownerOf;    // per-slab owning device idx, -1 = homeless
    void* handle = nullptr;      // opaque scheduler state; do not touch
};

// Runs the full-map sieve across ALL logical devices via the M2 backing pool:
//
//   - home assignment: each device owns a budget-sized backing region (slabs
//     [homeBase, homeBase+homeCount)), primed 0xff at setup. Home slabs live
//     in one device allocation per slab — except the residency device (see
//     below), whose home slabs sit at interior offsets of a single contiguous
//     map-sized buffer;
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
//   - drain: hostMap has no separate assembly pass — it is fed exclusively by
//     per-device copyStream D2Hs (home and relayed slabs on the OWNER's copy
//     stream, homeless slabs via self-completing sync copies). After the
//     worker threads join, every copyStream is fenced (recordEvent +
//     syncEvent) so hostMap AND every device-resident slab are final before
//     this function returns.
//
// Residency handoff (task 12): when residencyOut is non-null and device index
// residencyDev is feasible (its budget can back the contiguous map plus one
// staging slab), that device's home slabs are computed in place inside one
// contiguous g.mapBytes buffer and the scheduler SKIPS its teardown before
// returning, filling *residencyOut (see PullMapResidency). The caller closes
// the handoff with releasePullScheduler(). On any infeasibility, allocation
// failure or worker error the scheduler tears down internally and leaves
// residencyOut->valid false — validity is the only signal consumers need.
//
// devs/budgets must be index-aligned (both size N, the logical device list
// after --devices / --disable-vendor filtering). kernelPrimes is the small-
// prime list WITHOUT the leading 2 (SieveEngine::kernelPrimes()). hostMap is
// a pre-allocated g.mapBytes buffer.
//
// Returns maxPrimeMapValue on success, 0 on error. When wallUs is non-null it
// receives the sieve wall time (thread spawn through drain fence) in
// microseconds.
uint64_t runPullScheduler(const Config& cfg,
                          const std::vector<DeviceInfo>& devs,
                          const std::vector<DeviceBudget>& budgets,
                          const LegGeometry& g,
                          const uint32_t* kernelPrimes,
                          uint32_t kernelPrimeCount,
                          uint8_t* hostMap,
                          uint64_t* wallUs = nullptr,
                          int residencyDev = -1,
                          PullMapResidency* residencyOut = nullptr);

// Ends a deferred-teardown scheduler run: frees the residency device buffers,
// staging scratch, streams and events, then the scheduler state itself. Safe
// to call with valid=false / handle=nullptr (no-op) and idempotent (the
// handle is cleared); *residency is invalidated.
void releasePullScheduler(PullMapResidency* residency);

// Converts a hostMap produced by runPullScheduler (per-slab left-aligned
// wheel-30 internal bytes) IN PLACE into the canonical 16-values-per-byte
// format, so --dump-map / GpuPrime / stdout contracts see exactly the bytes
// the reference produces. Task-18 tiled scheduling: work is split into
// whole-superblock-group tiles of 64 MiB internal bytes; a region smaller
// than one tile keeps the original single-pass backward in-place widening
// (zero extra traffic). Multi-tile regions run in TWO EPOCHS separated by a
// thread join (the barrier): epoch 1 copies each tile's raw sources into a
// PER-ITEM arena slot (map reads only, so no writer can clobber sources
// mid-flight; tiling the backward in-place pass directly is racy: canonical
// destinations drift upward at 15/8x the source rate, so lower tiles' write
// ranges always overlap upper tiles' unread sources; proofs archived in
// .omo/start-work/evidence/e-tiling.md); epoch 2 widens each tile FORWARD
// from its arena slot into its own canonical window — ascending group order
// keeps write ceilings below unread shifted sources (15i+15 <= 7G'+8(i+1)
// iff i <= G'-1), windows partition each region exactly and regions partition
// the map, so every canonical byte is written exactly once. Pool size =
// min(hardware_concurrency [FF_EXPANSION_THREADS env cap], work items).
// Table-driven: four 65536-entry uint64-pair tables map each 16-input-bit
// quarter-superblock onto its pre-shifted 30-bit canonical image (~2 ops per
// input byte; built once). Multithreaded over tiles; the caller owns the pass
// timing (task 12: the "wheel expansion" stderr line reports the overlapped
// pass's critical-path residual — what the joining thread actually blocked
// past the kernel).
// slabSizeBytes must match the runPullScheduler call (8-byte aligned).
void expandSieveMapToCanonical(uint8_t* hostMap, const LegGeometry& g,
                               uint64_t slabSizeBytes);

}  // namespace ff

#endif  // FF_PULL_SCHEDULER_H
