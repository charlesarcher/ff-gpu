// g++ implementation of the full-map SieveEngine.
//
// Orchestrates:
//   1. computeLegGeometry → derive mapBytes / maxPrimeMapValue
//   2. Host sieve up to sqrt(maxPrimeMapValue) → small primes
//   3. Query per-arch SieveSlabEngineGetLaunchFn_<arch> (both arches)
//   4. Dispatch to the per-arch run function with the full map
//
// This TU is pure g++: no vendor headers, no __global__ / hip* types cross
// the boundary.  Device memory management lives entirely inside the
// per-arch TU (sieve_slab_engine.cpp compiled with hipcc).
//
// Note: The devabstraction.h include is available for potential future use
// (timing, device property queries) but the engine itself delegates memory
// operations to the per-arch run function which uses hipMalloc/hipMemcpy
// directly (same pattern as sieve_slab_kernel.cpp).

#include "sieve_engine.h"
#include "geometry.h"
#include "sieve_slab_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Simple host-side Eratosthenes sieve up to `limit`.
// Fills `out` with primes in ascending order.
void hostSievePrimes(uint64_t limit, std::vector<uint32_t>& out)
{
    if (limit < 2) return;
    std::vector<bool> composite(limit + 1, false);
    for (uint64_t p = 2; p * p <= limit; ++p) {
        if (!composite[p]) {
            for (uint64_t i = p * p; i <= limit; i += p)
                composite[static_cast<size_t>(i)] = true;
        }
    }
    out.clear();
    out.reserve(13000);  // pi(131072) ≈ 12334
    for (uint64_t p = 2; p <= limit; ++p) {
        if (!composite[static_cast<size_t>(p)])
            out.push_back(static_cast<uint32_t>(p));
    }
}

}  // namespace

// Same correction loop as main.cpp's static isqrt64 (sqrt() rounding guard).
namespace {
uint64_t isqrt64(uint64_t x) {
    if (x == 0) return 0;
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(x)));
    while ((r + 1) * (r + 1) <= x) ++r;
    while (r * r > x) --r;
    return r;
}
}  // namespace

SieveEngine::SieveEngine(int deviceIndex, const char* vendor)
    : deviceIndex_(deviceIndex), vendor_(vendor)
{
}

SieveEngine::~SieveEngine()
{
}

uint64_t SieveEngine::run(uint64_t sumLimit, uint8_t* hostMap)
{
    // ---- 1. Compute leg geometry ----
    ff::LegGeometry geom = ff::computeLegGeometry(5, sumLimit);

    // ---- 2. Generate small primes on host ----
    prepare(sumLimit);

    // ---- 3. Resolve per-arch engine run function by vendor ----
    bool nv = vendor_ != nullptr && std::strcmp(vendor_, "nvidia") == 0;
    SieveSlabEngineRunFn runFn = nv
        ? SieveSlabEngineGetLaunchFn_sm_120(deviceIndex_)
        : SieveSlabEngineGetLaunchFn_gfx1201(deviceIndex_);
    if (!runFn) {
        std::fprintf(stderr,
            "SieveEngine::run: no matching arch for device %d (vendor=%s)\n",
            deviceIndex_, vendor_ ? vendor_ : "(null)");
        return 0;
    }

    // ---- 4. Dispatch to per-arch full-map engine run ----
    // Reference Prime::SegmentFill's smallPrimes list excludes 2; the CPU
    // search's list keeps it (primes[0]=2, search indexes primes[2]=5).  Skip
    // the leading 2 for the kernel: with p=2 its odd-only adjustment is a
    // no-op, so it clears every value == 2 (mod 4), wiping all ==3 (mod 4)
    // primes from the map.
    const uint32_t* kernelPrimes = smallPrimes_.data();
    uint32_t kernelPrimeCount = static_cast<uint32_t>(smallPrimes_.size());
    if (kernelPrimeCount > 1 && kernelPrimes[0] == 2) {
        ++kernelPrimes;
        --kernelPrimeCount;
    }
    int rc = runFn(deviceIndex_,
                   kernelPrimes,
                   kernelPrimeCount,
                   geom.maxPrimeMapValue,
                   hostMap);
    if (rc != 0) {
        std::fprintf(stderr,
            "SieveEngine::run: per-arch engine failed (rc=%d)\n", rc);
        return 0;
    }

    return geom.maxPrimeMapValue;
}

uint64_t SieveEngine::prepare(uint64_t sumLimit)
{
    ff::LegGeometry geom = ff::computeLegGeometry(5, sumLimit);
    uint64_t primeLimit = geom.maxGeneratedPrime;
    if (primeLimit < 2) primeLimit = 2;
    hostSievePrimes(primeLimit, smallPrimes_);

    // The sieve kernel early-breaks at p*p >= bHi, so it never reads primes
    // beyond sqrt of the map's top value; trim the KERNEL list to
    // isqrt((mapBytes<<4)-1) — same geometry chain as everywhere else.  At 2M
    // this halves the H2D upload (~328 KB → ~172 KB).  The search phase keeps
    // reading the FULL list through getSmallPrimes().
    const uint64_t kernelPrimeMax = isqrt64((geom.mapBytes << 4) - 1);
    uint32_t kCount = 0;
    for (uint32_t v : smallPrimes_) {
        if (static_cast<uint64_t>(v) > kernelPrimeMax) break;
        ++kCount;
    }
    if (kCount > 0 && smallPrimes_[0] == 2) --kCount;  // kernel skips leading 2
    kernelPrimeCount_ = kCount;
    return geom.maxPrimeMapValue;
}

const uint32_t* SieveEngine::kernelPrimes(uint32_t* count) const
{
    const uint32_t* p = smallPrimes_.data();
    uint32_t n = static_cast<uint32_t>(smallPrimes_.size());
    if (n > 1 && p[0] == 2) {
        ++p;
        --n;
    }
    if (n > kernelPrimeCount_) n = kernelPrimeCount_;
    if (count) *count = n;
    return p;
}

const uint32_t* SieveEngine::getSmallPrimes(uint32_t* count) const
{
    if (count) *count = static_cast<uint32_t>(smallPrimes_.size());
    return smallPrimes_.data();
}