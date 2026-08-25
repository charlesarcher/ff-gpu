// g++ header for the full-map SieveEngine.
//
// The engine orchestrates: computeLegGeometry → host prime generation →
// dispatch to per-arch SieveSlabEngineRun (see sieve_slab_engine.h) → copy
// back to hostMap.  The g++ TU never includes vendor headers; it talks to the
// per-arch code through the extern "C" dispatch mechanism.
//
// Usage:
//   uint8_t map[mapBytes];
//   SieveEngine engine(0);           // device index 0
//   uint64_t result = engine.run(2097152, map);   // sumLimit = 2M
//   if (result != 0) { /* map[] holds the full sieve */ }

#ifndef FF_SIEVE_ENGINE_H
#define FF_SIEVE_ENGINE_H

#include <cstdint>
#include <vector>

// ---- Wheel-30 structural strip (tasks 6+7 pair: ACTIVATED) -----------------
// Under wheel-30 the sieve modulus 30 = 2*3*5 makes 2, 3, 5 STRUCTURAL: the
// internal map stores only residues coprime to 30, so multiples of 2/3/5 are
// unrepresentable by construction and the kernel marks via eight uniform
// 30p APs per prime (sieve_slab_kernel.h). Since the tasks-6+7 pair landed,
// kernelPrimes() STRIPS all three from the kernel MARKING list; the wheel
// expansion (expandSieveMapToCanonical) re-inserts the primes 3 and 5 into
// the canonical hostMap so --dump-map/stdout contracts are unchanged.
// getSmallPrimes() (the SEARCH list) KEEPS {2,3,5} untouched — it is the
// factor-advancement list for Freudenthal enumeration.
inline constexpr uint32_t kWheelStructuralPrimeCount = 3;
inline constexpr uint32_t kWheelStructuralPrimes[kWheelStructuralPrimeCount] = {
    2, 3, 5};

inline constexpr bool isWheelStructuralPrime(uint32_t p)
{
    return p == 2 || p == 3 || p == 5;
}

class SieveEngine {
public:
    // Construct for the given logical device index (0-based) and vendor
    // ("amd").  Always dispatches to the gfx1201 (AMD RDNA4) path.
    SieveEngine(int deviceIndex, const char* vendor);
    ~SieveEngine();

    // Run the full-map sieve for the given sumLimit on the selected device.
    //
    // sumLimit:      the sieving limit (same as LegGeometry input)
    // hostMap:       pre-allocated host buffer; on success holds the complete
    //                sieve map with mapBytes = ((geom.maxPrimeMapValue+1)+15)>>4
    //                bytes.  The caller must allocate this buffer before calling.
    //
    // Returns: maxPrimeMapValue on success, 0 on error.
    uint64_t run(uint64_t sumLimit, uint8_t* hostMap);

    // Device-independent half of run(): computes leg geometry and the host
    // small-prime list without dispatching to any device.  The M2 pull
    // scheduler (plan todo 10) calls this, then kernelPrimes()/getSmallPrimes()
    // and drives per-slab compute itself.  Returns maxPrimeMapValue (0 on
    // error).
    uint64_t prepare(uint64_t sumLimit);

    // Small-prime MARKING list for the GPU sieve kernel. Since the tasks-6+7
    // wheel pair: skips the whole {2,3,5} structural strip (modulus covers
    // them) AND is trimmed to primes <= sqrt(maxPrimeMapValue) (prepare()).
    // Valid after prepare()/run().
    const uint32_t* kernelPrimes(uint32_t* count = nullptr) const;

    // Returns the small-prime list used by the last run and (optionally) its
    // count.  Valid only after a successful run(). KEEPS {2,3,5}: this is the
    // search-side factor-advancement list.
    const uint32_t* getSmallPrimes(uint32_t* count = nullptr) const;

private:
    int deviceIndex_;
    const char* vendor_;
    std::vector<uint32_t> smallPrimes_;
    // kernelPrimes() count after the sqrt(map-span) trim and the {2,3,5}
    // structural skip; computed by prepare() from the leg geometry.
    uint32_t kernelPrimeCount_ = 0;
};

#endif  // FF_SIEVE_ENGINE_H