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

    // Small-prime list for the GPU kernel: the leading 2 is skipped (the
    // kernel's odd-only marking makes p=2 a destructive no-op — see
    // sieve_engine.cpp run()) and the tail is trimmed to primes <=
    // sqrt(maxPrimeMapValue) (see prepare()).  Valid after prepare()/run().
    const uint32_t* kernelPrimes(uint32_t* count = nullptr) const;

    // Returns the small-prime list used by the last run and (optionally) its
    // count.  Valid only after a successful run().
    const uint32_t* getSmallPrimes(uint32_t* count = nullptr) const;

private:
    int deviceIndex_;
    const char* vendor_;
    std::vector<uint32_t> smallPrimes_;
    // kernelPrimes() count after the sqrt(map-span) trim and the leading-2
    // skip; computed by prepare() from the leg geometry.
    uint32_t kernelPrimeCount_ = 0;
};

#endif  // FF_SIEVE_ENGINE_H