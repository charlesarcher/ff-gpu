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

// ---- Wheel-30 structural strip (task 5: PREPARED, INERT — do not wire) ----
// Under wheel-30 the sieve modulus 30 = 2*3*5 makes 2, 3, 5 STRUCTURAL: the
// internal map stores only residues coprime to 30, so multiples of 2/3/5 are
// unrepresentable by construction and the kernel will mark via the residue-8
// skip cycle G=[6,4,2,4,2,4,6,2] (sum 30) instead of per-prime APs for these
// three. Task 7 strips them from the kernel MARKING list and adds dev_IsPrime
// guards. Until that kernel lands these constants stay UNREFERENCED by the
// live list build below (Momus M1): the canonical odd-only kernel still
// consumes today's full list, and dropping 3/5 now would leave their odd
// multiples wrongly prime — dump-map sha256, slab_cmp and stdout would fail.
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

    // Small-prime MARKING list for the GPU sieve kernel. TODAY: the leading
    // 2 is skipped (the kernel's odd-only marking makes p=2 a destructive
    // no-op — see sieve_engine.cpp run()) and the tail is trimmed to primes
    // <= sqrt(maxPrimeMapValue) (see prepare()). FUTURE (task 7, wheel-30):
    // the same trim applies to smallPrimes_ minus kWheelStructuralPrimes —
    // 2/3/5 become structural (modulus 2*3*5, residue-cycle marking), so the
    // marking list drops all three; until then the strip above is inert and
    // this list's content is unchanged. Valid after prepare()/run().
    const uint32_t* kernelPrimes(uint32_t* count = nullptr) const;

    // Returns the small-prime list used by the last run and (optionally) its
    // count.  Valid only after a successful run().
    const uint32_t* getSmallPrimes(uint32_t* count = nullptr) const;

private:
    int deviceIndex_;
    const char* vendor_;
    std::vector<uint32_t> smallPrimes_;
    // kernelPrimes() count after the sqrt(map-span) trim and the leading-2
    // skip; computed by prepare() from the leg geometry. (Task 7 will extend
    // the skip to the whole {2,3,5} structural strip — see the wheel-30
    // block above.)
    uint32_t kernelPrimeCount_ = 0;
};

#endif  // FF_SIEVE_ENGINE_H