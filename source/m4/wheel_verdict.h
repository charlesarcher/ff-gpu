#ifndef FF_M4_WHEEL_VERDICT_H
#define FF_M4_WHEEL_VERDICT_H

// wheel_verdict.h — task 15 (D): scoped in-map-only primality decoder for
// the GPU-success EMIT path. Mirrors dev_IsPrime's in-map branch
// (gpu_search_kernel.h) bit-for-bit, but reads the HOST map, which holds the
// INTERNAL wheel-30 layout in the same PER-SLAB packing the H2D feed and
// expandSieveMapToCanonical use: slab s's internal bytes live at
// packedMap[(s*slabSizeBytes/8)*15 .. +cBi). A caller holding one contiguous
// internal image passes any 8-aligned slabSizeBytes >= its length (a single
// slab means s == 0 for every index, i.e. identity placement).
//
// Scope contract: this decoder serves ONLY the GPU-success emit overloads in
// gpu_search_emission.cpp. Dump-map and CPU-fallback consumers keep the real
// canonical expansion and GpuPrime.

#include <cassert>
#include <cstdint>

class Wheel30Verdict {
public:
    Wheel30Verdict(const uint8_t* packedMap, uint64_t maxPrimeMapValue,
                   uint64_t slabSizeBytes)
        : map_(packedMap), max_(maxPrimeMapValue), slab_(slabSizeBytes)
    {
        assert(map_ != nullptr);
        // Slab bases must be 8-aligned for the (s*slab/8)*15 placement to be
        // exact — the same alignment the H2D feed / expansion offsets assume.
        assert(slab_ % 8 == 0);
    }

    bool IsPrime(uint64_t n) const {
        // Emit-path terms are < sumLimit, far below the map bound, so
        // dev_IsPrime's above-bound Miller-Rabin quirk is unreachable here BY
        // CONSTRUCTION — hard-stop if a future consumer ever breaks that.
        // invariant: emit terms < sumLimit <= maxPrimeMapValue; tripping here means decoder called above-bound term, would need Miller-Rabin fallback
        assert(n <= max_);
        if (n == 2) return true;
        if (!(n & 1)) return false;
        if (n == 3 || n == 5) return true;
        // Deliberate literals mirroring gpu_search_kernel.h (task-8 decode
        // discipline): magic M = (2^68 + 14)/30 for division by 30; mask bit
        // r set <=> r coprime to 30 AND odd; slot = popcount-rank among the
        // ascending residues {1,7,11,13,17,19,23,29}.
        const uint64_t q =
            (uint64_t)(((__uint128_t)n * 0x8888888888888889ull) >> 64) >> 4;
        const uint32_t r = (uint32_t)(n - q * 30);
        if (!((0x208A2882u >> r) & 1u)) return false;
        const uint32_t slot = __builtin_popcount(0x208A2882u & ((1u << r) - 1u));
        const uint64_t s = q / slab_;
        const uint8_t byte =
            map_[(s * slab_ >> 3) * 15ull + (q - s * slab_)];
        return (byte >> slot) & 1u;
    }

private:
    const uint8_t* map_;
    uint64_t max_;
    uint64_t slab_;
};

#endif  // FF_M4_WHEEL_VERDICT_H
