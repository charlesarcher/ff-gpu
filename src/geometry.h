// Leg geometry derivation (plan todo 3) — replicates the reference's geometry
// chain EXACTLY (segmentedSieve.C main 851-855 + Prime ctor 136-142), so the
// GPU binary's mapBytes equals the reference's primeMapSize with no hardcoded
// sizes. Sanity (verified by tests/budget_selftest.cpp):
//   L = 2097152 -> mapBytes 17179869185 B (2^34+1, 16.0 GiB)
//   L = 1048576 -> mapBytes 4294967297 B  (2^32+1,  4.0 GiB)
//   L =   65536 -> mapBytes 16777217 B    (2^24+1, 16 MiB)

#ifndef FF_GEOMETRY_H
#define FF_GEOMETRY_H

#include <cstdint>

namespace ff {

struct LegGeometry {
    uint64_t sumStart = 5;
    uint64_t sumLimit = 2627;        // reference default when only one positional is given
    uint64_t maxGeneratedPrime = 0;  // (sumLimit+1)>>1
    uint64_t productLimit = 0;       // maxGeneratedPrime^2
    uint64_t primeLimit = 0;         // (productLimit>>2)+2
    uint64_t mapBytes = 0;           // ((primeLimit>1?primeLimit:2)+15)>>4 == primeMapSize
    uint64_t maxPrimeMapValue = 0;   // (mapBytes<<4)-1
};

// sumStart is forced odd via |1 and clamped >= 5, sumLimit >= sumStart
// (reference main, segmentedSieve.C:823-831).
LegGeometry computeLegGeometry(uint64_t sumStart, uint64_t sumLimit);

// M4 SEARCH-participation workspace (todo 14, unchanged): mirrors the
// reference's FreudenthalTools::factors sizing (segmentedSieve.C:398-402) —
// 8 B x floor(log2(productLimit)+0.5) x (that-1) x 2. At 2M: 24960 B (~24.4 KB).
uint64_t searchWorkspaceBytes(uint64_t leg);

}  // namespace ff

#endif  // FF_GEOMETRY_H
