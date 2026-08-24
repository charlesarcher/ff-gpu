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
    // CANONICAL sizing ((primeLimit>1?primeLimit:2)+15)>>4 == primeMapSize.
    // REMAINS the --dump-map/hostMap format and the sha256 contract.
    uint64_t mapBytes = 0;
    uint64_t maxPrimeMapValue = 0;   // (mapBytes<<4)-1 — numeric semantics unchanged
    // WHEEL-30 INTERNAL sizing ceil((mapBytes<<4)/30) over the same value span
    // (task 5: PREPARED only — no production path allocates or consumes it
    // until the scheduler/kernel wiring lands in tasks 6/7).
    uint64_t internalMapBytes = 0;
};

// sumStart is forced odd via |1 and clamped >= 5, sumLimit >= sumStart
// (reference main, segmentedSieve.C:823-831).
LegGeometry computeLegGeometry(uint64_t sumStart, uint64_t sumLimit);

// ---- Wheel-30 foundations (kernel-gap-closure task 5) ---------------------
//
// Two map layouts over the same integer-value span [0, span):
//
//   CANONICAL (shipped reference format): 16 integers/byte, one bit per odd
//   value v — byte v>>4, bit 0x80>>(v>>1 & 7). Feeds --dump-map/hostMap and
//   the sha256 contract; sized by canonicalMapBytes.
//
//   INTERNAL (wheel-30): 30 integers/byte, one bit per value coprime to 30.
//   Byte k covers values [30k, 30k+30); bit i (mask 1u<<i) is value
//   30k + kWheelResidues[i]. Sized by internalMapBytes — the honest 1.875x
//   density gain vs canonical (30/16), NOT 3.75x (that figure compares
//   against a naive all-integers bitmap).
//
// In LegGeometry, span == mapBytes<<4 == maxPrimeMapValue+1.

inline constexpr uint64_t kWheelModulus = 30;  // 2*3*5
inline constexpr unsigned kWheelResidueCount = 8;

// Ascending residues coprime to 30; internal bit i <-> kWheelResidues[i].
inline constexpr uint8_t kWheelResidues[kWheelResidueCount] = {
    1, 7, 11, 13, 17, 19, 23, 29};

uint64_t canonicalMapBytes(uint64_t span);  // ceil(span/16) — unchanged chain
uint64_t internalMapBytes(uint64_t span);   // ceil(span/30) — plan's bufMin

// Layout-conversion pair (pure functions; wired into production by tasks 6/9,
// exercised here only by ff_budget_selftest round-trip units).
//
// expandWheel30ToCanonical(src, dst, span): src holds internalMapBytes(span)
// readable bytes, dst receives canonicalMapBytes(span) FULLY DEFINED bytes:
//   - odd v < span coprime to 30: bit copied from src slot (v%30);
//   - odd v < span divisible by 3 or 5: set iff v==3 || v==5 (the primes
//     themselves survive; other multiples are structural composites);
//   - padding bits (odd v in [span, 16*canonicalMapBytes(span))): cleared.
// Precondition: src padding bits (slots with value >= span) must be zero.
//
// compactCanonicalToWheel30(src, dst, span): inverse. dst receives
// internalMapBytes(span) fully defined bytes (padding slots zeroed). Bits at
// odd multiples of 3 or 5 in src are never read.
//
// Round-trip guarantees (unit-tested in ff_budget_selftest):
//   compact(expand(x)) == x  for valid internal x (zero padding);
//   expand(compact(y)) == y  for canonical y respecting residues (0 at odd
//                            multiples of 3/5 except {3,5}, zero padding).
void expandWheel30ToCanonical(const uint8_t* src, uint8_t* dst, uint64_t span);
void compactCanonicalToWheel30(const uint8_t* src, uint8_t* dst, uint64_t span);

// M4 SEARCH-participation workspace (todo 14, unchanged): mirrors the
// reference's FreudenthalTools::factors sizing (segmentedSieve.C:398-402) —
// 8 B x floor(log2(productLimit)+0.5) x (that-1) x 2. At 2M: 24960 B (~24.4 KB).
uint64_t searchWorkspaceBytes(uint64_t leg);

}  // namespace ff

#endif  // FF_GEOMETRY_H
