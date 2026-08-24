#include "geometry.h"

#include <cmath>
#include <cstring>

namespace ff {

LegGeometry computeLegGeometry(uint64_t sumStart, uint64_t sumLimit)
{
    LegGeometry g;
    g.sumStart = sumStart | 1;   // forced odd (reference main: atoi(argv[1])|1)
    if (g.sumStart < 5) g.sumStart = 5;
    g.sumLimit = sumLimit;
    if (g.sumStart > g.sumLimit) g.sumLimit = g.sumStart;

    g.maxGeneratedPrime = (g.sumLimit + 1) >> 1;
    g.productLimit = g.maxGeneratedPrime * g.maxGeneratedPrime;
    g.primeLimit = (g.productLimit >> 2) + 2;  // Peter Product always gets a multiple of 4
    uint64_t cap = g.primeLimit > 1 ? g.primeLimit : 2;
    g.mapBytes = (cap + 15) >> 4;
    g.maxPrimeMapValue = (g.mapBytes << 4) - 1;
    g.internalMapBytes = internalMapBytes(g.mapBytes << 4);
    return g;
}

uint64_t searchWorkspaceBytes(uint64_t leg)
{
    uint64_t maxGeneratedPrime = (leg + 1) >> 1;
    uint64_t productLimit = maxGeneratedPrime * maxGeneratedPrime;
    // Reference: uint64_t greatestNumberPrimeFactors =
    //            log(productLimit)/log(2)+.5   (C-style truncation cast)
    double g = std::log2(double(productLimit)) + 0.5;
    uint64_t greatest = uint64_t(g);
    if (greatest < 2) greatest = 2;   // degenerate-leg guard
    uint64_t nFactors = (greatest * (greatest - 1)) << 1;
    return nFactors * sizeof(uint64_t);
}

uint64_t canonicalMapBytes(uint64_t span)
{
    return (span + 15) >> 4;
}

uint64_t internalMapBytes(uint64_t span)
{
    return (span + kWheelModulus - 1) / kWheelModulus;
}

namespace {

// Canonical bit placement for an odd value v: byte v>>4, MSB-first bit
// 0x80>>(v>>1 & 7) — mirrors dev_IsPrime's decode exactly.
inline void setCanonicalBit(uint8_t* dst, uint64_t v)
{
    dst[v >> 4] |= static_cast<uint8_t>(0x80u >> ((v >> 1) & 7));
}

inline bool canonicalBitSet(const uint8_t* src, uint64_t v)
{
    return (src[v >> 4] & (0x80u >> ((v >> 1) & 7))) != 0;
}

}  // namespace

void expandWheel30ToCanonical(const uint8_t* src, uint8_t* dst, uint64_t span)
{
    if (span == 0) return;
    const uint64_t nCan = canonicalMapBytes(span);
    const uint64_t nInt = internalMapBytes(span);
    std::memset(dst, 0, nCan);

    // Structural primes 3 and 5 survive wheel-30 (they are not represented
    // internally — multiples of 3/5 are unrepresentable by construction).
    if (span > 3) setCanonicalBit(dst, 3);
    if (span > 5) setCanonicalBit(dst, 5);

    // Fast path: whole 240-value superblocks (lcm(30,16)) map 8 internal
    // bytes onto exactly 15 canonical bytes.
    uint64_t k = 0;
    for (; (k + 8) * 30 <= span; k += 8) {
        for (unsigned j = 0; j < 8; ++j) {
            const uint64_t base = (k + j) * 30;
            for (unsigned i = 0; i < kWheelResidueCount; ++i) {
                if (src[k + j] & (1u << i))
                    setCanonicalBit(dst, base + kWheelResidues[i]);
            }
        }
    }
    // Tail blocks.
    for (; k < nInt; ++k) {
        const uint64_t base = k * 30;
        for (unsigned i = 0; i < kWheelResidueCount; ++i) {
            const uint64_t v = base + kWheelResidues[i];
            if (v >= span) break;  // residues ascending: rest are padding
            if (src[k] & (1u << i)) setCanonicalBit(dst, v);
        }
    }
}

void compactCanonicalToWheel30(const uint8_t* src, uint8_t* dst, uint64_t span)
{
    if (span == 0) return;
    const uint64_t nInt = internalMapBytes(span);
    for (uint64_t k = 0; k < nInt; ++k) {
        const uint64_t base = k * 30;
        uint8_t b = 0;
        for (unsigned i = 0; i < kWheelResidueCount; ++i) {
            const uint64_t v = base + kWheelResidues[i];
            if (v >= span) break;
            if (canonicalBitSet(src, v)) b |= static_cast<uint8_t>(1u << i);
        }
        dst[k] = b;
    }
}

}  // namespace ff
