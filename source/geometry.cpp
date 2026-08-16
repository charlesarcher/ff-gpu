#include "geometry.h"

#include <cmath>

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

}  // namespace ff
