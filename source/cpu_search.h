#ifndef FF_CPU_SEARCH_H
#define FF_CPU_SEARCH_H
#include "gpu_prime.h"
#include <cstdint>
#include <cmath>

class FreudenthalTools {
public:
    FreudenthalTools(uint64_t productLimit, const GpuPrime& primes,
                     const uint32_t* smallPrimes, uint32_t smallPrimeCount);
    ~FreudenthalTools();
    bool ProductOfTermPairsHasSingleFactorPair(uint64_t sum) const;
    bool AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(uint32_t power2, uint64_t oddProduct) const;
private:
    uint64_t productLimit_;
    const GpuPrime& primes_;
    const uint32_t* smallPrimes_;
    uint32_t smallPrimeCount_;
    uint64_t* factors_;
};

class Sammy2Loopy {
public:
    Sammy2Loopy(uint64_t sumLimit, const FreudenthalTools& fre, const GpuPrime& primes,
                const uint32_t* smallPrimes, uint32_t smallPrimeCount);
    ~Sammy2Loopy();
    bool DoesSammyKnow(uint64_t sum);
    bool DoesPeterKnow(uint32_t power2, uint64_t oddPartOfEven, uint64_t odd, uint64_t oddProduct);
    uint64_t termA() const { return termA_; }
    uint64_t termB() const { return termB_; }
private:
    uint64_t sumLimit_, termA_, termB_;
    uint32_t numProductsWithOnlyOneMultipleFactorPair_;
    bool termsFound_;
    uint32_t* compositePower2_;
    uint32_t numComposite_;
    const FreudenthalTools& fre_;
    const GpuPrime& primes_;
    const uint32_t* smallPrimes_;
    uint32_t smallPrimeCount_;

    uint32_t Power2Prime(uint64_t sum);
    uint32_t Power2Odd(uint64_t sum);
    uint32_t Power2Composite(uint64_t sum);
    uint32_t Power2Even(uint64_t sum);
};

void RunIt(const GpuPrime& prime, uint64_t sumStart, uint64_t sumLimit, uint64_t productLimit,
            const uint32_t* smallPrimes, uint32_t smallPrimeCount, int& countOut);

// PrintOutputTags is provided by m4/gpu_search_emission.cpp (included into main.cpp).
// cpu_search.cpp uses it for the CPU Freudenthal path — same byte-exact body.

#endif