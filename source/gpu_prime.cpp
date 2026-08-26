#include "gpu_prime.h"

GpuPrime::GpuPrime(const uint8_t* primeMap, uint64_t maxMapValue)
    : primeMap_(primeMap), maxPrimeMapValue_(maxMapValue) {}

GpuPrime::~GpuPrime() {}

GpuPrime::Boolean GpuPrime::IsPrime(uint64_t n) const {
#ifndef NDEBUG
    debugReads_.fetch_add(1, std::memory_order_relaxed);
#endif
    // Reference (segmentedSieve.C:281-294): n==2 -> True, even -> False,
    // n > maxPrimeMapValue -> Miller-Rabin, else map lookup. Even numbers
    // share a map bit with the following odd and are never cleared, so the
    // parity checks MUST precede the map lookup.
    if (n == 2) return True;
    if (!(n & 1)) return False;
    if (n > maxPrimeMapValue_) return AskMillerRabin(n);
    return (primeMap_[n >> 4] & (0x80 >> (n >> 1 & 7))) ? True : False;
}

inline GpuPrime::Boolean GpuPrime::AskMillerRabin(uint64_t n) const {
    // Reference (segmentedSieve.C:354-377) with EXACT thresholds/bases.
    // Invariant: n is odd and n > maxPrimeMapValue >= 3.
    static const uint64_t primeTestList[] = {2, 3, 5, 7, 11, 13, 17};
    uint32_t power2 = __builtin_ctzll(n >> 1) + 1;
    uint64_t nDiv2Odd = n >> power2;
    // For 'small' numbers, only the first few bases need to be checked.
    uint32_t maxTests =
        n < 3215031751ull ? 4 : n < 2152302898747ull ? 5 : n < 3474749660383ull ? 6 : 7;
    for (uint32_t i = 0; i < maxTests; ++i) {
        uint64_t p = primeTestList[i];
        uint64_t x = ModularPower(p, nDiv2Odd, n);
        if (x != 1 && x != n - 1) {
            for (uint32_t r = 1; r < power2; ++r) {
                x = ModularMulL(x, x, n);
                if (x == 1) return False;
                if (x == n - 1) break;
            }
            if (x != n - 1) return False;
        }
    }
    return True;
}

uint64_t GpuPrime::ModularMulL(uint64_t a, uint64_t b, uint64_t modulus) {
    __uint128_t result = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(result % (__uint128_t)modulus);
}

uint64_t GpuPrime::ModularPowerL(uint64_t base, uint64_t exp, uint64_t modulus) {
    uint64_t result = 1;
    base %= modulus;
    while (exp > 0) {
        if (exp & 1) {
            result = ModularMulL(result, base, modulus);
        }
        exp >>= 1;
        base = ModularMulL(base, base, modulus);
    }
    return result;
}

uint64_t GpuPrime::ModularPower(uint64_t base, uint64_t exp, uint64_t modulus) {
    // Reference (segmentedSieve.C:342-352): plain 64-bit path when
    // modulus < 2^32 (products stay < 2^64), ModularMulL path otherwise.
    if (modulus >= uint64_t(1) << 32) return ModularPowerL(base, exp, modulus);
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) result = (result * base) % modulus;
        exp >>= 1;
        base = (base * base) % modulus;
    }
    return result;
}