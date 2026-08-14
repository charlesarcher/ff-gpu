// GPU Freudenthal Search Kernel — device-side implementation.
//
// Ports the CPU Freudenthal search (cpu_search.cpp) to the GPU for massively
// parallel per-odd-sum evaluation. Each thread handles one odd sum value
// independently. No thread communication, no __constant__ memory, no FP64.
//
// The kernel body is defined here without a #ifdef guard — matching the
// sieve_slab_kernel.h pattern. hipcc compiles the __global__ function in
// device mode and the host entry point (SearchKernelRun_<arch>) in host mode,
// using hipLaunchKernelGGL to invoke it.
//
// Per-arch symbol rename: SEARCH_KERNEL pastes SIEVE_KERNEL_ARCH
// (gfx1201 or sm_120) into the __global__ symbol name.

#ifndef FF_GPU_SEARCH_KERNEL_H
#define FF_GPU_SEARCH_KERNEL_H

#include <cstdint>

// ---- Per-arch symbol rename (mirrors sieve_slab_kernel.h pattern) ----
// When SIEVE_KERNEL_ARCH is defined, SEARCH_KERNEL expands to the arch-tagged
// __global__ symbol name (e.g., FFSearchKernel_gfx1201).

#define FF_KERN_CAT2(a, b) a##b
#define FF_KERN_CAT(a, b)  FF_KERN_CAT2(a, b)

#ifdef SIEVE_KERNEL_ARCH
#define SEARCH_KERNEL FF_KERN_CAT(FFSearchKernel_, SIEVE_KERNEL_ARCH)
#endif

// ---- Constants ----

#define MAX_FACTORS                4096
#define MAX_COMP_MAX_POWER2        64

// ---- GpuRecord: one solution per record slot (host-visible) ----
typedef struct {
    uint32_t sum;     // 4 bytes
    uint64_t low;     // 8 bytes (offset 8, 8-byte aligned)
    uint64_t high;    // 8 bytes (offset 16, 8-byte aligned)
    uint8_t  tag;     // 1 byte  (offset 24, padded to 24-byte struct)
} GpuRecord;

// ================================================================
// Device-side code: only compiled when SIEVE_KERNEL_ARCH is defined.
// ================================================================

#ifdef SIEVE_KERNEL_ARCH

// Integer square root of a 64-bit value (Newton's method, integer only).
__device__ static uint64_t dev_isqrt64(uint64_t x) {
    if (x == 0) return 0;
    uint64_t r = (uint64_t)1 << 31;
    while (r * r > x) r >>= 1;
    uint64_t next = (r + x / r) >> 1;
    if (next < r) r = next;
    return r;
}

// Power-of-2 test: (-x & x) == x iff x is a power of 2 and x > 0.
__device__ static bool dev_IsPowerOf2(uint64_t x) {
    return x > 0 && ((-x) & x) == x;
}

// 64-bit modular multiplication using 128-bit intermediate (GPU-supported).
__device__ static uint64_t dev_ModularMulL(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(r % (__uint128_t)m);
}

// 64-bit modular exponentiation via repeated squaring.
__device__ static uint64_t dev_ModularPowerL(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = dev_ModularMulL(result, base, m);
        exp >>= 1;
        base = dev_ModularMulL(base, base, m);
    }
    return result;
}

// Bit-test primality: lookup in primeMap for n <= maxPrimeMapValue.
// For n > maxPrimeMapValue, falls back to Miller-Rabin with the same
// witness set and thresholds as GpuPrime::AskMillerRabin.
__device__ static bool dev_IsPrime(uint64_t n,
                                   const uint8_t* __restrict__ primeMap,
                                   uint64_t maxPrimeMapValue) {
    if (n == 2) return true;
    if (!(n & 1)) return false;
    if (n > maxPrimeMapValue) {
        static const uint64_t primeTestList[] = {2, 3, 5, 7, 11, 13, 17};
        uint32_t power2 = 1;
        uint64_t tmp = n >> 1;
        while ((tmp & 1) == 0) { tmp >>= 1; ++power2; }
        uint64_t nDiv2Odd = n >> power2;
        uint32_t maxTests =
            n < 3215031751ull ? 4 :
            n < 2152302898747ull ? 5 :
            n < 3474749660383ull ? 6 : 7;
        for (uint32_t i = 0; i < maxTests; ++i) {
            uint64_t p = primeTestList[i];
            uint64_t x = dev_ModularPowerL(p, nDiv2Odd, n);
            if (x != 1 && x != n - 1) {
                for (uint32_t r = 1; r < power2; ++r) {
                    x = dev_ModularMulL(x, x, n);
                    if (x == 1) return false;
                    if (x == n - 1) break;
                }
                if (x != n - 1) return false;
            }
        }
        return true;
    }
    return (primeMap[n >> 4] & (0x80 >> (n >> 1 & 7))) != 0;
}

// Product-of-term-pairs-has-single-factor-pair:
//   odd sum => sum-2 must be prime.
__device__ static bool dev_ProductOfTermPairsHasSingleFactorPair(uint64_t sum,
                                                                  const uint8_t* __restrict__ primeMap,
                                                                  uint64_t maxPrimeMapValue) {
    if (!(sum & 1)) return true;
    return dev_IsPrime(sum - 2, primeMap, maxPrimeMapValue);
}

// AllButOneProductOfTermPairsHasSingleFactorPair — the core Freudenthal
// factor enumeration. The `sum` parameter is the original odd sum (needed
// because ProductOfTermPairsHasSingleFactorPair checks n-2 is prime, and the
// CPU passes the full sum, not a partial product).
__device__ static bool dev_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
    uint32_t power2, uint64_t oddProduct, uint64_t sum,
    const uint8_t* __restrict__ primeMap, uint64_t maxPrimeMapValue)
{
    uint32_t primeFactor = 3,
             numFactors = 0,
             numPendingFactors = 0,
             numMultipleFactorPairs = 0;
    uint64_t product = ((uint64_t)1 << power2) * oddProduct,
             testProduct = oddProduct,
             temp = oddProduct / 3,
             testFactor = 3,
             factorLimit = dev_isqrt64(product);

    uint64_t factors[MAX_FACTORS];
    factors[0] = ((uint64_t)1 << power2);
    if (factors[0] <= factorLimit)
        numMultipleFactorPairs += !dev_ProductOfTermPairsHasSingleFactorPair(
            factors[0] + oddProduct, primeMap, maxPrimeMapValue);

    for (;;) {
        if (temp * primeFactor == testProduct) {
            if (testFactor <= factorLimit) {
                if ((numMultipleFactorPairs += !dev_ProductOfTermPairsHasSingleFactorPair(
                        testFactor + product / testFactor, primeMap, maxPrimeMapValue)) > 1)
                    return false;
                factors[numFactors + numPendingFactors++] = testFactor;
                for (uint32_t f = 0; f < numFactors; ++f) {
                    uint64_t multipleSave = factors[f] * testFactor;
                    if (multipleSave <= factorLimit) {
                        if ((numMultipleFactorPairs +=
                                !dev_ProductOfTermPairsHasSingleFactorPair(
                                    multipleSave + product / multipleSave,
                                    primeMap, maxPrimeMapValue)) > 1)
                            return false;
                        factors[numFactors + numPendingFactors++] = multipleSave;
                    } else {
                        for (uint64_t factor = factors[f];
                             f + 1 < numFactors && factor <= factors[f + 1]; ++f)
                            ; // skip unsorted entries
                    }
                }
                if (numMultipleFactorPairs > 1) return false;
            }
            testProduct = temp;
            temp /= primeFactor;
            testFactor *= primeFactor;
        } else {
            if (testProduct == 1) break;
            if (dev_IsPrime(testProduct, primeMap, maxPrimeMapValue)) {
                if (testProduct > factorLimit) break;
                primeFactor = (uint32_t)testProduct;
                temp = 1;
            } else {
                primeFactor = (uint32_t)(testFactor + 2);
                while (!dev_IsPrime(primeFactor, primeMap, maxPrimeMapValue))
                    primeFactor += 2;
                if (primeFactor > factorLimit) break;
                temp = testProduct / primeFactor;
            }
            numFactors += numPendingFactors;
            numPendingFactors = 0;
            testFactor = primeFactor;
        }
    }
    return numMultipleFactorPairs == 1;
}

// DoesPeterKnow: checks the Peter condition for a single tuple.
__device__ static bool dev_DoesPeterKnow(
    uint32_t power2, uint64_t oddPartOfEven, uint64_t oddTerm, uint64_t oddProduct,
    uint64_t sum,
    const uint8_t* __restrict__ primeMap, uint64_t maxPrimeMapValue)
{
    if (!dev_ProductOfTermPairsHasSingleFactorPair(
            ((uint64_t)1 << power2) + oddProduct, primeMap, maxPrimeMapValue))
        return false;
    if (oddPartOfEven != oddTerm &&
        !dev_ProductOfTermPairsHasSingleFactorPair(
            oddPartOfEven + (((uint64_t)1 << power2) * oddTerm),
            primeMap, maxPrimeMapValue))
        return false;
    return dev_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
        power2, oddProduct, sum, primeMap, maxPrimeMapValue);
}

// ---- SEARCH_KERNEL: one thread per odd sum ----

__global__ void SEARCH_KERNEL(
    const uint8_t* __restrict__  primeMap,
    uint64_t                     maxPrimeMapValue,
    uint64_t                     sumStart,
    uint64_t                     sumLimit,
    uint32_t* __restrict__       pAtomicCount,
    GpuRecord* __restrict__      pRecords)
{
    uint32_t tidx = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);

    uint64_t numOddSums = (sumLimit - sumStart) / 2 + 1;
    if ((uint64_t)tidx >= numOddSums) return;

    uint64_t sum = sumStart + (uint64_t)tidx * 2;

    // Thread-local storage.
    uint64_t factors[MAX_FACTORS];
    uint32_t compositePower2[MAX_COMP_MAX_POWER2];
    uint32_t numComposite = 0;
    uint32_t numValid = 0;
    uint64_t termA = 0, termB = 0;
    bool termsFound = false;

    // ---- Phase 1: Power2Prime ----
    for (uint64_t evenTerm = 4, power2 = 2; evenTerm < sum - 2; evenTerm <<= 1, ++power2) {
        uint64_t oddTerm = sum - evenTerm;
        if (dev_IsPrime(oddTerm, primeMap, maxPrimeMapValue)) {
            if (++numValid > 1) break;
            if (!termsFound && numValid == 1) {
                termA = evenTerm;
                termB = oddTerm;
                termsFound = true;
            }
        } else {
            if (numComposite < MAX_COMP_MAX_POWER2)
                compositePower2[numComposite++] = (uint32_t)power2;
        }
    }

    // ---- Phase 2: Power2Odd (k odd, from highest) ----
    if (numValid <= 1) {
        uint32_t power2;
        for (power2 = 2; 5u << power2 < sum - 2; ++power2)
            ;
        power2 = (power2 - 2) | 1;

        for (; power2 >= 2; power2 -= 2) {
            for (uint64_t oddPartOfEven = 5;
                 (oddPartOfEven << power2) < sum - 2;
                 oddPartOfEven += 2) {
                uint64_t evenTerm = oddPartOfEven << power2;
                uint64_t oddTerm = sum - evenTerm;
                uint64_t oddProd = oddPartOfEven * oddTerm;
                if (oddProd % 3) {
                    if ((numValid += dev_DoesPeterKnow(
                            power2, oddPartOfEven, oddTerm, oddProd, sum,
                            primeMap, maxPrimeMapValue)) > 1)
                        break;
                    if (!termsFound && numValid == 1) {
                        termA = evenTerm;
                        termB = oddTerm;
                        termsFound = true;
                    }
                }
            }
        }
    }

    // ---- Phase 3: Power2Composite ----
    if (numValid <= 1) {
        int32_t nc = (int32_t)numComposite - 1;
        while (nc >= 0) {
            int32_t idx = nc;
            --nc;
            uint64_t evenTerm = ((uint64_t)1 << compositePower2[idx]);
            uint64_t oddTerm = sum - evenTerm;
            if ((numValid += dev_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
                    compositePower2[idx], oddTerm, sum,
                    primeMap, maxPrimeMapValue)) > 1)
                break;
            if (!termsFound && numValid == 1) {
                termA = evenTerm;
                termB = oddTerm;
                termsFound = true;
            }
        }
    }

    // ---- Phase 4: Power2Even (k even, from highest) ----
    if (numValid <= 1) {
        uint32_t power2;
        for (power2 = 2; 3u << power2 < sum - 2; ++power2)
            ;
        power2 = (power2 - 1) & ~1u;

        for (; power2 >= 2; power2 -= 2) {
            for (uint64_t oddPartOfEven = 3;
                 (oddPartOfEven << power2) < sum - 2;
                 oddPartOfEven += 2) {
                uint64_t evenTerm = oddPartOfEven << power2;
                uint64_t oddTerm = sum - evenTerm;
                uint64_t oddProd = oddPartOfEven * oddTerm;
                if (!(oddProd % 3) || dev_IsPrime(oddTerm, primeMap, maxPrimeMapValue)) {
                    if ((numValid += dev_DoesPeterKnow(
                            power2, oddPartOfEven, oddTerm, oddProd, sum,
                            primeMap, maxPrimeMapValue)) > 1)
                        break;
                    if (!termsFound && numValid == 1) {
                        termA = evenTerm;
                        termB = oddTerm;
                        termsFound = true;
                    }
                }
            }
        }
    }

    bool skipSum = false;
    if (sum & 1) {
        uint64_t sumDiv3 = sum / 3;
        // Condition 1: ProductOfTermPairsHasSingleFactorPair must be false.
        // For odd sums, this means sum-2 must NOT be prime.
        if (dev_IsPrime(sum - 2, primeMap, maxPrimeMapValue))
            skipSum = true;
        // Condition 2: if sum is divisible by 3, sum/3 must be prime.
        else if (sum == 3 * sumDiv3 && !dev_IsPrime(sumDiv3, primeMap, maxPrimeMapValue))
            skipSum = true;
    }

    // ---- Write result if exactly one valid decomposition ----
    if (numValid == 1 && termsFound && !skipSum) {
        uint64_t lo = termA, hi = termB;
        if (lo > hi) { uint64_t tmp = lo; lo = hi; hi = tmp; }

        uint32_t count = atomicAdd(pAtomicCount, 1);
        GpuRecord rec;
        rec.sum = (uint32_t)sum;
        rec.low = lo;
        rec.high = hi;
        rec.tag = 0;
        pRecords[count] = rec;
    }
}

#endif  // SIEVE_KERNEL_ARCH

#endif  // FF_GPU_SEARCH_KERNEL_H