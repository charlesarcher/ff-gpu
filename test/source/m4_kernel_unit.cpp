// m4_kernel_unit.cpp — Unit test for the GPU Freudenthal search kernel.
//
// Verifies the GPU kernel output against a CPU reference (DoesSammyKnow) for
// sampled sums in [sumStart, sumLimit]. Uses DevAbstraction for all device
// operations (alloc, copy, launch).
//
// Build: make m4 && ./tests/m4_kernel_unit

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "devabstraction.h"
#include "m4/gpu_search_kernel.h"

// ---- Error handling ----

#define CHECK(expr) \
    do { \
        int _rc = (expr); \
        if (_rc != 0) { \
            std::fprintf(stderr, "CHECK FAILED: %s (rc=%d) at %s:%d\n", \
                         #expr, _rc, __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

// ---- CPU Miller-Rabin primality test (reference) ----

static uint64_t cpu_ModularMulL(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(r % (__uint128_t)m);
}

static uint64_t cpu_ModularPowerL(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = cpu_ModularMulL(result, base, m);
        exp >>= 1;
        base = cpu_ModularMulL(base, base, m);
    }
    return result;
}

static bool cpu_IsPrime(uint64_t n, const uint8_t* primeMap, uint64_t maxPrimeMapValue) {
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
            uint64_t x = cpu_ModularPowerL(p, nDiv2Odd, n);
            if (x != 1 && x != n - 1) {
                for (uint32_t r = 1; r < power2; ++r) {
                    x = cpu_ModularMulL(x, x, n);
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

// ---- CPU small-prime sieve ----

// Integer square root of a 64-bit value.
// Bit-by-bit long-division port of the reference isqrt64 (segmentedSieve.C:37-52).
static uint64_t cpu_isqrt64(uint64_t arg) {
    uint64_t result = 0, bitsToShift = 8 * sizeof(uint64_t), testValue = 0;
    do {
        testValue = (testValue << 2) | (arg >> (bitsToShift -= 2) & 3);
        result <<= 1;
        uint64_t divisor = result << 1;
        if (divisor < testValue) {
            result |= 1;
            testValue -= divisor | 1;
        }
    } while (bitsToShift);
    return result;
}

static std::vector<uint32_t> cpu_generateSmallPrimes(uint64_t limit) {
    if (limit < 2) return {2u};
    uint64_t sqrtLimit = cpu_isqrt64(limit);
    if (sqrtLimit < 2) sqrtLimit = 2;
    uint64_t mapSize = (sqrtLimit + 15) >> 4;
    std::vector<uint8_t> map(mapSize, 0xff);
    map[0] ^= 0x80;
    for (uint64_t p = 3; p <= cpu_isqrt64(sqrtLimit); p += 2)
        if (map[p >> 4] & (0x80 >> (p >> 1 & 7)))
            for (uint64_t i = p * p; i <= sqrtLimit; i += p << 1)
                map[i >> 4] &= ~(0x80 >> (i >> 1 & 7));
    std::vector<uint32_t> primes;
    primes.push_back(2u);
    for (uint64_t p = 3; p <= sqrtLimit; p += 2)
        if (map[p >> 4] & (0x80 >> (p >> 1 & 7)))
            primes.push_back(static_cast<uint32_t>(p));
    return primes;
}

// ---- CPU Freudenthal search (reference) ----

static uint64_t cpu_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
    uint32_t power2, uint64_t oddProduct,
    const uint8_t* primeMap, uint64_t maxPrimeMapValue)
{
    uint32_t compositePrime = 3, // cursor: last prime taken from the prime list (reference primes[primeIndex])
             primeFactor = 3,
             numFactors = 0,
             numPendingFactors = 0,
             numMultipleFactorPairs = 0;
    uint64_t product = ((uint64_t)1 << power2) * oddProduct,
             testProduct = oddProduct,
             temp = oddProduct / 3,
             testFactor = 3,
             factorLimit = cpu_isqrt64(product);

    uint64_t factors[256]; // CPU test only needs small factor lists
    factors[0] = ((uint64_t)1 << power2);
    if (factors[0] <= factorLimit)
        numMultipleFactorPairs += !(cpu_IsPrime(factors[numFactors++] + oddProduct - 2, primeMap, maxPrimeMapValue));

    for (;;) {
        if (temp * primeFactor == testProduct) {
            if (testFactor <= factorLimit) {
                if ((numMultipleFactorPairs += !(cpu_IsPrime(testFactor + product / testFactor - 2, primeMap, maxPrimeMapValue))) > 1)
                    return false;
                factors[numFactors + numPendingFactors++] = testFactor;
                for (uint32_t f = 0; f < numFactors; ++f) {
                    uint64_t ms = factors[f] * testFactor;
                    if (ms <= factorLimit) {
                        if ((numMultipleFactorPairs += !(cpu_IsPrime(ms + product / ms - 2, primeMap, maxPrimeMapValue))) > 1)
                            return false;
                        factors[numFactors + numPendingFactors++] = ms;
                    } else {
                        for (uint64_t factor = factors[f];
                             f + 1 < numFactors && factor <= factors[f + 1]; ++f)
                            ;
                    }
                }
                if (numMultipleFactorPairs > 1) return false;
            }
            testProduct = temp;
            temp /= primeFactor;
            testFactor *= primeFactor;
        } else {
            if (testProduct == 1) break;
            if (cpu_IsPrime(testProduct, primeMap, maxPrimeMapValue)) {
                if (testProduct > factorLimit) break;
                primeFactor = (uint32_t)testProduct;
                temp = 1;
            } else {
                // Advance through the prime list in order (reference: primeFactor = primes[++primeIndex]),
                // independent of testFactor, which accumulates prime powers in the if-branch.
                compositePrime += 2;
                while (!cpu_IsPrime(compositePrime, primeMap, maxPrimeMapValue))
                    compositePrime += 2;
                primeFactor = compositePrime;
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

static bool cpu_DoesPeterKnow(
    uint32_t power2, uint64_t oddPartOfEven, uint64_t oddTerm, uint64_t oddProduct,
    const uint8_t* primeMap, uint64_t maxPrimeMapValue)
{
    if (!(cpu_IsPrime(((uint64_t)1 << power2) + oddProduct - 2, primeMap, maxPrimeMapValue)))
        return false;
    if (oddPartOfEven != oddTerm &&
        !(cpu_IsPrime(oddPartOfEven + (((uint64_t)1 << power2) * oddTerm) - 2, primeMap, maxPrimeMapValue)))
        return false;
    return cpu_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
        power2, oddProduct, primeMap, maxPrimeMapValue);
}

static bool cpu_DoesSammyKnow(uint64_t sum, const uint8_t* primeMap, uint64_t maxPrimeMapValue) {
    // Reference RunIt filter (segmentedSieve.C:797), mirrored by the GPU kernel's skipSum:
    // skip odd sums where sum-2 is prime (single factor pair), or where sum == 3*sumDiv3
    // and sumDiv3 is composite.
    uint64_t sumDiv3 = sum / 3;
    if ((sum & 1) && (cpu_IsPrime(sum - 2, primeMap, maxPrimeMapValue) ||
        (sum == 3 * sumDiv3 && !cpu_IsPrime(sumDiv3, primeMap, maxPrimeMapValue))))
        return false;

    uint32_t numValid = 0;
    uint64_t termA = 0, termB = 0;
    bool termsFound = false;

    // Power2Prime
    uint32_t compositePower2[64];
    uint32_t numComposite = 0;
    for (uint64_t evenTerm = 4, power2 = 2; evenTerm < sum - 2; evenTerm <<= 1, ++power2) {
        uint64_t oddTerm = sum - evenTerm;
        if (cpu_IsPrime(oddTerm, primeMap, maxPrimeMapValue)) {
            if (++numValid > 1) break;
            if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
        } else {
            if (numComposite < 64) compositePower2[numComposite++] = (uint32_t)power2;
        }
    }
    if (numValid <= 1) {
        // Power2Odd
        uint32_t power2;
        for (power2 = 2; 5u << power2 < sum - 2; ++power2) ;
        power2 = (power2 - 2) | 1;
        for (; power2 >= 2; power2 -= 2) {
            for (uint64_t oddPartOfEven = 5;
                 (oddPartOfEven << power2) < sum - 2;
                 oddPartOfEven += 2) {
                uint64_t evenTerm = oddPartOfEven << power2;
                uint64_t oddTerm = sum - evenTerm;
                uint64_t oddProd = oddPartOfEven * oddTerm;
                if (oddProd % 3) {
                    if ((numValid += cpu_DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd, primeMap, maxPrimeMapValue)) > 1) break;
                    if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
                }
            }
        }
    }
    if (numValid <= 1) {
        // Power2Composite
        int32_t nc = (int32_t)numComposite - 1;
        while (nc >= 0) {
            int32_t idx = nc; --nc;
            uint64_t evenTerm = ((uint64_t)1 << compositePower2[idx]);
            uint64_t oddTerm = sum - evenTerm;
            if ((numValid += cpu_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
                    compositePower2[idx], oddTerm, primeMap, maxPrimeMapValue)) > 1) break;
            if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
        }
    }
    if (numValid <= 1) {
        // Power2Even
        uint32_t power2;
        for (power2 = 2; 3u << power2 < sum - 2; ++power2) ;
        power2 = (power2 - 1) & ~1u;
        for (; power2 >= 2; power2 -= 2) {
            for (uint64_t oddPartOfEven = 3;
                 (oddPartOfEven << power2) < sum - 2;
                 oddPartOfEven += 2) {
                uint64_t evenTerm = oddPartOfEven << power2;
                uint64_t oddTerm = sum - evenTerm;
                uint64_t oddProd = oddPartOfEven * oddTerm;
                if (!(oddProd % 3) || cpu_IsPrime(oddTerm, primeMap, maxPrimeMapValue)) {
                    if ((numValid += cpu_DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd, primeMap, maxPrimeMapValue)) > 1) break;
                    if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
                }
            }
        }
    }
    return numValid == 1 && termsFound;
}

// ---- Device kernel launch (per-arch, extern "C") ----

extern "C" {
    extern int SearchKernelRun_gfx1201(int deviceIndex,
                                       const uint8_t* d_primeMap, uint64_t d_maxPrimeMapValue,
                                       uint64_t d_sumStart, uint64_t d_sumLimit,
                                       uint32_t* d_pAtomicCount, GpuRecord* d_pRecords);
    extern int SearchKernelRun_sm_120(int deviceIndex,
                                      const uint8_t* d_primeMap, uint64_t d_maxPrimeMapValue,
                                      uint64_t d_sumStart, uint64_t d_sumLimit,
                                      uint32_t* d_pAtomicCount, GpuRecord* d_pRecords);
}

typedef int (*SearchKernelRunFn)(int, const uint8_t*, uint64_t,
                                  uint64_t, uint64_t,
                                  uint32_t*, GpuRecord*);

extern "C" SearchKernelRunFn SearchKernelGetLaunchFn_gfx1201(int deviceIndex);
extern "C" SearchKernelRunFn SearchKernelGetLaunchFn_sm_120(int deviceIndex);

// ---- Test: verify kernel output against CPU reference ----

struct ValidationResult {
    uint64_t sum;
    uint64_t gpuLow, gpuHigh;
    uint64_t cpuLow, cpuHigh;
    bool gpuHasResult;
    bool cpuHasResult;
    bool match;
};

static ValidationResult validateSum(uint64_t sum, const uint8_t* d_records, uint32_t dCount,
                                     const uint8_t* h_primeMap, uint64_t h_maxPrimeMapValue) {
    ValidationResult vr;
    vr.sum = sum;
    vr.gpuHasResult = false;
    vr.cpuHasResult = false;
    vr.gpuLow = vr.gpuHigh = vr.cpuLow = vr.cpuHigh = 0;
    vr.match = true;

    // Check GPU result.
    for (uint32_t i = 0; i < dCount; ++i) {
        const GpuRecord* rec = (const GpuRecord*)(d_records + i * sizeof(GpuRecord));
        if (rec->sum == (uint32_t)sum) {
            vr.gpuHasResult = true;
            vr.gpuLow = rec->low;
            vr.gpuHigh = rec->high;
            break;
        }
    }

    // Check CPU result.
    if (cpu_DoesSammyKnow(sum, h_primeMap, h_maxPrimeMapValue)) {
        vr.cpuHasResult = true;
        // Find the actual terms via the same algorithm.
        uint32_t numValid = 0;
        uint64_t termA = 0, termB = 0;
        bool termsFound = false;
        uint32_t compositePower2[64];
        uint32_t numComposite = 0;
        for (uint64_t evenTerm = 4, power2 = 2; evenTerm < sum - 2; evenTerm <<= 1, ++power2) {
            uint64_t oddTerm = sum - evenTerm;
            if (cpu_IsPrime(oddTerm, h_primeMap, h_maxPrimeMapValue)) {
                if (++numValid > 1) break;
                if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
            } else {
                if (numComposite < 64) compositePower2[numComposite++] = (uint32_t)power2;
            }
        }
        if (numValid <= 1) {
            uint32_t power2;
            for (power2 = 2; 5u << power2 < sum - 2; ++power2) ;
            power2 = (power2 - 2) | 1;
            for (; power2 >= 2; power2 -= 2) {
                for (uint64_t oddPartOfEven = 5;
                     (oddPartOfEven << power2) < sum - 2;
                     oddPartOfEven += 2) {
                    uint64_t evenTerm = oddPartOfEven << power2;
                    uint64_t oddTerm = sum - evenTerm;
                    uint64_t oddProd = oddPartOfEven * oddTerm;
                    if (oddProd % 3) {
                        if ((numValid += cpu_DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd, h_primeMap, h_maxPrimeMapValue)) > 1) break;
                        if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
                    }
                }
            }
        }
        if (numValid <= 1) {
            int32_t nc = (int32_t)numComposite - 1;
            while (nc >= 0) {
                int32_t idx = nc; --nc;
                uint64_t evenTerm = ((uint64_t)1 << compositePower2[idx]);
                uint64_t oddTerm = sum - evenTerm;
                if ((numValid += cpu_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
                        compositePower2[idx], oddTerm, h_primeMap, h_maxPrimeMapValue)) > 1) break;
                if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
            }
        }
        if (numValid <= 1) {
            uint32_t power2;
            for (power2 = 2; 3u << power2 < sum - 2; ++power2) ;
            power2 = (power2 - 1) & ~1u;
            for (; power2 >= 2; power2 -= 2) {
                for (uint64_t oddPartOfEven = 3;
                     (oddPartOfEven << power2) < sum - 2;
                     oddPartOfEven += 2) {
                    uint64_t evenTerm = oddPartOfEven << power2;
                    uint64_t oddTerm = sum - evenTerm;
                    uint64_t oddProd = oddPartOfEven * oddTerm;
                    if (!(oddProd % 3) || cpu_IsPrime(oddTerm, h_primeMap, h_maxPrimeMapValue)) {
                        if ((numValid += cpu_DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd, h_primeMap, h_maxPrimeMapValue)) > 1) break;
                        if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
                    }
                }
            }
        }
        if (numValid == 1 && termsFound) {
            vr.cpuLow = termA < termB ? termA : termB;
            vr.cpuHigh = termA < termB ? termB : termA;
        }
    }

    // Validate.
    if (vr.gpuHasResult != vr.cpuHasResult) {
        vr.match = false;
    } else if (vr.gpuHasResult && vr.cpuHasResult) {
        if (vr.gpuLow != vr.cpuLow || vr.gpuHigh != vr.cpuHigh) {
            vr.match = false;
        }
        // Also verify: low + high == sum
        if (vr.gpuLow + vr.gpuHigh != sum) vr.match = false;
    }
    return vr;
}

// ---- Main test entry point ----

static int runTest(uint32_t sumStart, uint64_t sumLimit, uint32_t testInterval) {
    std::printf("\n=== GPU Search Kernel Unit Test ===\n");
    std::printf("sumStart=%u, sumLimit=%llu, interval=%u\n",
                sumStart, (unsigned long long)sumLimit, testInterval);

    // 1. Initialize device abstraction.
    CHECK(ffdev::DevInit());

    // 2. Check device availability.
    int devCount = ffdev::DevGetDeviceCount();
    if (devCount == 0) {
        std::printf("  No GPU devices available — skipping kernel test.\n");
        return 0;
    }
    std::printf("  GPU devices: %d\n", devCount);

    ff::DeviceInfo di;
    CHECK(ffdev::DevGetDeviceProperties(0, &di));
    std::printf("  Device 0: %s (%s)\n", di.name, ffdev::backendName(ffdev::DevBackendOf(0)));

    // 3. Generate CPU prime map.
    uint64_t mapBytes = (sumLimit >> 4) + 1;
    std::vector<uint8_t> h_primeMap(mapBytes, 0xff);
    h_primeMap[0] ^= 0x80; // 1 is not prime
    for (uint64_t p = 3; p * p <= sumLimit; p += 2) {
        if (h_primeMap[p >> 4] & (0x80 >> (p >> 1 & 7))) {
            for (uint64_t i = p * p; i <= sumLimit; i += p << 1)
                h_primeMap[i >> 4] &= ~(0x80 >> (i >> 1 & 7));
        }
    }

    uint64_t maxPrimeMapValue = sumLimit;

    // 4. Compute kernel launch parameters.
    uint64_t numOddSums = (sumLimit - sumStart) / 2 + 1;
    uint32_t blockSize = 256;
    uint32_t numBlocks = (uint32_t)((numOddSums + blockSize - 1) / blockSize);

    // 5. Allocate device memory.
    ffdev::DevHandle dhPrimeMap, dhAtomicCount, dhRecords;

    size_t primeMapSize = (size_t)mapBytes;
    CHECK(ffdev::DevAlloc(0, primeMapSize, &dhPrimeMap));
    CHECK(ffdev::DevAlloc(0, sizeof(uint32_t), &dhAtomicCount));
    size_t recordSize = (size_t)numOddSums * sizeof(GpuRecord);
    CHECK(ffdev::DevAlloc(0, recordSize, &dhRecords));

    // 6. Copy prime map to device.
    CHECK(ffdev::DevCopy(&dhPrimeMap, h_primeMap.data(), primeMapSize, ffdev::DevCopyDir::H2D));

    // 7. Initialize atomic count on device to 0.
    uint32_t h_atomicCount = 0;
    CHECK(ffdev::DevCopy(&dhAtomicCount, &h_atomicCount, sizeof(uint32_t), ffdev::DevCopyDir::H2D));

    // 8. Resolve kernel launch function.
    SearchKernelRunFn launchFn = SearchKernelGetLaunchFn_gfx1201(0);
    if (!launchFn) launchFn = SearchKernelGetLaunchFn_sm_120(0);
    if (!launchFn) {
        std::printf("  Kernel launch function not available — skipping.\n");
        // Clean up.
        ffdev::DevFree(&dhRecords);
        ffdev::DevFree(&dhAtomicCount);
        ffdev::DevFree(&dhPrimeMap);
        return 0;
    }

    std::printf("  Launching kernel: grid=%u blocks x %u threads, %llu odd sums\n",
                numBlocks, blockSize, (unsigned long long)numOddSums);

    // 9. Launch kernel.
    int rc = launchFn(0,
                      (const uint8_t*)dhPrimeMap.ptr, (uint64_t)maxPrimeMapValue,
                      (uint64_t)sumStart, (uint64_t)sumLimit,
                      (uint32_t*)dhAtomicCount.ptr, (GpuRecord*)dhRecords.ptr);
    if (rc != 0) {
        std::fprintf(stderr, "  Kernel launch failed (rc=%d)\n", rc);
        ffdev::DevFree(&dhRecords);
        ffdev::DevFree(&dhAtomicCount);
        ffdev::DevFree(&dhPrimeMap);
        return 1;
    }

    // 10. Sync and copy results back.
   CHECK(ffdev::DevCopy(&dhAtomicCount, &h_atomicCount, sizeof(uint32_t), ffdev::DevCopyDir::D2H));
    std::printf("  Kernel results: %u valid sums\n", h_atomicCount);

    // Copy records back.
    std::vector<GpuRecord> h_records(numOddSums);
    CHECK(ffdev::DevCopy(&dhRecords, h_records.data(), recordSize, ffdev::DevCopyDir::D2H));

    // 11. Validate against CPU reference for sampled sums.
    int nSamples = 0;
    int nMatch = 0;
    int nMismatch = 0;

    for (uint64_t sum = sumStart; sum <= sumLimit; sum += 2) {
        if ((sum - sumStart) % (2 * testInterval) != 0) continue;
        ++nSamples;

        ValidationResult vr = validateSum(sum, (const uint8_t*)dhRecords.ptr, h_atomicCount,
                                          h_primeMap.data(), maxPrimeMapValue);

        if (vr.gpuHasResult && vr.cpuHasResult) {
            if (vr.match) {
                ++nMatch;
                if (nSamples <= 20)
                    std::printf("  sum=%llu GPU=(%llu,%llu) CPU=(%llu,%llu) MATCH\n",
                                (unsigned long long)sum,
                                (unsigned long long)vr.gpuLow, (unsigned long long)vr.gpuHigh,
                                (unsigned long long)vr.cpuLow, (unsigned long long)vr.cpuHigh);
            } else {
                ++nMismatch;
                std::printf("  sum=%llu GPU=(%llu,%llu) CPU=(%llu,%llu) MISMATCH\n",
                            (unsigned long long)sum,
                            (unsigned long long)vr.gpuLow, (unsigned long long)vr.gpuHigh,
                            (unsigned long long)vr.cpuLow, (unsigned long long)vr.cpuHigh);
            }
        } else if (!vr.gpuHasResult && !vr.cpuHasResult) {
            ++nMatch;
            if (nSamples <= 20)
                std::printf("  sum=%llu no solution (GPU=%s CPU=%s) MATCH\n",
                            (unsigned long long)sum,
                            vr.gpuHasResult ? "yes" : "no",
                            vr.cpuHasResult ? "yes" : "no");
        } else {
            ++nMismatch;
            std::printf("  sum=%llu GPU=%s CPU=%s MISMATCH (agreement)\n",
                        (unsigned long long)sum,
                        vr.gpuHasResult ? "has result" : "no result",
                        vr.cpuHasResult ? "has result" : "no result");
        }
    }

    // 12. Also verify GPU-reported counts for all sums.
    //     Count sums where GPU found a result.
    uint32_t gpuFoundCount = 0;
    for (uint32_t i = 0; i < h_atomicCount; ++i) {
        uint64_t sum = h_records[i].sum;
        if (sum < sumStart || sum > sumLimit) {
            std::printf("  ERROR: record sum %llu out of range [%llu, %llu]\n",
                        (unsigned long long)sum, (unsigned long long)sumStart, (unsigned long long)sumLimit);
            ++nMismatch;
            continue;
        }
        // Verify terms sum to the sum.
        if (h_records[i].low + h_records[i].high != sum) {
            std::printf("  ERROR: record sum %llu, low+high=%llu+%llu=%llu != sum\n",
                        (unsigned long long)sum,
                        (unsigned long long)h_records[i].low,
                        (unsigned long long)h_records[i].high,
                        (unsigned long long)(h_records[i].low + h_records[i].high));
            ++nMismatch;
        }
        ++gpuFoundCount;

        // Also verify this is a valid Freudenthal result.
        ValidationResult vr = validateSum(sum, (const uint8_t*)dhRecords.ptr, h_atomicCount,
                                          h_primeMap.data(), maxPrimeMapValue);
        if (vr.cpuHasResult && !vr.match) {
            ++nMismatch;
        } else if (vr.cpuHasResult && vr.match) {
            ++nMatch;
        }
    }

    std::printf("\n=== Results ===\n");
    std::printf("Sampled: %d, Matched: %d, Mismatched: %d\n",
                nSamples + (int)gpuFoundCount, nMatch, nMismatch);
    std::printf("GPU found: %u valid Freudenthal sums in [%u, %llu]\n",
                h_atomicCount, sumStart, (unsigned long long)sumLimit);

    // Clean up device memory.
    ffdev::DevFree(&dhRecords);
    ffdev::DevFree(&dhAtomicCount);
    ffdev::DevFree(&dhPrimeMap);

    return nMismatch > 0 ? 1 : 0;
}

int main(int argc, char** argv) {
    std::printf("=== m4_kernel_unit: GPU Freudenthal search kernel test ===\n");

    uint32_t sumStart = 5;
    uint64_t sumLimit = 65535;
    uint32_t testInterval = 100; // test every 100th odd sum

    if (argc >= 2) sumStart = (uint32_t)atol(argv[1]);
    if (argc >= 3) sumLimit = (uint64_t)atol(argv[2]);
    if (argc >= 4) testInterval = (uint32_t)atol(argv[3]);

    int rc = runTest(sumStart, sumLimit, testInterval);

    if (rc == 0) {
        std::printf("\n=== m4_kernel_unit: PASS ===\n");
    } else {
        std::printf("\n=== m4_kernel_unit: FAIL ===\n");
    }
    return rc;
}