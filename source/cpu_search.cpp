#include "cpu_search.h"
#include <cstdio>
#include <cmath>
#include <thread>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>

// PrintOutputTags is defined in m4/gpu_search_emission.cpp (included into main.cpp)
// and used by cpu_search.cpp for the CPU Freudenthal path.
void PrintOutputTags(const GpuPrime& prime, uint64_t term, bool isLast);

// ---- utility ----

uint64_t isqrt64(uint64_t x) {
    if (x == 0) return 0;
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(x)));
    while ((r + 1) * (r + 1) <= x) ++r;
    while (r * r > x) --r;
    return r;
}

// ============================================================
// FreudenthalTools
// ============================================================

FreudenthalTools::FreudenthalTools(uint64_t productLimit, const GpuPrime& primes,
                                    const uint32_t* smallPrimes, uint32_t smallPrimeCount)
    : productLimit_(productLimit > 3 ? productLimit + 1 : 4),
      primes_(primes),
      smallPrimes_(smallPrimes),
      smallPrimeCount_(smallPrimeCount)
{
    const uint64_t greatestNumberPrimeFactors =
        static_cast<uint64_t>(std::log(double(productLimit_)) / std::log(double(2)) + 0.5);
    const uint64_t greatestNumberFactors = greatestNumberPrimeFactors * (greatestNumberPrimeFactors - 1) * 2;
    factors_ = new uint64_t[greatestNumberFactors];
}

FreudenthalTools::~FreudenthalTools() {
    delete[] factors_;
}

bool FreudenthalTools::ProductOfTermPairsHasSingleFactorPair(uint64_t sum) const {
    // Even sums always satisfy the condition (Goldbach's conjecture)
    if (!(sum & 1)) return true;
    // Odd sums: only one valid factor pair iff sum-2 is prime
    return primes_.IsPrime(sum - 2);
}

bool FreudenthalTools::AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
    uint32_t power2, uint64_t oddProduct) const
{
    uint32_t primeIndex = 1,
             primeFactor = 3,
             numFactors = 0,
             numPendingFactors = 0,
             numMultipleFactorPairs = 0;
    uint64_t product = uint64_t(oddProduct) << power2,
             testProduct = oddProduct,
             temp = testProduct / 3,
             testFactor = 3,
             factorLimit = isqrt64(product);

    factors_[0] = uint64_t(1) << power2;
    if (factors_[0] <= factorLimit)
        numMultipleFactorPairs += !ProductOfTermPairsHasSingleFactorPair(factors_[numFactors++] + oddProduct);

    for (;;) {
        if (temp * primeFactor == testProduct) {
            if (testFactor <= factorLimit) {
                if ((numMultipleFactorPairs += !ProductOfTermPairsHasSingleFactorPair(
                        testFactor + product / testFactor)) > 1)
                    return false;
                factors_[numFactors + numPendingFactors++] = testFactor;
                for (uint32_t f = 0; f < numFactors; ++f) {
                    uint64_t multipleSave = factors_[f] * testFactor;
                    if (multipleSave <= factorLimit) {
                        if ((numMultipleFactorPairs +=
                                !ProductOfTermPairsHasSingleFactorPair(multipleSave + product / multipleSave)) > 1)
                            return false;
                        factors_[numFactors + numPendingFactors++] = multipleSave;
                    } else {
                        // factors[] is not sorted but subsets are; skip entries
                        for (uint64_t factor = factors_[f];
                             f + 1 < numFactors && factor <= factors_[f + 1]; ++f)
                            ; // skip until next sorted subset
                    }
                }
                if (numMultipleFactorPairs > 1) return false;
            }
            testProduct = temp;
            temp /= primeFactor;
            testFactor *= primeFactor;
        } else {
            if (testProduct == 1) break;
            if (primes_.IsPrime(testProduct)) {
                if (testProduct > factorLimit) break;
                primeFactor = static_cast<uint32_t>(testProduct);
                temp = 1;
            } else {
                primeFactor = smallPrimes_[++primeIndex]; // advance to next prime (reference primes[++primeIndex])
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

// ============================================================
// Sammy2Loopy
// ============================================================

Sammy2Loopy::Sammy2Loopy(uint64_t sumLimit, const FreudenthalTools& fre, const GpuPrime& primes,
                          const uint32_t* smallPrimes, uint32_t smallPrimeCount)
    : sumLimit_(sumLimit),
      termA_(0),
      termB_(0),
      numProductsWithOnlyOneMultipleFactorPair_(0),
      termsFound_(false),
      compositePower2_(new uint32_t[uint32_t(std::log(double(sumLimit - 2)) / std::log(double(2)) + 0.5)]),
      numComposite_(0),
      fre_(fre),
      primes_(primes),
      smallPrimes_(smallPrimes),
      smallPrimeCount_(smallPrimeCount)
{
}

Sammy2Loopy::~Sammy2Loopy() {
    delete[] compositePower2_;
}

bool Sammy2Loopy::DoesSammyKnow(uint64_t sum) {
    numProductsWithOnlyOneMultipleFactorPair_ = 0;
    termsFound_ = false;

    uint32_t r;
    if ((r = Power2Prime(sum)) <= 1)
        if ((r = Power2Odd(sum)) <= 1)
            if ((r = Power2Composite(sum)) <= 1)
                Power2Even(sum);
    return numProductsWithOnlyOneMultipleFactorPair_ == 1;
}

bool Sammy2Loopy::DoesPeterKnow(uint32_t power2, uint64_t oddPartOfEven, uint64_t odd, uint64_t oddProduct) {
    // Note: oddPartOfEven != 1 (never invoked if term is an exact power of 2)
    if (!fre_.ProductOfTermPairsHasSingleFactorPair((uint64_t(1) << power2) + oddProduct))
        return false;
    if (oddPartOfEven != odd &&
        !fre_.ProductOfTermPairsHasSingleFactorPair(oddPartOfEven + (uint64_t(1) << power2) * odd))
        return false;
    return fre_.AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(power2, oddProduct);
}

// Power of 2 prime: sum = 2^k + p  (evenTerm is a power of 2, oddTerm is prime)
uint32_t Sammy2Loopy::Power2Prime(uint64_t sum) {
    numComposite_ = 0;
    for (uint64_t evenTerm = 4, power2 = 2; evenTerm < sum - 2; evenTerm <<= 1, ++power2) {
        uint64_t oddTerm = sum - evenTerm;
        if (primes_.IsPrime(oddTerm)) {
            if (++numProductsWithOnlyOneMultipleFactorPair_ > 1) break;
            if (!termsFound_ && numProductsWithOnlyOneMultipleFactorPair_ == 1) {
                termA_ = evenTerm;
                termB_ = oddTerm;
                termsFound_ = true;
            }
        } else {
            compositePower2_[numComposite_++] = static_cast<uint32_t>(power2);
        }
    }
    return numProductsWithOnlyOneMultipleFactorPair_;
}

// Odd powers of 2: sum = (oddPartOfEven * 2^k) + oddTerm, where k is odd
uint32_t Sammy2Loopy::Power2Odd(uint64_t sum) {
    uint32_t power2;
    for (power2 = 2; 5u << power2 < sum - 2; ++power2)
        ;
    power2 = (power2 - 2) | 1; // odd turn for power2-1

    for (; power2 >= 2; power2 -= 2) {
        for (uint64_t oddPartOfEven = 5;
             (oddPartOfEven << power2) < sum - 2;
             oddPartOfEven += 2) {
            uint64_t evenTerm = oddPartOfEven << power2;
            uint64_t oddTerm = sum - evenTerm;
            uint64_t oddProd = uint64_t(oddPartOfEven) * oddTerm;
            if (oddProd % 3) { // can't have odd power of two and be multiple of 3
                if ((numProductsWithOnlyOneMultipleFactorPair_ +=
                         DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd)) > 1)
                    break;
                if (!termsFound_ && numProductsWithOnlyOneMultipleFactorPair_ == 1) {
                    termA_ = evenTerm;
                    termB_ = oddTerm;
                    termsFound_ = true;
                }
            }
        }
    }
    return numProductsWithOnlyOneMultipleFactorPair_;
}

// Power of 2 composite: sum = 2^k + compositeOddTerm (k is even, not in Power2Prime list)
uint32_t Sammy2Loopy::Power2Composite(uint64_t sum) {
    int32_t nc = numComposite_ - 1;
    while (nc >= 0) {
        int32_t idx = nc;
        --nc;
        uint64_t evenTerm = uint64_t(1) << compositePower2_[idx];
        uint64_t oddTerm = sum - evenTerm;
        if ((numProductsWithOnlyOneMultipleFactorPair_ +=
                 fre_.AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair(
                     compositePower2_[idx], oddTerm)) > 1)
            break;
        if (!termsFound_ && numProductsWithOnlyOneMultipleFactorPair_ == 1) {
            termA_ = evenTerm;
            termB_ = oddTerm;
            termsFound_ = true;
        }
    }
    return numProductsWithOnlyOneMultipleFactorPair_;
}

// Even powers of 2: sum = (oddPartOfEven * 2^k) + oddTerm, k even
uint32_t Sammy2Loopy::Power2Even(uint64_t sum) {
    uint32_t power2;
    for (power2 = 2; 3u << power2 < sum - 2; ++power2)
        ;
    power2 = (power2 - 1) & ~uint32_t(1); // even things up

    for (; power2 >= 2; power2 -= 2) {
        for (uint64_t oddPartOfEven = 3;
             (oddPartOfEven << power2) < sum - 2;
             oddPartOfEven += 2) {
            uint64_t evenTerm = oddPartOfEven << power2;
            uint64_t oddTerm = sum - evenTerm;
            uint64_t oddProd = uint64_t(oddPartOfEven) * oddTerm;
            if (!(oddProd % 3) || primes_.IsPrime(oddTerm)) {
                if ((numProductsWithOnlyOneMultipleFactorPair_ +=
                         DoesPeterKnow(power2, oddPartOfEven, oddTerm, oddProd)) > 1)
                    break;
                if (!termsFound_ && numProductsWithOnlyOneMultipleFactorPair_ == 1) {
                    termA_ = evenTerm;
                    termB_ = oddTerm;
                    termsFound_ = true;
                }
            }
        }
    }
    return numProductsWithOnlyOneMultipleFactorPair_;
}

// ============================================================
// OutputTags / RunIt
// ============================================================

// PrintOutputTags is defined in m4/gpu_search_emission.cpp and included
// into main.cpp.  The CPU path uses that same definition — the bodies are
// byte-identical (verified by the emission code's documentation).

// Format a result line without the counter prefix so per-thread buffers can
// be merged by the caller.  Byte-identical to the single-threaded printf
// sequence below.
static std::string FormatFreudenthalLine(const GpuPrime& prime, uint64_t sum,
                                          uint64_t lowTerm, uint64_t highTerm) {
    char buf[512];
    int n = std::snprintf(buf, sizeof(buf),
        " sum =%9llu, product =%16llu,  low term =%9llu",
        (unsigned long long)sum,
        (unsigned long long)(lowTerm * highTerm),
        (unsigned long long)lowTerm);

    // lowTerm tag (isLast == true: trailing comma-padded separator)
    if (prime.IsPrime(lowTerm)) {
        std::strncat(buf, " (prime), ", sizeof(buf) - (size_t)n - 1);
    } else if ((-lowTerm & lowTerm) == lowTerm) {
        double power2 = std::floor(std::log(double(lowTerm)) / std::log(double(2)) + 0.5);
        std::snprintf(buf + n, sizeof(buf) - (size_t)n, " (2^%.0f)", power2);
        int width = 3 - static_cast<int>(std::floor(std::log10(power2 + 0.001)));
        if (width < 1) width = 1;
        std::snprintf(buf + (int)std::strlen(buf), sizeof(buf) - (size_t)std::strlen(buf),
                      ",%*s", width, "");
    } else {
        std::strncat(buf, ",         ", sizeof(buf) - (size_t)n - 1);
    }

    // high term label
    std::snprintf(buf + (int)std::strlen(buf), sizeof(buf) - (size_t)std::strlen(buf),
        "high term =%9llu", (unsigned long long)highTerm);

    // highTerm tag (isLast == false: no trailing comma; composite ⇒ no tag)
    if (prime.IsPrime(highTerm)) {
        std::strncat(buf, " (prime)", sizeof(buf) - (size_t)std::strlen(buf) - 1);
    } else if ((-highTerm & highTerm) == highTerm) {
        double power2 = std::floor(std::log(double(highTerm)) / std::log(double(2)) + 0.5);
        std::snprintf(buf + (int)std::strlen(buf), sizeof(buf) - (size_t)std::strlen(buf),
                      " (2^%.0f)", power2);
    }

    return std::string(buf);
}

// Single-threaded core: iterate sums and print directly.  Used as a
// fallback when threading is disabled or the range is trivially small.
static void RunItSingle(const GpuPrime& prime, uint64_t sumStart, uint64_t sumLimit,
                        uint64_t productLimit, const uint32_t* smallPrimes,
                        uint32_t smallPrimeCount, int& countOut) {
    FreudenthalTools fre(productLimit, prime, smallPrimes, smallPrimeCount);
    Sammy2Loopy sammy(sumLimit, fre, prime, smallPrimes, smallPrimeCount);

    uint64_t count = 0;
    for (uint64_t sum = sumStart | 1; sum <= sumLimit; sum += 2) {
        uint64_t sumDiv3 = sum / 3;
        if (!fre.ProductOfTermPairsHasSingleFactorPair(sum) &&
            (sum != 3 * sumDiv3 || prime.IsPrime(sumDiv3)) &&
            sammy.DoesSammyKnow(sum))
        {
            uint64_t lowTerm = sammy.termA(),
                     highTerm = sammy.termB();
            if (lowTerm > highTerm) {
                lowTerm ^= highTerm;
                highTerm ^= lowTerm;
                lowTerm ^= highTerm;
            }

            std::printf("%7u) sum =%9llu, product =%16llu,  low term =%9llu",
                        static_cast<unsigned>(++count),
                        (unsigned long long)sum,
                        (unsigned long long)(lowTerm * highTerm),
                        (unsigned long long)lowTerm);
            PrintOutputTags(prime, lowTerm, true);
            std::printf("high term =%9llu", (unsigned long long)highTerm);
            PrintOutputTags(prime, highTerm, false);
            std::printf("\n");
        }
    }
    countOut = static_cast<int>(count);
}

// ============================================================
// Multi-threaded RunIt  (FreudenthalThreads pattern)
// ============================================================

// Per-thread work descriptor.  Mirrors ThreadData from segmentedSieve.C:682,
// but with a string buffer instead of a filename so no disk I/O is needed.
struct ThreadWork {
    uint64_t sumStart;
    uint64_t sumLimit;
    uint64_t productLimit;
    const GpuPrime* prime;
    const uint32_t* smallPrimes;
    uint32_t smallPrimeCount;
    std::vector<std::string>* outBuf;
    uint32_t localCount;
};

// Worker: iterate one [sumStart, sumLimit] range (both inclusive), appending
// formatted result lines to *outBuf.  Each physical thread runs this once for
// its front range and once for its back range (FreudenthalTwins pattern).
static void ProcessRange(const ThreadWork& w) {
    FreudenthalTools fre(w.productLimit, *w.prime, w.smallPrimes, w.smallPrimeCount);
    Sammy2Loopy sammy(w.sumLimit, fre, *w.prime, w.smallPrimes, w.smallPrimeCount);

    for (uint64_t sum = w.sumStart | 1; sum <= w.sumLimit; sum += 2) {
        uint64_t sumDiv3 = sum / 3;
        if (!fre.ProductOfTermPairsHasSingleFactorPair(sum) &&
            (sum != 3 * sumDiv3 || w.prime->IsPrime(sumDiv3)) &&
            sammy.DoesSammyKnow(sum))
        {
            uint64_t lowTerm = sammy.termA(),
                     highTerm = sammy.termB();
            if (lowTerm > highTerm) {
                lowTerm ^= highTerm;
                highTerm ^= lowTerm;
                lowTerm ^= highTerm;
            }

            w.outBuf->push_back(FormatFreudenthalLine(*w.prime, sum, lowTerm, highTerm));
        }
    }
}

void RunIt(const GpuPrime& prime, uint64_t sumStart, uint64_t sumLimit, uint64_t productLimit,
           const uint32_t* smallPrimes, uint32_t smallPrimeCount, int& countOut) {
    int threadCount = 31;
    const char* envThreads = std::getenv("FF_THREADS");
    if (envThreads) {
        int v = std::atoi(envThreads);
        if (v > 0) threadCount = v;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0) threadCount = std::min(threadCount, (int)hw);
    if (threadCount < 1) threadCount = 1;

    uint64_t oddStart = sumStart | 1;
    if (oddStart > sumLimit) {
        countOut = 0;
        return;
    }

    uint64_t numOdd = (sumLimit - oddStart) / 2 + 1;

    // Single-threaded fallback.
    if (threadCount <= 1 || numOdd <= 2) {
        RunItSingle(prime, sumStart, sumLimit, productLimit, smallPrimes, smallPrimeCount, countOut);
        return;
    }

    // ---- Multi-threaded path ----
    //
    // Two-pointer range split identical to segmentedSieve.C:696-813.
    // numChunks = 2 * numPhys gives the total number of sub-ranges;
    // each physical thread handles one front range + one back range.
    // Work is in absolute sum-value units (not odd-count units).

    const uint64_t numSum = sumLimit - sumStart + 1;
    const int numPhys = threadCount;
    const int numChunks = 2 * numPhys;
    const double delta = double(numSum) / double(numChunks);

    // Two-pointer split: front advances forward, back advances backward.
    // Both pointers start at sumStart/sumLimit respectively and only advance
    // when their raw value exceeds (or goes below) the previous value.
    uint64_t frontPrev = sumStart - 1;  // -1 so first iteration always moves
    uint64_t backPrev  = sumLimit + 1;   // +1 so first iteration always moves

    // Separate front and back buffers so we can output front ranges
    // in ascending sum order (t=0..19), then back ranges in ascending
    // sum order (t=19..0 reversed).
    std::vector<std::vector<std::string>> frontBufs(numPhys);
    std::vector<std::vector<std::string>> backBufs(numPhys);
    std::vector<std::thread> threadList;
    threadList.reserve(numPhys);

    for (int t = 0; t < numPhys; ++t) {
        // Front pointer advances forward.
        double frontRaw = t * delta + 0.5;
        uint64_t frontStart = sumStart + (uint64_t)frontRaw;
        if (frontStart > frontPrev) {
            frontPrev = frontStart;
        }

        // Back pointer advances backward.
        int t2Limit = numChunks - t - 1;
        double backRaw = t2Limit * delta + 0.5;
        uint64_t backStart = sumStart + (uint64_t)backRaw;
        if (backStart < backPrev) {
            backPrev = backStart;
        }

        // Front sub-range [frontStart, frontLimit].
        double frontLimitRaw = (t + 1) * delta - 0.5;
        uint64_t frontLimit = sumStart + (uint64_t)frontLimitRaw;
        if (frontLimit < frontStart) frontLimit = frontStart;
        if (frontLimit > sumLimit) frontLimit = sumLimit;

        // Back sub-range [backStart, backLimit].
        double backLimitRaw = (t2Limit + 1) * delta - 0.5;
        uint64_t backLimit = sumStart + (uint64_t)backLimitRaw;
        if (backLimit < backStart) backLimit = backStart;
        if (backLimit > sumLimit) backLimit = sumLimit;

        // Each physical thread processes front then back (FreudenthalTwins).
        // Capture by value to avoid dangling references to loop variables.
        ThreadWork fw_val = { frontStart, frontLimit, productLimit, &prime,
                              smallPrimes, smallPrimeCount, &frontBufs[t], 0 };
        ThreadWork bw_val = { backStart,  backLimit,  productLimit, &prime,
                              smallPrimes, smallPrimeCount, &backBufs[t], 0 };

        threadList.emplace_back([fw_val, bw_val]() mutable {
            ProcessRange(fw_val);
            ProcessRange(bw_val);
        });
    }

    for (auto& th : threadList) {
        if (th.joinable()) th.join();
    }

    // Output front ranges in thread order (ascending sum), then
    // back ranges in reversed thread order (also ascending sum).
    uint64_t globalCount = 0;
    for (int t = 0; t < numPhys; ++t) {
        for (const auto& line : frontBufs[t]) {
            std::printf("%7llu)%s\n", (unsigned long long)(++globalCount), line.c_str());
        }
    }
    for (int t = numPhys - 1; t >= 0; --t) {
        for (const auto& line : backBufs[t]) {
            std::printf("%7llu)%s\n", (unsigned long long)(++globalCount), line.c_str());
        }
    }
    countOut = static_cast<int>(globalCount);
}