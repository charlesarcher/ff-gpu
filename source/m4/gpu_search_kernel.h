// GPU Freudenthal Search Kernel — device-side implementation.
//
// Ports the CPU Freudenthal search (cpu_search.cpp) to the GPU for massively
// parallel per-odd-sum evaluation. Each thread handles one odd sum value
// independently.
//
// Optimizations vs. baseline:
//   1. smallPrimes: pre-computed prime list in __constant__ memory replaces
//      the O(n) odd-number scanning loop with O(1) array lookups.
//   2. No __constant__ memory was originally used; this version adds it
//      specifically for the smallPrimes array.
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

#include <cassert>
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

// Task-11 audit (instrumented atomicMax over every factors[] write site):
// max observed depth = 18 across ALL odd sums of the full production leg
// [5, 2097152] (14 on the m4_kernel_unit 65535 leg). 32 = pow2 headroom.
// The previous value, 256, cost ~2 KB/thread of local memory (the kernel's
// dominant resource) for depth that is never approached. Debug asserts below
// trap any future algebra change that would outgrow the table (dev preset
// never defines NDEBUG, so these stay live in production builds).
#define MAX_FACTORS                32
#define MAX_COMP_MAX_POWER2        64

// ---- Read-only load hints (task 15) -----------------------------------------
// FF_SEARCH_LD_RO_*: explicit read-only data load for values that are
// provably never written during kernel lifetime — primeMap is produced by the
// sieve phase (a different, earlier launch) and only read here; no concurrent
// writer exists in any launch in this tree. NVIDIA backend: __ldg lowers to
// ld.global.nc (non-coherent read-only path). AMD backend: plain dereference
// — RDNA L1/L2 are read-allocate anyway and the HIP porting guide documents
// __ldg as a no-op there (draft findings); guarded so we never depend on AMD
// intrinsic availability. Hints never change loaded VALUES.
#if defined(__HIP_PLATFORM_NVIDIA__)
#define FF_SEARCH_LD_RO_U8(p)  __ldg(p)
#else
#define FF_SEARCH_LD_RO_U8(p)  (*(p))
#endif

// ---- Occupancy clamp (kernel-gap-closure task 2) ----------------------------
//
// SEARCH_KERNEL carries __launch_bounds__(256, minBlocksPerSM). MACRO FORM IS
// MANDATORY: the raw __attribute__((launch_bounds(...))) spelling silently
// no-ops on ROCm 7.2 (live-verified in a prior session) while the macro form
// engages the backend's blocks-per-SM constraint.
//
// EFFECTIVE RUNG REALITY on gfx1201 (granule-24 VGPR allocation): the ladder
// collapses to TWO reachable points near the top — 16 waves/SIMD (VGPR <= 96)
// or 12 waves/SIMD (VGPR 97–120); requested rungs 13–15 are degenerate because
// the allocation granularity jumps straight from the 16-wave budget to the
// 12-wave budget. The baked value passed a ZERO-SPILL gate (one step across
// the boundary multiplied spills ~30x in live measurement); full ladder +
// forced-spill discrimination proof:
//   .omo/evidence/gpu-speedup/gap-closure/task-2-kernel-gap-closure/ladder.md
//
// NVIDIA side is intentionally UNCLAMPED: ptxas holds its natural 78 regs /
// 0 spills and clamping showed no benefit (parity check in ladder.md).
//
// BAKED RUNG = 12: N=16 compiles spill-free (VGPRs squeezed 100→88) but
// measured +2.2% slower @1M — this kernel is local-memory bound, so the
// occupancy win does not pay for the register squeeze; N=12 keeps native
// resources AND pins the floor (future VGPR growth must spill loudly here
// instead of silently dropping below 12 waves).
//
// Compile-time override for experiments, mirroring the FF_SIEVE_* geometry
// macro idiom: -DFF_SEARCH_MIN_BLOCKS_PER_SM=<n>. Values above the zero-spill
// boundary FORCE register spills (proven in forced-spill-rung.log).
#ifndef FF_SEARCH_MIN_BLOCKS_PER_SM
#if defined(__HIP_PLATFORM_AMD__)
#define FF_SEARCH_MIN_BLOCKS_PER_SM 12
#else
#define FF_SEARCH_MIN_BLOCKS_PER_SM 0
#endif
#endif

#if defined(__HIP_PLATFORM_AMD__) && FF_SEARCH_MIN_BLOCKS_PER_SM > 0
#define FF_SEARCH_LAUNCH_BOUNDS \
    __launch_bounds__(256, FF_SEARCH_MIN_BLOCKS_PER_SM)
#else
#define FF_SEARCH_LAUNCH_BOUNDS
#endif

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

// Integer square root of a 64-bit value.
// Bit-by-bit long-division port of the reference isqrt64 (segmentedSieve.C:37-52).
__device__ static uint64_t dev_isqrt64(uint64_t arg) {
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

// Power-of-2 test: (-x & x) == x iff x is a power of 2 and x > 0.
__device__ static bool dev_IsPowerOf2(uint64_t x) {
    return x > 0 && ((-x) & x) == x;
}

// ---- Montgomery REDC modular arithmetic (64-bit odd moduli) ----
//
// R = 2^64. For odd m < R, dev_MontMul returns a*b*R^{-1} mod m exactly, so
// every residue computed by the Miller-Rabin loop is identical to a plain
// (__uint128_t %) computation — only the representation differs. The MR
// branch of dev_IsPrime only ever sees ODD n (even n returns earlier), so no
// even-modulus fallback exists; the tiny-n base reduction below keeps n <= 17
// verdicts exact for non-production thresholds as well.

// -m^{-1} mod 2^64 via Newton iteration (seed correct mod 8, 5 doublings).
__device__ static uint64_t dev_MontMinv(uint64_t m) {
    uint64_t inv = m; // m*m == 1 (mod 8) for odd m
#pragma unroll
    for (int i = 0; i < 5; ++i) inv *= 2u - m * inv;
    return ~inv + 1u; // -inv mod 2^64
}

// Overflow-safe (a+b) mod m for a,b < m, valid up to m = 2^64-1: a wrap past
// 2^64 implies the true sum is in [2^64, 2m) so exactly one subtraction of m
// lands back in [0, m) after the uint64 wrap.
__device__ static uint64_t dev_MontAddMod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t s = a + b;
    if (s < a || s >= m) s -= m;
    return s;
}

// R^2 mod m for odd m: 128 overflow-safe doublings of 1 (= 2^128 mod m).
// Computed once per modulus; amortized over the whole witness loop.
__device__ static uint64_t dev_MontR2(uint64_t m) {
    uint64_t r = 1;
#pragma unroll 8
    for (int i = 0; i < 128; ++i) r = dev_MontAddMod(r, r, m);
    return r;
}

// Montgomery multiplication: a*b*R^{-1} mod m, exact for any odd m in [3, 2^64)
// with a,b < m. The carry term handles m > 2^63 where hi+umHi+c0 can exceed
// 64 bits: the true quotient t = c1*2^64 + rest is < 2m, so subtracting m from
// `rest` modulo 2^64 yields t - m exactly whenever c1 is set.
__device__ static uint64_t dev_MontMul(uint64_t a, uint64_t b, uint64_t m, uint64_t minv) {
    uint64_t lo   = a * b;
    uint64_t hi   = __umul64hi(a, b);
    uint64_t u    = lo * minv;            // (a*b) * (-m^{-1}) mod 2^64
    uint64_t umLo = u * m;
    uint64_t umHi = __umul64hi(u, m);
    uint64_t sumLo = lo + umLo;
    uint64_t c0 = sumLo < lo ? 1u : 0u;
    uint64_t t = hi + umHi;
    uint64_t c1 = t < hi ? 1u : 0u;
    t += c0;
    c1 += (t < c0) ? 1u : 0u;
    if (c1 || t >= m) t -= m;
    return t;
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
        // n is odd here, so REDC applies. Squaring loop stays in the
        // Montgomery domain; 1 and n-1 compare equal against their Montgomery
        // images (bijection on residues => identical comparison outcomes).
        const uint64_t minv = dev_MontMinv(n);
        const uint64_t r2   = dev_MontR2(n);
        const uint64_t oneM = dev_MontMul(1, r2, n, minv);
        const uint64_t nm1M = dev_MontMul(n - 1, r2, n, minv);
        for (uint32_t i = 0; i < maxTests; ++i) {
            uint64_t p = primeTestList[i];
            if (p >= n) p %= n; // unreachable in production; keeps tiny-n exact
            uint64_t xM = dev_MontMul(p, r2, n, minv);
            {
                uint64_t accM = oneM;
                uint64_t sqM = xM;
                uint64_t e = nDiv2Odd;
                while (e) {
                    if (e & 1) accM = dev_MontMul(accM, sqM, n, minv);
                    e >>= 1;
                    if (e) sqM = dev_MontMul(sqM, sqM, n, minv);
                }
                xM = accM;
            }
            if (xM != oneM && xM != nm1M) {
                for (uint32_t r = 1; r < power2; ++r) {
                    xM = dev_MontMul(xM, xM, n, minv);
                    if (xM == oneM) return false;
                    if (xM == nm1M) break;
                }
                if (xM != nm1M) return false;
            }
        }
        return true;
    }
    // In-map bit-test: read-only hinted load (task 15). The Montgomery branch
    // above performs NO map reads, so this is the only map touch in dev_IsPrime.
    return (FF_SEARCH_LD_RO_U8(primeMap + (n >> 4)) & (0x80 >> (n >> 1 & 7))) != 0;
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
    const uint8_t* __restrict__ primeMap, uint64_t maxPrimeMapValue,
    const uint32_t* __restrict__ smallPrimes, uint32_t smallPrimeCount,
    uint32_t primeIndex)
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
            factors[numFactors++] + oddProduct, primeMap, maxPrimeMapValue);

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
                // Use pre-computed smallPrimes for O(1) prime advancement.
                // Reference: primeFactor = smallPrimes_[++primeIndex]
                primeIndex++;
                if (primeIndex >= smallPrimeCount) {
                    break; // Safety: shouldn't happen if smallPrimes covers factorLimit
                }
                primeFactor = smallPrimes[primeIndex];
                if (primeFactor > factorLimit) break;
                temp = testProduct / primeFactor;
            }
            numFactors += numPendingFactors;
            numPendingFactors = 0;
            testFactor = primeFactor;
        }
    }
    // Depth (numFactors + numPendingFactors) is monotonically non-decreasing
    // within a call, so this single exit assert detects any MAX_FACTORS
    // overshoot at ANY write site above — one branch per call instead of one
    // per write. Gated behind FF_SEARCH_DEBUG_FACTORS because even this
    // trap-path stub costs ~1.3% on sm_120 (measured task 11); enable it when
    // touching the factor-enumeration algebra.
#if defined(FF_SEARCH_DEBUG_FACTORS)
    assert(numFactors + numPendingFactors <= MAX_FACTORS);
#endif
    return numMultipleFactorPairs == 1;
}

// DoesPeterKnow: checks the Peter condition for a single tuple.
__device__ static bool dev_DoesPeterKnow(
    uint32_t power2, uint64_t oddPartOfEven, uint64_t oddTerm, uint64_t oddProduct,
    uint64_t sum,
    const uint8_t* __restrict__ primeMap, uint64_t maxPrimeMapValue,
    const uint32_t* __restrict__ smallPrimes, uint32_t smallPrimeCount,
    uint32_t primeIndex)
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
        power2, oddProduct, sum, primeMap, maxPrimeMapValue, smallPrimes, smallPrimeCount, primeIndex);
}

// ---- SEARCH_KERNEL: persistent work-stealing grid ----
//
// Launch geometry (host side, gpu_search_kernel.cpp): k x multiProcessorCount
// blocks of 256 threads. Every thread loops pulling the next odd-sum index
// from ONE global device counter via atomicAdd(counter, 1) until the range is
// exhausted. Per-thread (not block-cooperative) pulling was chosen by
// measurement: block-chunk pulls force warps that finish early to idle at
// __syncthreads behind lagging block-mates instead of stealing work — that
// coupling cost ~15% on sm_120 at leg 2097152 (task-11 matrix evidence).
// Emission order is untouched because records still land in the sum-indexed
// slot (sum - sumStart) >> 1 regardless of which thread pulled the sum.
//
// The counter is a per-arch-module __device__ global; the host launcher zeroes
// it via hipMemcpyToSymbol before every launch (default-stream ordered).
#ifdef SIEVE_KERNEL_ARCH
static __device__ uint32_t FF_KERN_CAT(ffSearchWork_, SIEVE_KERNEL_ARCH);
#endif

__global__ void FF_SEARCH_LAUNCH_BOUNDS SEARCH_KERNEL(
    const uint8_t* __restrict__  primeMap,
    uint64_t                     maxPrimeMapValue,
    uint64_t                     sumStart,
    uint64_t                     sumLimit,
    uint32_t* __restrict__       pAtomicCount,
    GpuRecord* __restrict__      pRecords,
    const uint32_t* __restrict__ smallPrimes,
    uint32_t                     smallPrimeCount)
{
    // One-shot launch marker (was global thread 0 of the flat grid):
    // pAtomicCount keeps its test-visible semantics (0x100000 base + one
    // increment per emitted record).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(pAtomicCount, 0x100000);
        asm volatile("" ::: "memory");
    }

    const uint64_t numOddSums = (sumLimit - sumStart) / 2 + 1;

    for (;;) {
        const uint64_t tidx =
            (uint64_t)atomicAdd(&FF_KERN_CAT(ffSearchWork_, SIEVE_KERNEL_ARCH), 1u);
        if (tidx >= numOddSums) break;

        {
            uint64_t sum = sumStart + tidx * 2;

            // ---- CPU-order skip short-circuit (BEFORE any phase work) ----
            // Mirrors the cpu_search.cpp RunIt filter order:
            //   !ProductOfTermPairsHasSingleFactorPair(sum)
            //   && (sum != 3*sumDiv3 || IsPrime(sumDiv3))
            // For odd sums ProductOfTermPairs... reduces to IsPrime(sum-2), so
            // skipped sums can never reach the emission gate below — moving
            // the check ahead of phases 1-4 is output-invariant.
            bool skipSum = false;
            if (sum & 1) {
                uint64_t sumDiv3 = sum / 3;
                // Condition 1: sum-2 must NOT be prime.
                if (dev_IsPrime(sum - 2, primeMap, maxPrimeMapValue))
                    skipSum = true;
                // Condition 2: if sum is divisible by 3, sum/3 must be prime.
                else if (sum == 3 * sumDiv3 && !dev_IsPrime(sumDiv3, primeMap, maxPrimeMapValue))
                    skipSum = true;
            }

            if (!skipSum) {
            // STATE-RESET RULE (work-stealing): every per-sum local is declared
            // HERE and fully re-initialized for each pulled index — no state
            // may survive across loop iterations.
            uint32_t compositePower2[MAX_COMP_MAX_POWER2];
            uint32_t numComposite = 0;
            uint32_t numValid = 0;
            uint64_t termA = 0, termB = 0;
            bool termsFound = false;
            uint32_t primeIndex = 0; // smallPrimes has 2 stripped; index 0 = 3, 1 = 5, ...

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
                                    primeMap, maxPrimeMapValue, smallPrimes, smallPrimeCount, primeIndex)) > 1)
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
                            primeMap, maxPrimeMapValue, smallPrimes, smallPrimeCount, primeIndex)) > 1)
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
                                    primeMap, maxPrimeMapValue, smallPrimes, smallPrimeCount, primeIndex)) > 1)
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

            // ---- Write result if exactly one valid decomposition ----
            // (skipSum == false is guaranteed here by the early gate.)
            if (numValid == 1 && termsFound) {
                uint64_t lo = termA, hi = termB;
                if (lo > hi) { uint64_t tmp = lo; lo = hi; hi = tmp; }

                atomicAdd(pAtomicCount, 1);
                GpuRecord rec;
                rec.sum = (uint32_t)sum;
                rec.low = lo;
                rec.high = hi;
                rec.tag = 0;
                // Sum-indexed slot: each sum owns slot (sum-sumStart)/2, so slots
                // are in ascending-sum order — deterministic emission without a
                // sort, independent of pull order. Unsolved slots must read zero
                // (caller zero-fills the buffer first).
                pRecords[(sum - sumStart) >> 1] = rec;
            }
            }  // !skipSum
        }
    }
}

// ---- Test-only hook: batch Miller-Rabin verdicts through dev_IsPrime ----
// Driven exclusively by test/source/m4_mr_diff.cpp to exercise the REAL
// device MR path with arbitrary n. Never launched by production code.
// Parameter order mirrors SEARCH_KERNEL (pointers, then scalars, count last).
__global__ void FF_KERN_CAT(MRVerdictKernel_, SIEVE_KERNEL_ARCH)(
    const uint64_t* __restrict__ ns,
    uint8_t* __restrict__ verdicts,
    const uint8_t* __restrict__ primeMap,
    uint64_t maxPrimeMapValue,
    uint32_t count)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    verdicts[i] = dev_IsPrime(ns[i], primeMap, maxPrimeMapValue) ? 1 : 0;
}

#endif  // SIEVE_KERNEL_ARCH

#endif  // FF_GPU_SEARCH_KERNEL_H