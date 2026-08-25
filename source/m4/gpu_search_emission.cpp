// gpu_search_emission.cpp — Host-side formatting of GPU Freudenthal search
// results. Produces output byte-identical to the reference RunIt / OutputTags
// path (cpu_search.cpp, segmentedSieve.C).
//
// Three public functions:
//   PrintOutputTags     — prints the suffix tag for a single term
//   FormatGpuSearchResult — formats one GpuRecord as a single line
//   GpuSearchEmit       — scans a sum-indexed record array and prints results
//
// The kernel writes each solution to the sum-indexed slot (sum-sumStart)/2,
// so slot order IS ascending-sum order (no sort, no reordering).  Unsolved
// slots read zero; GpuSearchEmit skips them while numbering the printed
// solutions 1..N.  The caller passes the SLOT count (numOddSums), not the
// solution count.

#include "m4/gpu_search_kernel.h"
#include "m4/wheel_verdict.h"
#include "gpu_prime.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>

// ================================================================
// PrintOutputTags
//
// Matches PrintOutputTags in cpu_search.cpp byte-for-byte:
//   - prime term:  " (prime)"  + optionally ", " if isLast
//   - power-of-2:  " (2^N)" with fixed setprecision(0)
//                 then if isLast: ",%*s" with width=3-floor(log10(N))
//   - composite:   ",         " if isLast
//
// One template body serves both verdict sources: GpuPrime (canonical map,
// CPU search + fallback paths) and Wheel30Verdict (internal wheel-30 map,
// GPU-success emit path only — task 15/D).
// ================================================================

template <typename Verdict>
static void PrintOutputTagsImpl(const Verdict& prime, uint64_t term,
                                bool isLast) {
    if (prime.IsPrime(term)) {
        std::printf(" (prime)");
        if (isLast) std::printf(", ");
    } else if ((-term & term) == term) {
        double power2 = std::floor(std::log(double(term)) / std::log(double(2)) + 0.5);
        std::printf(" (2^%.0f)", power2);
        if (isLast) {
            int width = 3 - static_cast<int>(std::floor(std::log10(power2 + 0.001)));
            if (width < 1) width = 1;
            std::printf(",%*s", width, "");
        }
    } else {
        if (isLast) std::printf(",         ");
    }
}

void PrintOutputTags(const GpuPrime& prime, uint64_t term, bool isLast) {
    PrintOutputTagsImpl(prime, term, isLast);
}

void PrintOutputTags(const Wheel30Verdict& prime, uint64_t term, bool isLast) {
    PrintOutputTagsImpl(prime, term, isLast);
}

// ================================================================
// FormatGpuSearchResult
//
// Formats a single GpuRecord as one output line, byte-identical to
// cpu_search.cpp's FormatFreudenthalLine + printf wrapper.
//
// Format (with \n terminated):
//   %7u) sum =%9llu, product =%16llu,  low term =%9llu<lowTags>
//        high term =%9llu<highTags>\n
//
// Where lowTags has isLast=true, highTags has isLast=false.
// ================================================================

template <typename Verdict>
static void FormatGpuSearchResultImpl(const Verdict& prime, uint32_t count,
                                      const GpuRecord& rec) {
    // First line: counter + sum/product/lowTerm + lowTags
    std::printf("%7u) sum =%9llu, product =%16llu,  low term =%9llu",
                (unsigned)count,
                (unsigned long long)rec.sum,
                (unsigned long long)(rec.low * rec.high),
                (unsigned long long)rec.low);
    PrintOutputTags(prime, rec.low, true);   // isLast=true → trailing comma

    // Second line: highTerm + highTags (no trailing comma)
    std::printf("high term =%9llu", (unsigned long long)rec.high);
    PrintOutputTags(prime, rec.high, false); // isLast=false → no trailing comma

    std::printf("\n");
}

void FormatGpuSearchResult(const GpuPrime& prime, uint32_t count,
                           const GpuRecord& rec) {
    FormatGpuSearchResultImpl(prime, count, rec);
}

void FormatGpuSearchResult(const Wheel30Verdict& prime, uint32_t count,
                           const GpuRecord& rec) {
    FormatGpuSearchResultImpl(prime, count, rec);
}

// ================================================================
// GpuSearchEmit
//
// Scans the sum-indexed GpuRecord array in ascending slot order and prints
// each non-zero slot (1-based ordinal).  Unsolved slots (zero-filled before
// launch) are skipped.  slotCount = (sumLimit - sumStart)/2 + 1.
// ================================================================

template <typename Verdict>
static void GpuSearchEmitImpl(const Verdict& prime, const GpuRecord* records,
                              uint32_t slotCount) {
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < slotCount; ++i) {
        if (records[i].sum == 0) continue;   // unsolved slot
        ++ordinal;
        FormatGpuSearchResult(prime, ordinal, records[i]);
    }
}

void GpuSearchEmit(const GpuPrime& prime, const GpuRecord* records,
                   uint32_t slotCount) {
    GpuSearchEmitImpl(prime, records, slotCount);
}

void GpuSearchEmit(const Wheel30Verdict& prime, const GpuRecord* records,
                   uint32_t slotCount) {
    GpuSearchEmitImpl(prime, records, slotCount);
}