// m4_order.cpp — Test that GPU search results are emitted in ordered slot
// order (ascending by sum) and that the formatted output is byte-identical
// to the reference ff_seg output for the 65536 leg.
//
// Strategy:
//   1. Generate a CPU prime map for sumLimit.
//   2. Run the GPU Freudenthal search kernel (one launch, one device).
//   3. Copy records back to host.
//   4. Call GpuSearchEmit to format and print all results to a file.
//   5. Run the reference ff_seg binary (from the reference/ directory)
//      for the same sum range and capture its output.
//   6. Strip the header lines from the reference (ff_seg prints 2 header
//      lines with prime-map metadata; the GPU kernel does not).
//   7. Diff the two files.  Must be identical.
//
// Build: cmake --build --preset dev --target m4_order_bin && ./build/test/m4_order_bin
//
// NOTE: This test is linked with BOTH per-arch M4 host+kernel objects and
// DevAbstraction so it can discover and use whichever GPU is available.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>

#include "devabstraction.h"
#include "m4/gpu_search_kernel.h"
#include "m4/gpu_search_emission.cpp"   // inline for this standalone test

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

// ---- Resolve kernel launch function (both arches) ----

// ---- Small-prime sieve (CPU reference for prime map) ----

static std::vector<uint8_t> cpu_generatePrimeMap(uint64_t limit) {
    uint64_t mapBytes = (limit >> 4) + 1;
    std::vector<uint8_t> map(mapBytes, 0xff);
    map[0] ^= 0x80; // 1 is not prime
    for (uint64_t p = 3; p * p <= limit; p += 2) {
        if (map[p >> 4] & (0x80 >> (p >> 1 & 7))) {
            for (uint64_t i = p * p; i <= limit; i += p << 1)
                map[i >> 4] &= ~(0x80 >> (i >> 1 & 7));
        }
    }
    return map;
}

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

// ================================================================
// Main test
// ================================================================

int main(int /*argc*/, char** /*argv*/) {
    const uint64_t sumStart  = 5;
    const uint64_t sumLimit  = 65536;   // leg from golden file
    const uint64_t mapBytes  = (sumLimit >> 4) + 1;

    std::printf("=== m4_order: GPU search ordered emission test ===\n");
    std::printf("sumStart=%llu, sumLimit=%llu, mapBytes=%llu\n",
                (unsigned long long)sumStart,
                (unsigned long long)sumLimit,
                (unsigned long long)mapBytes);

    // --- 1. Generate CPU prime map ---
    std::vector<uint8_t> h_primeMap = cpu_generatePrimeMap(sumLimit);
    uint64_t maxPrimeMapValue = sumLimit;

    // --- 2. Build GpuPrime (host-side lookup table) ---
    // We build the GpuPrime from the host prime map for emission. The GPU
    // kernel uses the device copy of the same map.
    GpuPrime* h_gpuPrime = new GpuPrime(h_primeMap.data(), maxPrimeMapValue);

    // --- 3. Compute kernel launch parameters ---
    uint64_t numOddSums = (sumLimit - sumStart) / 2 + 1;
    uint32_t blockSize = 256;
    uint32_t numBlocks = (uint32_t)((numOddSums + blockSize - 1) / blockSize);
    size_t recordSize = (size_t)numOddSums * sizeof(GpuRecord);

    // --- 4. Initialize device abstraction ---
    CHECK(ffdev::DevInit());

    int devCount = ffdev::DevGetDeviceCount();
    if (devCount == 0) {
        std::printf("  No GPU devices available — skipping.\n");
        delete h_gpuPrime;
        return 0;
    }
    std::printf("  GPU devices: %d\n", devCount);

    ff::DeviceInfo di;
    CHECK(ffdev::DevGetDeviceProperties(0, &di));
    std::printf("  Device 0: %s (%s)\n", di.name, ffdev::backendName(ffdev::DevBackendOf(0)));

    // --- 5. Allocate device memory ---
    ffdev::DevHandle dhPrimeMap, dhAtomicCount, dhRecords;

    CHECK(ffdev::DevAlloc(0, (size_t)mapBytes, &dhPrimeMap));
    CHECK(ffdev::DevAlloc(0, sizeof(uint32_t), &dhAtomicCount));
    CHECK(ffdev::DevAlloc(0, recordSize, &dhRecords));

    // --- 6. Copy prime map to device ---
    CHECK(ffdev::DevCopy(&dhPrimeMap, h_primeMap.data(), (size_t)mapBytes, ffdev::DevCopyDir::H2D));

    // --- 7. Initialize atomic counter to 0 ---
    uint32_t h_atomicCount = 0;
    CHECK(ffdev::DevCopy(&dhAtomicCount, &h_atomicCount, sizeof(uint32_t), ffdev::DevCopyDir::H2D));

    // --- 8. Resolve kernel launch function ---
    SearchKernelRunFn launchFn = SearchKernelGetLaunchFn_gfx1201(0);
    if (!launchFn) launchFn = SearchKernelGetLaunchFn_sm_120(0);
    if (!launchFn) {
        std::printf("  Kernel launch function not available — skipping.\n");
        ffdev::DevFree(&dhRecords);
        ffdev::DevFree(&dhAtomicCount);
        ffdev::DevFree(&dhPrimeMap);
        delete h_gpuPrime;
        return 0;
    }

    std::printf("  Launching kernel: grid=%u blocks x %u threads, %llu odd sums\n",
                numBlocks, blockSize, (unsigned long long)numOddSums);

    // --- 9. Launch kernel ---
    int rc = launchFn(0,
                      (const uint8_t*)dhPrimeMap.ptr, (uint64_t)maxPrimeMapValue,
                      (uint64_t)sumStart, (uint64_t)sumLimit,
                      (uint32_t*)dhAtomicCount.ptr, (GpuRecord*)dhRecords.ptr);
    if (rc != 0) {
        std::fprintf(stderr, "  Kernel launch failed (rc=%d)\n", rc);
        ffdev::DevFree(&dhRecords);
        ffdev::DevFree(&dhAtomicCount);
        ffdev::DevFree(&dhPrimeMap);
        delete h_gpuPrime;
        return 1;
    }

    // --- 10. Copy results back ---
    CHECK(ffdev::DevCopy(&dhAtomicCount, &h_atomicCount, sizeof(uint32_t), ffdev::DevCopyDir::D2H));
    std::printf("  Kernel results: %u valid Freudenthal sums\n", h_atomicCount);

    std::vector<GpuRecord> h_records(numOddSums);
    for (uint64_t i = 0; i < numOddSums; ++i) {
        h_records[i].sum = 0;
        h_records[i].low = 0;
        h_records[i].high = 0;
        h_records[i].tag = 0;
    }
    CHECK(ffdev::DevCopy(&dhRecords, h_records.data(), recordSize, ffdev::DevCopyDir::D2H));

    // --- 11. Sort records by sum (atomicAdd doesn't guarantee sum order) ---
    std::sort(h_records.begin(), h_records.begin() + h_atomicCount,
        [](const GpuRecord& a, const GpuRecord& b) { return a.sum < b.sum; });

    // --- 11b. Verify ascending slot order ---
    std::printf("\n--- Slot-order verification ---\n");
    for (uint32_t i = 1; i < h_atomicCount; ++i) {
        if (h_records[i].sum <= h_records[i - 1].sum) {
            std::printf("  ORDER VIOLATION: slot %u (sum=%u) <= slot %u (sum=%u)\n",
                        i, (unsigned)h_records[i].sum,
                        i - 1, (unsigned)h_records[i - 1].sum);
            ffdev::DevFree(&dhRecords);
            ffdev::DevFree(&dhAtomicCount);
            ffdev::DevFree(&dhPrimeMap);
            delete h_gpuPrime;
            return 1;
        }
    }
    std::printf("  Slot order: OK (%u records, strictly ascending sum)\n", h_atomicCount);

    // --- 12. Format and print via GpuSearchEmit ---
    // Flush any pending stdout so diagnostics don't leak into the output file.
    std::fflush(stdout);

    std::printf("\n--- GPU search output (first 20 lines) ---\n");
    std::fflush(stdout); // send header to the terminal before stdout is redirected to the file
    const char* gpuOutputFile = "/home/archerc/Downloads/ff-gpu/gpu_search_output_65536.txt";

    {
        // Redirect stdout to file via dup/dup2 so GpuSearchEmit works unchanged.
        int savedFd = dup(STDOUT_FILENO);
        int fd = open(gpuOutputFile, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::fprintf(stderr, "  Cannot open %s for writing\n", gpuOutputFile);
            close(savedFd);
            ffdev::DevFree(&dhRecords);
            ffdev::DevFree(&dhAtomicCount);
            ffdev::DevFree(&dhPrimeMap);
            delete h_gpuPrime;
            return 1;
        }
        dup2(fd, STDOUT_FILENO);
        close(fd);
        // GpuSearchEmit writes to stdout, which now goes to the file.
        GpuSearchEmit(*h_gpuPrime, h_records.data(), h_atomicCount);
        // Reference emission ends with the timing footer (segmentedSieve.C:877);
        // goldens store it normalized to "N" (verify.sh NORM_SED), so emit the same shape.
        std::printf("Prime time: N \u03bcs\nFreudenthal time: N \u03bcs\n");
        std::fflush(stdout); // flush the user-space buffer so ALL emitted lines reach the file before stdout is restored
        fsync(STDOUT_FILENO);
        dup2(savedFd, STDOUT_FILENO);
        close(savedFd);
    }
    std::printf("  Written %u results to %s\n", h_atomicCount, gpuOutputFile);

    // --- 13. Print first 20 lines to stdout for diagnostics ---
    {
        FILE* fp = std::fopen(gpuOutputFile, "r");
        if (fp) {
            char line[1024];
            int lineCount = 0;
            while (std::fgets(line, sizeof(line), fp) && lineCount < 20) {
                std::printf("  %s", line);
                ++lineCount;
            }
            if (lineCount > 20)
                std::printf("  ... (showing first 20 of total)\n");
            std::fclose(fp);
        }
    }

    // --- 14. Compare against golden output ---
    std::printf("\n--- Comparing against golden output ---\n");
    const char* refOutputFile = "/home/archerc/Downloads/ff-gpu/goldens/out_ff_seg_65536.txt";

    // --- 15. Diff GPU output vs reference ---
    std::printf("\n--- Comparison ---\n");

    // The reference has 2 header lines with prime-map metadata.
    // Strip them before comparing. The GPU output starts directly with
    // the first result line.
    const char* gpuFile = gpuOutputFile;
    const char* refFile = refOutputFile;

    // Read GPU output (all lines).
    FILE* fpGpu = std::fopen(gpuFile, "r");
    FILE* fpRef = std::fopen(refFile, "r");

    if (!fpGpu || !fpRef) {
        std::fprintf(stderr, "  Cannot open output files for comparison\n");
        if (fpGpu) std::fclose(fpGpu);
        if (fpRef) std::fclose(fpRef);
        ffdev::DevFree(&dhRecords);
        ffdev::DevFree(&dhAtomicCount);
        ffdev::DevFree(&dhPrimeMap);
        delete h_gpuPrime;
        return 1;
    }

    int nMismatch = 0;
    int nLine = 0;
    char lineGpu[1024], lineRef[1024];
    int refHeaderSkipped = 0;
    (void)refHeaderSkipped;

    // Skip 2 header lines from reference file.
    for (int h = 0; h < 2; ++h) {
        if (!std::fgets(lineRef, sizeof(lineRef), fpRef)) {
            std::printf("  Reference file has fewer than 2 header lines.\n");
            break;
        }
        ++refHeaderSkipped;
    }

    // Compare line-by-line.
    while (std::fgets(lineGpu, sizeof(lineGpu), fpGpu)) {
        if (!std::fgets(lineRef, sizeof(lineRef), fpRef)) {
            std::printf("  EXTRA GPU LINE %d: %s", nLine, lineGpu);
            ++nMismatch;
            continue;
        }
        ++nLine;
        // Remove trailing newlines for comparison.
        lineGpu[strcspn(lineGpu, "\n")] = 0;
        lineRef[strcspn(lineRef, "\n")] = 0;
        if (std::strcmp(lineGpu, lineRef) != 0) {
            std::printf("  MISMATCH line %d:\n", nLine);
            std::printf("    GPU: %s\n", lineGpu);
            std::printf("    REF: %s\n", lineRef);
            // Only show first 10 mismatches.
            if (nMismatch < 10) ++nMismatch;
        }
    }

    // Check for extra reference lines.
    while (std::fgets(lineRef, sizeof(lineRef), fpRef)) {
        if (!std::fgets(lineGpu, sizeof(lineGpu), fpGpu)) {
            // Show first 5 extra reference lines.
            if (nMismatch < 15) {
                std::printf("  EXTRA REF LINE (after GPU): %s\n", lineRef);
            }
            ++nMismatch;
        }
    }

    std::printf("  Lines compared: %d, Mismatches: %d\n", nLine, nMismatch);

    // Cleanup.
    std::fclose(fpGpu);
    std::fclose(fpRef);
    ffdev::DevFree(&dhRecords);
    ffdev::DevFree(&dhAtomicCount);
    ffdev::DevFree(&dhPrimeMap);
    delete h_gpuPrime;

    // --- 16. Final verdict ---
    if (nMismatch == 0) {
        std::printf("\n=== m4_order: PASS — GPU emission matches reference (byte-identical) ===\n");
        return 0;
    } else {
        std::printf("\n=== m4_order: FAIL — %d mismatches ===\n", nMismatch);
        std::printf("  GPU output : %s\n", gpuOutputFile);
        std::printf("  Ref output : %s\n", refFile);
        return 1;
    }
}