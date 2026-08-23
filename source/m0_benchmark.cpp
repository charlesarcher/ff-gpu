// M0 bandwidth benchmark (GPU_PLAN §10, M0 exit criteria).
//
// Measures per-device:
//   * per-device VRAM write bandwidth via M0_MEMSET kernel (constant fill)
//   * PCIe H2D and D2H bandwidth via DevCopy
//
// Uses only the ffdev:: Dev* abstraction — no direct cuda*/hip* calls.
// All diagnostics to STDERR; JSON results written to config/m0-benchmarks.json.
//
// Each measurement: 1 warmup run + 3 timed runs, report average.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "devabstraction.h"

// --- constants (tunable hints, NOT measured values) ---
constexpr size_t kBenchBufElems  = 256 * 1024 * 1024;   // 1 GiB / 4 bytes
constexpr int      kTimedRuns    = 3;
constexpr int      kWarmupRuns   = 1;

namespace {

struct DevResult {
    std::string name;
    std::string vendor;
    double      writeBandwidthGbs = 0.0;
    double      h2dBandwidthGbs   = 0.0;
    double      d2hBandwidthGbs   = 0.0;
};

double now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Run a kernel (warmup + timed), return average time in ms.
double bench_kernel(int deviceIndex, int kernel_id, void* devBuf, int n)
{
    for (int w = 0; w < kWarmupRuns; ++w) {
        if (ffdev::DevLaunchM0(deviceIndex, kernel_id, devBuf, n, nullptr) != 0)
            return -1.0;
    }
    double total = 0.0;
    for (int r = 0; r < kTimedRuns; ++r) {
        double t0 = now_ms();
        if (ffdev::DevLaunchM0(deviceIndex, kernel_id, devBuf, n, nullptr) != 0)
            return -1.0;
        total += now_ms() - t0;
    }
    return total / kTimedRuns;
}

// Run a DevCopy (warmup + timed), return average time in ms.
double bench_copy(ffdev::DevHandle& h, void* host, size_t bytes, ffdev::DevCopyDir dir)
{
    for (int w = 0; w < kWarmupRuns; ++w) {
        if (ffdev::DevCopy(&h, host, bytes, dir) != 0)
            return -1.0;
    }
    double total = 0.0;
    for (int r = 0; r < kTimedRuns; ++r) {
        double t0 = now_ms();
        if (ffdev::DevCopy(&h, host, bytes, dir) != 0)
            return -1.0;
        total += now_ms() - t0;
    }
    return total / kTimedRuns;
}

std::string backendLabel(ffdev::DevBackend b)
{
    return b == ffdev::DevBackend::HipAmd ? "hip-amd"
         : b == ffdev::DevBackend::HipNv  ? "hip-nv"
                                          : "unknown";
}

}  // namespace

int main()
{
    std::fprintf(stderr, "=== M0 bandwidth benchmark ===\n");

    if (ffdev::DevInit() != 0) {
        std::fprintf(stderr, "FATAL: DevInit() failed\n");
        return 1;
    }

    int devCount = ffdev::DevGetDeviceCount();
    if (devCount <= 0) {
        std::fprintf(stderr, "FATAL: no devices enumerated (count=%d)\n", devCount);
        return 1;
    }
    std::fprintf(stderr, "Devices found: %d\n", devCount);

    std::vector<DevResult> results;
    results.reserve(static_cast<size_t>(devCount));

    for (int d = 0; d < devCount; ++d) {
        std::fprintf(stderr, "\n--- Device %d ---\n", d);

        ff::DeviceInfo props{};
        if (ffdev::DevGetDeviceProperties(d, &props) != 0) {
            std::fprintf(stderr, "  SKIP: DevGetDeviceProperties failed\n");
            continue;
        }

        size_t freeBytes = 0, totalBytes = 0;
        if (ffdev::DevGetMemInfo(d, &freeBytes, &totalBytes) != 0) {
            std::fprintf(stderr, "  SKIP: DevGetMemInfo failed\n");
            continue;
        }

        DevResult res;
        res.name = props.name;
        res.vendor = props.vendor;

        std::fprintf(stderr, "  name: %s\n", props.name);
        std::fprintf(stderr, "  vendor: %s  backend: %s\n",
                     props.vendor, backendLabel(ffdev::DevBackendOf(d)).c_str());
        std::fprintf(stderr, "  VRAM: %zu / %zu MiB (free/total)\n",
                     freeBytes >> 20, totalBytes >> 20);

        size_t allocBytes = kBenchBufElems * sizeof(unsigned int);
        if (allocBytes > freeBytes) {
            // Reduce buffer size to fit in available VRAM (half the requested).
            allocBytes = (freeBytes / 2 / sizeof(unsigned int)) * sizeof(unsigned int);
            if (allocBytes == 0) {
                std::fprintf(stderr, "  SKIP: insufficient VRAM even after reduction\n");
                continue;
            }
            std::fprintf(stderr, "  WARN: reduced allocation to %zu bytes (%zu elems)\n",
                         allocBytes, allocBytes / sizeof(unsigned int));
        }

        int nElems = static_cast<int>(allocBytes / sizeof(unsigned int));

        // Allocate device buffer.
        ffdev::DevHandle devH{};
        if (ffdev::DevAlloc(d, allocBytes, &devH) != 0) {
            std::fprintf(stderr, "  SKIP: DevAlloc failed\n");
            continue;
        }

        // Allocate host buffer for copy tests.
        void* hostBuf = std::malloc(allocBytes);
        if (!hostBuf) {
            std::fprintf(stderr, "  SKIP: host malloc failed\n");
            ffdev::DevFree(&devH);
            continue;
        }
        std::memset(hostBuf, 0xAA, allocBytes);  // non-zero pattern

        // --- Kernel: MEMSET (constant fill write bandwidth) ---
        double t_memset = bench_kernel(d, 0, devH.ptr, nElems);
        if (t_memset > 0.0) {
            double gbs = static_cast<double>(allocBytes) / (t_memset * 1e6);
            res.writeBandwidthGbs = gbs;
            std::fprintf(stderr, "  MEMSET write BW: %.2f GB/s  (%.3f ms for %.1f GiB)\n",
                         gbs, t_memset, allocBytes / (1024.0 * 1024 * 1024));
        } else {
            std::fprintf(stderr, "  MEMSET write BW: FAILED\n");
        }

        // --- Kernel: BW_SEQ (sequential pattern write bandwidth) ---
        double t_bw = bench_kernel(d, 1, devH.ptr, nElems);
        if (t_bw > 0.0) {
            double gbs = static_cast<double>(allocBytes) / (t_bw * 1e6);
            std::fprintf(stderr, "  BW_SEQ write BW: %.2f GB/s  (%.3f ms for %.1f GiB)\n",
                         gbs, t_bw, allocBytes / (1024.0 * 1024 * 1024));
        } else {
            std::fprintf(stderr, "  BW_SEQ write BW: FAILED\n");
        }

        // --- DevCopy H2D ---
        double t_h2d = bench_copy(devH, hostBuf, allocBytes, ffdev::DevCopyDir::H2D);
        if (t_h2d > 0.0) {
            double gbs = static_cast<double>(allocBytes) / (t_h2d * 1e6);
            res.h2dBandwidthGbs = gbs;
            std::fprintf(stderr, "  H2D  BW: %.2f GB/s  (%.3f ms for %.1f GiB)\n",
                         gbs, t_h2d, allocBytes / (1024.0 * 1024 * 1024));
        } else {
            std::fprintf(stderr, "  H2D  BW: FAILED\n");
        }

        // --- DevCopy D2H ---
        double t_d2h = bench_copy(devH, hostBuf, allocBytes, ffdev::DevCopyDir::D2H);
        if (t_d2h > 0.0) {
            double gbs = static_cast<double>(allocBytes) / (t_d2h * 1e6);
            res.d2hBandwidthGbs = gbs;
            std::fprintf(stderr, "  D2H  BW: %.2f GB/s  (%.3f ms for %.1f GiB)\n",
                         gbs, t_d2h, allocBytes / (1024.0 * 1024 * 1024));
        } else {
            std::fprintf(stderr, "  D2H  BW: FAILED\n");
        }

        // Cleanup.
        std::free(hostBuf);
        ffdev::DevFree(&devH);

        results.push_back(std::move(res));
    }

    std::fprintf(stderr, "\n=== Summary ===\n");
    for (const auto& r : results) {
        std::fprintf(stderr, "  %s (%s): write=%.1f  H2D=%.1f  D2H=%.1f GB/s\n",
                     r.name.c_str(), r.vendor.c_str(),
                     r.writeBandwidthGbs, r.h2dBandwidthGbs, r.d2hBandwidthGbs);
    }

    // --- Write JSON ---
    std::ofstream ofs("config/m0-benchmarks.json");
    if (!ofs.is_open()) {
        std::fprintf(stderr, "FATAL: cannot open config/m0-benchmarks.json for writing\n");
        return 1;
    }
    ofs << std::fixed << std::setprecision(2);
    ofs << "{\n";
    ofs << "  \"devices\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ofs << "    {\"name\": \"" << r.name << "\", \"vendor\": \"" << r.vendor
            << "\", \"writeBandwidthGbs\": " << r.writeBandwidthGbs
            << ", \"h2dBandwidthGbs\": " << r.h2dBandwidthGbs
            << ", \"d2hBandwidthGbs\": " << r.d2hBandwidthGbs << "}";
        if (i + 1 < results.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";
    ofs << "}\n";
    ofs.close();

    std::fprintf(stderr, "Wrote config/m0-benchmarks.json\n");
    return 0;
}