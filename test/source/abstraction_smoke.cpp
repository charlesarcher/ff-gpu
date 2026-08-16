// DevAbstraction smoke test (plan todo 6): ONE process exercises the
// vendor-neutral abstraction against BOTH GPUs — alloc, DevCopy H2D/D2H,
// DevEvent ordering, DevLaunch (trivial kernel) on every logical device —
// and asserts the logical device count is exactly 2 (bus-ID deduped, the
// 5090 is never double-counted; ff::mergeAndDedupe does the union inside
// DevInit).
//
// This binary is VENDOR-NEUTRAL: it includes only src/devabstraction.h and
// touches no cuda*/hip* type. It has its own main() (it is not linked with
// src/main.cpp — ff_sieve never sees this test's code).
//
// TEST-ONLY failure-QA override (never affects ff_sieve): FF_FORCE_BACKEND=
// hip|cuda forces the named backend onto the OTHER vendor's device via
// DevForceBackendForBus. On this machine neither runtime sees the other
// vendor's card, so the forced backend cannot be located in its own
// enumeration -> DevForceBackendForBus refuses with an explicit message and
// this binary exits rc=1 (NO silent fallback).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "devabstraction.h"

namespace {

constexpr int kElems = 1 << 20;   // 4 MiB of unsigned ints

int runDeviceOps(int idx, const ff::DeviceInfo& info)
{
    ffdev::DevHandle h;
    if (ffdev::DevAlloc(idx, kElems * sizeof(unsigned int), &h) != 0) {
        std::fprintf(stderr, "[abs-smoke] device[%d] DevAlloc FAILED\n", idx);
        return -1;
    }

    std::vector<unsigned int> host(kElems);
    for (int i = 0; i < kElems; ++i) host[i] = static_cast<unsigned int>(i);
    if (ffdev::DevCopy(&h, host.data(), kElems * sizeof(unsigned int),
                       ffdev::DevCopyDir::H2D) != 0) {
        std::fprintf(stderr, "[abs-smoke] device[%d] DevCopy H2D FAILED\n", idx);
        ffdev::DevFree(&h);
        return -1;
    }

    // DevEvent ordering: e0 -> DevLaunch on stream -> e1; syncing e1 only
    // completes after the kernel it follows, so elapsed(e0,e1) is well
    // defined only when the events completed in record order.
    ffdev::DevStream s;
    ffdev::DevEvent e0, e1;
    if (ffdev::DevStreamCreate(idx, &s) != 0 ||
        ffdev::DevEventCreate(idx, &e0) != 0 ||
        ffdev::DevEventCreate(idx, &e1) != 0) {
        std::fprintf(stderr, "[abs-smoke] device[%d] stream/event create FAILED\n",
                     idx);
        ffdev::DevFree(&h);
        return -1;
    }
    float elapsedMs = -1.0f;
    if (ffdev::DevEventRecord(&e0, &s) != 0 ||
        ffdev::DevLaunch(idx, h.ptr, kElems, &s) != 0 ||
        ffdev::DevEventRecord(&e1, &s) != 0 ||
        ffdev::DevEventSync(&e1) != 0 ||
        ffdev::DevEventElapsedMs(&e0, &e1, &elapsedMs) != 0) {
        std::fprintf(stderr,
                     "[abs-smoke] device[%d] DevEvent ordering / DevLaunch "
                     "FAILED\n", idx);
        ffdev::DevEventDestroy(&e1);
        ffdev::DevEventDestroy(&e0);
        ffdev::DevStreamDestroy(&s);
        ffdev::DevFree(&h);
        return -1;
    }
    ffdev::DevEventDestroy(&e1);
    ffdev::DevEventDestroy(&e0);
    ffdev::DevStreamDestroy(&s);

    if (ffdev::DevCopy(&h, host.data(), kElems * sizeof(unsigned int),
                       ffdev::DevCopyDir::D2H) != 0) {
        std::fprintf(stderr, "[abs-smoke] device[%d] DevCopy D2H FAILED\n", idx);
        ffdev::DevFree(&h);
        return -1;
    }
    bool ok = true;
    for (int i = 0; i < kElems; ++i) {
        if (host[i] != static_cast<unsigned int>(i) + 1u) {
            std::fprintf(stderr,
                         "[abs-smoke] device[%d] kernel verify FAILED at %d "
                         "(got %u)\n", idx, i, host[i]);
            ok = false;
            break;
        }
    }
    if (ffdev::DevFree(&h) != 0) {
        std::fprintf(stderr, "[abs-smoke] device[%d] DevFree FAILED\n", idx);
        return -1;
    }
    if (!ok) return -1;

    std::fprintf(stderr,
                 "[abs-smoke] device[%d] %s : alloc OK DevCopy H2D OK "
                 "DevLaunch OK DevEvent ordering OK (elapsed=%.3f ms) "
                 "DevCopy D2H OK verify OK (%d elems)\n",
                 idx, info.name, elapsedMs, kElems);
    return 0;
}

}  // namespace

int main()
{
    std::fprintf(stderr,
                 "== ffdev abstraction smoke (one process, two GPUs) ==\n");

    const char* force = std::getenv("FF_FORCE_BACKEND");
    if (force && force[0] != '\0' && std::strcmp(force, "hip") != 0 &&
        std::strcmp(force, "cuda") != 0) {
        std::fprintf(stderr,
                     "[abs-smoke] error: FF_FORCE_BACKEND='%s' is not "
                     "'hip' or 'cuda' (test-only override)\n", force);
        return 1;
    }

    if (ffdev::DevInit() != 0) {
        std::fprintf(stderr, "[abs-smoke] DevInit FAILED\n");
        return 1;
    }
    const int count = ffdev::DevGetDeviceCount();
    if (count < 1) {
        std::fprintf(stderr, "[abs-smoke] no logical devices enumerated\n");
        return 1;
    }

    std::vector<ff::DeviceInfo> devs(static_cast<size_t>(count));
    std::fprintf(stderr,
                 "[abs-smoke] logical device count: %d (expect 2, NOT 3 — "
                 "deduped by PCI bus ID via ff::mergeAndDedupe)\n", count);
    for (int i = 0; i < count; ++i) {
        if (ffdev::DevGetDeviceProperties(i, &devs[static_cast<size_t>(i)]) != 0) {
            std::fprintf(stderr, "[abs-smoke] DevGetDeviceProperties(%d) "
                                 "FAILED\n", i);
            return 1;
        }
        const ff::DeviceInfo& d = devs[static_cast<size_t>(i)];
        size_t freeBytes = 0, totalBytes = 0;
        if (ffdev::DevGetMemInfo(i, &freeBytes, &totalBytes) != 0) {
            std::fprintf(stderr, "[abs-smoke] DevGetMemInfo(%d) FAILED\n", i);
            return 1;
        }
        std::fprintf(stderr,
                     "[abs-smoke] device[%d] %s (vendor=%s bus=%s "
                     "backend=%s): free=%llu B total=%llu B compute=%d.%d "
                     "maxThreads=%d sharedMem=%llu B smem/mp=%llu B smp=%d\n",
                     i, d.name, d.vendor, d.busId,
                     ffdev::backendName(ffdev::DevBackendOf(i)),
                     static_cast<unsigned long long>(freeBytes),
                     static_cast<unsigned long long>(totalBytes),
                     d.computeMajor, d.computeMinor, d.maxThreadsPerBlock,
                     static_cast<unsigned long long>(d.sharedMemPerBlock),
                     static_cast<unsigned long long>(
                         d.sharedMemPerMultiprocessor),
                     d.multiProcessorCount);
    }

    if (count != 2) {
        std::fprintf(stderr,
                     "[abs-smoke] FAIL: expected exactly 2 logical devices, "
                     "got %d\n", count);
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        if (runDeviceOps(i, devs[static_cast<size_t>(i)]) != 0) return 1;
    }

    // ---- TEST-ONLY failure-QA override (never affects ff_sieve) ----
    if (force && force[0] != '\0') {
        // Force the OTHER vendor's backend onto a device: hip -> the
        // nvidia device, cuda -> the amd device. On this machine the forced
        // backend cannot see that PCI bus -> must refuse loudly, rc=1.
        const ff::DeviceInfo* target = nullptr;
        for (int i = 0; i < count; ++i) {
            const ff::DeviceInfo& d = devs[static_cast<size_t>(i)];
            if (std::strcmp(force, "hip") == 0 &&
                std::strcmp(d.vendor, "nvidia") == 0)
                target = &d;
            if (std::strcmp(force, "cuda") == 0 &&
                std::strcmp(d.vendor, "amd") == 0)
                target = &d;
        }
        if (!target) {
            std::fprintf(stderr,
                         "[abs-smoke] QA SKIP (explicit): FF_FORCE_BACKEND=%s "
                         "but no %s device exists to force — override not "
                         "exercised, failing loudly\n",
                         force,
                         std::strcmp(force, "hip") == 0 ? "nvidia" : "amd");
            return 1;
        }
        const ffdev::DevBackend forced =
            std::strcmp(force, "hip") == 0 ? ffdev::DevBackend::Hip
                                           : ffdev::DevBackend::Cuda;
        std::fprintf(stderr,
                     "[abs-smoke] QA: FF_FORCE_BACKEND=%s -> forcing '%s' "
                     "backend onto PCI bus %s (logical device %s)\n",
                     force, force, target->busId, target->name);
        int rc = ffdev::DevForceBackendForBus(target->busId, forced);
        if (rc != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] QA FAILURE (expected): forcing '%s' "
                         "backend onto PCI bus %s was refused — no silent "
                         "fallback, exiting rc=1\n",
                         force, target->busId);
            return 1;
        }
        // Defensive: if a machine DOES see the card through both runtimes,
        // the remap applies and dispatch through the forced backend must
        // still work — prove it with a live query.
        size_t freeBytes = 0, totalBytes = 0;
        if (ffdev::DevGetMemInfo(0, &freeBytes, &totalBytes) != 0) {
            std::fprintf(stderr, "[abs-smoke] QA FAIL: forced-backend query "
                                 "FAILED\n");
            return 1;
        }
        std::fprintf(stderr, "[abs-smoke] QA: override applied and dispatch "
                             "through forced backend OK (free=%llu B)\n",
                     static_cast<unsigned long long>(freeBytes));
        return 0;
    }

    int nHip = 0, nCuda = 0;
    for (int i = 0; i < count; ++i)
        (ffdev::DevBackendOf(i) == ffdev::DevBackend::Hip ? nHip : nCuda)++;
    std::fprintf(stderr,
                 "[abs-smoke] ABSTRACTION SMOKE OK: %d logical devices "
                 "(backend hip=%d cuda=%d) exercised alloc/copy/event/launch "
                 "in one process\n",
                 count, nHip, nCuda);
    return 0;
}
