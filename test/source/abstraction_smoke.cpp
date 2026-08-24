// DevAbstraction smoke test (plan todo 6): ONE process exercises the
// vendor-neutral abstraction against BOTH GPUs — alloc, DevCopy H2D/D2H,
// DevEvent ordering, DevLaunch (trivial kernel) on every logical device —
// and asserts the logical device count is exactly 2 (bus-ID deduped, each
// card appears once; ff::mergeAndDedupe does the union inside DevInit).
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
//
// Task 7 adds the seed-from-enumeration phase: DevInitFromDevices must seed
// WITHOUT any ff_enum_hip_* call, resolveLogical post-seed must serve from
// the persistent cache (100x resolves -> zero enumerations), and an EMPTY
// seed list must be refused leaving runtimes uninitialized. Init state is
// process-global, so these phases run in a CHILD PROCESS (the binary re-execs
// itself with FF_SMOKE_SEED_CHILD=1) while this parent keeps validating the
// legacy DevInit() path unchanged.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "devabstraction.h"
#include "device_registry.h"

extern "C" int ff_enum_hip_gfx1201(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_enum_hip_sm_120(ff::DeviceInfo* out, int maxDevices, int* outCount);

namespace {

constexpr int kElems = 1 << 20;   // 4 MiB of unsigned ints
constexpr int kMaxDevices = 64;

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

// Task 7 phases: empty-seed refusal, seed-from-enumeration with zero
// re-enumeration proof, cached resolveLogical, alloc/free per seeded device.
int seedPhaseMain()
{
    std::fprintf(stderr,
                 "\n== ffdev seed-from-enumeration phase (child process) ==\n");

    // ---- EMPTY-seed failure QA: refuse WITHOUT initializing runtimes ----
    long long enumCalls = ffdev::DevTestEnumCallCount();
    if (ffdev::DevInitFromDevices(std::vector<ff::DeviceInfo>()) == 0) {
        std::fprintf(stderr,
                     "[abs-smoke] EMPTY-SEED QA FAIL: empty list accepted\n");
        return 1;
    }
    if (ffdev::DevTestEnumCallCount() != enumCalls) {
        std::fprintf(stderr,
                     "[abs-smoke] EMPTY-SEED QA FAIL: %lld ff_enum_hip_* "
                     "call(s) during refused seed\n",
                     ffdev::DevTestEnumCallCount() - enumCalls);
        return 1;
    }
    if (ffdev::DevGetDeviceCount() != -1 ||
        ffdev::DevBackendOf(0) != ffdev::DevBackend::Unknown) {
        std::fprintf(stderr,
                     "[abs-smoke] EMPTY-SEED QA FAIL: abstraction state was "
                     "mutated by the refused seed\n");
        return 1;
    }
    std::fprintf(stderr,
                 "[abs-smoke] EMPTY-SEED QA OK: DevInitFromDevices({}) "
                 "refused rc!=0, ZERO ff_enum_hip_* calls (%lld total), "
                 "runtimes uninitialized (DevGetDeviceCount=-1, "
                 "DevBackendOf=unknown)\n",
                 ffdev::DevTestEnumCallCount());

    // ---- Build an already-enumerated list exactly like DevInit would ----
    ff::DeviceInfo amdBuf[kMaxDevices] = {};
    ff::DeviceInfo nvBuf[kMaxDevices] = {};
    int amdCount = 0, nvCount = 0, skipped = 0;
    if (ff_enum_hip_gfx1201(amdBuf, kMaxDevices, &amdCount) != 0)
        std::fprintf(stderr,
                     "[abs-smoke] warning: AMD enumeration failed\n");
    if (ff_enum_hip_sm_120(nvBuf, kMaxDevices, &nvCount) != 0)
        std::fprintf(stderr,
                     "[abs-smoke] warning: NVIDIA enumeration failed\n");
    std::vector<ff::DeviceInfo> seeded =
        ff::mergeAndDedupe(amdBuf, amdCount, nvBuf, nvCount, &skipped);
    // The two direct extern calls above deliberately BYPASS the abstraction
    // (that is the point: enumeration happened OUTSIDE ffdev), so the
    // abstraction-side counter must still be zero here.
    if (seeded.empty()) {
        std::fprintf(stderr,
                     "[abs-smoke] SEED FAIL: test-side enumeration produced "
                     "no devices\n");
        return 1;
    }
    if (ffdev::DevTestEnumCallCount() != 0) {
        std::fprintf(stderr,
                     "[abs-smoke] SEED FAIL: abstraction performed %lld "
                     "enumeration call(s) before any init\n",
                     ffdev::DevTestEnumCallCount());
        return 1;
    }

    // ---- Seed: must perform ZERO ff_enum_hip_* calls ----
    const long long preSeed = ffdev::DevTestEnumCallCount();
    if (ffdev::DevInitFromDevices(seeded) != 0) {
        std::fprintf(stderr, "[abs-smoke] SEED FAIL: DevInitFromDevices "
                             "refused a valid list\n");
        return 1;
    }
    if (ffdev::DevTestEnumCallCount() != preSeed) {
        std::fprintf(stderr,
                     "[abs-smoke] SEED FAIL: %lld ff_enum_hip_* call(s) "
                     "inside DevInitFromDevices\n",
                     ffdev::DevTestEnumCallCount() - preSeed);
        return 1;
    }
    std::fprintf(stderr,
                 "[abs-smoke] SEED OK: %zu logical device(s) seeded, "
                 "ff_enum_hip_* calls during seed = 0\n",
                 seeded.size());

    // ---- Zero-reenum proof: 100x resolveLogical post-seed ----
    const int count = ffdev::DevGetDeviceCount();
    if (count != static_cast<int>(seeded.size())) {
        std::fprintf(stderr,
                     "[abs-smoke] SEED FAIL: count %d != seeded %zu\n",
                     count, seeded.size());
        return 1;
    }
    const long long preResolve = ffdev::DevTestEnumCallCount();
    for (int rep = 0; rep < 100; ++rep) {
        for (int i = 0; i < count; ++i) {
            if (ffdev::DevBackendOf(i) == ffdev::DevBackend::Unknown) {
                std::fprintf(stderr,
                             "[abs-smoke] RESOLVE FAIL: device[%d] unknown "
                             "backend at rep %d\n", i, rep);
                return 1;
            }
        }
    }
    if (ffdev::DevTestEnumCallCount() != preResolve) {
        std::fprintf(stderr,
                     "[abs-smoke] ZERO-REENUM PROOF FAIL: %d resolves over "
                     "%d device(s) triggered %lld ff_enum_hip_* call(s)\n",
                     100 * count, count,
                     ffdev::DevTestEnumCallCount() - preResolve);
        return 1;
    }
    std::fprintf(stderr,
                 "[abs-smoke] ZERO-REENUM PROOF OK: %d resolveLogical ops "
                 "(100x over %d device(s)) performed %lld ff_enum_hip_* "
                 "calls\n",
                 100 * count, count,
                 ffdev::DevTestEnumCallCount() - preResolve);

    // ---- Alloc trivial buffer on each seeded device -> free ----
    for (int i = 0; i < count; ++i) {
        const ff::DeviceInfo& want = seeded[static_cast<size_t>(i)];
        ff::DeviceInfo live {};
        if (ffdev::DevGetDeviceProperties(i, &live) != 0 ||
            std::strcmp(want.name, live.name) != 0 ||
            std::strcmp(want.busId, live.busId) != 0 ||
            std::strcmp(want.vendor, live.vendor) != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED FAIL: device[%d] live query does "
                         "not match seeded info (wrong runtime serving this "
                         "logical index?)\n", i);
            return 1;
        }
        ffdev::DevHandle h;
        if (ffdev::DevAlloc(i, 4096, &h) != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED FAIL: device[%d] trivial "
                         "DevAlloc FAILED\n", i);
            return 1;
        }
        const ffdev::DevBackend wantBackend =
            std::strcmp(want.vendor, "nvidia") == 0
                ? ffdev::DevBackend::HipNv
                : ffdev::DevBackend::HipAmd;
        if (h.backend != wantBackend || h.vendorIndex != want.runtimeIndex) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED FAIL: device[%d] handle routes to "
                         "%s vendorIndex %d (want %s/%d)\n",
                         i, ffdev::backendName(h.backend), h.vendorIndex,
                         ffdev::backendName(wantBackend), want.runtimeIndex);
            ffdev::DevFree(&h);
            return 1;
        }
        if (ffdev::DevFree(&h) != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED FAIL: device[%d] DevFree FAILED\n",
                         i);
            return 1;
        }
        std::fprintf(stderr,
                     "[abs-smoke] SEEDED device[%d] %s (bus=%s backend=%s "
                     "vendorIndex=%d): live props match, trivial alloc+free "
                     "OK\n",
                     i, want.name, want.busId,
                     ffdev::backendName(wantBackend), want.runtimeIndex);
    }

    std::fprintf(stderr,
                 "[abs-smoke] SEED-PHASE OK: %d device(s) seeded without "
                 "re-enumeration, cache-persistent resolve proven, "
                 "alloc/free exercised on every seeded device\n",
                 count);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    // Child dispatch: the seed-from-enumeration phases need virgin init
    // state, so the parent re-execs this binary with FF_SMOKE_SEED_CHILD=1.
    const char* seedChild = std::getenv("FF_SMOKE_SEED_CHILD");
    if (seedChild && seedChild[0] == '1') return seedPhaseMain();

    std::fprintf(stderr,
                 "== ffdev abstraction smoke (one process, two GPUs) ==\n");

    const char* force = std::getenv("FF_FORCE_BACKEND");
    if (force && force[0] != '\0' && std::strcmp(force, "hip") != 0) {
        std::fprintf(stderr,
                     "[abs-smoke] error: FF_FORCE_BACKEND='%s' is not "
                     "'hip' (test-only override)\n", force);
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
                 "[abs-smoke] logical device count: %d (expect 2, NOT 3 or 4 — "
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
                     "[abs-smoke] FAIL: expected exactly 2 logical devices "
                     "(AMD + NVIDIA), got %d\n", count);
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        if (runDeviceOps(i, devs[static_cast<size_t>(i)]) != 0) return 1;
    }

    // ---- TEST-ONLY failure-QA override (never affects ff_sieve) ----
    if (force && force[0] != '\0') {
        // Force hip backend: on this machine the forced backend cannot see
        // the PCI bus -> must refuse loudly, rc=1.
        const ff::DeviceInfo* target = nullptr;
        for (int i = 0; i < count; ++i) {
            const ff::DeviceInfo& d = devs[static_cast<size_t>(i)];
            if (std::strcmp(force, "hip") == 0 &&
                std::strcmp(d.vendor, "nvidia") == 0)
                target = &d;
        }
        if (!target) {
            std::fprintf(stderr,
                         "[abs-smoke] QA SKIP (explicit): FF_FORCE_BACKEND=%s "
                         "but no nvidia device exists to force — override not "
                         "exercised, failing loudly\n",
                         force);
            return 1;
        }
        const ffdev::DevBackend forced = ffdev::DevBackend::HipAmd;
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

    int nAmd = 0, nNv = 0;
    for (int i = 0; i < count; ++i) {
        if (ffdev::DevBackendOf(i) == ffdev::DevBackend::HipAmd) ++nAmd;
        if (ffdev::DevBackendOf(i) == ffdev::DevBackend::HipNv) ++nNv;
    }
    std::fprintf(stderr,
                 "[abs-smoke] ABSTRACTION SMOKE OK: %d logical device(s) "
                 "(backend hip-amd=%d hip-nv=%d) exercised alloc/copy/event/"
                 "launch in one process\n",
                 count, nAmd, nNv);

    // ---- Task 7: exercise DevInitFromDevices in a fresh child process ----
    // Init state is process-global; this parent keeps validating legacy
    // DevInit() while the child proves the seed path + cached resolve.
    if (argc > 0 && argv[0] && argv[0][0] != '\0') {
        if (setenv("FF_SMOKE_SEED_CHILD", "1", 1) != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED-PHASE FAIL: setenv failed\n");
            return 1;
        }
        const std::string cmd =
            std::string("'") + argv[0] + "' 2>&1";
        const int childRc = std::system(cmd.c_str());
        if (childRc != 0) {
            std::fprintf(stderr,
                         "[abs-smoke] SEED-PHASE FAIL: child exited with "
                         "status %d\n", childRc);
            return 1;
        }
        std::fprintf(stderr,
                     "[abs-smoke] SEED-PHASE (child) OK: DevInitFromDevices "
                     "seeded without re-enumeration\n");
    }
    return 0;
}
