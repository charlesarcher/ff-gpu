// Vendor-neutral device abstraction host TU (GPU_PLAN §5.2, plan todo 6).
// Compiled with g++ — NO vendor headers in this TU, ever. Owns:
//   * the logical device list (todo-3 ff_enum_hip/ff_enum_cuda unioned with
//     ff::mergeAndDedupe by PCI bus ID — a card both runtimes see counts
//     once, so the 5090 is never double-counted),
//   * the busId -> {backend, vendor runtime index} map,
//   * dispatch of every ffdev:: op to the right extern "C" backend TU
//     (src/hip_devabstraction.cpp | src/cuda_devabstraction.cpp).
// The backend TUs are selected at runtime by PCI bus ID; the mapping is
// built from which enumeration actually reported each bus ID.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "devabstraction.h"
#include "device_registry.h"

extern "C" int ff_enum_hip(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_enum_cuda(ff::DeviceInfo* out, int maxDevices, int* outCount);

namespace {

constexpr int kMaxDevices = 64;

struct DevMapEntry {
    ffdev::DevBackend backend = ffdev::DevBackend::Unknown;
    int vendorIndex = -1;   // index within the vendor runtime (runtimeIndex)
};

std::vector<ff::DeviceInfo> g_devices;
std::map<std::string, DevMapEntry> g_busMap;   // busId -> backend mapping
int g_initialized = 0;

std::string mapKey(const ff::DeviceInfo& d)
{
    // PCI bus ID is the dedup/mapping key; devices without one get a
    // synthetic vendor:index key so they still resolve (real GPUs always
    // report a bus ID — this is a defensive fallback only).
    if (d.busId[0] != '\0') return std::string(d.busId);
    return std::string("vendor:") + std::string(d.vendor) + ":" +
           std::to_string(d.runtimeIndex);
}

ffdev::DevBackend backendFromVendor(const ff::DeviceInfo& d)
{
    return std::strcmp(d.vendor, "amd") == 0 ? ffdev::DevBackend::Hip
                                             : ffdev::DevBackend::Cuda;
}

// Re-enumerate a single backend and return the vendor runtime index for the
// given PCI bus ID. Used to remap logical-device indices to the correct
// vendor-specific index when a backend only sees a subset of devices.
static int findVendorIndex(ffdev::DevBackend backend, const std::string& busId)
{
    ff::DeviceInfo buf[64] = {};
    int count = 0;
    int rc = backend == ffdev::DevBackend::Hip
                 ? ff_enum_hip(buf, 64, &count)
                 : ff_enum_cuda(buf, 64, &count);
    if (rc != 0) return -1;
    for (int i = 0; i < count; ++i)
        if (std::string(buf[i].busId) == busId) return i;
    return -1;
}

int resolveLogical(int deviceIndex, DevMapEntry* e)
{
    if (!g_initialized) {
        std::fprintf(stderr, "[ffdev] error: DevInit() not called\n");
        return -1;
    }
    if (deviceIndex < 0 ||
        static_cast<size_t>(deviceIndex) >= g_devices.size()) {
        std::fprintf(stderr,
                     "[ffdev] error: logical device index %d out of range "
                     "(count=%zu)\n",
                     deviceIndex, g_devices.size());
        return -1;
    }
    const std::string key = mapKey(g_devices[deviceIndex]);
    auto it = g_busMap.find(key);
    if (it != g_busMap.end()) {
        // Remap: the stored vendorIndex is from the enumeration that FIRST
        // reported this bus. Re-enumerate to get the correct index within
        // the chosen backend (a CUDA-only run sees only 1 device at index 0,
        // not the logical index in the merged list).
        int remapped = findVendorIndex(it->second.backend, key);
        if (remapped >= 0) {
            DevMapEntry fixed = it->second;
            fixed.vendorIndex = remapped;
            *e = fixed;
            return 0;
        }
        *e = it->second;
        return 0;
    }
    // Fallback (should not happen: the map is built from the same list).
    e->backend = backendFromVendor(g_devices[deviceIndex]);
    e->vendorIndex = g_devices[deviceIndex].runtimeIndex;
    return 0;
}

const char* backendTag(ffdev::DevBackend b)
{
    return b == ffdev::DevBackend::Hip ? "hip" : "cuda";
}

}  // namespace

namespace ffdev {

const char* backendName(DevBackend b)
{
    switch (b) {
        case DevBackend::Hip: return "hip";
        case DevBackend::Cuda: return "cuda";
        default: return "unknown";
    }
}

int DevInit(void)
{
    if (g_initialized) return 0;   // idempotent

    ff::DeviceInfo hipBuf[kMaxDevices] = {};
    ff::DeviceInfo cudaBuf[kMaxDevices] = {};
    int hipCount = 0, cudaCount = 0;
    if (ff_enum_hip(hipBuf, kMaxDevices, &hipCount) != 0)
        std::fprintf(stderr,
                     "[ffdev] warning: HIP (AMD) enumeration failed; no AMD "
                     "devices will participate\n");
    if (ff_enum_cuda(cudaBuf, kMaxDevices, &cudaCount) != 0)
        std::fprintf(stderr,
                     "[ffdev] warning: CUDA (NVIDIA) enumeration failed; no "
                     "NVIDIA devices will participate\n");

    int skipped = 0;
    g_devices = ff::mergeAndDedupe(hipBuf, hipCount, cudaBuf, cudaCount,
                                   &skipped);
    g_busMap.clear();
    for (size_t i = 0; i < g_devices.size(); ++i) {
        DevMapEntry e;
        e.backend = backendFromVendor(g_devices[i]);
        e.vendorIndex = g_devices[i].runtimeIndex;
        g_busMap[mapKey(g_devices[i])] = e;
    }

    std::fprintf(stderr,
                 "[ffdev] DevInit: %zu logical device(s) after bus-ID dedup, "
                 "%d duplicate(s) skipped\n",
                 g_devices.size(), skipped);
    g_initialized = 1;
    return 0;
}

int DevGetDeviceCount(void)
{
    if (!g_initialized) {
        std::fprintf(stderr, "[ffdev] error: DevInit() not called\n");
        return -1;
    }
    return static_cast<int>(g_devices.size());
}

int DevGetDeviceProperties(int deviceIndex, ff::DeviceInfo* out)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_deviceprops(e.vendorIndex, out)
                 : ff_dev_cuda_deviceprops(e.vendorIndex, out);
    if (rc != 0)
        std::fprintf(stderr,
                     "[ffdev] DevGetDeviceProperties(%d) failed via %s "
                     "backend\n",
                     deviceIndex, backendName(e.backend));
    return rc;
}

int DevGetMemInfo(int deviceIndex, size_t* freeBytes, size_t* totalBytes)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_meminfo(e.vendorIndex, freeBytes, totalBytes)
                 : ff_dev_cuda_meminfo(e.vendorIndex, freeBytes, totalBytes);
    if (rc != 0)
        std::fprintf(stderr,
                     "[ffdev] DevGetMemInfo(%d) failed via %s backend\n",
                     deviceIndex, backendName(e.backend));
    return rc;
}

DevBackend DevBackendOf(int deviceIndex)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return DevBackend::Unknown;
    return e.backend;
}

int DevAlloc(int deviceIndex, size_t bytes, DevHandle* out)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    void* p = nullptr;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_alloc(e.vendorIndex, bytes, &p)
                 : ff_dev_cuda_alloc(e.vendorIndex, bytes, &p);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevAlloc(%d, %zu B) failed via %s backend\n",
                     deviceIndex, bytes, backendName(e.backend));
        return -1;
    }
    out->ptr = p;
    out->deviceIndex = deviceIndex;
    out->vendorIndex = e.vendorIndex;
    out->backend = e.backend;
    return 0;
}

int DevFree(DevHandle* h)
{
    if (!h || !h->ptr) {
        std::fprintf(stderr, "[ffdev] DevFree: null handle\n");
        return -1;
    }
    int rc = h->backend == DevBackend::Hip
                 ? ff_dev_hip_free(h->vendorIndex, h->ptr)
                 : ff_dev_cuda_free(h->vendorIndex, h->ptr);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevFree(%d) failed via %s backend\n",
                     h->deviceIndex, backendName(h->backend));
        return -1;
    }
    h->ptr = nullptr;
    return 0;
}

int DevCopy(DevHandle* dev, void* host, size_t bytes, DevCopyDir dir)
{
    if (!dev || !dev->ptr || !host) {
        std::fprintf(stderr, "[ffdev] DevCopy: null handle/pointer\n");
        return -1;
    }
    int rc = -1;
    if (dev->backend == DevBackend::Hip)
        rc = dir == DevCopyDir::H2D
                 ? ff_dev_hip_copy_h2d(dev->vendorIndex, dev->ptr, host, bytes)
                 : ff_dev_hip_copy_d2h(dev->vendorIndex, host, dev->ptr, bytes);
    else
        rc = dir == DevCopyDir::H2D
                 ? ff_dev_cuda_copy_h2d(dev->vendorIndex, dev->ptr, host, bytes)
                 : ff_dev_cuda_copy_d2h(dev->vendorIndex, host, dev->ptr, bytes);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevCopy(%s, %zu B) failed via %s backend\n",
                     dir == DevCopyDir::H2D ? "H2D" : "D2H", bytes,
                     backendName(dev->backend));
        return -1;
    }
    return 0;
}

int DevStreamCreate(int deviceIndex, DevStream* out)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    void* token = nullptr;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_stream_create(e.vendorIndex, &token)
                 : ff_dev_cuda_stream_create(e.vendorIndex, &token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevStreamCreate(%d) failed via %s backend\n",
                     deviceIndex, backendName(e.backend));
        return -1;
    }
    out->token = token;
    out->deviceIndex = deviceIndex;
    out->vendorIndex = e.vendorIndex;
    out->backend = e.backend;
    return 0;
}

int DevStreamDestroy(DevStream* s)
{
    if (!s || !s->token) {
        std::fprintf(stderr, "[ffdev] DevStreamDestroy: null stream\n");
        return -1;
    }
    int rc = s->backend == DevBackend::Hip
                 ? ff_dev_hip_stream_destroy(s->vendorIndex, s->token)
                 : ff_dev_cuda_stream_destroy(s->vendorIndex, s->token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevStreamDestroy(%d) failed via %s backend\n",
                     s->deviceIndex, backendName(s->backend));
        return -1;
    }
    s->token = nullptr;
    return 0;
}

int DevStreamSync(DevStream* s)
{
    if (!s || !s->token) {
        std::fprintf(stderr, "[ffdev] DevStreamSync: null stream\n");
        return -1;
    }
    int rc = s->backend == DevBackend::Hip
                 ? ff_dev_hip_stream_sync(s->vendorIndex, s->token)
                 : ff_dev_cuda_stream_sync(s->vendorIndex, s->token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevStreamSync(%d) failed via %s backend\n",
                     s->deviceIndex, backendName(s->backend));
        return -1;
    }
    return 0;
}

int DevEventCreate(int deviceIndex, DevEvent* out)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    void* token = nullptr;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_event_create(e.vendorIndex, &token)
                 : ff_dev_cuda_event_create(e.vendorIndex, &token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevEventCreate(%d) failed via %s backend\n",
                     deviceIndex, backendName(e.backend));
        return -1;
    }
    out->token = token;
    out->deviceIndex = deviceIndex;
    out->vendorIndex = e.vendorIndex;
    out->backend = e.backend;
    return 0;
}

int DevEventDestroy(DevEvent* ev)
{
    if (!ev || !ev->token) {
        std::fprintf(stderr, "[ffdev] DevEventDestroy: null event\n");
        return -1;
    }
    int rc = ev->backend == DevBackend::Hip
                 ? ff_dev_hip_event_destroy(ev->vendorIndex, ev->token)
                 : ff_dev_cuda_event_destroy(ev->vendorIndex, ev->token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevEventDestroy(%d) failed via %s backend\n",
                     ev->deviceIndex, backendName(ev->backend));
        return -1;
    }
    ev->token = nullptr;
    return 0;
}

int DevEventRecord(DevEvent* ev, DevStream* s)
{
    if (!ev || !ev->token) {
        std::fprintf(stderr, "[ffdev] DevEventRecord: null event\n");
        return -1;
    }
    void* sTok = s ? s->token : nullptr;
    int rc = ev->backend == DevBackend::Hip
                 ? ff_dev_hip_event_record(ev->vendorIndex, ev->token, sTok)
                 : ff_dev_cuda_event_record(ev->vendorIndex, ev->token, sTok);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevEventRecord(%d) failed via %s backend\n",
                     ev->deviceIndex, backendName(ev->backend));
        return -1;
    }
    return 0;
}

int DevEventSync(DevEvent* ev)
{
    if (!ev || !ev->token) {
        std::fprintf(stderr, "[ffdev] DevEventSync: null event\n");
        return -1;
    }
    int rc = ev->backend == DevBackend::Hip
                 ? ff_dev_hip_event_sync(ev->vendorIndex, ev->token)
                 : ff_dev_cuda_event_sync(ev->vendorIndex, ev->token);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevEventSync(%d) failed via %s backend\n",
                     ev->deviceIndex, backendName(ev->backend));
        return -1;
    }
    return 0;
}

int DevEventElapsedMs(DevEvent* start, DevEvent* end, float* ms)
{
    if (!start || !start->token || !end || !end->token) {
        std::fprintf(stderr, "[ffdev] DevEventElapsedMs: null event\n");
        return -1;
    }
    if (start->backend != end->backend ||
        start->deviceIndex != end->deviceIndex) {
        std::fprintf(stderr,
                     "[ffdev] DevEventElapsedMs: events from different "
                     "devices/backends\n");
        return -1;
    }
    int rc = start->backend == DevBackend::Hip
                 ? ff_dev_hip_event_elapsed_ms(start->vendorIndex,
                                               start->token, end->token, ms)
                 : ff_dev_cuda_event_elapsed_ms(start->vendorIndex,
                                                start->token, end->token, ms);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevEventElapsedMs(%d) failed via %s backend\n",
                     start->deviceIndex, backendName(start->backend));
        return -1;
    }
    return 0;
}

int DevLaunch(int deviceIndex, void* devBuf, int n, DevStream* s)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    void* sTok = s ? s->token : nullptr;
    int rc = e.backend == DevBackend::Hip
                 ? ff_dev_hip_launch(e.vendorIndex, devBuf, n, sTok)
                 : ff_dev_cuda_launch(e.vendorIndex, devBuf, n, sTok);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevLaunch(%d) failed via %s backend\n",
                     deviceIndex, backendName(e.backend));
        return -1;
    }
    return 0;
}

int DevForceBackendForBus(const char* busId, DevBackend backend)
{
    if (!g_initialized) {
        std::fprintf(stderr, "[ffdev] error: DevInit() not called\n");
        return -1;
    }
    if (backend != DevBackend::Hip && backend != DevBackend::Cuda) {
        std::fprintf(stderr, "[ffdev] error: DevForceBackendForBus: unknown "
                             "backend\n");
        return -1;
    }
    // The bus must be a known logical device.
    const std::string bus(busId ? busId : "");
    bool known = false;
    for (const ff::DeviceInfo& d : g_devices)
        if (bus == mapKey(d)) {
            known = true;
            break;
        }
    if (!known) {
        std::fprintf(stderr,
                     "[ffdev] DevForceBackendForBus: PCI bus '%s' is not a "
                     "known logical device — refusing\n",
                     bus.c_str());
        return -1;
    }

    // Re-enumerate the forced backend and locate the bus INSIDE it. This is
    // the no-silent-fallback gate: if the backend cannot see the bus, forcing
    // would dispatch vendor ops at the wrong runtime index and silently run
    // on the wrong device — refuse loudly instead. On success the vendor
    // runtime index is recomputed within the forced backend, so the remap is
    // genuinely correct when it applies.
    ff::DeviceInfo buf[kMaxDevices] = {};
    int count = 0;
    int rc = backend == DevBackend::Hip
                 ? ff_enum_hip(buf, kMaxDevices, &count)
                 : ff_enum_cuda(buf, kMaxDevices, &count);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevForceBackendForBus: re-enumerating forced "
                     "backend '%s' failed — refusing\n", backendName(backend));
        return -1;
    }
    int vendorIndex = -1;
    std::string seen;
    for (int i = 0; i < count; ++i) {
        if (!seen.empty()) seen += ", ";
        seen += buf[i].busId;
        if (bus == buf[i].busId) vendorIndex = i;
    }
    if (vendorIndex < 0) {
        std::fprintf(stderr,
                     "[ffdev] DevForceBackendForBus: forced backend '%s' "
                     "CANNOT see PCI bus %s (it sees: %s) — forcing would "
                     "silently run on the wrong device; refusing (no silent "
                     "fallback)\n",
                     backendName(backend), bus.c_str(), seen.c_str());
        return -1;
    }

    DevMapEntry e;
    e.backend = backend;
    e.vendorIndex = vendorIndex;
    g_busMap[bus] = e;
    std::fprintf(stderr,
                 "[ffdev] DevForceBackendForBus: remapped PCI bus %s to "
                 "'%s' backend (vendor index %d)\n",
                 bus.c_str(), backendName(backend), vendorIndex);
    return 0;
}

}  // namespace ffdev
