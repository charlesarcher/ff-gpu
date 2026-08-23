// Device abstraction host TU (GPU_PLAN §5.2, plan todo 6).
// Compiled with g++ — NO vendor headers in this TU, ever. Owns:
//   * the logical device list (both per-arch enum TUs merged + bus-deduped),
//   * the busId -> {backend, vendor runtime index} map,
//   * dispatch of every ffdev:: op to the arch-tagged extern "C" backend
//     entry points (HIP-amd objects or HIP-nvidia objects).
//
// Both backends are the SAME HIP source compiled twice; this TU never calls
// cuda*/hip* directly.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "devabstraction.h"
#include "device_registry.h"

extern "C" int ff_enum_hip_gfx1201(ff::DeviceInfo* out, int maxDevices, int* outCount);
extern "C" int ff_enum_hip_sm_120(ff::DeviceInfo* out, int maxDevices, int* outCount);

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

// Re-enumerate the given backend and return the vendor runtime index for the
// given PCI bus ID. Used to remap logical-device indices to the correct
// vendor-specific index when the backend only sees a subset of devices.
static int findVendorIndexFor(const std::string& busId, ffdev::DevBackend backend)
{
    ff::DeviceInfo buf[kMaxDevices] = {};
    int count = 0;
    int rc = backend == ffdev::DevBackend::HipNv
                 ? ff_enum_hip_sm_120(buf, kMaxDevices, &count)
                 : ff_enum_hip_gfx1201(buf, kMaxDevices, &count);
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
        // reported this bus. Re-enumerate THAT backend to get the correct
        // index within it.
        int remapped = findVendorIndexFor(key, it->second.backend);
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
    e->backend = std::strcmp(g_devices[deviceIndex].vendor, "nvidia") == 0
                     ? ffdev::DevBackend::HipNv
                     : ffdev::DevBackend::HipAmd;
    e->vendorIndex = g_devices[deviceIndex].runtimeIndex;
    return 0;
}

}  // namespace

namespace ffdev {

const char* backendName(DevBackend b)
{
    switch (b) {
        case DevBackend::HipAmd: return "hip-amd";
        case DevBackend::HipNv: return "hip-nv";
        default: return "unknown";
    }
}

int DevInit(void)
{
    if (g_initialized) return 0;   // idempotent

    ff::DeviceInfo amdBuf[kMaxDevices] = {};
    ff::DeviceInfo nvBuf[kMaxDevices] = {};
    int amdCount = 0, nvCount = 0;
    if (ff_enum_hip_gfx1201(amdBuf, kMaxDevices, &amdCount) != 0)
        std::fprintf(stderr,
                     "[ffdev] warning: HIP (AMD) enumeration failed; no AMD "
                     "devices will participate\n");
    if (ff_enum_hip_sm_120(nvBuf, kMaxDevices, &nvCount) != 0)
        std::fprintf(stderr,
                     "[ffdev] warning: HIP-NV enumeration failed; no NVIDIA "
                     "devices will participate\n");

    int skipped = 0;
    g_devices = ff::mergeAndDedupe(amdBuf, amdCount, nvBuf, nvCount, &skipped);
    g_busMap.clear();
    for (size_t i = 0; i < g_devices.size(); ++i) {
        DevMapEntry e;
        e.backend = std::strcmp(g_devices[i].vendor, "nvidia") == 0
                        ? DevBackend::HipNv
                        : DevBackend::HipAmd;
        e.vendorIndex = g_devices[i].runtimeIndex;
        g_busMap[mapKey(g_devices[i])] = e;
    }

    std::fprintf(stderr,
                 "[ffdev] DevInit: %zu logical device(s), "
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_deviceprops_sm_120(e.vendorIndex, out)
                 : ff_dev_hip_deviceprops_gfx1201(e.vendorIndex, out);
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_meminfo_sm_120(e.vendorIndex, freeBytes, totalBytes)
                 : ff_dev_hip_meminfo_gfx1201(e.vendorIndex, freeBytes, totalBytes);
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_alloc_sm_120(e.vendorIndex, bytes, &p)
                 : ff_dev_hip_alloc_gfx1201(e.vendorIndex, bytes, &p);
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
    int rc = h->backend == DevBackend::HipNv
                 ? ff_dev_hip_free_sm_120(h->vendorIndex, h->ptr)
                 : ff_dev_hip_free_gfx1201(h->vendorIndex, h->ptr);
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
    bool nv = dev->backend == DevBackend::HipNv;
    int rc = dir == DevCopyDir::H2D
                  ? (nv ? ff_dev_hip_copy_h2d_sm_120(dev->vendorIndex, dev->ptr, host, bytes)
                        : ff_dev_hip_copy_h2d_gfx1201(dev->vendorIndex, dev->ptr, host, bytes))
                  : (nv ? ff_dev_hip_copy_d2h_sm_120(dev->vendorIndex, host, dev->ptr, bytes)
                        : ff_dev_hip_copy_d2h_gfx1201(dev->vendorIndex, host, dev->ptr, bytes));
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_stream_create_sm_120(e.vendorIndex, &token)
                 : ff_dev_hip_stream_create_gfx1201(e.vendorIndex, &token);
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
    int rc = s->backend == DevBackend::HipNv
                 ? ff_dev_hip_stream_destroy_sm_120(s->vendorIndex, s->token)
                 : ff_dev_hip_stream_destroy_gfx1201(s->vendorIndex, s->token);
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
    int rc = s->backend == DevBackend::HipNv
                 ? ff_dev_hip_stream_sync_sm_120(s->vendorIndex, s->token)
                 : ff_dev_hip_stream_sync_gfx1201(s->vendorIndex, s->token);
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_event_create_sm_120(e.vendorIndex, &token)
                 : ff_dev_hip_event_create_gfx1201(e.vendorIndex, &token);
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
    int rc = ev->backend == DevBackend::HipNv
                 ? ff_dev_hip_event_destroy_sm_120(ev->vendorIndex, ev->token)
                 : ff_dev_hip_event_destroy_gfx1201(ev->vendorIndex, ev->token);
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
    int rc = ev->backend == DevBackend::HipNv
                 ? ff_dev_hip_event_record_sm_120(ev->vendorIndex, ev->token, sTok)
                 : ff_dev_hip_event_record_gfx1201(ev->vendorIndex, ev->token, sTok);
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
    int rc = ev->backend == DevBackend::HipNv
                 ? ff_dev_hip_event_sync_sm_120(ev->vendorIndex, ev->token)
                 : ff_dev_hip_event_sync_gfx1201(ev->vendorIndex, ev->token);
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
    int rc = start->backend == DevBackend::HipNv
                 ? ff_dev_hip_event_elapsed_ms_sm_120(start->vendorIndex,
                                                      start->token, end->token, ms)
                 : ff_dev_hip_event_elapsed_ms_gfx1201(start->vendorIndex,
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
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_launch_sm_120(e.vendorIndex, devBuf, n, sTok)
                 : ff_dev_hip_launch_gfx1201(e.vendorIndex, devBuf, n, sTok);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevLaunch(%d) failed via %s backend\n",
                     deviceIndex, backendName(e.backend));
        return -1;
    }
    return 0;
}

int DevLaunchM0(int deviceIndex, int kernel_id, void* devBuf, int n, DevStream* s)
{
    DevMapEntry e;
    if (resolveLogical(deviceIndex, &e) != 0) return -1;
    void* sTok = s ? s->token : nullptr;
    int rc = e.backend == DevBackend::HipNv
                 ? ff_dev_hip_m0_launch_sm_120(e.vendorIndex, kernel_id, devBuf, n, sTok)
                 : ff_dev_hip_m0_launch_gfx1201(e.vendorIndex, kernel_id, devBuf, n, sTok);
    if (rc != 0) {
        std::fprintf(stderr,
                     "[ffdev] DevLaunchM0(%d, kernel=%d) failed via %s backend\n",
                     deviceIndex, kernel_id, backendName(e.backend));
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
    if (backend != DevBackend::HipAmd && backend != DevBackend::HipNv) {
        std::fprintf(stderr, "[ffdev] error: DevForceBackendForBus: only HIP "
                             "backends are supported\n");
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

    // Re-enumerate the forced backend and locate the bus INSIDE it.
    ff::DeviceInfo buf[kMaxDevices] = {};
    int count = 0;
    int rc = backend == DevBackend::HipNv
                 ? ff_enum_hip_sm_120(buf, kMaxDevices, &count)
                 : ff_enum_hip_gfx1201(buf, kMaxDevices, &count);
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
                     "CANNOT see PCI bus %s (it sees: %s) — refusing\n",
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
