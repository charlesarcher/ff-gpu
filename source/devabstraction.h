// Device abstraction layer (GPU_PLAN §5.2, plan todo 6):
// DevAlloc / DevCopy / DevStream / DevEvent / DevLaunch plus
// DevGetDeviceCount / DevGetDeviceProperties / DevGetMemInfo. Host code
// (scheduler, overlap engine, search) talks ONLY to this header — no
// hip* type ever crosses into host-facing code.
//
// Two backends, both pure HIP source compiled per platform by hipcc:
//   HipAmd — HIP_PLATFORM=amd objects   (gfx1201, RX 9070 XT)
//   HipNv  — HIP_PLATFORM=nvidia objects (sm_120, RTX 5090; hip* entry
//            points resolve to the CUDA runtime at the symbol level)
// The vendor-neutral TU (src/devabstraction.cpp) owns the logical device
// list: it enumerates both backends with ff::mergeAndDedupe and maps each
// bus ID to a backend + vendor runtime index. Every entry point below is
// declared only in terms of POD / opaque tokens, so this header compiles
// under g++ and the HIP clang frontend.

#ifndef FF_DEVABSTRACTION_H
#define FF_DEVABSTRACTION_H

#include <cstddef>

#include "device_info.h"

namespace ffdev {

enum class DevBackend : int {
    Unknown = 0,
    HipAmd = 1, // ROCm/HIP, AMD platform (hip_runtime.h TU, gfx1201)
    HipNv = 2,  // HIP over CUDA runtime, NVIDIA platform TU (sm_120)
};

// Opaque tokens: the host TU only moves these around. The backend TU that
// created them owns the pointed-to runtime objects; DevFree/DevStreamDestroy/
// DevEventDestroy return them to their backend.
struct DevHandle {
    void* ptr = nullptr;
    int deviceIndex = -1;          // logical device index (0-based)
    int vendorIndex = -1;          // runtime index within the chosen backend
    DevBackend backend = DevBackend::Unknown;
};

struct DevStream {
    void* token = nullptr;
    int deviceIndex = -1;
    int vendorIndex = -1;          // runtime index within the chosen backend
    DevBackend backend = DevBackend::Unknown;
};

struct DevEvent {
    void* token = nullptr;
    int deviceIndex = -1;
    int vendorIndex = -1;          // runtime index within the chosen backend
    DevBackend backend = DevBackend::Unknown;
};

enum class DevCopyDir : int {
    H2D = 0,   // host -> device (dev = destination, host = source)
    D2H = 1,   // device -> host (dev = source, host = destination)
};

const char* backendName(DevBackend b);

// ---- discovery / queries (logical device indices throughout) ----
// Initializes the abstraction: enumerates HIP devices through the todo-3
// extern "C" entry point and builds the busId -> backend mapping. Call once
// before anything else; idempotent. Returns 0 on success, -1 on error
// (stderr detail).
int DevInit(void);

int DevGetDeviceCount(void);

// Fills `out` with the device's LIVE properties + mem-info (the backend TU
// reports compute capability, maxThreadsPerBlock, sharedMem, SM count,
// free/total VRAM into the shared ff::DeviceInfo POD so the abstraction and
// todo 3's CLI agree). Returns 0 on success, -1 on error.
int DevGetDeviceProperties(int deviceIndex, ff::DeviceInfo* out);

// Current-device-scoped free/total VRAM (memGetInfo is current-device
// scoped — the backend selects the device, queries, restores the previous
// current device). Returns 0 on success, -1 on error.
int DevGetMemInfo(int deviceIndex, size_t* freeBytes, size_t* totalBytes);

// Backend resolved for a logical device (diagnostics / test use).
DevBackend DevBackendOf(int deviceIndex);

// ---- allocation ----
int DevAlloc(int deviceIndex, size_t bytes, DevHandle* out);
int DevFree(DevHandle* h);

// ---- copies (synchronous, default-stream semantics) ----
int DevCopy(DevHandle* dev, void* host, size_t bytes, DevCopyDir dir);

// ---- streams ----
int DevStreamCreate(int deviceIndex, DevStream* out);
int DevStreamDestroy(DevStream* s);
int DevStreamSync(DevStream* s);

// ---- events (timing enabled; elapsed requires ordered completion) ----
int DevEventCreate(int deviceIndex, DevEvent* out);
int DevEventDestroy(DevEvent* e);
int DevEventRecord(DevEvent* e, DevStream* s);   // s may be null = default stream
int DevEventSync(DevEvent* e);
int DevEventElapsedMs(DevEvent* start, DevEvent* end, float* ms);

// ---- launch ----
// Launches the trivial arch-renamed kernel (smoke/smoke_kernel.h,
// SieveSlab_<arch>: buf[i] += 1, bounds-checked) on the device. s may be
// null = default stream; the backend synchronizes the used stream so the
// launch is observable before returning.
int DevLaunch(int deviceIndex, void* devBuf, int n, DevStream* s);

// M0 benchmark kernel launch (smoke/m0_kernel.h). kernel_id: 0 = MEMSET
// (constant 0xDEADBEEF fill), 1 = BW_SEQ (sequential index-based pattern).
// Synchronizes the used stream before returning — same contract as DevLaunch.
int DevLaunchM0(int deviceIndex, int kernel_id, void* devBuf, int n, DevStream* s);

// ---- TEST-ONLY override (never used by ff_sieve; the abstraction is not
// even linked into it) ----
// Forces the backend for a PCI bus ID, re-enumerating the forced backend to
// (a) verify it can actually see that bus and (b) recompute the vendor
// runtime index within it. Returns 0 if the remap was applied, -1 with an
// explicit stderr message if the forced backend cannot see the bus — there
// is NO silent fallback: refusing loudly beats running on the wrong device.
int DevForceBackendForBus(const char* busId, DevBackend backend);

}  // namespace ffdev

// Backend entry points (extern "C", implemented by hip_devabstraction.cpp /
// hip_enum.cpp / hip_m0_kernels.cpp / smoke/hip_smoke.cpp, compiled once per
// platform with SIEVE_KERNEL_ARCH=gfx1201 | sm_120 so each object set exports
// uniquely arch-tagged symbols). `dev` is the VENDOR runtime index
// (DeviceInfo.runtimeIndex). POD-only signatures so they cross the
// g++/hipcc/nvcc TU boundary safely.
extern "C" {

// ---- AMD backend (HIP_PLATFORM=amd objects) ----
int ff_dev_hip_deviceprops_gfx1201(int dev, ff::DeviceInfo* out);
int ff_dev_hip_meminfo_gfx1201(int dev, size_t* freeBytes, size_t* totalBytes);
int ff_dev_hip_alloc_gfx1201(int dev, size_t bytes, void** out);
int ff_dev_hip_free_gfx1201(int dev, void* p);
int ff_dev_hip_copy_h2d_gfx1201(int dev, void* dst, const void* src, size_t bytes);
int ff_dev_hip_copy_d2h_gfx1201(int dev, void* dst, const void* src, size_t bytes);
int ff_dev_hip_stream_create_gfx1201(int dev, void** out);
int ff_dev_hip_stream_destroy_gfx1201(int dev, void* s);
int ff_dev_hip_stream_sync_gfx1201(int dev, void* s);
int ff_dev_hip_event_create_gfx1201(int dev, void** out);
int ff_dev_hip_event_destroy_gfx1201(int dev, void* e);
int ff_dev_hip_event_record_gfx1201(int dev, void* e, void* s);
int ff_dev_hip_event_sync_gfx1201(int dev, void* e);
int ff_dev_hip_event_elapsed_ms_gfx1201(int dev, void* e0, void* e1, float* ms);
int ff_dev_hip_launch_gfx1201(int dev, void* buf, int n, void* s);
int ff_dev_hip_m0_launch_gfx1201(int dev, int kernel_id, void* buf, int n, void* s);

// ---- NVIDIA backend (HIP_PLATFORM=nvidia objects; HIP source, CUDA symbols) ----
int ff_dev_hip_deviceprops_sm_120(int dev, ff::DeviceInfo* out);
int ff_dev_hip_meminfo_sm_120(int dev, size_t* freeBytes, size_t* totalBytes);
int ff_dev_hip_alloc_sm_120(int dev, size_t bytes, void** out);
int ff_dev_hip_free_sm_120(int dev, void* p);
int ff_dev_hip_copy_h2d_sm_120(int dev, void* dst, const void* src, size_t bytes);
int ff_dev_hip_copy_d2h_sm_120(int dev, void* dst, const void* src, size_t bytes);
int ff_dev_hip_stream_create_sm_120(int dev, void** out);
int ff_dev_hip_stream_destroy_sm_120(int dev, void* s);
int ff_dev_hip_stream_sync_sm_120(int dev, void* s);
int ff_dev_hip_event_create_sm_120(int dev, void** out);
int ff_dev_hip_event_destroy_sm_120(int dev, void* e);
int ff_dev_hip_event_record_sm_120(int dev, void* e, void* s);
int ff_dev_hip_event_sync_sm_120(int dev, void* e);
int ff_dev_hip_event_elapsed_ms_sm_120(int dev, void* e0, void* e1, float* ms);
int ff_dev_hip_launch_sm_120(int dev, void* buf, int n, void* s);
int ff_dev_hip_m0_launch_sm_120(int dev, int kernel_id, void* buf, int n, void* s);

}  // extern "C"

#endif  // FF_DEVABSTRACTION_H
