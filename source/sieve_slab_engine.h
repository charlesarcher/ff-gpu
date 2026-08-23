// g++ header for the per-arch SieveSlabEngineRun dispatch.
//
// Compiled separately from the per-arch TU (sieve_slab_engine.cpp) — this
// header is included by host code (g++) only. The per-arch TU (compiled with
// hipcc) uses SIEVE_KERNEL_ARCH to tag function names via the same two-level
// macro-paste mechanism as sieve_slab_kernel.cpp, so the AMD and NVIDIA
// objects never export colliding symbols.
//
// Usage from g++ host code:
//   SieveSlabEngineRunFn fn = SieveSlabEngineGetLaunchFn_gfx1201(deviceIndex);
//   if (!fn) { /* no matching arch */ }
//   int rc = fn(deviceIndex, primes.data(), numPrimes, maxPrimeMapValue, hostMap);

#ifndef FF_SIEVE_SLAB_ENGINE_H
#define FF_SIEVE_SLAB_ENGINE_H

#include <cstdint>

// Per-arch engine run function pointer type.
//   deviceIndex:   logical device index (0-based)
//   h_primes:      host pointer to small-prime list (uint32_t)
//   numPrimes:     number of primes in h_primes
//   maxPrimeMapValue: the top value covered by the map
//   h_out:         host buffer to receive the full sieve map
//   returns:       0 on success, -1 on any HIP/CUDA error
typedef int (*SieveSlabEngineRunFn)(int deviceIndex,
                                     const uint32_t* h_primes,
                                     uint32_t numPrimes,
                                     uint64_t maxPrimeMapValue,
                                     uint8_t* h_out);

// Per-arch dispatch getter (gfx1201 / AMD RDNA4).
extern "C" SieveSlabEngineRunFn SieveSlabEngineGetLaunchFn_gfx1201(int deviceIndex);

// Per-arch dispatch getter (sm_120 / NVIDIA, HIP-platform objects).
extern "C" SieveSlabEngineRunFn SieveSlabEngineGetLaunchFn_sm_120(int deviceIndex);

// ---------------------------------------------------------------------------
// M2 backing pool + weighted pulls (plan todo 10): per-arch pool operation
// table.  The g++ pull scheduler (src/pull_scheduler.cpp) drives per-slab
// compute through these function pointers; device memory never crosses the
// g++/vendor-TU boundary as a typed handle.
//
// vendorIndex: index WITHIN the vendor runtime (DeviceInfo.runtimeIndex), NOT
// the logical device index.  Every op switches the calling thread's current
// device to vendorIndex first.
// ---------------------------------------------------------------------------
typedef struct SievePoolOps {
    // Allocate `bytes` on device vendorIndex; returns buffer or NULL.
    void* (*alloc)(int vendorIndex, uint64_t bytes);
    // Fill `bytes` bytes of d_ptr with host value `v`. Returns 0 / -1.
    int (*memset)(int vendorIndex, void* d_ptr, int v, uint64_t bytes);
    // Free a buffer allocated by alloc(). Returns 0 / -1.
    int (*free)(int vendorIndex, void* d_ptr);
    // Copy host -> device. d_dst must live on vendorIndex's device.
    int (*copyH2D)(int vendorIndex, void* d_dst, const void* h_src,
                   uint64_t bytes);
    // Copy device -> host. d_src must live on vendorIndex's device.
    int (*copyD2H)(int vendorIndex, void* h_dst, const void* d_src,
                   uint64_t bytes);
    // Sieve one slab covering values [segLo, segHi) into d_slab (a buffer of
    // >= slabBytes allocated on vendorIndex's device). primeSlab != 0 -> the
    // buffer is NOT yet primed: memset 0xff first (and clear the bit-0 of
    // byte 0 — value 1 — when segLo == 0) before launching the kernel.
    // h_primes is the host small-prime list (uploaded per call).
    int (*slabCompute)(int vendorIndex, const uint32_t* h_primes,
                       uint32_t numPrimes, uint64_t segLo, uint64_t segHi,
                       void* d_slab, uint64_t slabBytes, int primeSlab);
    // ---- M3 overlap engine (plan todo 12) ----
    // Stream/event handles are opaque vendor pointers (hipStream_t on both
    // platforms), never dereferenced in g++ host code. Every op
    // switches the calling thread's current device to vendorIndex first.
    void* (*createStream)(int vendorIndex);
    int (*destroyStream)(int vendorIndex, void* stream);
    void* (*createEvent)(int vendorIndex);
    int (*destroyEvent)(int vendorIndex, void* ev);
    int (*recordEvent)(int vendorIndex, void* ev, void* stream);
    int (*waitEvent)(int vendorIndex, void* ev, void* stream);
    int (*syncEvent)(int vendorIndex, void* ev);
    int (*elapsedEvents)(int vendorIndex, void* startEv, void* endEv, double* msOut);
    // Host pinned tier (hipHostMalloc on both platforms).
    int (*allocPinned)(int vendorIndex, void** h, uint64_t bytes);
    int (*freePinned)(int vendorIndex, void* h);
    // Async copies on an explicit stream (pinned host memory required).
    int (*copyH2DAsync)(int vendorIndex, void* d_dst, const void* h_src,
                        uint64_t bytes, void* stream);
    int (*copyD2HAsync)(int vendorIndex, void* h_dst, const void* d_src,
                        uint64_t bytes, void* stream);
    // Async slab compute: make computeStream wait on waitEvent (may be NULL),
    // then launch the existing slab kernel on computeStream. NO device sync,
    // NO memset priming (caller pre-primed d_slab via copyH2DAsync). The
    // per-vendor small-prime cache is uploaded once per device and reused.
    int (*slabComputeAsync)(int vendorIndex, const uint32_t* h_primes,
                            uint32_t numPrimes, uint64_t segLo, uint64_t segHi,
                            void* d_slab, uint64_t slabBytes, void* computeStream,
                            void* waitEvent);
} SievePoolOps;

// gfx1201 (AMD RDNA4) pool operations getter.
extern "C" const SievePoolOps* SievePoolGet_gfx1201(int vendorIndex);

// sm_120 (NVIDIA, HIP-platform objects) pool operations getter.
extern "C" const SievePoolOps* SievePoolGet_sm_120(int vendorIndex);

// Vendor-dispatch helper (g++ side): returns the ops table for the device's
// vendor ("amd" -> gfx1201, "nvidia" -> sm_120), or NULL if unknown/unavailable.
const SievePoolOps* SievePoolGetForVendor(const char* vendor, int vendorIndex);

#endif  // FF_SIEVE_SLAB_ENGINE_H