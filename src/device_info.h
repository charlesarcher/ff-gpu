// Vendor-neutral device descriptor (plan todo 3). Shared by every TU —
// including the vendor TUs (src/hip_enum.cpp, src/cuda_enum.cpp) which fill it
// through extern "C" entry points.
//
// POD only (fixed-size char arrays + plain integral fields, no std:: types) so
// the object layout is identical whether a TU is compiled by g++, the HIP
// clang frontend, or the CUDA nvcc host frontend — the struct crosses the
// vendor-TU boundary by pointer, never by value.

#ifndef FF_DEVICE_INFO_H
#define FF_DEVICE_INFO_H

namespace ff {

struct DeviceInfo {
    char vendor[8];          // "amd" | "nvidia"
    char name[256];          // runtime-reported device name (hipDeviceProp_t/cudaDeviceProp .name)
    char busId[32];          // PCI bus-ID dedup key: "%04x:%02x:%02x" (domain:bus:device)
    int  pciDomain = 0;      // raw PCI domain
    int  pciBus = 0;         // raw PCI bus
    int  pciDevice = 0;      // raw PCI device
    unsigned long long freeBytes = 0;   // free VRAM at init (hipMemGetInfo / cudaMemGetInfo)
    unsigned long long totalBytes = 0;  // total VRAM (hipMemGetInfo / cudaMemGetInfo)
    int  computeMajor = 0;   // compute capability (major)
    int  computeMinor = 0;   // compute capability (minor)
    int  maxThreadsPerBlock = 0;
    unsigned long long sharedMemPerBlock = 0;
    unsigned long long sharedMemPerMultiprocessor = 0;
    int  multiProcessorCount = 0;
    int  maxGridDimX = 0;
    int  maxGridDimY = 0;
    int  maxGridDimZ = 0;
    int  warpSize = 0;
    int  runtimeIndex = 0;   // index within the vendor runtime that enumerated it
};

}  // namespace ff

#endif  // FF_DEVICE_INFO_H
