// Logical device registry (plan todo 3): unions the HIP(AMD-platform) and CUDA
// enumerations into one list, deduped by PCI bus ID (Metis MUST-COVER — a card
// visible to both runtimes must be counted once, never assumed away).

#ifndef FF_DEVICE_REGISTRY_H
#define FF_DEVICE_REGISTRY_H

#include <vector>

#include "device_info.h"

namespace ff {

// Merges the HIP and CUDA enumerations into ONE logical device list, deduped
// by busId (domain:bus:device). HIP devices come first (logical index 0..),
// then CUDA devices whose bus ID was not already seen. Devices with an empty
// busId are kept (cannot dedup — assume unique). *skippedDuplicates receives
// the number of bus-ID collisions dropped.
std::vector<DeviceInfo> mergeAndDedupe(const DeviceInfo* hipDevs, int hipCount,
                                       const DeviceInfo* cudaDevs, int cudaCount,
                                       int* skippedDuplicates);

}  // namespace ff

#endif  // FF_DEVICE_REGISTRY_H
