// Logical device registry (plan todo 3): merges BOTH per-arch HIP
// enumerations (AMD-platform objects + NVIDIA-platform objects) into the
// logical device list, deduped by PCI bus ID.

#ifndef FF_DEVICE_REGISTRY_H
#define FF_DEVICE_REGISTRY_H

#include <vector>

#include "device_info.h"

namespace ff {

// Merges the AMD-backend and NVIDIA-backend enumerations into ONE logical
// device list, deduped by busId (domain:bus:device). Devices with an empty
// busId are kept (cannot dedup — assume unique). *skippedDuplicates receives
// the number of bus-ID collisions dropped.
std::vector<DeviceInfo> mergeAndDedupe(const DeviceInfo* hipDevs, int hipCount,
                                       const DeviceInfo* nvDevs, int nvCount,
                                       int* skippedDuplicates);

}  // namespace ff

#endif  // FF_DEVICE_REGISTRY_H
