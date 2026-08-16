#include "device_registry.h"

#include <map>
#include <string>

namespace ff {

std::vector<DeviceInfo> mergeAndDedupe(const DeviceInfo* hipDevs, int hipCount,
                                       const DeviceInfo* cudaDevs, int cudaCount,
                                       int* skippedDuplicates)
{
    std::vector<DeviceInfo> out;
    std::map<std::string, int> seen;   // busId -> index into out
    int skipped = 0;

    auto add = [&](const DeviceInfo& d) {
        std::string bus(d.busId);
        if (!bus.empty() && seen.count(bus)) {
            ++skipped;   // same physical card already registered
            return;
        }
        out.push_back(d);
        if (!bus.empty()) seen[bus] = static_cast<int>(out.size()) - 1;
    };

    for (int i = 0; i < hipCount; ++i) add(hipDevs[i]);
    for (int i = 0; i < cudaCount; ++i) add(cudaDevs[i]);

    if (skippedDuplicates) *skippedDuplicates = skipped;
    return out;
}

}  // namespace ff
