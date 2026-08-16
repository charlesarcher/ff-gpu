// GPU_PLAN §4 configuration surface (plan todo 3): CLI flags + FF_* env vars,
// precedence CLI > env > default. Hand-rolled parsing — no third-party
// libraries (guardrail).

#ifndef FF_CONFIG_H
#define FF_CONFIG_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ff {

struct Config {
    double globalFraction = 0.90;                    // --vram-fraction / FF_VRAM_FRACTION
    std::map<std::string, double> deviceFractions;   // --device-vram-fraction: "amd"|"nvidia"|<logical index> -> f
    bool hasBudgetCap = false;                       // --vram-budget / FF_VRAM_BUDGET
    uint64_t budgetCapBytes = 0;                     // absolute per-device cap
    bool hasScratch = false;                         // --scratch / FF_SCRATCH
    uint64_t scratchBytes = 0;
    uint64_t slabSizeBytes = 1ull << 30;             // --slab-size, default 1 GiB
    bool hasHostTierCap = false;                     // --host-tier-cap / FF_HOST_TIER_CAP
    bool hostTierAuto = false;                       // "auto" = host RAM - 4 GiB (resolved in validateConfig)
    uint64_t hostTierCapBytes = 0;                   // 0 = disabled (default)
    bool noHostTier = false;                         // --no-host-tier: force-disable the host overflow tier
    bool listDevices = false;                        // --list-devices (recon flag, plan-defined)
    std::string deviceFilter;                        // --devices <backend>: "amd"|"nvidia" ("" = all)
    // M2 weighted pulls (plan todo 10): vendor name -> bandwidth weight (> 0).
    // Populated by loadPullWeights() from config/m0-benchmarks.json
    // (writeBandwidthGbs), overridden by FF_PULL_WEIGHTS. Vendors absent from
    // the map fall back to weight 1.0 (uniform pulls).
    std::map<std::string, double> pullWeights;
    double pullWeightRatio = 0.0;                    // max/min of pullWeights (0 = not loaded)
    std::string disableVendors;                      // FF_DISABLE_DEVICE: "amd"|"nvidia" ("" = none)
    std::string dumpMapFile;                        // --dump-map <file> (todo 13)
    bool noGpu = false;                             // --no-gpu: force CPU-only search
    bool gpuSearch = false;                         // --gpu-search: try GPU when VRAM allows
};

// Size parser: "<num>[unit]" — unit in {b, k, m, g, ki, mi, gi, kib, mib, gib}
// (case-insensitive); bare number = bytes. Decimal values allowed ("1.5GiB").
// "auto" is NOT a size (host-tier-cap handles it specially). Returns false on
// malformed/empty/negative input. Rounds fractional bytes to nearest whole.
bool parseSize(const std::string& s, uint64_t* out);

// Fraction parser: non-negative decimal string; the caller validates the
// §4.3 [0.10, 1.0] band so the error can name the offending knob.
bool parseFraction(const std::string& s, double* out);

// Device-vram-fraction spec: comma-separated "key=value" (key = vendor name
// "amd"/"nvidia" or a logical device index). Malformed -> false.
bool parseDeviceFractionSpec(const std::string& spec,
                             std::map<std::string, double>* out);

// Reads the FF_* env vars into cfg (defaults already set). Non-empty but
// malformed values are a HARD error (message written to stderr) — a silently
// ignored budget env could oversubscribe; never silently oversubscribe (§4.3).
// Returns 0 on success, -1 on error.
int loadEnv(Config* cfg);

// Loads pull weights (M2, plan todo 10) into cfg->pullWeights: per-vendor
// writeBandwidthGbs from config/m0-benchmarks.json (hand-rolled scanner, no
// third-party libraries — guardrail). FF_PULL_WEIGHTS values already merged by
// loadEnv() take precedence; vendors absent from both fall back to 1.0 at use
// time. A missing/unreadable JSON file is a warning (uniform pulls), but a
// weight <= 0 from any source is a HARD error. Returns 0 on success, -1 on
// error (message on stderr).
int loadPullWeights(Config* cfg);

// Parses the CLI: flags (+ optional =value form) and positionals. CLI values
// override env. Unknown --* flags are rejected (plan guardrail: any other new
// flag is forbidden). Returns 0 on success, -1 on a usage error (message on
// stderr).
int parseArgs(int argc, char** argv, Config* cfg,
              std::vector<std::string>* positionals);

// GPU_PLAN §4.3 validation: reject f not in [0.10,1.0], zero/negative caps,
// non-16-value-aligned or zero slab-size. Resolves --host-tier-cap auto and
// applies --no-host-tier. Returns 0 on success, -1 on error (message on stderr).
int validateConfig(Config* cfg);

// True when slabSizeBytes*8 is divisible by 16, i.e. every slab edge lands on
// a byte boundary (§4.3 correctness invariant).
bool slabSizeValueAligned(uint64_t slabSizeBytes);

}  // namespace ff

#endif  // FF_CONFIG_H
