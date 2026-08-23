#include "config.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace ff {
namespace {

std::string toLower(std::string s)
{
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parseSizeImpl(const std::string& s, uint64_t* out)
{
    if (s.empty()) return false;
    if (toLower(s) == "auto") return false;   // not a size; caller handles it
    char* end = nullptr;
    double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return false;       // no leading digits
    if (value < 0.0) return false;
    std::string unit = toLower(std::string(end));
    if (!unit.empty() && unit.back() == 'b') unit.pop_back();
    double mult = 1.0;
    if (unit.empty() || unit == "byte") mult = 1.0;
    else if (unit == "k" || unit == "ki") mult = 1024.0;
    else if (unit == "m" || unit == "mi") mult = 1048576.0;
    else if (unit == "g" || unit == "gi") mult = 1073741824.0;
    else return false;
    double bytes = value * mult;
    if (bytes > 18446744073709551615.0) return false;   // uint64 overflow
    uint64_t rounded = uint64_t(bytes + 0.5);
    if (rounded == 0 && bytes > 0.0) rounded = 1;       // e.g. "0.1B"
    *out = rounded;
    return true;
}

// --pull-weights spec: comma-separated "vendor=weight" pairs (vendor
// case-insensitive, weight > 0). Malformed/zero/negative -> false.
bool parsePullWeightsSpec(const std::string& spec,
                          std::map<std::string, double>* out)
{
    out->clear();
    size_t pos = 0;
    for (;;) {
        size_t comma = spec.find(',', pos);
        std::string item = spec.substr(
            pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (item.empty()) return false;
        size_t eq = item.find('=');
        if (eq == std::string::npos) return false;
        std::string vendor = toLower(item.substr(0, eq));
        std::string val = item.substr(eq + 1);
        if (vendor.empty() || val.empty()) return false;
        double w = 0.0;
        if (!parseFraction(val, &w)) return false;
        if (w <= 0.0) return false;
        (*out)[vendor] = w;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return true;
}

bool readFileIntoString(const char* path, std::string* out)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    if (len < 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out->resize(static_cast<size_t>(len));
    size_t got = len > 0
                     ? std::fread(&(*out)[0], 1, static_cast<size_t>(len), f)
                     : 0;
    std::fclose(f);
    return got == static_cast<size_t>(len);
}

// Scans config/m0-benchmarks.json for {"vendor": "...", ...,
// "writeBandwidthGbs": <num>} device entries. Hand-rolled — the JSON library
// guardrail forbids third-party deps and the document is a fixed small file.
bool scanWriteBandwidths(const std::string& text,
                         std::map<std::string, double>* out)
{
    bool found = false;
    size_t pos = 0;
    for (;;) {
        size_t vk = text.find("\"vendor\"", pos);
        if (vk == std::string::npos) break;
        size_t colon = text.find(':', vk);
        if (colon == std::string::npos) break;
        size_t q1 = text.find('"', colon + 1);
        if (q1 == std::string::npos) break;
        size_t q2 = text.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string vendor = text.substr(q1 + 1, q2 - q1 - 1);
        size_t wk = text.find("\"writeBandwidthGbs\"", q2);
        if (wk == std::string::npos) break;
        size_t wcolon = text.find(':', wk);
        if (wcolon == std::string::npos) break;
        char* end = nullptr;
        double val = std::strtod(text.c_str() + wcolon + 1, &end);
        if (end == text.c_str() + wcolon + 1) break;
        (*out)[vendor] = val;
        found = true;
        pos = q2 + 1;
    }
    return found;
}

}  // namespace

bool parseSize(const std::string& s, uint64_t* out) { return parseSizeImpl(s, out); }

bool parseFraction(const std::string& s, double* out)
{
    if (s.empty()) return false;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return false;
    if (v < 0.0) return false;
    *out = v;
    return true;
}

bool parseDeviceFractionSpec(const std::string& spec,
                             std::map<std::string, double>* out)
{
    out->clear();
    size_t pos = 0;
    for (;;) {
        size_t comma = spec.find(',', pos);
        std::string item = spec.substr(
            pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (item.empty()) return false;                 // trailing/double comma
        size_t eq = item.find('=');
        if (eq == std::string::npos) return false;
        std::string key = item.substr(0, eq);
        std::string val = item.substr(eq + 1);
        if (key.empty() || val.empty()) return false;
        double f = 0.0;
        if (!parseFraction(val, &f)) return false;
        (*out)[key] = f;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return true;
}

int parseArgs(int argc, char** argv, Config* cfg,
              std::vector<std::string>* positionals)
{
    for (int i = 1; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok.rfind("--", 0) != 0) {
            positionals->push_back(tok);
            continue;
        }
        std::string name, value;
        bool hasValue = false;
        size_t eq = tok.find('=');
        if (eq != std::string::npos) {
            name = tok.substr(0, eq);
            value = tok.substr(eq + 1);
            hasValue = true;
        } else {
            name = tok;
        }

        auto needValue = [&](const char* flag) -> const char* {
            if (hasValue) return value.c_str();
            if (i + 1 < argc) { ++i; return argv[i]; }
            std::fprintf(stderr, "[ff_sieve] error: %s requires a value\n", flag);
            return nullptr;
        };

        if (name == "--threads") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            char* end = nullptr;
            long n = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || n < 0) {
                std::fprintf(stderr, "[ff_sieve] error: --threads must be a non-negative integer\n");
                return -1;
            }
            cfg->threads = static_cast<int>(n);
            continue;
        }
        if (name == "--pull-weights") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            std::map<std::string, double> w;
            if (!parsePullWeightsSpec(v, &w)) {
                std::fprintf(stderr,
                             "[ff_sieve] error: invalid --pull-weights '%s' "
                             "(expected comma-separated 'vendor=weight', e.g. "
                             "'amd=598.75'; weights must be > 0)\n",
                             v);
                return -1;
            }
            cfg->pullWeights = w;
            continue;
        }
        if (name == "--disable-vendor") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            std::string b = toLower(v);
            if (b != "amd") {
                std::fprintf(stderr,
                             "[ff_sieve] error: invalid --disable-vendor '%s' "
                             "(expected 'amd')\n", v);
                return -1;
            }
            cfg->disableVendor = b;
            continue;
        }
        if (name == "--list-devices") {
            cfg->listDevices = true;
            continue;
        }
        if (name == "--no-host-tier") {
            cfg->noHostTier = true;
            continue;
        }
        if (name == "--vram-fraction") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            double f = 0.0;
            if (!parseFraction(v, &f)) {
                std::fprintf(stderr, "[ff_sieve] error: invalid fraction '%s'\n", v);
                return -1;
            }
            cfg->globalFraction = f;
            continue;
        }
        if (name == "--device-vram-fraction") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            std::map<std::string, double> m;
            if (!parseDeviceFractionSpec(v, &m)) {
                std::fprintf(stderr,
                             "[ff_sieve] error: invalid --device-vram-fraction '%s' "
                             "(expected 'amd=0.9')\n", v);
                return -1;
            }
            cfg->deviceFractions = m;
            continue;
        }
        if (name == "--vram-budget") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            uint64_t sz = 0;
            if (!parseSize(v, &sz)) {
                std::fprintf(stderr, "[ff_sieve] error: invalid size '%s' "
                             "(expected e.g. 20GiB or 20971520)\n", v);
                return -1;
            }
            cfg->hasBudgetCap = true;
            cfg->budgetCapBytes = sz;
            continue;
        }
        if (name == "--scratch") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            uint64_t sz = 0;
            if (!parseSize(v, &sz)) {
                std::fprintf(stderr, "[ff_sieve] error: invalid size '%s'\n", v);
                return -1;
            }
            cfg->hasScratch = true;
            cfg->scratchBytes = sz;
            continue;
        }
        if (name == "--slab-size") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            uint64_t sz = 0;
            if (!parseSize(v, &sz)) {
                std::fprintf(stderr, "[ff_sieve] error: invalid size '%s'\n", v);
                return -1;
            }
            cfg->slabSizeBytes = sz;
            continue;
        }
        if (name == "--host-tier-cap") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            if (toLower(v) == "auto") {
                cfg->hasHostTierCap = true;
                cfg->hostTierAuto = true;
            } else {
                uint64_t sz = 0;
                if (!parseSize(v, &sz)) {
                    std::fprintf(stderr, "[ff_sieve] error: invalid size '%s' "
                                 "(expected a size or 'auto')\n", v);
                    return -1;
                }
                cfg->hasHostTierCap = true;
                cfg->hostTierCapBytes = sz;
            }
            continue;
        }
        if (name == "--devices") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            std::string b = toLower(v);
            if (b != "amd" && b != "nvidia") {
                std::fprintf(stderr, "[ff_sieve] error: invalid --devices "
                             "backend '%s' (expected 'amd' or 'nvidia')\n", v);
                return -1;
            }
            cfg->deviceFilter = b;
            continue;
        }
if (name == "--dump-map") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            cfg->dumpMapFile = v;
            continue;
        }
        if (name == "--no-gpu") {
            cfg->noGpu = true;
            continue;
        }
        if (name == "--gpu-search") {
            cfg->gpuSearch = true;
            continue;
        }
        if (name == "--gpu-search-device") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            char* end = nullptr;
            long idx = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || idx < 0) {
                std::fprintf(stderr, "[ff_sieve] error: --gpu-search-device must be a non-negative integer\n");
                return -1;
            }
            cfg->gpuSearchDevice = static_cast<int>(idx);
            cfg->gpuSearch = true;
            continue;
        }
        if (name == "--sieve-device") {
            const char* v = needValue(name.c_str());
            if (!v) return -1;
            char* end = nullptr;
            long idx = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || idx < 0) {
                std::fprintf(stderr, "[ff_sieve] error: --sieve-device must be a non-negative integer\n");
                return -1;
            }
            cfg->sieveDevice = static_cast<int>(idx);
            continue;
        }
        std::fprintf(stderr, "[ff_sieve] error: unknown option '%s'\n", name.c_str());
        return -1;
    }
    return 0;
}

bool slabSizeValueAligned(uint64_t slabSizeBytes)
{
    if (slabSizeBytes > UINT64_MAX / 8) return false;   // 8x overflow guard
    return (slabSizeBytes * 8) % 16 == 0;               // 16 values per byte pair -> byte-aligned edges
}

int validateConfig(Config* cfg)
{
    if (cfg->globalFraction < 0.10 || cfg->globalFraction > 1.0) {
        std::fprintf(stderr,
                     "[ff_sieve] validation error: --vram-fraction %.6g is outside "
                     "[0.10, 1.0]\n", cfg->globalFraction);
        return -1;
    }
    for (const auto& kv : cfg->deviceFractions) {
        if (kv.second < 0.10 || kv.second > 1.0) {
            std::fprintf(stderr,
                         "[ff_sieve] validation error: device fraction '%s=%.6g' is "
                         "outside [0.10, 1.0]\n", kv.first.c_str(), kv.second);
            return -1;
        }
    }
    if (cfg->hasBudgetCap && cfg->budgetCapBytes == 0) {
        std::fprintf(stderr,
                     "[ff_sieve] validation error: --vram-budget must be > 0 "
                     "(omit the flag for no cap)\n");
        return -1;
    }
    if (cfg->hasScratch && cfg->scratchBytes == 0) {
        std::fprintf(stderr, "[ff_sieve] validation error: --scratch must be > 0\n");
        return -1;
    }
    if (cfg->slabSizeBytes == 0) {
        std::fprintf(stderr, "[ff_sieve] validation error: --slab-size must be > 0\n");
        return -1;
    }
    if (!slabSizeValueAligned(cfg->slabSizeBytes)) {
        std::fprintf(stderr,
                     "[ff_sieve] validation error: --slab-size %llu B is not "
                     "16-value aligned (slabSize*8 must be divisible by 16 so slab "
                     "edges land on byte boundaries)\n",
                     static_cast<unsigned long long>(cfg->slabSizeBytes));
        return -1;
    }
    if (cfg->noHostTier) {
        cfg->hostTierCapBytes = 0;   // --no-host-tier force-disables the overflow tier
        cfg->hostTierAuto = false;
    }
    if (cfg->hostTierAuto) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long psz = sysconf(_SC_PAGE_SIZE);
        uint64_t ram = (pages > 0 && psz > 0) ? uint64_t(pages) * uint64_t(psz) : 0;
        const uint64_t kReserve = 4ull << 30;   // "auto" = host RAM - 4 GiB
        cfg->hostTierCapBytes = ram > kReserve ? ram - kReserve : 0;
        cfg->hostTierAuto = false;
        if (cfg->hostTierCapBytes == 0)
            std::fprintf(stderr, "[ff_sieve] warning: host RAM < 4 GiB; "
                                 "--host-tier-cap auto resolves to 0\n");
    }
    return 0;
}

int loadPullWeights(Config* cfg)
{
    std::map<std::string, double> jsonWeights;
    std::string text;
    if (readFileIntoString("config/m0-benchmarks.json", &text)) {
        if (scanWriteBandwidths(text, &jsonWeights)) {
            for (const auto& kv : jsonWeights) {
                if (cfg->pullWeights.find(kv.first) == cfg->pullWeights.end())
                    cfg->pullWeights[kv.first] = kv.second;
            }
        } else {
            std::fprintf(stderr,
                         "[ff_sieve] warning: config/m0-benchmarks.json contains "
                         "no writeBandwidthGbs entries; pull weights are uniform\n");
        }
    } else {
        std::fprintf(stderr,
                     "[ff_sieve] warning: config/m0-benchmarks.json unreadable; "
                     "pull weights are uniform\n");
    }
    double minW = 0.0, maxW = 0.0;
    for (const auto& kv : cfg->pullWeights) {
        if (kv.second <= 0.0) {
            std::fprintf(stderr, "[ff_sieve] error: pull weight '%s=%.6g' must "
                                 "be > 0\n", kv.first.c_str(), kv.second);
            return -1;
        }
        if (minW == 0.0 || kv.second < minW) minW = kv.second;
        if (kv.second > maxW) maxW = kv.second;
    }
    if (!cfg->pullWeights.empty()) cfg->pullWeightRatio = maxW / minW;
    return 0;
}

}  // namespace ff
