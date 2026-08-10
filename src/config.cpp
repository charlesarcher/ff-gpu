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

int loadEnv(Config* cfg)
{
    const char* v = nullptr;
    double d = 0.0;
    if ((v = std::getenv("FF_VRAM_FRACTION")) && *v) {
        if (!parseFraction(v, &d)) {
            std::fprintf(stderr, "[ff_sieve] error: malformed FF_VRAM_FRACTION='%s'\n", v);
            return -1;
        }
        cfg->globalFraction = d;
    }
    if ((v = std::getenv("FF_DEVICE_VRAM_FRACTION")) && *v) {
        std::map<std::string, double> m;
        if (!parseDeviceFractionSpec(v, &m)) {
            std::fprintf(stderr,
                         "[ff_sieve] error: malformed FF_DEVICE_VRAM_FRACTION='%s' "
                         "(expected 'amd=0.9,nvidia=0.8')\n", v);
            return -1;
        }
        cfg->deviceFractions = m;
    }
    uint64_t sz = 0;
    if ((v = std::getenv("FF_VRAM_BUDGET")) && *v) {
        if (!parseSize(v, &sz)) {
            std::fprintf(stderr, "[ff_sieve] error: malformed FF_VRAM_BUDGET='%s' "
                         "(expected e.g. 20GiB)\n", v);
            return -1;
        }
        cfg->hasBudgetCap = true;
        cfg->budgetCapBytes = sz;
    }
    if ((v = std::getenv("FF_SCRATCH")) && *v) {
        if (!parseSize(v, &sz)) {
            std::fprintf(stderr, "[ff_sieve] error: malformed FF_SCRATCH='%s'\n", v);
            return -1;
        }
        cfg->hasScratch = true;
        cfg->scratchBytes = sz;
    }
    if ((v = std::getenv("FF_SLAB_SIZE")) && *v) {
        if (!parseSize(v, &sz)) {
            std::fprintf(stderr, "[ff_sieve] error: malformed FF_SLAB_SIZE='%s'\n", v);
            return -1;
        }
        cfg->slabSizeBytes = sz;
    }
    if ((v = std::getenv("FF_HOST_TIER_CAP")) && *v) {
        if (toLower(v) == "auto") {
            cfg->hasHostTierCap = true;
            cfg->hostTierAuto = true;
        } else if (!parseSize(v, &sz)) {
            std::fprintf(stderr, "[ff_sieve] error: malformed FF_HOST_TIER_CAP='%s' "
                         "(expected a size or 'auto')\n", v);
            return -1;
        } else {
            cfg->hasHostTierCap = true;
            cfg->hostTierCapBytes = sz;
        }
    }
    return 0;
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
                             "(expected 'amd=0.9,nvidia=0.8')\n", v);
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

}  // namespace ff
