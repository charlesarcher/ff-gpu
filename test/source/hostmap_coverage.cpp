// hostmap_coverage.cpp — task 9 (B2) + task 18 extension: runtime enforcement
// of the hostmap write-before-read invariant, now across BOTH expansion
// schedules (task-18 tiled two-phase epochs AND the adaptive in-place path).
//
// Strategy:
//   1. Run the REAL ff_sieve binary built in POISON mode (ff_sieve_poison,
//      compiled with FF_POISON_HOSTMAP=1): its host prime map starts filled
//      with the 0xA5 sentinel instead of zero. Same allocation path as
//      production; only the fill pattern differs.
//   2. Search-mode cases assert rc==0 and that stdout, after timing
//      normalization (the same rewrite scripts/verify.sh applies:
//      'Prime|Freudenthal time: <digits>' -> 'time: N'), is byte-identical to
//      the leg's golden. The run matrix spans:
//        - leg 65536 default + --gpu-search (expansion is single-region
//          ADAPTIVE IN-PLACE here; gpu-search SKIPS expansion entirely),
//        - leg 524288 default slab (ONE 546 MiB region -> 9 whole-group
//          64 MiB tiles -> the TILED two-phase epoch path),
//        - leg 524288 --slab-size=128MiB (mixed grid: four 2-tile regions +
//          one single-tile in-place region in the same run),
//        - leg 524288 --slab-size=32MiB (every region single-tile: adaptive
//          everywhere).
//   3. Dump-map case: leg 524288 with --dump-map under sentinel fill must be
//      BYTE-IDENTICAL to the CPU reference map (reference/ff_seg --dump-map).
//      Unlike verdict-driven stdout equality, this proves EVERY canonical byte
//      was producer-written — a single unwritten tile window would carry 0xA5.
//
// A single canonical byte read before its producer writes it carries 0xA5
// bits into a primality verdict or the dumped map, flipping "(prime)"
// annotations, solution lines, or map bytes with near-certainty.
//
// Build: cmake --build --preset dev --target hostmap_coverage_test

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef FF_COVERAGE_POISON_BIN
#error "FF_COVERAGE_POISON_BIN must point at the ff_sieve_poison binary"
#endif
#ifndef FF_COVERAGE_REF_BIN
#error "FF_COVERAGE_REF_BIN must point at the reference ff_seg binary"
#endif
#ifndef FF_COVERAGE_GOLDENS_DIR
#error "FF_COVERAGE_GOLDENS_DIR must point at the goldens directory"
#endif

namespace {

std::string normalizeTiming(std::string s) {
    // Reproduces scripts/verify.sh NORM_SED exactly:
    //   s/(Prime|Freudenthal) time: [0-9]+/\1 time: N/
    static const char* kMarkers[] = {"Prime time: ", "Freudenthal time: "};
    for (const char* marker : kMarkers) {
        const size_t mlen = std::strlen(marker);
        size_t pos = 0;
        while ((pos = s.find(marker, pos)) != std::string::npos) {
            size_t digitsEnd = pos + mlen;
            const size_t digitsBegin = digitsEnd;
            while (digitsEnd < s.size() && s[digitsEnd] >= '0' &&
                   s[digitsEnd] <= '9')
                ++digitsEnd;
            if (digitsEnd > digitsBegin) {
                s.replace(digitsBegin, digitsEnd - digitsBegin, "N");
                pos = digitsBegin + 1;
            } else {
                pos = digitsBegin;   // no digit run: leave untouched (as sed)
            }
        }
    }
    return s;
}

bool readWholeFile(const std::string& path, std::string* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

std::string makeTempPath(const char* prefix) {
    std::string tmpl = std::string("/tmp/") + prefix + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const int fd = ::mkstemp(buf.data());
    if (fd < 0) return std::string();
    ::close(fd);   // reopened by the child with O_TRUNC
    return std::string(buf.data());
}

struct RunResult {
    int rc = -1;
    bool spawned = false;
    std::string stdoutText;
    std::string stderrText;
};

bool runCapture(const std::string& bin, const std::vector<std::string>& args,
                RunResult* out) {
    const std::string outPath = makeTempPath("ff_cov_out");
    const std::string errPath = makeTempPath("ff_cov_err");
    if (outPath.empty() || errPath.empty()) return false;

    const pid_t pid = ::fork();
    if (pid < 0) return false;
    if (pid == 0) {
        int fo = ::open(outPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        int fe = ::open(errPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fo < 0 || fe < 0) ::_exit(127);
        ::dup2(fo, STDOUT_FILENO);
        ::dup2(fe, STDERR_FILENO);
        ::close(fo);
        ::close(fe);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(bin.c_str()));
        for (const std::string& a : args)
            argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        ::execv(bin.c_str(), argv.data());
        ::_exit(127);
    }

    int status = 0;
    ::waitpid(pid, &status, 0);
    out->spawned = true;
    out->rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    readWholeFile(outPath, &out->stdoutText);
    readWholeFile(errPath, &out->stderrText);
    ::unlink(outPath.c_str());
    ::unlink(errPath.c_str());
    return true;
}

void printFirstDiff(const std::string& got, const std::string& want) {
    const size_t n = std::min(got.size(), want.size());
    size_t i = 0;
    while (i < n && got[i] == want[i]) ++i;
    std::fprintf(stderr,
                 "  first diff at byte %zu (got %zu B, golden %zu B)\n",
                 i, got.size(), want.size());
    const size_t lo = i >= 80 ? i - 80 : 0;
    const size_t hi = std::min(i + 120, std::max(got.size(), want.size()));
    std::fprintf(stderr, "  got  : %.120s\n",
                 got.substr(lo, hi - lo).c_str());
    std::fprintf(stderr, "  golden: %.120s\n",
                 want.substr(lo, hi - lo).c_str());
}

}  // namespace

int main() {
    const std::string poisonBin = FF_COVERAGE_POISON_BIN;
    const std::string refBin = FF_COVERAGE_REF_BIN;

    std::printf("=== hostmap_coverage: poison-build write-before-read trap "
                "(task 9 + task 18 tiled matrix) ===\n");
    std::printf("  poison binary: %s\n", poisonBin.c_str());
    std::printf("  reference binary: %s\n", refBin.c_str());

    struct Case {
        const char* name;
        std::vector<std::string> args;
        const char* leg;
        bool dumpMap;   // true: byte-compare the dumped map vs reference
        const char* envThreads;  // "" or FF_EXPANSION_THREADS value
    };
    const Case cases[] = {
        {"cpu-search(default) leg=65536 adaptive-inplace", {}, "65536",
         false, ""},
        {"gpu-search(skip-expansion) leg=65536", {"--gpu-search"}, "65536",
         false, ""},
        {"cpu-search(default) leg=524288 tiled-9-tiles", {}, "524288",
         false, ""},
        {"cpu-search(slab=128MiB) leg=524288 mixed-grid",
         {"--slab-size=134217728"}, "524288", false, ""},
        {"cpu-search(slab=32MiB) leg=524288 adaptive-all",
         {"--slab-size=33554432"}, "524288", false, ""},
        {"dump-map(default) leg=524288 tiled full-byte", {}, "524288", true,
         ""},
        {"dump-map(threads=1) leg=1048576 multi-region serialized", {},
         "1048576", true, "1"},
        {"dump-map(threads=4) leg=1048576 multi-region oversubscribed", {},
         "1048576", true, "4"},
    };

    int fails = 0;
    for (const Case& c : cases) {
        // Pool cap for this case (children inherit at exec; reference runs
        // don't care but inherit harmlessly).
        if (!*c.envThreads) ::unsetenv("FF_EXPANSION_THREADS");
        else ::setenv("FF_EXPANSION_THREADS", c.envThreads, 1);

        std::vector<std::string> args = c.args;
        args.push_back("5");
        args.push_back(c.leg);

        std::string goldenPath;
        std::string golden;
        std::string gotMapPath;
        std::string refMapPath;
        if (c.dumpMap) {
            gotMapPath = makeTempPath("ff_cov_map_got");
            refMapPath = makeTempPath("ff_cov_map_ref");
            if (gotMapPath.empty() || refMapPath.empty()) {
                std::fprintf(stderr, "FAIL [%s]: cannot create temp files\n",
                             c.name);
                ++fails;
                continue;
            }
            args.push_back("--dump-map");
            args.push_back(gotMapPath);
        } else {
            goldenPath = std::string(FF_COVERAGE_GOLDENS_DIR) +
                         "/out_ff_seg_" + c.leg + ".txt";
            if (!readWholeFile(goldenPath, &golden)) {
                std::fprintf(stderr, "FAIL [%s]: cannot read golden %s\n",
                             c.name, goldenPath.c_str());
                ++fails;
                continue;
            }
        }

        RunResult r;
        if (!runCapture(poisonBin, args, &r)) {
            std::fprintf(stderr, "FAIL [%s]: spawn failed (errno=%d)\n",
                         c.name, errno);
            ++fails;
            continue;
        }
        if (r.rc != 0) {
            std::fprintf(stderr,
                         "FAIL [%s]: poison binary exited rc=%d\n"
                         "  stderr head: %.600s\n",
                         c.name, r.rc, r.stderrText.c_str());
            ++fails;
            continue;
        }

        if (c.dumpMap) {
            std::vector<std::string> refArgs = {"--dump-map", refMapPath,
                                                "5", c.leg};
            RunResult rr;
            if (!runCapture(refBin, refArgs, &rr) || rr.rc != 0) {
                std::fprintf(stderr,
                             "FAIL [%s]: reference dump-map run failed "
                             "(rc=%d)\n",
                             c.name, rr.rc);
                ++fails;
                continue;
            }
            std::string gotMap, refMap;
            if (!readWholeFile(gotMapPath, &gotMap) ||
                !readWholeFile(refMapPath, &refMap)) {
                std::fprintf(stderr,
                             "FAIL [%s]: cannot read dumped map file(s)\n",
                             c.name);
                ++fails;
                continue;
            }
            ::unlink(gotMapPath.c_str());
            ::unlink(refMapPath.c_str());
            if (gotMap.size() != refMap.size()) {
                std::fprintf(stderr,
                             "FAIL [%s]: map size %zu != reference %zu\n",
                             c.name, gotMap.size(), refMap.size());
                ++fails;
                continue;
            }
            if (gotMap != refMap) {
                size_t diff = 0;
                while (diff < gotMap.size() && gotMap[diff] == refMap[diff])
                    ++diff;
                std::fprintf(stderr,
                             "FAIL [%s]: dumped map differs from reference "
                             "at byte %zu (sentinel leak through an unwritten "
                             "tile window?)\n",
                             c.name, diff);
                ++fails;
                continue;
            }
            std::printf("  PASS [%s]: rc=0, dumped map byte-identical to CPU "
                        "reference (%zu B) under 0xA5 sentinel fill\n",
                        c.name, gotMap.size());
            continue;
        }

        const std::string normalized = normalizeTiming(r.stdoutText);
        if (normalized != golden) {
            std::fprintf(stderr,
                         "FAIL [%s]: stdout not golden-identical after timing "
                         "normalization — a sentinel byte influenced a "
                         "verdict\n",
                         c.name);
            printFirstDiff(normalized, golden);
            ++fails;
            continue;
        }
        std::printf("  PASS [%s]: rc=0, stdout byte-identical to golden "
                    "(%zu B) under 0xA5 sentinel fill\n",
                    c.name, normalized.size());
    }

    if (fails == 0) {
        std::printf("\n=== hostmap_coverage: PASS — every map byte read was "
                    "producer-written before the sentinel could leak "
                    "(tiled + adaptive schedules) ===\n");
        return 0;
    }
    std::fprintf(stderr, "\n=== hostmap_coverage: FAIL (%d case(s)) ===\n",
                 fails);
    return 1;
}
