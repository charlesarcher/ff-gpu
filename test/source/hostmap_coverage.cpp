// hostmap_coverage.cpp — task 9 (B2): runtime enforcement of the hostmap
// write-before-read invariant.
//
// Strategy:
//   1. Run the REAL ff_sieve binary built in POISON mode (ff_sieve_poison,
//      compiled with FF_POISON_HOSTMAP=1): its host prime map starts filled
//      with the 0xA5 sentinel instead of zero. Same allocation path as
//      production; only the fill pattern differs.
//   2. Run leg 65536 in BOTH search modes (default CPU search + --gpu-search).
//   3. Assert rc==0 and that stdout, after timing normalization (the same
//      rewrite scripts/verify.sh applies: 'Prime|Freudenthal time: <digits>'
//      -> 'time: N'), is byte-identical to goldens/out_ff_seg_65536.txt.
//
// A single canonical byte read before its producer writes it carries 0xA5
// bits into a primality verdict, flipping "(prime)" annotations or solution
// lines with near-certainty — so golden equality proves no sentinel byte ever
// influenced a verdict on any consumer path (CPU RunIt, GPU-search H2D feed +
// emit, dump-map expansion ordering).
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
    const std::string goldenPath =
        std::string(FF_COVERAGE_GOLDENS_DIR) + "/out_ff_seg_65536.txt";

    std::printf("=== hostmap_coverage: poison-build write-before-read trap ===\n");
    std::printf("  poison binary: %s\n", poisonBin.c_str());

    std::string golden;
    if (!readWholeFile(goldenPath, &golden)) {
        std::fprintf(stderr, "FAIL: cannot read golden %s\n",
                     goldenPath.c_str());
        return 1;
    }

    struct Case {
        const char* name;
        std::vector<std::string> args;
    };
    const Case cases[] = {
        {"cpu-search(default)", {}},
        {"gpu-search", {"--gpu-search"}},
    };

    int fails = 0;
    for (const Case& c : cases) {
        std::vector<std::string> args = c.args;
        args.push_back("5");
        args.push_back("65536");

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
                    "producer-written before the sentinel could leak ===\n");
        return 0;
    }
    std::fprintf(stderr, "\n=== hostmap_coverage: FAIL (%d case(s)) ===\n",
                 fails);
    return 1;
}
