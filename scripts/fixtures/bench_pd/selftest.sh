#!/usr/bin/env bash
# selftest.sh — GPU-free fixture tests for scripts/bench_per_device.sh helpers.
#
# Extracts the marker-delimited helper block VERBATIM from the production
# script (the real code is tested, never a copy) and exercises it against
# fixed text fixtures in this directory. Deterministic by construction:
# fixed inputs, no sleeps, no GPUs, no network.
#
# Usage: bash scripts/fixtures/bench_pd/selftest.sh
# Exit:  0 = all tests passed, 1 = at least one failure, 2 = harness error.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCRIPT="$ROOT/scripts/bench_per_device.sh"

pass=0 fail=0
ok()  { printf 'PASS: %s\n' "$1"; pass=$((pass+1)); }
bad() { printf 'FAIL: %s\n' "$1"; fail=$((fail+1)); }
check_eq() { # <desc> <expected> <actual>
    if [[ "$2" == "$3" ]]; then ok "$1"
    else bad "$1 | expected [$2] got [$3]"; fi
}

[[ -f "$SCRIPT" ]] || { echo "FATAL: $SCRIPT missing"; exit 2; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

awk '/^# >>> BEGIN bench_pd testable helpers/,/^# <<< END bench_pd testable helpers/' \
    "$SCRIPT" > "$TMP/helpers.sh"
grep -q 'phase_ms()' "$TMP/helpers.sh" || { echo "FATAL: helper-block extraction failed"; exit 2; }
# shellcheck disable=SC1090
source "$TMP/helpers.sh"

FX="$HERE"

echo "== (a) phase_ms: all five timer names (+ legacy phases), prefix-disambiguation trap =="
E="$FX/stderr_five_timers.txt"
check_eq "device enumeration skips earlier (--list-devices) variant line" \
    "124.200" "$(phase_ms "$E" "device enumeration")"
check_eq "budget computation"      "3.410"   "$(phase_ms "$E" "budget computation")"
check_eq "sieve phase"             "268.512" "$(phase_ms "$E" "sieve phase")"
check_eq "search phase"            "1500.001" "$(phase_ms "$E" "search phase")"
check_eq "total"                   "2100.000" "$(phase_ms "$E" "total")"
check_eq "wheel expansion (A1)"    "377.900" "$(phase_ms "$E" "wheel expansion")"
check_eq "hostmap zero-fill"       "41.007"  "$(phase_ms "$E" "hostmap zero-fill")"
check_eq "scheduler teardown"      "12.340"  "$(phase_ms "$E" "scheduler teardown")"
check_eq "search device setup"     "88.250"  "$(phase_ms "$E" "search device setup")"
check_eq "search device teardown"  "7.750"   "$(phase_ms "$E" "search device teardown")"
check_eq "search kernel sub-timer" "1200.000" "$(phase_ms "$E" "search kernel")"

printf 'ff_sieve timing: device enumeration (--list-devices) = 999.123 ms\n' > "$TMP/variant_only.txt"
check_eq "trap: variant-only file satisfies NO plain lookup (empty, not 999.123)" \
    "" "$(phase_ms "$TMP/variant_only.txt" "device enumeration")"

echo "== (b) missing timer line -> EMPTY cell (never zero) =="
M="$FX/stderr_missing_lines.txt"
check_eq "absent search kernel -> empty"  "" "$(phase_ms "$M" "search kernel")"
check_eq "absent total -> empty"          "" "$(phase_ms "$M" "total")"
check_eq "absent wheel expansion -> empty" "" "$(phase_ms "$M" "wheel expansion")"
check_eq "present line still parses"      "268.512" "$(phase_ms "$M" "sieve phase")"
R="$FX/stderr_no_timers.txt"
for name in "device enumeration" "wheel expansion" "hostmap zero-fill" \
            "scheduler teardown" "search device setup" "search device teardown" "total"; do
    check_eq "ref-like stderr: '$name' -> empty" "" "$(phase_ms "$R" "$name")"
done
check_eq "nonexistent stderr file -> empty" "" "$(phase_ms "$TMP/does_not_exist.txt" "total")"

echo "== (c) normalization + sha256 gate semantics =="
HA=$(stdout_sha256 "$FX/stdout_leg_a.txt")
HB=$(stdout_sha256 "$FX/stdout_leg_b.txt")
HC=$(stdout_sha256 "$FX/stdout_leg_corrupt.txt")
check_eq "identical stdout, different timing digits -> SAME hash" "$HA" "$HB"
[[ "$HA" != "$HC" ]] && ok "corrupted solution line -> DIFFERENT hash" \
                    || bad "corrupted solution line must change the hash"
check_eq "normalize_stdout rewrites Prime digits" \
    "Prime time: N μs" "$(printf 'Prime time: 42 μs\n' | normalize_stdout)"
check_eq "normalize_stdout rewrites Freudenthal digits" \
    "Freudenthal time: N μs" "$(printf 'Freudenthal time: 7 μs\n' | normalize_stdout)"
check_eq "normalization is idempotent (already-N golden text unchanged)" \
    "Prime time: N μs" "$(printf 'Prime time: N μs\n' | normalize_stdout)"
sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/' "$FX/stdout_leg_a.txt" > "$TMP/golden_style.txt"
check_eq "live stdout and pre-normalized golden digest identically (gate symmetry)" \
    "$HA" "$(stdout_sha256 "$TMP/golden_style.txt")"

echo "== (d) unaccounted_ms arithmetic on canned rows =="
check_eq "fully-populated row: 1000 − 950" \
    "50.000" "$(unaccounted_ms 1000.000 100.000 50.000 300.000 200.000 150.000 40.000 30.000 60.000 20.000)"
check_eq "one missing component -> EMPTY (NA), not partial-sum guess" \
    "" "$(unaccounted_ms 1000.000 100.000 "" 300.000 200.000 150.000 40.000 30.000 60.000 20.000)"
check_eq "empty total -> EMPTY" \
    "" "$(unaccounted_ms "" 100.000 50.000 300.000 200.000 150.000 40.000 30.000 60.000 20.000)"
check_eq "components exceeding total report NEGATIVE honestly (no clamping)" \
    "-20.000" "$(unaccounted_ms 100.000 60.000 60.000 0 0 0 0 0 0 0)"
check_eq "zero components: unaccounted == total" \
    "1000.000" "$(unaccounted_ms 1000.000 0 0 0 0 0 0 0 0 0)"

echo "== sanity: median helper (summary path) =="
check_eq "median of 3" "2.000" "$(median 1 2 3)"
check_eq "median of 5" "3.000" "$(median 5 1 4 2 3)"

echo ""
echo "selftest summary: $pass passed, $fail failed"
[[ "$fail" -eq 0 ]] || exit 1
exit 0
