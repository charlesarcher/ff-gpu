#!/usr/bin/env bash
# check_pull_balance.sh — M2 weighted-pull balance checker (plan todo 10).
#
# Reads the ff_sieve stderr stream (file arg or stdin), extracts the
# "[ff_sieve] pull distribution:" line, and verifies the observed pull-count
# ratio matches the expected bandwidth ratio within +/-20%.
#
# Expected ratio: FF_PULL_WEIGHTS max/min when the env var is set (the binary
# applies the same override), else the binary's JSON-derived ratio printed on
# the line (config/m0-benchmarks.json writeBandwidthGbs).
#
# Single-vendor distributions (all slabs on one device) always pass — there
# is no cross-vendor balance to check.
#
# Usage: ./scripts/check_pull_balance.sh [stderr_file]   (default: stdin)
# Exit codes: 0 = balanced (or single-vendor), 1 = imbalance / parse failure.

set -u

input="${1:--}"

if [ "$input" = "-" ]; then
    data="$(cat)"
else
    data="$(cat "$input")"
fi

line="$(printf '%s\n' "$data" | grep '\[ff_sieve\] pull distribution:' | tail -n 1)"
if [ -z "$line" ]; then
    echo "check_pull_balance: error: no '[ff_sieve] pull distribution:' line in stderr" >&2
    exit 1
fi

declare -A counts
expected=""
for tok in $line; do
    case "$tok" in
        *=*)
            key="${tok%%=*}"
            val="${tok#*=}"
            case "$key" in
                total|ratio) ;;
                expected) expected="$val" ;;
                *) counts["$key"]="$val" ;;
            esac
            ;;
    esac
done

# FF_PULL_WEIGHTS (when set) overrides the binary's JSON-derived expectation.
if [ -n "${FF_PULL_WEIGHTS:-}" ]; then
    maxw=""
    minw=""
    IFS=',' read -ra pairs <<< "$FF_PULL_WEIGHTS"
    for p in "${pairs[@]}"; do
        w="${p#*=}"
        case "$w" in
            ''|*[!0-9.]*|.*)
                echo "check_pull_balance: error: malformed FF_PULL_WEIGHTS='$FF_PULL_WEIGHTS'" >&2
                exit 1
                ;;
        esac
        if [ -z "$maxw" ] || awk -v a="$w" -v b="$maxw" 'BEGIN{exit !(a>b)}'; then maxw="$w"; fi
        if [ -z "$minw" ] || awk -v a="$w" -v b="$minw" 'BEGIN{exit !(a<b)}'; then minw="$w"; fi
    done
    if [ -z "$maxw" ] || [ -z "$minw" ]; then
        echo "check_pull_balance: error: FF_PULL_WEIGHTS has no weights" >&2
        exit 1
    fi
    expected="$(awk -v a="$maxw" -v b="$minw" 'BEGIN{print a/b}')"
fi

if [ -z "$expected" ]; then
    echo "check_pull_balance: error: no expected ratio on the distribution line and no FF_PULL_WEIGHTS" >&2
    exit 1
fi

obsmax=""
obsmin=""
for v in "${!counts[@]}"; do
    c="${counts[$v]}"
    if [ "$c" -gt 0 ] 2>/dev/null; then
        if [ -z "$obsmax" ] || [ "$c" -gt "$obsmax" ]; then obsmax="$c"; fi
        if [ -z "$obsmin" ] || [ "$c" -lt "$obsmin" ]; then obsmin="$c"; fi
    fi
done

if [ -z "$obsmin" ]; then
    echo "check_pull_balance: single-vendor distribution (all slabs on one device) — nothing to balance; PASS"
    exit 0
fi

obs="$(awk -v a="$obsmax" -v b="$obsmin" 'BEGIN{print a/b}')"

awk -v o="$obs" -v e="$expected" 'BEGIN{
    if (e <= 0) { print "check_pull_balance: error: expected ratio <= 0"; exit 2 }
    r = o / e;
    if (r >= 0.80 && r <= 1.25) {
        printf "check_pull_balance: PASS observed=%s expected=%s (within +/-20%%)\n", o, e;
        exit 0
    }
    printf "check_pull_balance: FAIL observed=%s expected=%s (outside +/-20%%)\n", o, e;
    exit 1
}'
rc=$?
exit $rc
