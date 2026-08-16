#!/usr/bin/env bash
# Overlap engine verification (todo 12)
# Runs ff_sieve with host-tier-cap to force overlap activity,
# parses stderr for overlap stats, exits 0 if stats present.

set -euo pipefail
cd "$(dirname "$0")/.."

LOGFILE="run/check_overlap.log"
mkdir -p run

echo "[check_overlap] running ff_sieve 5 2097152 with --host-tier-cap=16GiB"
# HIP_VISIBLE_DEVICES=1 forces NVIDIA-only (AMD ROCm JIT compilation is extremely slow on this system)
HIP_VISIBLE_DEVICES=1 ./ff_sieve 5 2097152 --host-tier-cap=16GiB 2>"$LOGFILE" &
PID=$!

# Wait for completion (with 20min timeout for large runs)
( sleep 1200 && kill $PID 2>/dev/null ) &
WATCHER_PID=$!
wait $PID
kill $WATCHER_PID 2>/dev/null
wait $WATCHER_PID 2>/dev/null || true

echo "[check_overlap] checking overlap stats..."
if grep -q "overlap engine" "$LOGFILE"; then
    echo "[check_overlap] PASS: overlap stats present"
    grep "overlap engine" "$LOGFILE" | head -5
    exit 0
else
    echo "[check_overlap] FAIL: no overlap stats found"
    echo "[check_overlap] Last 20 lines of stderr:"
    tail -20 "$LOGFILE"
    exit 1
fi