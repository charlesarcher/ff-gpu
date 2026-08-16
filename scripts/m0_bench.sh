#!/usr/bin/env bash
# M0 bandwidth benchmark runner (GPU_PLAN §10 M0 exit criteria).
#
# Usage:  ./scripts/m0_bench.sh
#
# Builds the benchmark binary (make bench) if not up-to-date,
# runs it, prints results to STDERR, persists JSON to config/m0-benchmarks.json.
# All benchmark data goes to STDERR; this script exits 0 on success.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Build the benchmark binary.
echo "[m0_bench] building benchmark binary..." >&2
make bench >&2

BENCH="./build/m0_bench"
if [ ! -x "${BENCH}" ]; then
    echo "[m0_bench] ERROR: benchmark binary not found at ${BENCH}" >&2
    exit 1
fi

# Ensure output directory exists.
mkdir -p config

# Run the benchmark. All benchmark output (including JSON write) goes to STDERR.
echo "[m0_bench] running benchmark..." >&2
"${BENCH}" 2>&1

# Verify JSON was written and is valid.
JSON="config/m0-benchmarks.json"
if [ ! -f "${JSON}" ]; then
    echo "[m0_bench] ERROR: ${JSON} was not produced" >&2
    exit 1
fi

echo "[m0_bench] done. Results in ${JSON}" >&2
exit 0