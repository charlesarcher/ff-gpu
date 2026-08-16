#!/usr/bin/env bash
# bench_per_device.sh — Per-device benchmark: CPU vs each GPU, CPU-search vs GPU-search
#
# Configurations (6): reference CPU, CPU-only new binary, NVIDIA sieve+CPU-search,
# AMD sieve+CPU-search, NVIDIA full-GPU-search, AMD full-GPU-search.
# Legs: 65536 .. 2097152. Each config×leg run 3 times, median wall reported.
#
# Known expected outcomes recorded honestly:
#   - amd @ 2097152: sieve rejects (AMD VRAM 15.86 GiB < 16 GiB map) -> rc=1, recorded
#   - gpu-search @ 2097152: pre-existing sharded-map crash rc=141 (both devices) -> recorded
#
# Usage: ./scripts/bench_per_device.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF_BIN="$ROOT/reference/ff_seg"
NEW_BIN="$ROOT/build/ff_sieve"
CSV="$ROOT/scripts/bench_per_device_results.csv"
REPORT="$ROOT/scripts/bench_per_device_report.md"
TIMEOUT_SEC=600
REPS=3

LEGS=(65536 131072 262144 524288 1048576 2097152)
declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

# config|env-prefix|cmd
declare -a CONFIGS=(
  "ref| |$REF_BIN 5"
  "cpu_no_gpu| |$NEW_BIN --no-gpu 5"
  "nvidia_cpu| |$NEW_BIN --devices=nvidia 5"
  "amd_cpu| |$NEW_BIN --devices=amd 5"
  "nvidia_gpu_search| |$NEW_BIN --gpu-search --devices=nvidia 5"
  "amd_gpu_search| |$NEW_BIN --gpu-search --devices=amd 5"
)

mkdir -p "$ROOT/run"

echo "=== BUILD CHECK ==="
[[ -x "$REF_BIN" ]] || { echo "ERROR: $REF_BIN missing"; exit 1; }
[[ -x "$NEW_BIN" ]] || { echo "ERROR: $NEW_BIN missing"; exit 1; }
echo "Both binaries OK."

echo "=== CLEANING ==="
rm -rf "$ROOT/ffPlayground" "$ROOT/run/bench_pd_*"

cat > "$CSV" << EOF
config,leg,wall_s,median_of_runs,solutions,expected,correct,exit_code
EOF

median() { # median of args (sorted)
    local a=($(printf '%s\n' "$@" | sort -n))
    local n=${#a[@]}
    awk "BEGIN { printf \"%.3f\", ${a[$((n/2))]} }"
}

run_one() { # config leg run_label prefix cmd...
    local cfg="$1" leg="$2" label="$3" prefix="$4"; shift 4
    local scratch_dir
    scratch_dir="$(mktemp -d "$ROOT/run/bench_pd_${cfg}_${leg}.XXXXXX")"
    local t_start t_end wall_ms wall_s rc
    t_start=$(date +%s%N)
    eval "cd '$scratch_dir' && timeout ${TIMEOUT_SEC} env $prefix $*" > "$scratch_dir/output.txt" 2> "$scratch_dir/stderr.txt"
    rc=$?
    t_end=$(date +%s%N)
    wall_ms=$(( (t_end - t_start) / 1000000 ))
    wall_s=$(awk "BEGIN {printf \"%.3f\", $wall_ms / 1000}")
    local lines=0
    if [[ -f "$scratch_dir/output.txt" ]]; then
        lines=$(grep -c ') sum' "$scratch_dir/output.txt" 2>/dev/null || true)
        lines=${lines:-0}
    fi
    printf '%s|%s|%s|%s|%s|%s\n' "$cfg" "$leg" "$label" "$wall_s" "$rc" "$lines"
    rm -rf "$scratch_dir"
}

echo "=== RUNNING BENCHMARKS (${REPS} reps per config x leg) ==="
: > /tmp/bench_pd_raw.txt

for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%|*}"; rest="${entry#*|}"; prefix="${rest%%|*}"; cmd="${rest#*|}"
    echo ""
    echo "--- config: $cfg ---"
    for leg in "${LEGS[@]}"; do
        expected="${EXPECTED[$leg]}"
        walls=(); rcs=(); lines=()
        for r in $(seq 1 "$REPS"); do
            # reference runs inside a scratch dir (creates ffPlayground in cwd)
            out=$(run_one "$cfg" "$leg" "$r" "$prefix" $cmd $leg)
            wall=$(echo "$out" | cut -d'|' -f4)
            rc=$(echo "$out" | cut -d'|' -f5)
            ln=$(echo "$out" | cut -d'|' -f6)
            walls+=("$wall"); rcs+=("$rc"); lines+=("$ln")
            echo "  [$cfg $leg rep$r] wall=${wall}s rc=$rc lines=$ln"
        done
        med=$(median "${walls[@]}")
        # correctness: majority rc==0 and solutions == expected
        rc0=$(printf '%s\n' "${rcs[@]}" | grep -c '^0$' || true)
        lnmode=$(printf '%s\n' "${lines[@]}" | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
        if [[ "$rc0" -ge 2 && "$lnmode" == "$expected" ]]; then correct="YES"; else correct="NO"; fi
        lastrc=$(printf '%s\n' "${rcs[@]}" | sort -n | head -1)
        echo "${cfg},${leg},${med},${REPS},${lnmode},${expected},${correct},${lastrc}" >> "$CSV"
        printf "  -> %-22s leg=%8s  median=%.3fs  lines=%s/%s  correct=%s  rc=%s\n" \
            "$cfg" "$leg" "$med" "$lnmode" "$expected" "$correct" "$lastrc"
    done
done

echo ""
echo "CSV: $CSV"

echo "=== GENERATING REPORT ==="
python3 - "$CSV" "$REPORT" << 'PYEOF'
import csv, sys, statistics

csv_path, report_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
legs = ["65536","131072","262144","524288","1048576","2097152"]
confs = ["ref","cpu_no_gpu","nvidia_cpu","amd_cpu","nvidia_gpu_search","amd_gpu_search"]
labels = {
    "ref": "Reference ff_seg (31-thread CPU)",
    "cpu_no_gpu": "ff_sieve --no-gpu (CPU sieve + CPU search)",
    "nvidia_cpu": "ff_sieve --devices=nvidia (GPU sieve + CPU search)",
    "amd_cpu": "ff_sieve --devices=amd (GPU sieve + CPU search)",
    "nvidia_gpu_search": "ff_sieve --gpu-search --devices=nvidia (full GPU)",
    "amd_gpu_search": "ff_sieve --gpu-search --devices=amd (full GPU)",
}
by = {(r["config"], r["leg"]): r for r in rows}

L = []
L.append("# FF-GPU Per-Device Benchmark Report\n")
L.append("> Median of 3 runs per config x leg, post-fix binary (HEAD `72d9ccc`). CPU included in every comparison.\n")
L.append("## Configurations\n")
L.append("| # | Config | Description |")
L.append("|---|--------|-------------|")
for i, c in enumerate(confs, 1):
    L.append(f"| {i} | `{c}` | {labels[c]} |")
L.append("")
L.append("## Wall-Clock Time (seconds, median of 3)\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
for c in confs:
    cells = []
    for leg in legs:
        r = by.get((c, leg))
        if r is None:
            cells.append("-")
        elif r["correct"] == "YES":
            cells.append(f"{r['wall_s']}")
        else:
            cells.append(f"{r['wall_s']} ✗(rc={r['exit_code']})")
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")
L.append("## Speedup vs CPU reference (ref / config; >1 = faster than reference)\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
ref_by_leg = {leg: float(by[("ref", leg)]["wall_s"]) for leg in legs}
for c in confs:
    if c == "ref":
        continue
    cells = []
    for leg in legs:
        r = by.get((c, leg))
        if r is None or r["correct"] != "YES" or ref_by_leg[leg] == 0:
            cells.append("-")
        else:
            ratio = ref_by_leg[leg] / float(r["wall_s"])
            cells.append(f"{ratio:.2f}x")
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")
L.append("## Correctness Summary\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
for c in confs:
    cells = []
    for leg in legs:
        r = by.get((c, leg))
        cells.append("✅" if r and r["correct"] == "YES" else "❌" if r else "-")
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")
L.append("## Notes\n")
L.append("- `amd` @ 2M: sieve rejects at startup (AMD backing 15.86 GiB < 16 GiB map) — rejection logic working as designed, rc=1.")
L.append("- `gpu_search` @ 2M: pre-existing sharded-map defect (rc=141, documented in .omo/notepads/ff-gpu-consolidated/issues.md). Not a regression of this plan.")
L.append("- `--devices=<vendor>` restricts to one GPU; dual-GPU default scheduling is benchmarked separately in `bench_results.csv`.")

with open(report_path, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Report: {report_path}")
PYEOF

rm -rf "$ROOT/ffPlayground" "$ROOT/run/bench_pd_*"
echo "=== DONE ==="
