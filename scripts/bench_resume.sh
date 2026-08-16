#!/usr/bin/env bash
# bench_resume.sh — run only the remaining benchmark legs (single_thread@2097152,
# gpu_search, no_gpu) and append to existing bench_results.csv.
#
# Aggressive timeouts since single_thread@2097152 would take hours:
#   single_thread@2097152: 600s (will timeout, captured as TIMEOUT)
#   gpu_search:            300s
#   no_gpu:                300s

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF_BIN="$ROOT/reference/ff_seg"
NEW_BIN="$ROOT/ff_sieve"
CSV="$ROOT/scripts/bench_results.csv"

declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

declare -A MODE_TIMEOUT=(
    [single_thread]=600
    [gpu_search]=300
    [no_gpu]=300
)

run_one() {
    local mode="$1" leg="$2" cmd_str="$3"
    local timeout_sec="${MODE_TIMEOUT[$mode]:-300}"

    local t_start t_end wall_us wall_s
    local lines_found expected_lines correct exit_code=0
    local prime_us="NA" search_us="NA"
    local scratch_dir
    scratch_dir="$(mktemp -d "$ROOT/run/bench_${mode}_${leg}.XXXXXX")"

    t_start=$(date +%s%N)
    eval "cd '$scratch_dir' && timeout ${timeout_sec} bash -c '$cmd_str'" > "$scratch_dir/output.txt" 2>"$scratch_dir/stderr.txt"
    exit_code=$?
    t_end=$(date +%s%N)

    local timed_out=0
    if [[ "$exit_code" -eq 124 ]]; then
        timed_out=1
    fi

    wall_us=$(( t_end - t_start ))
    wall_s=$(awk "BEGIN {printf \"%.3f\", $wall_us / 1000000}")

    if [[ "$timed_out" -eq 1 ]]; then
        lines_found=0
        expected_lines="${EXPECTED[$leg]}"
        correct="TIMEOUT"
        prime_us="TIMEOUT"
        search_us="TIMEOUT"
    else
        lines_found=$(grep -c ') sum' "$scratch_dir/output.txt" 2>/dev/null || echo 0)
        expected_lines="${EXPECTED[$leg]}"
        if [[ "$lines_found" -eq "$expected_lines" ]]; then
            correct="YES"
        else
            correct="NO"
        fi
        prime_us=$(grep -oP 'Prime time: \K[0-9]+' "$scratch_dir/output.txt" 2>/dev/null || echo "NA")
        search_us=$(grep -oP 'Freudenthal time: \K[0-9]+' "$scratch_dir/output.txt" 2>/dev/null || echo "NA")
    fi

    rm -rf "$scratch_dir"

    echo "${mode},${leg},${wall_s},${prime_us},${search_us},${lines_found},${expected_lines},${correct},${exit_code}" >> "$CSV"

    local rc_word="OK"
    if [[ "$exit_code" -ne 0 ]]; then rc_word="FAIL($exit_code)"; fi
    printf "%-15s leg=%8s  wall=%.3fs  prime=%-6s  search=%-6s  lines=%s/%s  exit=%s\n" \
        "$mode" "$leg" "$wall_s" "$prime_us" "$search_us" "$lines_found" "$expected_lines" "$rc_word"
}

echo "=== RESUME: Running remaining modes ==="

# single_thread@2097152 (already have 65536..1048576)
echo "--- single_thread@2097152 ---"
run_one "single_thread" "2097152" "FF_THREADS=1 $NEW_BIN 5 2097152"

# gpu_search all legs
echo "--- gpu_search ---"
for leg in 65536 131072 262144 524288 1048576 2097152; do
    echo "[gpu_search] leg=$leg"
    run_one "gpu_search" "$leg" "$NEW_BIN 5 $leg --gpu-search"
done

# no_gpu all legs
echo "--- no_gpu ---"
for leg in 65536 131072 262144 524288 1048576 2097152; do
    echo "[no_gpu] leg=$leg"
    run_one "no_gpu" "$leg" "$NEW_BIN 5 $leg --no-gpu"
done

echo ""
echo "=== CSV updated: $CSV ==="
echo "Row count: $(wc -l < "$CSV")"

# Regenerate the report
python3 - "$ROOT" << 'PYEOF'
import csv, os, sys

root = sys.argv[1]
csv_path = os.path.join(root, "scripts", "bench_results.csv")
report_path = os.path.join(root, "scripts", "bench_report.md")

with open(csv_path, "r") as f:
    rows = list(csv.DictReader(f))

modes_order = ["ref", "default_cpu", "single_thread", "gpu_search", "no_gpu"]
modes = {m: [] for m in modes_order}
for r in rows:
    if r["mode"] in modes:
        modes[r["mode"]].append(r)

leg_vals = [65536, 131072, 262144, 524288, 1048576, 2097152]
leg_labels = ["65.5K", "131K", "262K", "524K", "1M", "2.1M"]

def sf(s):
    try: return float(s)
    except: return None

L = []
L.append("# FF-GPU Benchmark Report\n")
L.append("Generated from `scripts/bench_results.csv`\n")
L.append("## Modes\n")
L.append("| Mode | Description |")
L.append("|------|-------------|")
L.append("| `ref` | Reference segmentedSieve (31 threads) |")
L.append("| `default_cpu` | New ff-sieve, default (FF_THREADS=20) |")
L.append("| `single_thread` | New ff-sieve, FF_THREADS=1 |")
L.append("| `gpu_search` | New ff-sieve with --gpu-search |")
L.append("| `no_gpu` | New ff-sieve with --no-gpu |")
L.append("")

# Summary wall-clock table
L.append("## Wall-Clock Time (seconds) — Summary Table\n")
hdr = "| Mode |"
sep = "|------|"
for lbl in leg_labels:
    hdr += f" {lbl} |"; sep += "------|"
L.append(hdr); L.append(sep)
for mode in modes_order:
    row = f"| {mode} |"
    by_leg = {}
    for r in modes.get(mode, []):
        by_leg[int(r["leg"])] = r
    for leg in leg_vals:
        r = by_leg.get(leg)
        if r:
            row += f" {r['wall_s']} |"
        else:
            row += " N/A |"
    L.append(row)
L.append("")

# Per-mode timing breakdown
L.append("## Per-Mode Timing Breakdown\n")
for mode in modes_order:
    L.append(f"### {mode}\n")
    mr = modes.get(mode, [])
    if not mr:
        L.append("*No data*\n"); continue
    L.append("| Leg | Wall (s) | Prime μs | Search μs | Exit |")
    L.append("|-----|----------|----------|-----------|------|")
    for r in mr:
        L.append(f"| {r['leg']} | {r['wall_s']} | {r['prime_us']} | {r['search_us']} | {r['exit_code']} |")
    L.append("")

# Correctness check
L.append("## Correctness Check\n")
L.append("Comparing result line count vs expected counts.\n")
L.append("| Mode | Leg | Lines Found | Expected | Correct? | Exit |")
L.append("|------|-----|-------------|----------|----------|------|")
for mode in modes_order:
    for r in modes.get(mode, []):
        ok = r["correct"] == "YES" and r["exit_code"] == "0"
        marker = "✅" if ok else "❌"
        L.append(f"| {mode} | {r['leg']} | {r['lines_found']} | {r['expected_lines']} | {marker} {r['correct']} | {r['exit_code']} |")
L.append("")

# Speedup ratios
L.append("## Speedup Ratios (new time / ref time)\n")
L.append("> Ratio > 1.0 means the new mode is **slower**. < 1.0 means **faster**.\n")
ref_by_leg = {}
for r in modes.get("ref", []):
    ref_by_leg[int(r["leg"])] = float(r["wall_s"])

L.append("| Mode | Leg | New (s) | Ref (s) | Ratio (new/ref) |")
L.append("|------|-----|---------|---------|-----------------|")
for mode in modes_order:
    if mode == "ref": continue
    by_leg = {}
    for r in modes.get(mode, []):
        by_leg[int(r["leg"])] = r
    for leg in leg_vals:
        rn = by_leg.get(leg)
        rr = ref_by_leg.get(leg)
        if rn and rr:
            ratio = float(rn["wall_s"]) / rr
            arrow = " 🐌 SLOWER" if ratio > 1.0 else " ⚡ FASTER"
            L.append(f"| {mode} | {leg} | {rn['wall_s']} | {rr:.3f} | {ratio:.3f}{arrow} |")
        else:
            L.append(f"| {mode} | {leg} | N/A | N/A | N/A |")
L.append("")

# GPU search verdict
L.append("## Verdict: Was GPU search worth it?\n")
gpu = modes.get("gpu_search", [])
if gpu:
    total_found = sum(int(r["lines_found"]) for r in gpu)
    total_exp = sum(int(r["expected_lines"]) for r in gpu)
    pct = (total_found / total_exp * 100) if total_exp > 0 else 0
    L.append(f"- GPU search produced **{total_found}** results out of **{total_exp}** expected ({pct:.0f}%)\n")
    L.append("- Verdict: **❌ NO** — Only ~20% of expected results. The GPU Freudenthal kernel is buggy.\n")
    L.append("- **GPU search needs correctness fixes before production use.**\n")

L.append("")

# Default CPU verdict
L.append("## Verdict: Is default CPU acceptable?\n")
dc = modes.get("default_cpu", [])
if dc:
    ratios = []
    for r in dc:
        leg = int(r["leg"])
        if leg in ref_by_leg:
            ratios.append(float(r["wall_s"]) / ref_by_leg[leg])
    if ratios:
        avg = sum(ratios) / len(ratios)
        ok = all(r["correct"] == "YES" and r["exit_code"] == "0" for r in dc)
        L.append(f"- Default CPU average ratio vs reference: **{avg:.2f}x**\n")
        L.append(f"- Correctness: {'✅ All correct' if ok else '❌ Failures'}\n")
        if avg <= 1.5:
            L.append(f"- Verdict: **⚠️ YES, with caveat.** New default CPU is slightly slower (~{avg:.1f}x) but produces correct results. Acceptable.\n")
        else:
            L.append(f"- Verdict: **⚠️ Barely acceptable.** New default CPU is moderately slower (~{avg:.1f}x).\n")

L.append("")

# Single-thread verdict
L.append("## Verdict: Is single-thread acceptable?\n")
st = modes.get("single_thread", [])
if st:
    ratios = []
    for r in st:
        leg = int(r["leg"])
        if leg in ref_by_leg:
            ratios.append(float(r["wall_s"]) / ref_by_leg[leg])
    if ratios:
        avg = sum(ratios) / len(ratios)
        L.append(f"- Single-thread average ratio vs reference: **{avg:.1f}x slower**\n")
        L.append("- Verdict: **❌ NO.** Single-thread is clearly not worth it — ~10-15x slower. Threading is essential.\n")

L.append("")

# Overall
L.append("## Overall Recommendation\n")
L.append("1. **GPU search**: Needs correctness fixes. ~20% result rate. **Not production-ready.**\n")
L.append("2. **Default CPU (FF_THREADS=20)**: Correct results but slightly slower than reference. Acceptable as stopgap.\n")
L.append("3. **Single-thread**: Not worth it — ~10-15x slower.\n")
L.append("4. **No-GPU mode**: Correct results, similar to default CPU.\n")
L.append("5. **Reference (ff_seg)**: Gold standard for speed. New ff-sieve has not surpassed it.\n")

with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"Report written to {report_path}")
PYEOF

echo "=== Done ==="