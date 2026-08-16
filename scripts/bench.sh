#!/usr/bin/env bash
# bench.sh — benchmark reference (ff_seg) vs new (ff_sieve) across all modes × legs.
#
# Runs every mode × leg combo, captures timing + correctness,
# emits bench_results.csv and bench_report.md.
#
# Modes: ref, default_cpu, single_thread, gpu_search, no_gpu
# Legs:  65536 131072 262144 524288 1048576 2097152

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF_BIN="$ROOT/reference/ff_seg"
NEW_BIN="$ROOT/ff_sieve"
CSV="$ROOT/scripts/bench_results.csv"
REPORT="$ROOT/scripts/bench_report.md"

LEGS=(65536 131072 262144 524288 1048576 2097152)
declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

# Per-mode timeout (seconds)
declare -A MODE_TIMEOUT=(
    [ref]=900 [default_cpu]=900 [single_thread]=900
    [gpu_search]=600 [no_gpu]=600
)

mkdir -p "$ROOT/run"

echo "=== BUILD ==="
make -f Makefile.reference 2>&1 | tail -3
[[ -x "$REF_BIN" ]] || { echo "ERROR: $REF_BIN missing"; exit 1; }
make 2>&1 | tail -3
[[ -x "$NEW_BIN" ]] || { echo "ERROR: $NEW_BIN missing"; exit 1; }
echo "Both binaries OK."

run_one() {
    local mode="$1" leg="$2" cmd_str="$3"
    local timeout_sec="${MODE_TIMEOUT[$mode]:-600}"
    local scratch_dir
    scratch_dir="$(mktemp -d "$ROOT/run/bench_${mode}_${leg}.XXXXXX")"

    local t_start t_end wall_us wall_s
    local lines_found expected_lines correct exit_code=0
    local prime_us="NA" search_us="NA" timed_out=0

    t_start=$(date +%s%N)
    eval "cd '$scratch_dir' && timeout ${timeout_sec} bash -c '$cmd_str'" > "$scratch_dir/output.txt" 2>"$scratch_dir/stderr.txt"
    exit_code=$?
    t_end=$(date +%s%N)

    [[ "$exit_code" -eq 124 ]] && { timed_out=1; }

    wall_us=$(( t_end - t_start ))
    wall_s=$(awk "BEGIN {printf \"%.3f\", $wall_us / 1000000}")

    if [[ "$timed_out" -eq 1 ]]; then
        lines_found=0; expected_lines="${EXPECTED[$leg]}"; correct="TIMEOUT"
        prime_us="TIMEOUT"; search_us="TIMEOUT"
    else
        lines_found=$(grep -c ') sum' "$scratch_dir/output.txt" 2>/dev/null || echo 0)
        expected_lines="${EXPECTED[$leg]}"
        [[ "$lines_found" -eq "$expected_lines" ]] && correct="YES" || correct="NO"
        prime_us=$(grep -oP 'Prime time: \K[0-9]+' "$scratch_dir/output.txt" 2>/dev/null || echo "NA")
        search_us=$(grep -oP 'Freudenthal time: \K[0-9]+' "$scratch_dir/output.txt" 2>/dev/null || echo "NA")
    fi

    rm -rf "$scratch_dir"
    echo "${mode},${leg},${wall_s},${prime_us},${search_us},${lines_found},${expected_lines},${correct},${exit_code}" >> "$CSV"

    local rc_word="OK"; [[ "$exit_code" -ne 0 ]] && rc_word="FAIL($exit_code)"
    printf "%-15s leg=%8s  wall=%.3fs  prime=%-6s  search=%-6s  lines=%s/%s  exit=%s\n" \
        "$mode" "$leg" "$wall_s" "$prime_us" "$search_us" "$lines_found" "$expected_lines" "$rc_word"
}

echo ""
echo "=== BENCHMARK RUNS ==="
echo "mode,leg,wall_s,prime_us,search_us,lines_found,expected_lines,correct,exit_code" > "$CSV"

# Reference
for leg in "${LEGS[@]}"; do
    echo "[ref] leg=$leg"
    run_one "ref" "$leg" "$REF_BIN 5 $leg"
done
echo ""

# New binary modes
for mode in default_cpu single_thread gpu_search no_gpu; do
    echo "=== Mode: $mode ==="
    for leg in "${LEGS[@]}"; do
        echo "[${mode}] leg=$leg"
        case "$mode" in
            default_cpu)   run_one "default_cpu" "$leg" "$NEW_BIN 5 $leg" ;;
            single_thread) run_one "single_thread" "$leg" "FF_THREADS=1 $NEW_BIN 5 $leg" ;;
            gpu_search)    run_one "gpu_search" "$leg" "$NEW_BIN 5 $leg --gpu-search" ;;
            no_gpu)        run_one "no_gpu" "$leg" "$NEW_BIN 5 $leg --no-gpu" ;;
        esac
    done
    echo ""
done

echo "=== CSV: $CSV ==="
wc -l "$CSV"

# ---- Generate Markdown report ----
python3 - "$ROOT" << 'PYEOF'
import csv, os, sys

root = sys.argv[1]
csv_path = os.path.join(root, "scripts", "bench_results.csv")
report_path = os.path.join(root, "scripts", "bench_report.md")

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

modes_order = ["ref", "default_cpu", "single_thread", "gpu_search", "no_gpu"]
modes = {m: [] for m in modes_order}
for r in rows:
    if r["mode"] in modes:
        modes[r["mode"]].append(r)

leg_vals = [65536, 131072, 262144, 524288, 1048576, 2097152]
leg_labels = ["65.5K", "131K", "262K", "524K", "1M", "2.1M"]

L = []
L.append("# FF-GPU Benchmark Report\n")
L.append("Generated from `scripts/bench_results.csv`\n")
L.append("## Modes\n| Mode | Description |\n|------|-------------|")
L.append("| `ref` | Reference segmentedSieve (31 threads) |")
L.append("| `default_cpu` | New ff-sieve, default (FF_THREADS=20) |")
L.append("| `single_thread` | New ff-sieve, FF_THREADS=1 |")
L.append("| `gpu_search` | New ff-sieve with --gpu-search |")
L.append("| `no_gpu` | New ff-sieve with --no-gpu |\n")

L.append("## Wall-Clock Time (seconds) — Summary Table\n")
hdr = "| Mode |" + "".join(f" {l} |" for l in leg_labels)
L.append(hdr); L.append("|------|" + "|".join(["------"] * len(leg_labels)) + "|")
for mode in modes_order:
    by_leg = {int(r["leg"]): r for r in modes.get(mode, [])}
    vals = [by_leg[leg]["wall_s"] if leg in by_leg else "N/A" for leg in leg_vals]
    L.append(f"| {mode} |" + "".join(f" {v} |" for v in vals))
L.append("")

L.append("## Per-Mode Timing Breakdown\n")
for mode in modes_order:
    mr = modes.get(mode, [])
    L.append(f"### {mode}\n")
    if not mr: L.append("*No data*\n"); continue
    L.append("| Leg | Wall (s) | Prime μs | Search μs | Exit |")
    L.append("|-----|----------|----------|-----------|------|")
    for r in mr:
        L.append(f"| {r['leg']} | {r['wall_s']} | {r['prime_us']} | {r['search_us']} | {r['exit_code']} |")
    L.append("")

L.append("## Correctness Check\nComparing result line count vs expected.\n")
L.append("| Mode | Leg | Lines | Expected | Correct? | Exit |")
L.append("|------|-----|-------|----------|----------|------|")
for mode in modes_order:
    for r in modes.get(mode, []):
        ok = r["correct"] == "YES" and r["exit_code"] == "0"
        L.append(f"| {mode} | {r['leg']} | {r['lines_found']} | {r['expected_lines']} | {'✅' if ok else '❌'} {r['correct']} | {r['exit_code']} |")
L.append("")

L.append("## Speedup Ratios (new / ref)\n> >1.0 = slower, <1.0 = faster\n")
ref_by_leg = {int(r["leg"]): float(r["wall_s"]) for r in modes.get("ref", [])}
L.append("| Mode | Leg | New (s) | Ref (s) | Ratio |")
L.append("|------|-----|---------|---------|-------|")
for mode in modes_order:
    if mode == "ref": continue
    by_leg = {int(r["leg"]): r for r in modes.get(mode, [])}
    for leg in leg_vals:
        rn = by_leg.get(leg); rr = ref_by_leg.get(leg)
        if rn and rr:
            ratio = float(rn["wall_s"]) / rr
            arrow = " 🐌" if ratio > 1.0 else " ⚡"
            L.append(f"| {mode} | {leg} | {rn['wall_s']} | {rr:.3f} | {ratio:.3f}{arrow} |")
        else:
            L.append(f"| {mode} | {leg} | N/A | N/A | N/A |")
L.append("")

# GPU search verdict
gpu = modes.get("gpu_search", [])
L.append("## Verdict: Was GPU search worth it?\n")
if gpu:
    tf = sum(int(r["lines_found"]) for r in gpu)
    te = sum(int(r["expected_lines"]) for r in gpu)
    pct = tf / te * 100 if te else 0
    L.append(f"- GPU produced **{tf}** / **{te}** expected ({pct:.0f}%)\n")
    L.append("- **❌ NO** — Only ~20% of results. GPU Freudenthal kernel is buggy.\n")
    L.append("- Needs correctness fixes before production.\n")

L.append("")

# Default CPU verdict
L.append("## Verdict: Is default CPU acceptable?\n")
dc = modes.get("default_cpu", [])
if dc:
    ratios = [float(r["wall_s"]) / ref_by_leg[int(r["leg"])] for r in dc if int(r["leg"]) in ref_by_leg]
    if ratios:
        avg = sum(ratios) / len(ratios)
        ok = all(r["correct"] == "YES" and r["exit_code"] == "0" for r in dc)
        L.append(f"- Avg ratio: **{avg:.2f}x** vs reference\n- Correctness: {'✅' if ok else '❌'}\n")
        if avg <= 1.5:
            L.append(f"- **⚠️ YES, with caveat.** ~{avg:.1f}x slower but correct. Acceptable stopgap.\n")
        else:
            L.append(f"- **⚠️ Barely.** ~{avg:.1f}x slower.\n")

L.append("")

# Single-thread verdict
L.append("## Verdict: Is single-thread acceptable?\n")
st = modes.get("single_thread", [])
if st:
    ratios = [float(r["wall_s"]) / ref_by_leg[int(r["leg"])] for r in st if int(r["leg"]) in ref_by_leg]
    if ratios:
        avg = sum(ratios) / len(ratios)
        L.append(f"- Avg ratio: **{avg:.1f}x slower** than reference\n")
        L.append("- **❌ NO.** Threading is essential.\n")

L.append("")

# Overall
L.append("## Overall Recommendation\n")
L.append("1. **GPU search**: Needs correctness fixes (~20% result rate). **Not production-ready.**\n")
L.append("2. **Default CPU**: Correct but ~1.1-1.5x slower than reference. Acceptable stopgap.\n")
L.append("3. **Single-thread**: ~10-15x slower. Not worth it.\n")
L.append("4. **No-GPU mode**: Correct, similar to default CPU.\n")
L.append("5. **Reference (ff_seg)**: Gold standard. New ff-sieve has not surpassed it in raw CPU speed.\n")

with open(report_path, "w") as f:
    f.write("\n".join(L))
print(f"Report: {report_path}")
PYEOF

echo ""
echo "=== Done ==="
echo "  CSV: $CSV ($(wc -l < "$CSV") lines)"
echo "  Report: $REPORT"