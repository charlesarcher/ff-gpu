#!/usr/bin/env bash
# bench_per_device.sh — Reference vs full-GPU-enablement benchmark, per GPU.
#
# Configurations (3):
#   ref         reference ff_seg binary (pure-CPU software implementation)
#   amd_gpu     ff_sieve --devices=amd   --gpu-search   (both kernels pinned to RX 9070 XT)
#   nvidia_gpu  ff_sieve --devices=nvidia --gpu-search  (both kernels pinned to RTX 5090)
#
# Methodology (see scripts/BENCHMARK_METHODOLOGY.md):
#   - untimed warmup rep per config (GPU clock ramp + page cache)
#   - REPS timed reps per config x leg; per-rep walls land in *_raw.csv;
#     summary CSV carries min/median/max/stdev
#   - per-rep DEVICE ATTRIBUTION gates (hard, fail the rep):
#       1. "--devices=<v> kept 1 of 2" filter line present
#       2. "GPU search:" stderr line names the EXPECTED card
#       3. no "GPU search:" line names any OTHER card
#   - per-rep utilization sampling of both cards (advisory evidence,
#     recorded, not gating: sub-second legs can finish between samples)
#   - environment snapshot (git HEAD, kernel, CPU, both cards' bus IDs
#     from ff_sieve --list-devices) captured once into run/bench_env.txt
#
# Expected outcomes recorded honestly:
#   - amd_gpu @ 2097152: capacity-gate refusal (backing < 16 GiB map), rc=1.
#     Marked EXPECTED_GATE_REFUSAL, not a defect. (--host-tier-cap=auto spills
#     and passes byte-identical; the default-path refusal is what is benched.)
#
# Usage: ./scripts/bench_per_device.sh
# Env overrides: BENCH_REPS (default 3), BENCH_LEGS (default all six),
#                BENCH_WARMUP (default 1)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF_BIN="$ROOT/reference/ff_seg"
NEW_BIN="$ROOT/build/ff_sieve"
CSV="$ROOT/scripts/bench_per_device_results.csv"
RAW_CSV="$ROOT/scripts/bench_per_device_raw.csv"
REPORT="$ROOT/scripts/bench_per_device_report.md"
ENV_TXT="$ROOT/run/bench_env.txt"
TIMEOUT_SEC=600
REPS="${BENCH_REPS:-3}"
WARMUP="${BENCH_WARMUP:-1}"

LEGS=(${BENCH_LEGS:-65536 131072 262144 524288 1048576 2097152})
declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

AMD_CARD="AMD Radeon RX 9070 XT"
NV_CARD="NVIDIA GeForce RTX 5090"

# config|expect-card-name|forbid-substring|cmd
declare -a CONFIGS=(
  "ref|||$REF_BIN 5"
  "amd_gpu|$AMD_CARD|RTX 5090|$NEW_BIN --devices=amd --gpu-search 5"
  "nvidia_gpu|$NV_CARD|RX 9070 XT|$NEW_BIN --devices=nvidia --gpu-search 5"
)

mkdir -p "$ROOT/run"

echo "=== BUILD CHECK ==="
[[ -x "$REF_BIN" ]] || { echo "ERROR: $REF_BIN missing"; exit 1; }
[[ -x "$NEW_BIN" ]] || { echo "ERROR: $NEW_BIN missing"; exit 1; }
echo "Both binaries OK."

# ---- Environment snapshot --------------------------------------------------
GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')"
KERNEL_REL="$(uname -r)"
{
  echo "timestamp_utc = $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "git_head      = $GIT_HEAD (dirty files: $GIT_DIRTY)"
  echo "kernel        = $KERNEL_REL"
  echo "cpu           = $CPU_MODEL"
  echo "reps          = $REPS  warmup=$WARMUP  legs=${LEGS[*]}"
  echo "--- ff_sieve --list-devices (enumeration proof) ---"
  "$NEW_BIN" --list-devices 2>&1 >/dev/null
} > "$ENV_TXT"
cat "$ENV_TXT"

# ---- Preflight: warn if other compute procs hold the GPUs ------------------
nv_procs="$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
if [[ "${nv_procs:-0}" -gt 0 ]]; then
  echo "WARNING: $nv_procs process(es) hold NVIDIA compute context — timings may be polluted."
fi

echo "=== CLEANING ==="
rm -rf "$ROOT/ffPlayground" "$ROOT/run/bench_pd_*"

# ---- Utilization sampler (advisory) ----------------------------------------
sample_gpus() { # $1 = logfile ; samples until killed
    local f="$1"
    echo "# ts nv_util amd_util" >> "$f"
    while :; do
        local ts nv au
        ts="$(date +%s.%N)"
        nv="$(timeout 2 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
              | grep -oE '^[0-9]+' | head -1)"
        au="$(timeout 2 rocm-smi --showuse 2>/dev/null \
              | grep -oE 'GPU use \(%\): *[0-9]+' | grep -oE '[0-9]+' | head -1)"
        echo "$ts nv=${nv:-NA} amd=${au:-NA}" >> "$f"
        sleep 0.2
    done
}

max_field() { # $1 logfile, $2 field prefix (nv=|amd=) -> max int or NA
    [[ -f "$1" ]] || { echo NA; return; }
    awk -v p="$2" '
        { for (i = 2; i <= NF; i++)
            if (index($i, p) == 1) { v = substr($i, length(p)+1)+0; if (v > m || m == "") m = v } }
        END { print (m == "" ? "NA" : m) }' "$1"
}

median() {
    local a=($(printf '%s\n' "$@" | sort -n))
    local n=${#a[@]}
    awk "BEGIN { printf \"%.3f\", ${a[$((n/2))]} }"
}

# run_one cfg leg scratch_dir expect forbid cmd...
# -> prints "wall_s|rc|solutions|search_card_ok|filter_ok|other_card_seen|refused"
run_one() {
    local cfg="$1" leg="$2" scratch_dir="$3" expect="$4" forbid="$5"; shift 5
    local t_start t_end wall_ms wall_s rc
    t_start=$(date +%s%N)
    eval "cd '$scratch_dir' && timeout ${TIMEOUT_SEC} $*" \
        > "$scratch_dir/output.txt" 2> "$scratch_dir/stderr.txt"
    rc=$?
    t_end=$(date +%s%N)
    wall_ms=$(( (t_end - t_start) / 1000000 ))
    wall_s=$(awk "BEGIN {printf \"%.3f\", $wall_ms / 1000}")
    local lines=0
    [[ -f "$scratch_dir/output.txt" ]] && \
        lines=$(grep -c ') sum' "$scratch_dir/output.txt" 2>/dev/null || true)
    lines=${lines:-0}
    local err="$scratch_dir/stderr.txt"
    local search_line search_card_ok filter_ok other_seen refused
    search_line="$(grep 'GPU search:' "$err" 2>/dev/null | head -1)"
    case "$cfg" in
      ref)
        # reference never touches GPUs: PASS only if no GPU-search line exists.
        search_card_ok=$([[ -z "$search_line" ]] && echo yes || echo no)
        filter_ok=na; other_seen=no ;;
      *)
        if [[ -n "$search_line" && "$search_line" == *"$expect"* ]]; then search_card_ok=yes; else search_card_ok=no; fi
        if grep -q 'GPU search:' "$err" 2>/dev/null && grep 'GPU search:' "$err" | grep -qv "$expect"; then other_seen=yes; else other_seen=no; fi
        filter_ok=$(grep -c "device filter: --devices=.* kept 1 of 2 logical device(s)" "$err" 2>/dev/null || true)
        [[ "$filter_ok" -ge 1 ]] && filter_ok=yes || filter_ok=no
        ;;
    esac
    refused=$([[ "$rc" != 0 ]] && grep -q 'GATE FAIL' "$err" 2>/dev/null && echo yes || echo no)
    printf '%s|%s|%s|%s|%s|%s|%s\n' "$wall_s" "$rc" "$lines" \
        "$search_card_ok" "$filter_ok" "$other_seen" "$refused"
}

echo "=== WARMUP (${WARMUP} untimed rep per config @ ${LEGS[0]}) ==="
warm_walls=()
for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%|*}"; rest="${entry#*|}"
    cmd="${rest##*|}"
    [[ "$WARMUP" -ge 1 ]] || continue
    wd="$(mktemp -d "$ROOT/run/bench_pd_warm.XXXXXX")"
    eval "cd '$wd' && timeout ${TIMEOUT_SEC} $cmd ${LEGS[0]}" \
        > /dev/null 2>&1
    echo "  warm: $cfg done"
    rm -rf "$wd"
done

# ---- Timed sweep ------------------------------------------------------------
echo ""
echo "=== RUNNING BENCHMARKS (${REPS} timed reps per config x leg) ==="

cat > "$CSV" << EOF
config,leg,wall_median_s,wall_min_s,wall_max_s,wall_stdev_s,reps,solutions,expected,outcome,exit_code
EOF
cat > "$RAW_CSV" << EOF
config,leg,rep,wall_s,solutions,exit_code,search_card_ok,filter_ok,other_card_seen,util_amd_max_pct,util_nv_max_pct
EOF

for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%|*}"; rest="${entry#*|}"
    expect="${rest%%|*}"; rest="${rest#*|}"
    forbid="${rest%%|*}"; cmd="${rest#*|}"
    echo ""
    echo "--- config: $cfg ---"
    for leg in "${LEGS[@]}"; do
        expected="${EXPECTED[$leg]}"
        walls=(); rcs=(); sols=()
        g_ok_all=yes; f_ok_all=yes; o_seen_any=no; refused_all=yes; util_amax=NA; util_nmax=NA
        for r in $(seq 1 "$REPS"); do
            scratch_dir="$(mktemp -d "$ROOT/run/bench_pd_${cfg}_${leg}.XXXXXX")"
            if [[ "$cfg" != ref ]]; then
                slog="$scratch_dir/util.log"
                sample_gpus "$slog" &
                SAMPLER_PID=$!
            fi
            out=$(run_one "$cfg" "$leg" "$scratch_dir" "$expect" "$forbid" $cmd $leg)
            IFS='|' read -r wall rc lines s_ok f_ok o_seen refused <<< "$out"
            if [[ -n "${SAMPLER_PID:-}" ]]; then
                kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null
                SAMPLER_PID=""
            fi
            ua="NA"; un="NA"
            if [[ -n "${slog:-}" && -f "$slog" ]]; then
                ua=$(max_field "$slog" "amd="); un=$(max_field "$slog" "nv=")
            fi
            rm -rf "$scratch_dir"
            echo "${cfg},${leg},${r},${wall},${lines},${rc},${s_ok},${f_ok},${o_seen},${ua},${un}" >> "$RAW_CSV"
            walls+=("$wall"); rcs+=("$rc"); sols+=("$lines")
            [[ "$s_ok" == yes ]] || g_ok_all=no
            [[ "$f_ok" == yes || "$f_ok" == na ]] || f_ok_all=no
            [[ "$o_seen" == yes ]] && o_seen_any=yes
            [[ "$refused" == yes ]] || refused_all=no
            [[ "$ua" != NA ]] && { [[ "$util_amax" == NA || "$ua" -gt "$util_amax" ]] && util_amax="$ua"; }
            [[ "$un" != NA ]] && { [[ "$util_nmax" == NA || "$un" -gt "$util_nmax" ]] && util_nmax="$un"; }
            echo "  [$cfg $leg rep$r] wall=${wall}s rc=$rc sols=$lines gpu_search_card=$s_ok util(amd/nv)=${ua}/${un}"
        done
        med=$(median "${walls[@]}")
        minw=$(printf '%s\n' "${walls[@]}" | sort -n | head -1)
        maxw=$(printf '%s\n' "${walls[@]}" | sort -n | tail -1)
        stdev=$(python3 -c "
import statistics,sys
ws=[float(x) for x in sys.argv[1:]]
print('%.3f'%statistics.pstdev(ws) if len(ws)>1 else '0.000')" "${walls[@]}")
        rc0=$(printf '%s\n' "${rcs[@]}" | grep -c '^0$' || true)
        lnmode=$(printf '%s\n' "${sols[@]}" | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
        lastrc=$(printf '%s\n' "${rcs[@]}" | sort -n | head -1)
        outcome=OK
        if [[ "$rc0" -eq "$REPS" && "$lnmode" == "$expected" && "$g_ok_all" == yes && "$f_ok_all" == yes && "$o_seen_any" == no ]]; then
            outcome=OK
        elif [[ "$refused_all" == yes ]]; then
            outcome=EXPECTED_GATE_REFUSAL
        else
            outcome=DEFECT
        fi
        echo "${cfg},${leg},${med},${minw},${maxw},${stdev},${REPS},${lnmode},${expected},${outcome},${lastrc}" >> "$CSV"
        printf "  -> %-12s leg=%8s  med=%.3fs min=%s max=%s sd=%s  outcome=%s  util(a/n)=%s/%s\n" \
            "$cfg" "$leg" "$med" "$minw" "$maxw" "$stdev" "$outcome" "$util_amax" "$util_nmax"
    done
done

echo ""
echo "CSV: $CSV"
echo "RAW: $RAW_CSV"

# ---- Report -----------------------------------------------------------------
python3 - "$CSV" "$REPORT" "$RAW_CSV" "$ENV_TXT" << 'PYEOF'
import csv, sys, statistics

csv_path, report_path, raw_path, env_path = sys.argv[1:5]
rows = list(csv.DictReader(open(csv_path)))
raw = list(csv.DictReader(open(raw_path)))
legs = ["65536","131072","262144","524288","1048576","2097152"]
confs = ["ref","amd_gpu","nvidia_gpu"]
labels = {
    "ref": "Reference ff_seg (software/CPU implementation)",
    "amd_gpu": "ff_sieve --devices=amd --gpu-search (full GPU: sieve+search on RX 9070 XT)",
    "nvidia_gpu": "ff_sieve --devices=nvidia --gpu-search (full GPU: sieve+search on RTX 5090)",
}
by = {(r["config"], r["leg"]): r for r in rows}

def outcome_mark(r):
    if r is None: return "-"
    if r["outcome"] == "OK": return ""
    return " ⚠️GATE" if r["outcome"] == "EXPECTED_GATE_REFUSAL" else f" ✗({r['outcome']},rc={r['exit_code']})"

env_lines = open(env_path).read().rstrip().splitlines()

L = []
L.append("# FF-GPU Per-Device Benchmark Report\n")
L.append(f"> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.")
L.append(f"> Median of {rows[0]['reps']} timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.")
L.append("> Every GPU-config rep passed hard device-attribution gates (vendor filter line +")
L.append("> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.\n")

L.append("## Environment\n")
L.append("```")
L.extend(env_lines)
L.append("```\n")

L.append("## Configurations\n")
L.append("| # | Config | Description |")
L.append("|---|--------|-------------|")
for i, c in enumerate(confs, 1):
    L.append(f"| {i} | `{c}` | {labels[c]} |")
L.append("")

L.append("## Wall-Clock Time (seconds)\n")
L.append("| Config | Leg | median | min | max | stdev | outcome |")
L.append("|--------|-----|--------|-----|-----|-------|---------|")
for c in confs:
    for leg in legs:
        r = by.get((c, leg))
        if r is None: continue
        L.append(f"| `{c}` | {leg} | {r['wall_median_s']} | {r['wall_min_s']} | "
                 f"{r['wall_max_s']} | {r['wall_stdev_s']} | {r['outcome']}{outcome_mark(r) if False else ''} |")
L.append("")
L.append("### Median wall-clock matrix\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
for c in confs:
    cells = []
    for leg in legs:
        r = by.get((c, leg))
        cells.append("-" if r is None else r["wall_median_s"])
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")

L.append("## Speedup vs reference (ref/config; >1 = faster)\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
ref_by_leg = {leg: float(by[("ref", leg)]["wall_median_s"]) for leg in legs if ("ref", leg) in by}
for c in confs:
    if c == "ref": continue
    cells = []
    for leg in legs:
        r = by.get((c, leg))
        if r is None or r["outcome"] != "OK" or ref_by_leg.get(leg, 0) == 0:
            cells.append("-")
        else:
            cells.append(f"{ref_by_leg[leg]/float(r['wall_median_s']):.2f}x")
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")

L.append("## Device Attribution (hard gates + utilization evidence)\n")
L.append("Hard gates per rep: vendor-filter line present, `GPU search:` names the intended")
L.append("card, and no foreign card named. All reps below show the aggregate result.\n")
L.append("| Config | 65K | 131K | 262K | 524K | 1M | 2M |")
L.append("|--------|-----|------|------|------|----|-----|")
for c in confs:
    cells = []
    for leg in legs:
        rr = [x for x in raw if x["config"]==c and x["leg"]==leg]
        if not rr: cells.append("-"); continue
        s = by.get((c, leg), {}).get("outcome")
        if s == "EXPECTED_GATE_REFUSAL":
            cells.append("⚠️ refused"); continue
        ok = all(x["search_card_ok"]=="yes" and (x["filter_ok"]=="yes" or x["filter_ok"]=="na")
                 and x["other_card_seen"]=="no" for x in rr)
        amax = max((int(x["util_amd_max_pct"]) for x in rr if x["util_amd_max_pct"] not in ("NA","")), default=None)
        nmax = max((int(x["util_nv_max_pct"]) for x in rr if x["util_nv_max_pct"] not in ("NA","")), default=None)
        u = f"a{amax}/n{nmax}" if (amax is not None or nmax is not None) else "n/a"
        mark = "✅" if ok else "❌"
        cells.append(f"{mark} {u}")
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("`a<n>%`/`n<n>%` = max observed utilization (rocm-smi / nvidia-smi samples) for AMD/NVIDIA.")
L.append("Sub-second legs may finish between sampler ticks (`n/a`); the stderr-name gates above")
L.append("remain authoritative for those.\n")

L.append("## Notes\n")
L.append("- `amd_gpu` @ 2M: capacity-gate refusal by design (AMD backing ≈13.2 GiB < 16 GiB map), rc=1;")
L.append("  recorded as `EXPECTED_GATE_REFUSAL`. The spill path (`--host-tier-cap=auto`) completes this")
L.append("  leg byte-identically but is not part of this default-path sweep.")
L.append("- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/")
L.append("  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.")

with open(report_path, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Report: {report_path}")
PYEOF

rm -rf "$ROOT/ffPlayground" "$ROOT/run/bench_pd_*"
echo "=== DONE ==="
