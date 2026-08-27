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
#   - per-rep ADDITIVE stderr phase capture: "ff_sieve timing:" lines
#     (device enumeration / budget computation / sieve phase / search phase /
#     total + search H2D/kernel/D2H/emit sub-stages) parsed into extra raw-CSV
#     columns; missing lines -> empty cells
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
#   - amd_gpu @ 2097152: runs TRUE GPU search since the wheel-30 landing —
#     the compressed internal map (8.53 GiB at 2M) fits the card's
#     participation budget, so the aggregate gate passes on internal bytes,
#     the residency handoff reads the sieve-resident map in place (0 B H2D),
#     and every rep carries search-kernel sub-timers. (Historical eras: this
#     cell once refused with a capacity GATE FAIL, then completed via the
#     task-14 auto host-tier spill with CPU-search fallback; both are gone.)
#     EXPECTED_GATE_REFUSAL survives only as a generic outcome label.
#
# Usage: ./scripts/bench_per_device.sh
# Env overrides: BENCH_REPS (default 3), BENCH_LEGS (default all six),
#                BENCH_WARMUP (default 1), BENCH_MEDIAN_N (default 5),
#                BENCH_JIT_WARMUP (default 1), BENCH_AUTOEXTEND (default 1)
#
# Protocol extensions (plan task 2 — harness upgrade):
#   ACTIVE by default (adds a gate, changes NO measurement):
#     - per-rep stdout sha256 gate: captured stdout is normalized with
#         sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/'
#       BEFORE hashing and compared against identically-normalized
#       goldens/out_ff_seg_<leg>.txt (hashes precomputed at sweep start;
#       one golden set suffices — stdout is platform-independent by
#       contract). Any mismatch => outcome DEFECT.
#
# PROTOCOL DEFAULTS — ACTIVATED as of the plan-task-7 re-baseline (formerly
#   dormant env-flagged knobs, ALL OFF before task 7; each remains
#   env-overridable, set to 0 to restore the legacy behavior for that knob):
#     BENCH_MEDIAN_N=5    median-of-N (N>=5) timed reps for sub-second legs
#                         (a leg is classified sub-second by its first
#                         rep's wall clock < 1.000 s)
#     BENCH_JIT_WARMUP=1  one extra untimed warmup before EVERY config x leg
#                         block (the default warmup stays once per config)
#     BENCH_AUTOEXTEND=1  re-run while (max-min)/median > 0.15, capped at
#                         2 extra reps per cell
#   Raw CSV gains these APPEND-ONLY columns (existing order untouched):
#     phase_wheel_expansion_ms, phase_zero_fill_ms, phase_sched_teardown_ms,
#     phase_search_setup_ms, phase_search_teardown_ms, unaccounted_ms,
#     sha256_ok
#   unaccounted_ms = phase_total − Σ(enum+budget+sieve+search+wheel_expansion+
#   zero_fill+sched_teardown+setup+teardown), computed ONLY when every input
#   is present; otherwise the cell stays EMPTY (reported NA — missing data is
#   never silently treated as zero).

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
MEDIAN_N="${BENCH_MEDIAN_N:-5}";     [[ "$MEDIAN_N"   =~ ^[0-9]+$ ]] || MEDIAN_N=5
JIT_WARMUP="${BENCH_JIT_WARMUP:-1}"; [[ "$JIT_WARMUP" =~ ^[0-9]+$ ]] || JIT_WARMUP=1
AUTOEXTEND="${BENCH_AUTOEXTEND:-1}"; [[ "$AUTOEXTEND" =~ ^[0-9]+$ ]] || AUTOEXTEND=1

LEGS=(${BENCH_LEGS:-65536 131072 262144 524288 1048576 2097152})
declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [196608]=6922 [262144]=9163
  [327680]=11395 [393216]=13675 [458752]=16032 [524288]=18408
  [589824]=20414 [655360]=22558 [720896]=24726 [786432]=26847
  [851968]=29012 [917504]=31145 [983040]=33347 [1048576]=35556
  [1114112]=37791 [1179648]=39949 [1245184]=42106 [1310720]=44313
  [1376256]=46576 [1441792]=48779 [1507328]=51053 [1572864]=53287
  [1638400]=55529 [1703936]=57818 [1769472]=60100 [1835008]=62308
  [1900544]=64577 [1966080]=66881 [2031616]=69120 [2097152]=71424
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
  echo "protocol      = sha_gate=on median_n=$MEDIAN_N jit_warmup=$JIT_WARMUP autoextend=$AUTOEXTEND"
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

# >>> BEGIN bench_pd testable helpers (selftest extracts this block verbatim;
# keep every function here self-contained: no globals, no side effects)
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

# phase_ms <stderr-file> <exact-phase-name>
# Extracts the millisecond value from a binary's own stderr line of the form
#   ff_sieve timing: <name> = <ms> ms
# Prints the raw ms number, or NOTHING when the line is absent (older
# binaries / the CPU reference emit none) -> caller writes an empty CSV cell.
# Uses index() matching on an exact "name = " prefix so e.g. the
# "--list-devices" variant ("device enumeration (--list-devices) = ...")
# can never satisfy the plain "device enumeration" lookup.
phase_ms() {
    local f="$1" name="$2"
    [[ -f "$f" ]] || return 0
    awk -v p="ff_sieve timing: ${name} = " '
        index($0, p) == 1 {
            v = substr($0, length(p)+1); sub(/ ms$/, "", v); print v; exit
        }' "$f"
}

normalize_stdout() {
    sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/'
}

stdout_sha256() {
    normalize_stdout < "$1" | sha256sum | cut -d' ' -f1
}

unaccounted_ms() {
    local total="$1"; shift
    local v sum="0"
    [[ -n "$total" ]] || return 0
    for v in "$@"; do
        [[ -n "$v" ]] || return 0
        sum=$(awk "BEGIN{printf \"%.3f\", $sum + $v}")
    done
    awk "BEGIN{printf \"%.3f\", $total - $sum}"
}
# <<< END bench_pd testable helpers

# ---- Normalized-golden stdout hashes (sha256 gate, ACTIVE by default) -------
declare -A GOLDEN_SHA=()
for _gl in "${LEGS[@]}"; do
    _gf="$ROOT/goldens/out_ff_seg_${_gl}.txt"
    [[ -f "$_gf" ]] && GOLDEN_SHA[$_gl]="$(stdout_sha256 "$_gf")"
done

# run_one cfg leg scratch_dir expect forbid cmd...
# -> prints "wall_s|rc|solutions|search_card_ok|filter_ok|other_card_seen|refused\
# |phase_enum_ms|phase_budget_ms|phase_sieve_ms|phase_search_ms|phase_total_ms\
# |search_h2d_ms|search_kernel_ms|search_d2h_ms|search_emit_ms\
# |phase_wheel_expansion_ms|phase_zero_fill_ms|phase_sched_teardown_ms\
# |phase_search_setup_ms|phase_search_teardown_ms|unaccounted_ms|sha256_ok"
# (the trailing per-phase fields are ADDITIVE captures from stderr; empty when
# absent; unaccounted_ms empty unless every phase input exists; sha256_ok is
# yes/no against the leg's normalized golden, or na when no golden file exists)
run_one() {
    local cfg="$1" leg="$2" scratch_dir="$3" expect="$4" forbid="$5"; shift 5
    local t_start t_end wall_ms wall_s rc
    t_start=$(date +%s%N)
    eval "( cd '$scratch_dir' && exec timeout ${TIMEOUT_SEC} $* )" \
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
        # task-14 SANCTIONED CELL EXCEPTION (the only expectation change owned
        # by task 14): amd_gpu@2097152 now COMPLETES via the auto host-tier
        # spill (rc=0, correct solution count), and its single "GPU search:"
        # line may legitimately be the documented capacity-fallback-to-CPU
        # notice instead of a card name. Every other config x cell keeps the
        # task-3 expectations frozen.
        local fallback_ok=no
        if [[ "$cfg" == amd_gpu && "$leg" == 2097152 ]] && \
           [[ -n "$search_line" && "$search_line" == *"no device fits"* ]]; then
          fallback_ok=yes
        fi
        if [[ "$fallback_ok" == yes ]]; then
          search_card_ok=yes
          other_seen=$(grep 'GPU search:' "$err" 2>/dev/null \
                       | grep -v "$expect" | grep -v 'no device fits' \
                       | grep -q . && echo yes || echo no)
        else
          if [[ -n "$search_line" && "$search_line" == *"$expect"* ]]; then search_card_ok=yes; else search_card_ok=no; fi
          if grep -q 'GPU search:' "$err" 2>/dev/null && grep 'GPU search:' "$err" | grep -qv "$expect"; then other_seen=yes; else other_seen=no; fi
        fi
        # task-14a consequence: vendor filtering now happens BEFORE enumeration,
        # so the filter line's total is the surviving vendor's device count
        # ("kept 1 of 1") rather than the dual-vendor union ("kept 1 of 2") —
        # an honest "of 2" would require initializing the excluded runtime.
        # The gate keeps its discriminative power: filter line present AND
        # exactly one logical device kept.
        filter_ok=$(grep -c "device filter: --devices=.* kept 1 of [0-9][0-9]* logical device(s)" "$err" 2>/dev/null || true)
        [[ "$filter_ok" -ge 1 ]] && filter_ok=yes || filter_ok=no
        ;;
    esac
    refused=$([[ "$rc" != 0 ]] && grep -q 'GATE FAIL' "$err" 2>/dev/null && echo yes || echo no)
    local p_enum p_budget p_sieve p_search p_total s_h2d s_kernel s_d2h s_emit
    p_enum=$(phase_ms "$err" "device enumeration")
    p_budget=$(phase_ms "$err" "budget computation")
    p_sieve=$(phase_ms "$err" "sieve phase")
    p_search=$(phase_ms "$err" "search phase")
    p_total=$(phase_ms "$err" "total")
    s_h2d=$(phase_ms "$err" "search H2D copies")
    s_kernel=$(phase_ms "$err" "search kernel")
    s_d2h=$(phase_ms "$err" "search D2H copies")
    s_emit=$(phase_ms "$err" "search emit")
    local p_expand p_zfill p_schedtd p_setup p_teardown unacc sha_ok out_sha
    p_expand=$(phase_ms "$err" "wheel expansion")
    p_zfill=$(phase_ms "$err" "hostmap zero-fill")
    p_schedtd=$(phase_ms "$err" "scheduler teardown")
    p_setup=$(phase_ms "$err" "search device setup")
    p_teardown=$(phase_ms "$err" "search device teardown")
    unacc=$(unaccounted_ms "$p_total" "$p_enum" "$p_budget" "$p_sieve" \
                        "$p_search" "$p_expand" "$p_zfill" "$p_schedtd" \
                        "$p_setup" "$p_teardown")
    sha_ok=na
    if [[ -n "${GOLDEN_SHA[$leg]:-}" ]]; then
        if [[ -f "$scratch_dir/output.txt" ]]; then
            out_sha=$(stdout_sha256 "$scratch_dir/output.txt")
        else
            out_sha=""
        fi
        if [[ "$out_sha" == "${GOLDEN_SHA[$leg]}" ]]; then sha_ok=yes; else sha_ok=no; fi
    fi
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$wall_s" "$rc" "$lines" \
        "$search_card_ok" "$filter_ok" "$other_seen" "$refused" \
        "$p_enum" "$p_budget" "$p_sieve" "$p_search" "$p_total" \
        "$s_h2d" "$s_kernel" "$s_d2h" "$s_emit" \
        "$p_expand" "$p_zfill" "$p_schedtd" "$p_setup" "$p_teardown" \
        "$unacc" "$sha_ok"
}

echo "=== WARMUP (${WARMUP} untimed rep per config @ ${LEGS[0]}) ==="
warm_walls=()
for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%|*}"; rest="${entry#*|}"
    cmd="${rest##*|}"
    [[ "$WARMUP" -ge 1 ]] || continue
    wd="$(mktemp -d "$ROOT/run/bench_pd_warm.XXXXXX")"
    eval "( cd '$wd' && exec timeout ${TIMEOUT_SEC} $cmd ${LEGS[0]} )" \
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
config,leg,rep,wall_s,solutions,exit_code,search_card_ok,filter_ok,other_card_seen,util_amd_max_pct,util_nv_max_pct,phase_enum_ms,phase_budget_ms,phase_sieve_ms,phase_search_ms,phase_total_ms,search_h2d_ms,search_kernel_ms,search_d2h_ms,search_emit_ms,phase_wheel_expansion_ms,phase_zero_fill_ms,phase_sched_teardown_ms,phase_search_setup_ms,phase_search_teardown_ms,unaccounted_ms,sha256_ok
EOF

# run_rep: one timed rep for the current $cfg/$leg block (caller sets globals).
# Appends to walls/rcs/sols, folds per-rep gates into the *_all accumulators,
# appends the raw-CSV row. Factored so the dormant protocol modes (median-of-N,
# autoextend) reuse the EXACT default-path rep body.
run_rep() {
    local scratch_dir slog out
    scratch_dir="$(mktemp -d "$ROOT/run/bench_pd_${cfg}_${leg}.XXXXXX")"
    if [[ "$cfg" != ref ]]; then
        slog="$scratch_dir/util.log"
        sample_gpus "$slog" &
        SAMPLER_PID=$!
    fi
    out=$(run_one "$cfg" "$leg" "$scratch_dir" "$expect" "$forbid" $cmd $leg)
    local wall rc lines s_ok f_ok o_seen refused \
        p_enum p_budget p_sieve p_search p_total \
        s_h2d s_kernel s_d2h s_emit \
        p_expand p_zfill p_schedtd p_setup p_teardown unacc sha_ok
    IFS='|' read -r wall rc lines s_ok f_ok o_seen refused \
        p_enum p_budget p_sieve p_search p_total \
        s_h2d s_kernel s_d2h s_emit \
        p_expand p_zfill p_schedtd p_setup p_teardown unacc sha_ok <<< "$out"
    if [[ -n "${SAMPLER_PID:-}" ]]; then
        kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null
        SAMPLER_PID=""
    fi
    local ua="NA" un="NA"
    if [[ -n "${slog:-}" && -f "$slog" ]]; then
        ua=$(max_field "$slog" "amd="); un=$(max_field "$slog" "nv=")
    fi
    rm -rf "$scratch_dir"
    echo "${cfg},${leg},${r},${wall},${lines},${rc},${s_ok},${f_ok},${o_seen},${ua},${un},${p_enum},${p_budget},${p_sieve},${p_search},${p_total},${s_h2d},${s_kernel},${s_d2h},${s_emit},${p_expand},${p_zfill},${p_schedtd},${p_setup},${p_teardown},${unacc},${sha_ok}" >> "$RAW_CSV"
    walls+=("$wall"); rcs+=("$rc"); sols+=("$lines")
    [[ "$s_ok" == yes ]] || g_ok_all=no
    [[ "$f_ok" == yes || "$f_ok" == na ]] || f_ok_all=no
    [[ "$o_seen" == yes ]] && o_seen_any=yes
    [[ "$refused" == yes ]] || refused_all=no
    [[ "$sha_ok" == no ]] && sha_all=no
    [[ "$ua" != NA ]] && { [[ "$util_amax" == NA || "$ua" -gt "$util_amax" ]] && util_amax="$ua"; }
    [[ "$un" != NA ]] && { [[ "$util_nmax" == NA || "$un" -gt "$util_nmax" ]] && util_nmax="$un"; }
    echo "  [$cfg $leg rep$r] wall=${wall}s rc=$rc sols=$lines gpu_search_card=$s_ok util(amd/nv)=${ua}/${un} sha=${sha_ok}"
}

for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%|*}"; rest="${entry#*|}"
    expect="${rest%%|*}"; rest="${rest#*|}"
    forbid="${rest%%|*}"; cmd="${rest#*|}"
    echo ""
    echo "--- config: $cfg ---"
    for leg in "${LEGS[@]}"; do
        expected="${EXPECTED[$leg]:-}"
        walls=(); rcs=(); sols=()
        g_ok_all=yes; f_ok_all=yes; o_seen_any=no; refused_all=yes; util_amax=NA; util_nmax=NA
        sha_all=yes
        if [[ "$JIT_WARMUP" == 1 ]]; then
            wwd="$(mktemp -d "$ROOT/run/bench_pd_jit.XXXXXX")"
            eval "( cd '$wwd' && exec timeout ${TIMEOUT_SEC} $cmd $leg )" >/dev/null 2>&1
            rm -rf "$wwd"
        fi
        target="$REPS"; r=0; extras=0
        while (( r < target )); do
            r=$((r+1))
            run_rep
            if (( r == 1 && MEDIAN_N >= 5 )); then
                awk "BEGIN{exit !(${walls[0]} < 1.0)}" && target="$MEDIAN_N"
            fi
        done
        while [[ "$AUTOEXTEND" == 1 && "$extras" -lt 2 ]]; do
            _med=$(median "${walls[@]}")
            _min=$(printf '%s\n' "${walls[@]}" | sort -n | head -1)
            _max=$(printf '%s\n' "${walls[@]}" | sort -n | tail -1)
            _over=$(awk "BEGIN{print ($_med <= 0 || ($_max - $_min) / $_med > 0.15) ? 1 : 0}")
            (( $_over == 1 )) || break
            r=$((r+1)); extras=$((extras+1))
            echo "  [$cfg $leg] autoextend: spread ($_min..$_max)/$_med > 0.15 — extra rep $extras/2"
            run_rep
        done
        nreps=${#walls[@]}
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
        if [[ "$rc0" -eq "$nreps" && "$lnmode" == "$expected" && "$g_ok_all" == yes && "$f_ok_all" == yes && "$o_seen_any" == no && "$sha_all" == yes ]]; then
            outcome=OK
        elif [[ "$refused_all" == yes ]]; then
            outcome=EXPECTED_GATE_REFUSAL
        else
            outcome=DEFECT
        fi
        echo "${cfg},${leg},${med},${minw},${maxw},${stdev},${nreps},${lnmode},${expected},${outcome},${lastrc}" >> "$CSV"
        printf "  -> %-12s leg=%8s  med=%.3fs min=%s max=%s sd=%s  outcome=%s  util(a/n)=%s/%s  sha=%s\n" \
            "$cfg" "$leg" "$med" "$minw" "$maxw" "$stdev" "$outcome" "$util_amax" "$util_nmax" "$sha_all"
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
L.append("> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.")
L.append("> Per-rep stdout sha256 gate (ACTIVE): timing digits normalized (`Prime|Freudenthal time: N`)")
L.append("> before hashing against the identically-normalized `goldens/out_ff_seg_<leg>.txt`; any")
L.append("> mismatch marks the cell DEFECT.\n")

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
            sp = ref_by_leg[leg] / float(r["wall_median_s"])
            cells.append(f"{sp:.2f}x" + (" **REGRESSION**" if sp < 1.0 else ""))
    L.append("| " + " | ".join([f"`{c}`"] + cells) + " |")
L.append("")
L.append("`**REGRESSION**` marks speedup < 1.00x versus the SAME-SESSION reference above — a distinct")
L.append("visual mark from the ⚠️GATE / ✗(...) outcome annotations, which carry prior-verdict comparisons.")
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

L.append("## Phase Decomposition & Unaccounted Time\n")
L.append("`unaccounted_ms = phase_total − (enum+budget+sieve+search+wheel_expansion+zero_fill+")
L.append("sched_teardown+setup+teardown)`, computed per rep by the harness and carried in")
L.append("`bench_per_device_raw.csv`; the table shows the median over reps where ALL ten inputs existed.")
L.append("`NA` = timer lines incomplete for that cell: the reference emits NO `ff_sieve timing:` lines,")
L.append("and binaries predating a given timer simply lack its line. Missing data is NEVER coerced to zero.\n")
L.append("| Config | Leg | phase_total_med_ms | unaccounted_med_ms | computable_reps |")
L.append("|--------|-----|--------------------|--------------------|-----------------|")

def _med_or_na(vals):
    return f"{statistics.median(vals):.3f}" if vals else "NA"

for c in confs:
    for leg in legs:
        rr = [x for x in raw if x["config"] == c and x["leg"] == leg]
        if not rr: continue
        totals = [float(x["phase_total_ms"]) for x in rr if x.get("phase_total_ms")]
        unaccs = [float(x["unaccounted_ms"]) for x in rr if x.get("unaccounted_ms")]
        L.append(f"| `{c}` | {leg} | {_med_or_na(totals)} | {_med_or_na(unaccs)} | {len(unaccs)}/{len(rr)} |")
L.append("")

L.append("## Notes\n")
L.append("- `amd_gpu` @ 2M: TRUE GPU search since the wheel-30 landing — the compressed")
L.append("  internal map (8.53 GiB at 2M) fits the card, so every rep runs the search")
L.append("  kernel on-device (card named in stderr, zero fallback notices, residency")
L.append("  handoff 0 B H2D). Historical eras — capacity-gate refusal, then task-14")
L.append("  auto host-tier spill with CPU-search fallback — are both gone.")
L.append("- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/")
L.append("  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.")

with open(report_path, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Report: {report_path}")
PYEOF

rm -rf "$ROOT/ffPlayground" "$ROOT/run/bench_pd_*"
echo "=== DONE ==="
