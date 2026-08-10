#!/usr/bin/env bash
# verify.sh — golden-diff harness for the ff-gpu port (GPU_PLAN §9.4).
#
# Signature (pinned by plan todo 1):
#   verify.sh [<gpu_bin>] [<limit>] [--all-legs] [--gpu-search] [--devices <backend>] [--self-test]
#
#   <gpu_bin>       optional (self-test runs without one), else FIRST positional.
#                   The GPU binary is invoked as: <gpu_bin> [--devices <backend>] 5 <leg>
#   <limit>         SECOND positional — single golden leg (e.g. 2097152).
#   --all-legs      run all 6 golden legs (limit omitted).
#   --gpu-search    M4 mode — implies all 6 legs (limit omitted).
#   --devices <b>   harness passthrough: forwarded to the GPU binary (e.g. amd).
#   --self-test     standalone harness self-check; no gpu_bin; exits 0 on pass.
#
# Per leg the harness:
#   1. captures the GPU binary stdout on leg `5 <leg>`, asserts rc==0,
#      and timing-normalizes it (`Prime time: N μs` / `Freudenthal time: N μs`).
#   2. writes the captured-and-normalized stdout to out.txt at the repo root
#      (defined ONCE here; consumed by todo 8's `grep -c ') sum' out.txt`).
#   3. E2E diff: full stdout vs goldens/out_ff_seg_<leg>.txt (byte-exact),
#      and solution blocks vs goldens/out_pen_<leg>.txt + out_pen2_<leg>.txt.
#   4. asserts the solution count (2357/4776/9163/18408/35556/71424).
#   5. (map sha256) if a `--dump-map <file>` token is passed through to the GPU
#      binary, also dumps the CPU reference map with the same grammar and
#      compares sha256 — skipped with a notice while the reference lacks the
#      --dump-map patch (pre-todo-2 state); todo 2/13 own the full map contract.
#
# Exits non-zero with the first differing line printed on any mismatch.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOLDENS="$ROOT/goldens"
REF="$ROOT/reference"
OUT_TXT="$ROOT/out.txt"          # pinned once — repo-root out.txt

NORM_SED='s/(Prime|Freudenthal) time: [0-9]+/\1 time: N/'
RAW_CAP="$ROOT/run/verify_capture.raw"
ERR_LOG="$ROOT/run/verify_stderr.log"

declare -A COUNTS=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)
ALL_LEGS="65536 131072 262144 524288 1048576 2097152"

# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
GPU_BIN=""
LIMIT=""
ALL_LEGS_FLAG=0
GPU_SEARCH=0
SELF_TEST=0
DEVICES=""
PASSTHROUGH=()          # extra tokens forwarded to the GPU binary

i=1
while [[ $i -le $# ]]; do
  arg="${!i}"
  case "$arg" in
    --self-test)   SELF_TEST=1 ;;
    --all-legs)    ALL_LEGS_FLAG=1 ;;
    --gpu-search)  GPU_SEARCH=1 ;;
    --devices)     i=$((i+1)); DEVICES="${!i}" ;;
    --*)           PASSTHROUGH+=("$arg") ;;   # e.g. --dump-map handled below
    *)
      if [[ -z "$GPU_BIN" ]]; then
        GPU_BIN="$arg"
      elif [[ -z "$LIMIT" && "$arg" =~ ^[0-9]+$ ]]; then
        LIMIT="$arg"
      else
        echo "verify.sh: unexpected argument '$arg'" >&2
        echo "usage: verify.sh [<gpu_bin>] [<limit>] [--all-legs] [--gpu-search] [--devices <backend>] [--self-test]" >&2
        exit 2
      fi
      ;;
  esac
  i=$((i+1))
done

# --dump-map <file> is a passthrough pair (its file operand is the next token).
DUMP_MAP=""
for ((j=0; j<${#PASSTHROUGH[@]}; j++)); do
  if [[ "${PASSTHROUGH[$j]}" == "--dump-map" && $((j+1)) -lt ${#PASSTHROUGH[@]} ]]; then
    DUMP_MAP="${PASSTHROUGH[$((j+1))]}"
  fi
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
normalize_timing() { sed -E "$NORM_SED"; }

first_diff() {   # $1=label  $2=fileA  $3=fileB
  local label="$1" a="$2" b="$3"
  diff "$a" "$b" > "$ROOT/run/verify_diff.out" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  MISMATCH ($label): first differing lines:" >&2
    head -8 "$ROOT/run/verify_diff.out" >&2
    return 1
  fi
  return 0
}

run_one_leg() {
  local bin="$1" leg="$2" fails=0
  local golden_ff="$GOLDENS/out_ff_seg_${leg}.txt"
  local golden_pen="$GOLDENS/out_pen_${leg}.txt"
  local golden_pen2="$GOLDENS/out_pen2_${leg}.txt"
  local want="${COUNTS[$leg]}"

  mkdir -p "$ROOT/run"
  rm -f "$OUT_TXT" "$RAW_CAP"

  local devarg=()
  [[ -n "$DEVICES" ]] && devarg=(--devices "$DEVICES")

  # Capture + assert rc. Diagnostics from the GPU binary go to stderr only.
  local rc=0
  "$bin" "${devarg[@]}" "${PASSTHROUGH[@]}" 5 "$leg" > "$RAW_CAP" 2> "$ERR_LOG" || rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL leg $leg: '$bin' exited rc=$rc (see $ERR_LOG)" >&2
    sed -n '1,10p' "$ERR_LOG" >&2
    return 1
  fi
  normalize_timing < "$RAW_CAP" > "$OUT_TXT"

  # solution count
  local count
  count="$(grep -c ') sum' "$OUT_TXT" || true)"
  if [[ "$count" != "$want" ]]; then
    echo "FAIL leg $leg: solution count $count != expected $want" >&2
    fails=1
  fi

  # E2E: full stdout vs ff_seg golden (byte-exact, incl. headers + timing shape)
  first_diff "full stdout vs out_ff_seg_${leg}" "$OUT_TXT" "$golden_ff" || fails=1

  # E2E: solution blocks vs pen/pen2 goldens
  local sol_out sol_p
  sol_out="$ROOT/run/verify_sol_out.txt"
  sol_p="$ROOT/run/verify_sol_pen.txt"
  grep ') sum' "$OUT_TXT" > "$sol_out"
  grep ') sum' "$golden_pen" > "$sol_p"
  first_diff "solution block vs out_pen_${leg}" "$sol_out" "$sol_p" || fails=1
  grep ') sum' "$golden_pen2" > "$sol_p"
  first_diff "solution block vs out_pen2_${leg}" "$sol_out" "$sol_p" || fails=1

  # map sha256 (when --dump-map passthrough present)
  if [[ -n "$DUMP_MAP" && $fails -eq 0 ]]; then
    local refdump="$ROOT/run/verify_ref_map.bin"
    rm -f "$refdump"
    if "$REF/ff_seg" --dump-map "$refdump" 5 "$leg" >/dev/null 2>&1 && [[ -f "$refdump" ]]; then
      local hg hr
      hg="$(sha256sum "$DUMP_MAP" | awk '{print $1}')"
      hr="$(sha256sum "$refdump" | awk '{print $1}')"
      echo "  map sha256 gpu=$hg"
      echo "  map sha256 ref=$hr"
      if [[ "$hg" != "$hr" ]]; then
        echo "FAIL leg $leg: GPU map sha256 != CPU reference map sha256" >&2
        fails=1
      fi
    else
      echo "  note: reference ff_seg has no --dump-map yet (pre-todo-2); map sha256 skipped"
    fi
  fi

  if [[ $fails -eq 0 ]]; then
    echo "PASS leg $leg: '$bin' byte-identical, $count solutions (out.txt)"
  else
    echo "FAIL leg $leg" >&2
  fi
  return $fails
}

# ---------------------------------------------------------------------------
# --self-test: exercise the harness's own diff path on a known-good fixture
# ---------------------------------------------------------------------------
if [[ $SELF_TEST -eq 1 ]]; then
  mkdir -p "$ROOT/run"
  fails=0
  echo "self-test: known-good fixture (goldens/out_ff_seg_65536.txt via scripts/fixtures/fake_golden_cat.sh)"
  if run_one_leg "$ROOT/scripts/fixtures/fake_golden_cat.sh" 65536; then
    echo "self-test: positive path OK"
  else
    echo "self-test: FAILED on known-good fixture (harness broken)" >&2
    fails=1
  fi

  echo "self-test: corrupted fixture must be DETECTED (scripts/fixtures/fake_golden_broken.sh)"
  if run_one_leg "$ROOT/scripts/fixtures/fake_golden_broken.sh" 65536 >/dev/null 2>&1; then
    echo "self-test: FAILED — corrupted fixture passed (diff path broken)" >&2
    fails=1
  else
    echo "self-test: corruption detected OK"
  fi

  if [[ $fails -eq 0 ]]; then
    echo "verify.sh --self-test: PASS"
    exit 0
  fi
  echo "verify.sh --self-test: FAIL" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# normal runs
# ---------------------------------------------------------------------------
if [[ -z "$GPU_BIN" ]]; then
  echo "verify.sh: no gpu_bin given (only --self-test runs without one)" >&2
  echo "usage: verify.sh [<gpu_bin>] [<limit>] [--all-legs] [--gpu-search] [--devices <backend>] [--self-test]" >&2
  exit 2
fi
if [[ ! -x "$GPU_BIN" && ! -x "$ROOT/$GPU_BIN" ]]; then
  echo "verify.sh: GPU binary not found/executable: $GPU_BIN" >&2
  exit 2
fi

if [[ -n "$LIMIT" ]]; then
  LEGS="$LIMIT"
else
  LEGS="$ALL_LEGS"
fi

if [[ $GPU_SEARCH -eq 1 ]]; then
  echo "verify.sh: M4 GPU-search mode (--gpu-search), all 6 legs"
fi

fail=0
for leg in $LEGS; do
  run_one_leg "$GPU_BIN" "$leg" || fail=1
done

if [[ $fail -eq 0 ]]; then
  echo "verify.sh: ALL LEGS PASS"
  exit 0
fi
echo "verify.sh: FAILED" >&2
exit 1
