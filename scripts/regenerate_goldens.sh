#!/usr/bin/env bash
# regenerate_goldens.sh — regenerate the 18 committed golden files from the
# untouched vendored reference sources (the byte contract).
#
#   ff-gpu/goldens/out_<prog>_<leg>.txt   (prog in ff_seg pen pen2, leg in
#                                         65536 131072 262144 524288 1048576 2097152)
#
# The reference programs chdir into `ffPlayground` in their cwd and EXIT -2 if
# that directory already exists, so EVERY leg runs from a FRESH empty cwd
# (run/<prog>_<leg>/, cleaned up afterwards). The pinned CLI is `5 <leg>`
# (sumStart=5, sumLimit=leg).
#
# Timing lines are value-normalized at save time (raw wall-clock μs would break
# the reproducibility sha256): `Prime time: N μs` / `Freudenthal time: N μs`.
# Solution lines and headers are stored byte-for-byte. Per-leg solution counts
# are asserted (2357/4776/9163/18408/35556/71424) so source drift is caught
# early — Metis build-reproducibility requirement.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/reference"
GOLDENS="$ROOT/goldens"
SCRATCH="$ROOT/run"

LEGS="65536 131072 262144 524288 1048576 2097152"
PROGS="ff_seg pen pen2"

declare -A EXPECTED_COUNTS=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

# Timing normalization — the ONLY non-deterministic lines in the output.
NORM_SED='s/(Prime|Freudenthal) time: [0-9]+/\1 time: N/'

mkdir -p "$GOLDENS" "$SCRATCH"

fail=0
for prog in $PROGS; do
  bin="$REF/$prog"
  if [[ ! -x "$bin" ]]; then
    echo "ERROR: reference binary not found/executable: $bin (run 'make -f Makefile.reference' first)" >&2
    fail=1
    continue
  fi
  for leg in $LEGS; do
    cwd="$SCRATCH/${prog}_${leg}"
    rm -rf "$cwd"
    mkdir -p "$cwd"
    out="$GOLDENS/out_${prog}_${leg}.txt"
    # Fresh cwd per leg (ffPlayground must not pre-exist); absolute binary path.
    ( cd "$cwd" && "$bin" 5 "$leg" | sed -E "$NORM_SED" > "$out" )
    rc=$?
    # Clean up the per-leg scratch dir (holds ffpartNNN partials + ffPlayground).
    rm -rf "$cwd"
    if [[ $rc -ne 0 ]]; then
      echo "ERROR: $bin 5 $leg exited $rc (see $out)" >&2
      fail=1
      continue
    fi
    count="$(grep -c ') sum' "$out")"
    want="${EXPECTED_COUNTS[$leg]}"
    if [[ "$count" != "$want" ]]; then
      echo "ERROR: $out solution count $count != expected $want" >&2
      fail=1
      continue
    fi
    echo "OK  $out  (${count} solutions)"
  done
done

# Sanity: all 18 files present
missing=0
for prog in $PROGS; do
  for leg in $LEGS; do
    [[ -f "$GOLDENS/out_${prog}_${leg}.txt" ]] || { echo "ERROR: missing $GOLDENS/out_${prog}_${leg}.txt" >&2; missing=1; }
  done
done
[[ $missing -eq 1 ]] && fail=1

if [[ $fail -ne 0 ]]; then
  echo "regenerate_goldens.sh: FAILED" >&2
  exit 1
fi
echo "regenerate_goldens.sh: all 18 goldens regenerated OK"
