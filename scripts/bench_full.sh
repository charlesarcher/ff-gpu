#!/usr/bin/env bash
# bench_full.sh — Comprehensive GPU vs CPU benchmark sweep
#
# Tests 5 configurations across 6 legs (65536 to 2097152)
#
# Configurations:
#   1. Original (ff_seg reference)
#   2. GPU + CPU search, RTX 5090 only
#   3. GPU + CPU search, RX 9070 XT only
#   4. GPU all (sieve+search), RTX 5090 only
#   5. GPU all (sieve+search), RX 9070 XT only
#
# Usage: ./scripts/bench_full.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF_BIN="$ROOT/reference/ff_seg"
NEW_BIN="$ROOT/build/ff_sieve"
CSV="$ROOT/scripts/bench_full_results.csv"
REPORT="$ROOT/scripts/bench_full_report.md"
TIMEOUT_SEC=600  # Per-run timeout (10 min)

LEGS=(65536 131072 262144 524288 1048576 2097152)
# Expected solution counts
declare -A EXPECTED=(
  [65536]=2357 [131072]=4776 [262144]=9163
  [524288]=18408 [1048576]=35556 [2097152]=71424
)

# Configuration definitions: name, flags, env
declare -a CONFIGS=(
  "original| |"
  "gpu+cpu_5090|FF_THREADS=31 --devices=nvidia|"
  "gpu+cpu_9070|FF_THREADS=31 --devices=amd|"
  "gpu_all_5090|FF_THREADS=31 --gpu-search --devices=nvidia|"
  "gpu_all_9070|FF_THREADS=31 --gpu-search --devices=amd|"
)

mkdir -p "$ROOT/run"

echo "=== BUILD ==="
cmake --build --preset dev 2>&1 | tail -3
[[ -x "$REF_BIN" ]] || { echo "ERROR: $REF_BIN missing"; exit 1; }
[[ -x "$NEW_BIN" ]] || { echo "ERROR: $NEW_BIN missing"; exit 1; }
echo "Both binaries OK."

echo ""
echo "=== CLEANING PREVIOUS RUNS ==="
rm -rf "$ROOT/ffPlayground"

# Write CSV header
cat > "$CSV" << EOF
config,leg,wall_s,exit_code,solutions,expected,correct
EOF

echo "=== RUNNING BENCHMARKS ==="

for cfg_entry in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg_name cfg_flags cfg_env <<< "$cfg_entry"
  echo ""
  echo "--- Config: $cfg_name ---"
  
  for leg in "${LEGS[@]}"; do
    # Skip 2M for AMD configs (VRAM too small, even with spill it may fail)
    if [[ "$cfg_name" == *"9070"* && "$leg" -eq 2097152 ]]; then
      echo "  leg=$leg: SKIPPED (AMD VRAM insufficient for 2M)"
      continue
    fi
    
    # Build command - use absolute paths
    if [[ "$cfg_name" == "original" ]]; then
      cmd="$REF_BIN 5 $leg"
    else
      cmd="$NEW_BIN 5 $leg"
      if [[ "$cfg_name" == *"gpu_all"* ]]; then
        cmd="$cmd --gpu-search"
      fi
      # Add device filter
      if [[ "$cfg_name" == *"5090"* ]]; then
        cmd="$cmd --devices=nvidia"
      elif [[ "$cfg_name" == *"9070"* ]]; then
        cmd="$cmd --devices=amd"
      fi
    fi
    
    # Build full command with env vars
    full_cmd="$cmd"
    if [[ "$cfg_name" != "original" ]]; then
      full_cmd="FF_THREADS=31 $full_cmd"
    fi
    
    # Run with timeout
    scratch_dir="$(mktemp -d "$ROOT/run/bench_${cfg_name}_${leg}.XXXXXX")"
    
    t_start=$(python3 -c "import time; print(time.monotonic())")
    
    # Run the command
    if [[ "$cfg_name" == "original" ]]; then
      (cd "$scratch_dir" && eval "$cmd") > "$scratch_dir/output.txt" 2>"$scratch_dir/stderr.txt"
      exit_code=$?
    else
      (cd "$scratch_dir" && timeout $TIMEOUT_SEC bash -c "$cmd") > "$scratch_dir/output.txt" 2>"$scratch_dir/stderr.txt"
      exit_code=$?
    fi
    
    t_end=$(python3 -c "import time; print(time.monotonic())")
    wall_s=$(python3 -c "print(f'{$t_end - $t_start:.3f}')")
    
    # Count solutions
    if [[ $exit_code -eq 0 ]]; then
      solutions=$(grep -c ') sum' "$scratch_dir/output.txt" 2>/dev/null || echo "0")
    else
      solutions=0
    fi
    
    expected="${EXPECTED[$leg]}"
    if [[ "$exit_code" -eq 0 && "$solutions" -eq "$expected" ]]; then
      correct="YES"
    else
      correct="NO"
    fi
    
    echo "  leg=$leg  wall=${wall_s}s  exit=$exit_code  solutions=$solutions/$expected  $correct"
    
    # Append to CSV
    echo "$cfg_name,$leg,$wall_s,$exit_code,$solutions,$expected,$correct" >> "$CSV"
    
    # Clean up
    rm -rf "$scratch_dir"
    rm -rf "$ROOT/ffPlayground"
  done
done

echo ""
echo "=== GENERATING REPORT ==="

# Generate markdown report
cat > "$REPORT" << 'HEADER'
# FF-GPU Full Benchmark Report

## Configurations

| # | Config | Description |
|---|--------|-------------|
| 1 | original | Reference ff_seg (31 threads) |
| 2 | gpu+cpu_5090 | GPU sieve + CPU search, RTX 5090 only |
| 3 | gpu+cpu_9070 | GPU sieve + CPU search, RX 9070 XT only |
| 4 | gpu_all_5090 | GPU sieve + GPU search, RTX 5090 only |
| 5 | gpu_all_9070 | GPU sieve + GPU search, RX 9070 XT only |

## Wall-Clock Time (seconds)

HEADER

# Add timing table
python3 << 'PYEOF'
import csv

configs = set()
data = {}

with open('scripts/bench_full_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        leg = row['leg']
        wall = row['wall_s']
        configs.add(config)
        if config not in data:
            data[config] = {}
        data[config][leg] = wall

legs = sorted(set(int(l) for l in data.values()[0].keys()))
config_names = {
    'original': 'Original (ff_seg)',
    'gpu+cpu_5090': 'GPU+CPU RTX 5090',
    'gpu+cpu_9070': 'GPU+CPU RX 9070 XT',
    'gpu_all_5090': 'GPU All RTX 5090',
    'gpu_all_9070': 'GPU All RX 9070 XT'
}

print('| Config | 65K | 131K | 262K | 524K | 1M | 2M |')
print('|--------|-----|------|------|------|----|-----|')

for config in ['original', 'gpu+cpu_5090', 'gpu+cpu_9070', 'gpu_all_5090', 'gpu_all_9070']:
    if config not in data:
        continue
    name = config_names.get(config, config)
    row = [name]
    for leg in legs:
        leg_str = str(leg)
        if leg == 2097152:
            leg_str = '2M'
        elif leg == 1048576:
            leg_str = '1M'
        elif leg == 524288:
            leg_str = '524K'
        elif leg == 262144:
            leg_str = '262K'
        elif leg == 131072:
            leg_str = '131K'
        elif leg == 65536:
            leg_str = '65K'
        if leg_str in data[config]:
            row.append(f"{data[config][leg_str]:.3f}s")
        else:
            row.append('-')
    print('| ' + ' | '.join(row) + ' |')

print()

# Speedup vs reference
print('## Speedup vs Reference (ratio >1 = faster)')
print()
print('| Config | 65K | 131K | 262K | 524K | 1M | 2M |')
print('|--------|-----|------|------|------|----|-----|')

ref_data = data.get('original', {})
for config in ['gpu+cpu_5090', 'gpu+cpu_9070', 'gpu_all_5090', 'gpu_all_9070']:
    if config not in data:
        continue
    name = config_names.get(config, config)
    row = [name]
    for leg in legs:
        leg_str = str(leg)
        if leg == 2097152:
            leg_str = '2M'
        elif leg == 1048576:
            leg_str = '1M'
        elif leg == 524288:
            leg_str = '524K'
        elif leg == 262144:
            leg_str = '262K'
        elif leg == 131072:
            leg_str = '131K'
        elif leg == 65536:
            leg_str = '65K'
        
        gpu_t = data[config].get(leg_str, None)
        ref_t = ref_data.get(leg_str, None)
        
        if gpu_t and ref_t and float(gpu_t) > 0:
            speedup = float(ref_t) / float(gpu_t)
            row.append(f"{speedup:.2f}x")
        else:
            row.append('-')
    print('| ' + ' | '.join(row) + ' |')

PYEOF

echo "" >> "$REPORT"
echo "## Speedup vs Reference (ratio >1 = faster)" >> "$REPORT"
echo "" >> "$REPORT"
echo '| Config | 65K | 131K | 262K | 524K | 1M | 2M |' >> "$REPORT"
echo '|--------|-----|------|------|------|----|-----|' >> "$REPORT"

python3 << 'PYEOF'
import csv

data = {}
with open('scripts/bench_full_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        leg = row['leg']
        wall = row['wall_s']
        if config not in data:
            data[config] = {}
        data[config][leg] = wall

config_names = {
    'original': 'Original (ff_seg)',
    'gpu+cpu_5090': 'GPU+CPU RTX 5090',
    'gpu+cpu_9070': 'GPU+CPU RX 9070 XT',
    'gpu_all_5090': 'GPU All RTX 5090',
    'gpu_all_9070': 'GPU All RX 9070 XT'
}

ref_data = data.get('original', {})
legs = ['65536', '131072', '262144', '524288', '1048576', '2097152']
leg_labels = ['65K', '131K', '262K', '524K', '1M', '2M']

for config in ['gpu+cpu_5090', 'gpu+cpu_9070', 'gpu_all_5090', 'gpu_all_9070']:
    if config not in data:
        continue
    name = config_names.get(config, config)
    row = [name]
    for i, leg in enumerate(legs):
        gpu_t = data[config].get(leg, None)
        ref_t = ref_data.get(leg, None)
        if gpu_t and ref_t and float(gpu_t) > 0:
            speedup = float(ref_t) / float(gpu_t)
            row.append(f"{speedup:.2f}x")
        else:
            row.append('-')
    print('| ' + ' | '.join(row) + ' |')

PYEOF

echo "" >> "$REPORT"
echo "## Correctness Summary" >> "$REPORT"
echo "" >> "$REPORT"

python3 << 'PYEOF'
import csv

data = {}
with open('scripts/bench_full_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        leg = row['leg']
        correct = row['correct']
        if config not in data:
            data[config] = {}
        data[config][leg] = correct

config_names = {
    'original': 'Original (ff_seg)',
    'gpu+cpu_5090': 'GPU+CPU RTX 5090',
    'gpu+cpu_9070': 'GPU+CPU RX 9070 XT',
    'gpu_all_5090': 'GPU All RTX 5090',
    'gpu_all_9070': 'GPU All RX 9070 XT'
}

legs = ['65536', '131072', '262144', '524288', '1048576', '2097152']
leg_labels = ['65K', '131K', '262K', '524K', '1M', '2M']

for config in ['original', 'gpu+cpu_5090', 'gpu+cpu_9070', 'gpu_all_5090', 'gpu_all_9070']:
    if config not in data:
        continue
    name = config_names.get(config, config)
    row = [name]
    for i, leg in enumerate(legs):
        if leg in data[config]:
            row.append(data[config][leg])
        else:
            row.append('-')
    print('| ' + ' | '.join(row) + ' |')

PYEOF

echo ""
echo "=== BENCHMARK COMPLETE ==="
echo "CSV: $CSV"
echo "Report: $REPORT"