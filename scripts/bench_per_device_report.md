# FF-GPU Per-Device Benchmark Report

> Median of 3 runs per config x leg, post-fix binary (HEAD `72d9ccc`). CPU included in every comparison.

## Configurations

| # | Config | Description |
|---|--------|-------------|
| 1 | `ref` | Reference ff_seg (31-thread CPU) |
| 2 | `cpu_no_gpu` | ff_sieve --no-gpu (CPU sieve + CPU search) |
| 3 | `nvidia_cpu` | ff_sieve --devices=nvidia (GPU sieve + CPU search) |
| 4 | `amd_cpu` | ff_sieve --devices=amd (GPU sieve + CPU search) |
| 5 | `nvidia_gpu_search` | ff_sieve --gpu-search --devices=nvidia (full GPU) |
| 6 | `amd_gpu_search` | ff_sieve --gpu-search --devices=amd (full GPU) |

## Wall-Clock Time (seconds, median of 3)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.021 | 0.101 | 0.478 | 2.446 | 9.891 | 48.578 |
| `cpu_no_gpu` | 0.417 | 0.504 | 0.932 | 4.243 | 15.441 | 92.146 |
| `nvidia_cpu` | 0.391 | 0.456 | 0.786 | 2.731 | 11.696 | 64.849 |
| `amd_cpu` | 0.419 | 0.503 | 0.926 | 4.152 | 22.516 | 0.333 ✗(rc=1) |
| `nvidia_gpu_search` | 0.517 | 0.692 | 1.174 | 3.010 | 10.720 | 64.498 |
| `amd_gpu_search` | 0.528 | 0.703 | 1.245 | 4.252 | 21.469 | 0.234 ✗(rc=1) |

## Speedup vs CPU reference (ref / config; >1 = faster than reference)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `cpu_no_gpu` | 0.05x | 0.20x | 0.51x | 0.58x | 0.64x | 0.53x |
| `nvidia_cpu` | 0.05x | 0.22x | 0.61x | 0.90x | 0.85x | 0.75x |
| `amd_cpu` | 0.05x | 0.20x | 0.52x | 0.59x | 0.44x | - |
| `nvidia_gpu_search` | 0.04x | 0.15x | 0.41x | 0.81x | 0.92x | 0.75x |
| `amd_gpu_search` | 0.04x | 0.14x | 0.38x | 0.58x | 0.46x | - |

## Correctness Summary

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cpu_no_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `nvidia_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `amd_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `nvidia_gpu_search` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `amd_gpu_search` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Notes

- `amd` @ 2M: sieve rejects at startup (AMD backing 15.86 GiB < 16 GiB map) — rejection logic working as designed, rc=1.
- `gpu_search` @ 2M on NVIDIA only: **works correctly** (64.5s, 71424/71424) — single-device map avoids the pre-existing sharded-map defect (rc=141) seen in dual-GPU mode.
- `--devices=<vendor>` restricts to one GPU; dual-GPU default scheduling is benchmarked separately in `bench_results.csv`.
