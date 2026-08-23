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
| `ref` | 0.024 | 0.100 | 0.471 | 2.414 | 9.883 | 48.073 |
| `cpu_no_gpu` | 0.464 | 0.728 | 1.656 | 4.449 | 11.902 | 51.286 |
| `nvidia_cpu` | 0.407 | 0.483 | 0.831 | 2.685 | 10.065 | 46.367 |
| `amd_cpu` | 0.469 | 0.714 | 1.599 | 4.216 | 13.495 | 0.231 ✗(rc=1) |
| `nvidia_gpu_search` | 0.483 ✗(rc=0) | 0.755 ✗(rc=0) | 2.805 ✗(rc=0) | 12.849 ✗(rc=0) | 57.102 ✗(rc=0) | 45.992 |
| `amd_gpu_search` | 0.568 | 0.907 | 1.873 | 4.306 | 11.818 | 0.243 ✗(rc=1) |

## Speedup vs CPU reference (ref / config; >1 = faster than reference)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `cpu_no_gpu` | 0.05x | 0.14x | 0.28x | 0.54x | 0.83x | 0.94x |
| `nvidia_cpu` | 0.06x | 0.21x | 0.57x | 0.90x | 0.98x | 1.04x |
| `amd_cpu` | 0.05x | 0.14x | 0.29x | 0.57x | 0.73x | - |
| `nvidia_gpu_search` | - | - | - | - | - | 1.05x |
| `amd_gpu_search` | 0.04x | 0.11x | 0.25x | 0.56x | 0.84x | - |

## Correctness Summary

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cpu_no_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `nvidia_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `amd_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `nvidia_gpu_search` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `amd_gpu_search` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Notes

- `amd` @ 2M: sieve rejects at startup (AMD backing 15.86 GiB < 16 GiB map) — rejection logic working as designed, rc=1.
- `gpu_search` @ 2M: pre-existing sharded-map defect (rc=141, documented in .omo/notepads/ff-gpu-consolidated/issues.md). Not a regression of this plan.
- `--devices=<vendor>` restricts to one GPU; dual-GPU default scheduling is benchmarked separately in `bench_results.csv`.
