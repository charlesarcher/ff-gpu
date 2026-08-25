# FF-GPU Per-Device Benchmark Report

> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.
> Median of 3 timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.
> Every GPU-config rep passed hard device-attribution gates (vendor filter line +
> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.

## Environment

```
timestamp_utc = 2026-08-25T01:50:26Z
git_head      = 8fb8dcd (dirty files: 5)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 604.705 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 604.746 ms
```

## Configurations

| # | Config | Description |
|---|--------|-------------|
| 1 | `ref` | Reference ff_seg (software/CPU implementation) |
| 2 | `amd_gpu` | ff_sieve --devices=amd --gpu-search (full GPU: sieve+search on RX 9070 XT) |
| 3 | `nvidia_gpu` | ff_sieve --devices=nvidia --gpu-search (full GPU: sieve+search on RTX 5090) |

## Wall-Clock Time (seconds)

| Config | Leg | median | min | max | stdev | outcome |
|--------|-----|--------|-----|-----|-------|---------|
| `ref` | 65536 | 0.019 | 0.019 | 0.020 | 0.000 | OK |
| `ref` | 131072 | 0.085 | 0.085 | 0.086 | 0.000 | OK |
| `ref` | 262144 | 0.441 | 0.433 | 0.451 | 0.007 | OK |
| `ref` | 524288 | 2.240 | 2.238 | 2.263 | 0.011 | OK |
| `ref` | 1048576 | 9.473 | 9.373 | 9.482 | 0.049 | OK |
| `ref` | 2097152 | 41.340 | 41.042 | 41.527 | 0.200 | OK |
| `amd_gpu` | 65536 | 0.113 | 0.098 | 0.497 | 0.185 | OK |
| `amd_gpu` | 131072 | 0.154 | 0.149 | 0.163 | 0.006 | OK |
| `amd_gpu` | 262144 | 0.362 | 0.359 | 0.375 | 0.007 | OK |
| `amd_gpu` | 524288 | 1.193 | 1.113 | 1.287 | 0.071 | OK |
| `amd_gpu` | 1048576 | 3.283 | 3.229 | 3.296 | 0.029 | OK |
| `amd_gpu` | 2097152 | 12.483 | 12.078 | 12.588 | 0.220 | OK |
| `nvidia_gpu` | 65536 | 0.263 | 0.245 | 0.348 | 0.045 | OK |
| `nvidia_gpu` | 131072 | 0.288 | 0.286 | 0.289 | 0.001 | OK |
| `nvidia_gpu` | 262144 | 0.450 | 0.434 | 0.454 | 0.009 | OK |
| `nvidia_gpu` | 524288 | 1.029 | 0.942 | 1.038 | 0.043 | OK |
| `nvidia_gpu` | 1048576 | 1.974 | 1.956 | 1.978 | 0.010 | OK |
| `nvidia_gpu` | 2097152 | 5.452 | 5.372 | 5.470 | 0.043 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.085 | 0.441 | 2.240 | 9.473 | 41.340 |
| `amd_gpu` | 0.113 | 0.154 | 0.362 | 1.193 | 3.283 | 12.483 |
| `nvidia_gpu` | 0.263 | 0.288 | 0.450 | 1.029 | 1.974 | 5.452 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.17x | 0.55x | 1.22x | 1.88x | 2.89x | 3.31x |
| `nvidia_gpu` | 0.07x | 0.30x | 0.98x | 2.18x | 4.80x | 7.58x |

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a46/n0 | ✅ a50/n0 | ✅ a67/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a77/n7 | ✅ a0/n18 | ✅ a0/n1 | ✅ a0/n42 | ✅ a0/n100 | ✅ a0/n100 |
`a<n>%`/`n<n>%` = max observed utilization (rocm-smi / nvidia-smi samples) for AMD/NVIDIA.
Sub-second legs may finish between sampler ticks (`n/a`); the stderr-name gates above
remain authoritative for those.

## Notes

- `amd_gpu` @ 2M: completes by default since task 14 — the aggregate capacity gate
  auto-enables the host overflow tier (rc=0, byte-identical, outcome OK); GPU search
  falls back to CPU on this leg via the documented "no device fits" capacity notice.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
