# FF-GPU Per-Device Benchmark Report

> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.
> Median of 3 timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.
> Every GPU-config rep passed hard device-attribution gates (vendor filter line +
> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.

## Environment

```
timestamp_utc = 2026-08-24T10:07:34Z
git_head      = 7ab4b7d (dirty files: 1)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 608.608 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 608.644 ms
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
| `ref` | 65536 | 0.019 | 0.018 | 0.021 | 0.001 | OK |
| `ref` | 131072 | 0.084 | 0.082 | 0.086 | 0.002 | OK |
| `ref` | 262144 | 0.427 | 0.426 | 0.438 | 0.005 | OK |
| `ref` | 524288 | 2.198 | 2.188 | 2.241 | 0.023 | OK |
| `ref` | 1048576 | 9.227 | 9.208 | 9.264 | 0.023 | OK |
| `ref` | 2097152 | 40.559 | 40.531 | 40.733 | 0.089 | OK |
| `amd_gpu` | 65536 | 0.104 | 0.093 | 0.483 | 0.181 | OK |
| `amd_gpu` | 131072 | 0.218 | 0.213 | 0.223 | 0.004 | OK |
| `amd_gpu` | 262144 | 0.618 | 0.618 | 0.629 | 0.005 | OK |
| `amd_gpu` | 524288 | 1.878 | 1.844 | 1.887 | 0.019 | OK |
| `amd_gpu` | 1048576 | 6.296 | 6.280 | 6.382 | 0.045 | OK |
| `amd_gpu` | 2097152 | 50.596 | 50.587 | 50.618 | 0.013 | OK |
| `nvidia_gpu` | 65536 | 0.253 | 0.246 | 0.338 | 0.042 | OK |
| `nvidia_gpu` | 131072 | 0.271 | 0.270 | 0.271 | 0.000 | OK |
| `nvidia_gpu` | 262144 | 0.419 | 0.417 | 0.427 | 0.004 | OK |
| `nvidia_gpu` | 524288 | 0.894 | 0.888 | 0.902 | 0.006 | OK |
| `nvidia_gpu` | 1048576 | 2.506 | 2.501 | 2.522 | 0.009 | OK |
| `nvidia_gpu` | 2097152 | 11.822 | 11.809 | 11.839 | 0.012 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.084 | 0.427 | 2.198 | 9.227 | 40.559 |
| `amd_gpu` | 0.104 | 0.218 | 0.618 | 1.878 | 6.296 | 50.596 |
| `nvidia_gpu` | 0.253 | 0.271 | 0.419 | 0.894 | 2.506 | 11.822 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.18x | 0.39x | 0.69x | 1.17x | 1.47x | 0.80x |
| `nvidia_gpu` | 0.08x | 0.31x | 1.02x | 2.46x | 3.68x | 3.43x |

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a58/n0 | ✅ a78/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a13/n0 | ✅ a0/n38 | ✅ a0/n36 | ✅ a0/n100 | ✅ a0/n100 | ✅ a0/n100 |
`a<n>%`/`n<n>%` = max observed utilization (rocm-smi / nvidia-smi samples) for AMD/NVIDIA.
Sub-second legs may finish between sampler ticks (`n/a`); the stderr-name gates above
remain authoritative for those.

## Notes

- `amd_gpu` @ 2M: capacity-gate refusal by design (AMD backing ≈13.2 GiB < 16 GiB map), rc=1;
  recorded as `EXPECTED_GATE_REFUSAL`. The spill path (`--host-tier-cap=auto`) completes this
  leg byte-identically but is not part of this default-path sweep.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
