# FF-GPU Per-Device Benchmark Report

> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.
> Median of 3 timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.
> Every GPU-config rep passed hard device-attribution gates (vendor filter line +
> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.

## Environment

```
timestamp_utc = 2026-08-23T20:03:19Z
git_head      = 5e74fde (dirty files: 6)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 602.493 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 602.536 ms
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
| `ref` | 65536 | 0.019 | 0.018 | 0.023 | 0.002 | OK |
| `ref` | 131072 | 0.082 | 0.080 | 0.087 | 0.003 | OK |
| `ref` | 262144 | 0.440 | 0.404 | 0.441 | 0.017 | OK |
| `ref` | 524288 | 2.188 | 2.185 | 2.254 | 0.032 | OK |
| `ref` | 1048576 | 9.385 | 9.331 | 9.525 | 0.082 | OK |
| `ref` | 2097152 | 40.845 | 40.639 | 40.883 | 0.107 | OK |
| `amd_gpu` | 65536 | 1.354 | 1.003 | 1.395 | 0.176 | OK |
| `amd_gpu` | 131072 | 1.742 | 1.690 | 1.754 | 0.028 | OK |
| `amd_gpu` | 262144 | 2.682 | 2.674 | 2.711 | 0.016 | OK |
| `amd_gpu` | 524288 | 5.017 | 4.930 | 5.056 | 0.053 | OK |
| `amd_gpu` | 1048576 | 12.131 | 12.105 | 12.161 | 0.023 | OK |
| `amd_gpu` | 2097152 | 1.181 | 1.175 | 1.229 | 0.024 | EXPECTED_GATE_REFUSAL |
| `nvidia_gpu` | 65536 | 1.281 | 1.271 | 1.292 | 0.009 | OK |
| `nvidia_gpu` | 131072 | 1.316 | 1.281 | 1.342 | 0.025 | OK |
| `nvidia_gpu` | 262144 | 1.517 | 1.430 | 1.541 | 0.048 | OK |
| `nvidia_gpu` | 524288 | 2.102 | 2.094 | 2.110 | 0.007 | OK |
| `nvidia_gpu` | 1048576 | 4.479 | 4.446 | 4.502 | 0.023 | OK |
| `nvidia_gpu` | 2097152 | 14.435 | 14.421 | 14.597 | 0.080 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.082 | 0.440 | 2.188 | 9.385 | 40.845 |
| `amd_gpu` | 1.354 | 1.742 | 2.682 | 5.017 | 12.131 | 1.181 |
| `nvidia_gpu` | 1.281 | 1.316 | 1.517 | 2.102 | 4.479 | 14.435 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.01x | 0.05x | 0.16x | 0.44x | 0.77x | - |
| `nvidia_gpu` | 0.01x | 0.06x | 0.29x | 1.04x | 2.10x | 2.83x |

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a59/n1 | ✅ a100/n1 | ✅ a100/n1 | ✅ a100/n0 | ✅ a100/n1 | ⚠️ refused |
| `nvidia_gpu` | ✅ a3/n36 | ✅ a3/n36 | ✅ a2/n96 | ✅ a2/n100 | ✅ a2/n100 | ✅ a3/n100 |
`a<n>%`/`n<n>%` = max observed utilization (rocm-smi / nvidia-smi samples) for AMD/NVIDIA.
Sub-second legs may finish between sampler ticks (`n/a`); the stderr-name gates above
remain authoritative for those.

## Notes

- `amd_gpu` @ 2M: capacity-gate refusal by design (AMD backing ≈13.2 GiB < 16 GiB map), rc=1;
  recorded as `EXPECTED_GATE_REFUSAL`. The spill path (`--host-tier-cap=auto`) completes this
  leg byte-identically but is not part of this default-path sweep.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
