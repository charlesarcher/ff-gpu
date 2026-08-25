# FF-GPU Per-Device Benchmark Report

> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.
> Median of 3 timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.
> Every GPU-config rep passed hard device-attribution gates (vendor filter line +
> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.
> Per-rep stdout sha256 gate (ACTIVE): timing digits normalized (`Prime|Freudenthal time: N`)
> before hashing against the identically-normalized `goldens/out_ff_seg_<leg>.txt`; any
> mismatch marks the cell DEFECT.

## Environment

```
timestamp_utc = 2026-08-25T16:25:06Z
git_head      = 32b5859 (dirty files: 5)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
protocol      = sha_gate=on median_n=0 jit_warmup=0 autoextend=0
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 602.668 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 602.707 ms
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
| `ref` | 65536 | 0.018 | 0.018 | 0.019 | 0.000 | OK |
| `ref` | 131072 | 0.083 | 0.079 | 0.085 | 0.002 | OK |
| `ref` | 262144 | 0.429 | 0.411 | 0.432 | 0.009 | OK |
| `ref` | 524288 | 2.194 | 2.178 | 2.254 | 0.033 | OK |
| `ref` | 1048576 | 9.315 | 9.219 | 9.350 | 0.055 | OK |
| `ref` | 2097152 | 41.020 | 40.829 | 41.355 | 0.217 | OK |
| `amd_gpu` | 65536 | 0.111 | 0.100 | 0.492 | 0.182 | OK |
| `amd_gpu` | 131072 | 0.156 | 0.151 | 0.168 | 0.007 | OK |
| `amd_gpu` | 262144 | 0.361 | 0.350 | 0.383 | 0.014 | OK |
| `amd_gpu` | 524288 | 1.148 | 1.114 | 1.171 | 0.023 | OK |
| `amd_gpu` | 1048576 | 3.320 | 3.310 | 3.441 | 0.060 | OK |
| `amd_gpu` | 2097152 | 12.469 | 12.467 | 12.673 | 0.097 | OK |
| `nvidia_gpu` | 65536 | 0.254 | 0.250 | 0.341 | 0.042 | OK |
| `nvidia_gpu` | 131072 | 0.285 | 0.282 | 0.288 | 0.002 | OK |
| `nvidia_gpu` | 262144 | 0.442 | 0.429 | 0.461 | 0.013 | OK |
| `nvidia_gpu` | 524288 | 0.982 | 0.941 | 0.990 | 0.021 | OK |
| `nvidia_gpu` | 1048576 | 1.984 | 1.945 | 2.045 | 0.041 | OK |
| `nvidia_gpu` | 2097152 | 5.469 | 5.269 | 5.520 | 0.108 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.018 | 0.083 | 0.429 | 2.194 | 9.315 | 41.020 |
| `amd_gpu` | 0.111 | 0.156 | 0.361 | 1.148 | 3.320 | 12.469 |
| `nvidia_gpu` | 0.254 | 0.285 | 0.442 | 0.982 | 1.984 | 5.469 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.16x **REGRESSION** | 0.53x **REGRESSION** | 1.19x | 1.91x | 2.81x | 3.29x |
| `nvidia_gpu` | 0.07x **REGRESSION** | 0.29x **REGRESSION** | 0.97x **REGRESSION** | 2.23x | 4.70x | 7.50x |

`**REGRESSION**` marks speedup < 1.00x versus the SAME-SESSION reference above — a distinct
visual mark from the ⚠️GATE / ✗(...) outcome annotations, which carry prior-verdict comparisons.

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a44/n0 | ✅ a51/n0 | ✅ a66/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a67/n13 | ✅ a0/n13 | ✅ a0/n28 | ✅ a0/n100 | ✅ a0/n100 | ✅ a0/n100 |
`a<n>%`/`n<n>%` = max observed utilization (rocm-smi / nvidia-smi samples) for AMD/NVIDIA.
Sub-second legs may finish between sampler ticks (`n/a`); the stderr-name gates above
remain authoritative for those.

## Phase Decomposition & Unaccounted Time

`unaccounted_ms = phase_total − (enum+budget+sieve+search+wheel_expansion+zero_fill+
sched_teardown+setup+teardown)`, computed per rep by the harness and carried in
`bench_per_device_raw.csv`; the table shows the median over reps where ALL ten inputs existed.
`NA` = timer lines incomplete for that cell: the reference emits NO `ff_sieve timing:` lines,
and binaries predating a given timer simply lack its line. Missing data is NEVER coerced to zero.

| Config | Leg | phase_total_med_ms | unaccounted_med_ms | computable_reps |
|--------|-----|--------------------|--------------------|-----------------|
| `ref` | 65536 | NA | NA | 0/3 |
| `ref` | 131072 | NA | NA | 0/3 |
| `ref` | 262144 | NA | NA | 0/3 |
| `ref` | 524288 | NA | NA | 0/3 |
| `ref` | 1048576 | NA | NA | 0/3 |
| `ref` | 2097152 | NA | NA | 0/3 |
| `amd_gpu` | 65536 | 82.235 | 0.204 | 3/3 |
| `amd_gpu` | 131072 | 129.669 | 0.370 | 3/3 |
| `amd_gpu` | 262144 | 333.590 | 0.634 | 3/3 |
| `amd_gpu` | 524288 | 1120.556 | 1.823 | 3/3 |
| `amd_gpu` | 1048576 | 3289.008 | 5.963 | 3/3 |
| `amd_gpu` | 2097152 | 12438.834 | 22.354 | 3/3 |
| `nvidia_gpu` | 65536 | 162.415 | 0.183 | 3/3 |
| `nvidia_gpu` | 131072 | 196.959 | 0.173 | 3/3 |
| `nvidia_gpu` | 262144 | 344.493 | 0.638 | 3/3 |
| `nvidia_gpu` | 524288 | 877.863 | 1.100 | 3/3 |
| `nvidia_gpu` | 1048576 | 1899.352 | 3.863 | 3/3 |
| `nvidia_gpu` | 2097152 | 5354.283 | 14.358 | 3/3 |

## Notes

- `amd_gpu` @ 2M: TRUE GPU search since the wheel-30 landing — the compressed
  internal map (8.53 GiB at 2M) fits the card, so every rep runs the search
  kernel on-device (card named in stderr, zero fallback notices, residency
  handoff 0 B H2D). Historical eras — capacity-gate refusal, then task-14
  auto host-tier spill with CPU-search fallback — are both gone.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
