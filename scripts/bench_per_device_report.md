# FF-GPU Per-Device Benchmark Report

> Full-GPU-enablement sweep: reference vs `--devices=<vendor> --gpu-search`.
> Median of 5 timed reps after 1 untimed warmup; per-rep walls in `bench_per_device_raw.csv`.
> Every GPU-config rep passed hard device-attribution gates (vendor filter line +
> `GPU search:` stderr line naming exactly the intended card); utilization samples recorded.
> Per-rep stdout sha256 gate (ACTIVE): timing digits normalized (`Prime|Freudenthal time: N`)
> before hashing against the identically-normalized `goldens/out_ff_seg_<leg>.txt`; any
> mismatch marks the cell DEFECT.

## Environment

```
timestamp_utc = 2026-08-25T16:50:49Z
git_head      = 74c5b20 (dirty files: 3)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
protocol      = sha_gate=on median_n=5 jit_warmup=1 autoextend=1
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 611.003 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 611.043 ms
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
| `ref` | 65536 | 0.019 | 0.018 | 0.020 | 0.001 | OK |
| `ref` | 131072 | 0.088 | 0.084 | 0.090 | 0.003 | OK |
| `ref` | 262144 | 0.422 | 0.403 | 0.447 | 0.013 | OK |
| `ref` | 524288 | 2.245 | 2.194 | 2.302 | 0.043 | OK |
| `ref` | 1048576 | 9.407 | 9.360 | 9.439 | 0.032 | OK |
| `ref` | 2097152 | 41.105 | 41.090 | 41.328 | 0.109 | OK |
| `amd_gpu` | 65536 | 0.101 | 0.099 | 0.116 | 0.006 | OK |
| `amd_gpu` | 131072 | 0.157 | 0.150 | 0.171 | 0.006 | OK |
| `amd_gpu` | 262144 | 0.357 | 0.355 | 0.380 | 0.008 | OK |
| `amd_gpu` | 524288 | 1.151 | 1.089 | 1.255 | 0.062 | OK |
| `amd_gpu` | 1048576 | 3.259 | 3.171 | 3.279 | 0.047 | OK |
| `amd_gpu` | 2097152 | 12.467 | 12.134 | 12.504 | 0.166 | OK |
| `nvidia_gpu` | 65536 | 0.252 | 0.251 | 0.287 | 0.014 | OK |
| `nvidia_gpu` | 131072 | 0.296 | 0.281 | 0.306 | 0.010 | OK |
| `nvidia_gpu` | 262144 | 0.445 | 0.437 | 0.467 | 0.011 | OK |
| `nvidia_gpu` | 524288 | 0.983 | 0.950 | 1.042 | 0.031 | OK |
| `nvidia_gpu` | 1048576 | 1.984 | 1.981 | 1.993 | 0.005 | OK |
| `nvidia_gpu` | 2097152 | 5.528 | 5.456 | 5.632 | 0.072 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.088 | 0.422 | 2.245 | 9.407 | 41.105 |
| `amd_gpu` | 0.101 | 0.157 | 0.357 | 1.151 | 3.259 | 12.467 |
| `nvidia_gpu` | 0.252 | 0.296 | 0.445 | 0.983 | 1.984 | 5.528 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.19x **REGRESSION** | 0.56x **REGRESSION** | 1.18x | 1.95x | 2.89x | 3.30x |
| `nvidia_gpu` | 0.08x **REGRESSION** | 0.30x **REGRESSION** | 0.95x **REGRESSION** | 2.28x | 4.74x | 7.44x |

`**REGRESSION**` marks speedup < 1.00x versus the SAME-SESSION reference above — a distinct
visual mark from the ⚠️GATE / ✗(...) outcome annotations, which carry prior-verdict comparisons.

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a50/n0 | ✅ a52/n0 | ✅ a67/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a14/n3 | ✅ a0/n30 | ✅ a0/n77 | ✅ a0/n100 | ✅ a0/n100 | ✅ a0/n100 |
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
| `ref` | 65536 | NA | NA | 0/5 |
| `ref` | 131072 | NA | NA | 0/7 |
| `ref` | 262144 | NA | NA | 0/7 |
| `ref` | 524288 | NA | NA | 0/5 |
| `ref` | 1048576 | NA | NA | 0/3 |
| `ref` | 2097152 | NA | NA | 0/3 |
| `amd_gpu` | 65536 | 72.930 | 0.213 | 5/5 |
| `amd_gpu` | 131072 | 130.396 | 0.335 | 7/7 |
| `amd_gpu` | 262144 | 330.288 | 0.665 | 7/7 |
| `amd_gpu` | 524288 | 1121.255 | 1.922 | 5/5 |
| `amd_gpu` | 1048576 | 3231.337 | 6.090 | 3/3 |
| `amd_gpu` | 2097152 | 12435.086 | 24.209 | 3/3 |
| `nvidia_gpu` | 65536 | 161.325 | 0.155 | 5/5 |
| `nvidia_gpu` | 131072 | 198.611 | 0.197 | 5/5 |
| `nvidia_gpu` | 262144 | 348.627 | 0.466 | 5/5 |
| `nvidia_gpu` | 524288 | 889.958 | 1.159 | 5/5 |
| `nvidia_gpu` | 1048576 | 1900.272 | 3.951 | 3/3 |
| `nvidia_gpu` | 2097152 | 5410.866 | 14.515 | 3/3 |

## Notes

- `amd_gpu` @ 2M: TRUE GPU search since the wheel-30 landing — the compressed
  internal map (8.53 GiB at 2M) fits the card, so every rep runs the search
  kernel on-device (card named in stderr, zero fallback notices, residency
  handoff 0 B H2D). Historical eras — capacity-gate refusal, then task-14
  auto host-tier spill with CPU-search fallback — are both gone.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
