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
timestamp_utc = 2026-08-26T23:22:31Z
git_head      = b5caf5d (dirty files: 3)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 196608 262144 327680 393216 458752 524288 589824 655360 720896 786432 851968 917504 983040 1048576 1114112 1179648 1245184 1310720 1376256 1441792 1507328 1572864 1638400 1703936 1769472 1835008 1900544 1966080 2031616 2097152
protocol      = sha_gate=on median_n=5 jit_warmup=1 autoextend=1
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 602.428 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 602.466 ms
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
| `ref` | 65536 | 0.019 | 0.019 | 0.019 | 0.000 | OK |
| `ref` | 131072 | 0.088 | 0.086 | 0.089 | 0.001 | OK |
| `ref` | 262144 | 0.419 | 0.404 | 0.432 | 0.010 | OK |
| `ref` | 524288 | 2.177 | 2.170 | 2.197 | 0.011 | OK |
| `ref` | 1048576 | 9.282 | 9.196 | 9.324 | 0.053 | OK |
| `ref` | 2097152 | 40.437 | 40.436 | 40.686 | 0.118 | OK |
| `amd_gpu` | 65536 | 0.085 | 0.083 | 0.099 | 0.005 | OK |
| `amd_gpu` | 131072 | 0.131 | 0.127 | 0.139 | 0.005 | OK |
| `amd_gpu` | 262144 | 0.243 | 0.235 | 0.254 | 0.006 | OK |
| `amd_gpu` | 524288 | 0.724 | 0.697 | 0.750 | 0.019 | OK |
| `amd_gpu` | 1048576 | 2.457 | 2.437 | 2.495 | 0.024 | OK |
| `amd_gpu` | 2097152 | 10.733 | 10.731 | 11.208 | 0.224 | OK |
| `nvidia_gpu` | 65536 | 0.226 | 0.222 | 0.232 | 0.004 | OK |
| `nvidia_gpu` | 131072 | 0.252 | 0.244 | 0.254 | 0.004 | OK |
| `nvidia_gpu` | 262144 | 0.335 | 0.321 | 0.341 | 0.007 | OK |
| `nvidia_gpu` | 524288 | 0.585 | 0.581 | 0.613 | 0.012 | OK |
| `nvidia_gpu` | 1048576 | 1.331 | 1.321 | 1.334 | 0.006 | OK |
| `nvidia_gpu` | 2097152 | 4.243 | 4.234 | 4.279 | 0.019 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.088 | 0.419 | 2.177 | 9.282 | 40.437 |
| `amd_gpu` | 0.085 | 0.131 | 0.243 | 0.724 | 2.457 | 10.733 |
| `nvidia_gpu` | 0.226 | 0.252 | 0.335 | 0.585 | 1.331 | 4.243 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.22x **REGRESSION** | 0.67x **REGRESSION** | 1.72x | 3.01x | 3.78x | 3.77x |
| `nvidia_gpu` | 0.08x **REGRESSION** | 0.35x **REGRESSION** | 1.25x | 3.72x | 6.97x | 9.53x |

`**REGRESSION**` marks speedup < 1.00x versus the SAME-SESSION reference above — a distinct
visual mark from the ⚠️GATE / ✗(...) outcome annotations, which carry prior-verdict comparisons.

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a100/n0 | ✅ a61/n0 | ✅ a74/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a28/n1 | ✅ a0/n34 | ✅ a0/n88 | ✅ a0/n100 | ✅ a0/n100 | ✅ a0/n100 |
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
| `ref` | 131072 | NA | NA | 0/5 |
| `ref` | 262144 | NA | NA | 0/5 |
| `ref` | 524288 | NA | NA | 0/3 |
| `ref` | 1048576 | NA | NA | 0/3 |
| `ref` | 2097152 | NA | NA | 0/3 |
| `amd_gpu` | 65536 | 56.719 | NA | 0/7 |
| `amd_gpu` | 131072 | 103.041 | NA | 0/5 |
| `amd_gpu` | 262144 | 213.867 | NA | 0/5 |
| `amd_gpu` | 524288 | 696.115 | NA | 0/5 |
| `amd_gpu` | 1048576 | 2426.096 | NA | 0/3 |
| `amd_gpu` | 2097152 | 10704.961 | NA | 0/3 |
| `nvidia_gpu` | 65536 | 143.899 | NA | 0/5 |
| `nvidia_gpu` | 131072 | 165.502 | NA | 0/5 |
| `nvidia_gpu` | 262144 | 229.973 | NA | 0/5 |
| `nvidia_gpu` | 524288 | 504.465 | NA | 0/5 |
| `nvidia_gpu` | 1048576 | 1245.288 | NA | 0/3 |
| `nvidia_gpu` | 2097152 | 4166.618 | NA | 0/3 |

## Notes

- `amd_gpu` @ 2M: TRUE GPU search since the wheel-30 landing — the compressed
  internal map (8.53 GiB at 2M) fits the card, so every rep runs the search
  kernel on-device (card named in stderr, zero fallback notices, residency
  handoff 0 B H2D). Historical eras — capacity-gate refusal, then task-14
  auto host-tier spill with CPU-search fallback — are both gone.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
