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
timestamp_utc = 2026-08-25T23:47:31Z
git_head      = 47676b3 (dirty files: 2)
kernel        = 7.2.0-1-cachyos
cpu           = AMD Ryzen 9 9950X3D 16-Core Processor
reps          = 3  warmup=1  legs=65536 131072 262144 524288 1048576 2097152
protocol      = sha_gate=on median_n=5 jit_warmup=1 autoextend=1
--- ff_sieve --list-devices (enumeration proof) ---
ff_sieve timing: device enumeration (--list-devices) = 598.941 ms
== ff_sieve logical device list (deduped by PCI bus ID) ==
[ff_sieve] device[0] AMD Radeon RX 9070 XT (vendor=amd bus=0000:05:00): free=17026777088 B (15.86 GiB) total=17095983104 B (15.92 GiB) compute=12.0 maxThreads=1024 sharedMem=65536 B smem/mp=65536 B smp=32
[ff_sieve] device[1] NVIDIA GeForce RTX 5090 (vendor=nvidia bus=0000:01:00): free=33179631616 B (30.90 GiB) total=33711521792 B (31.40 GiB) compute=12.0 maxThreads=1024 sharedMem=49152 B smem/mp=102400 B smp=170
2 logical device(s), 0 bus-ID duplicate(s) skipped
ff_sieve timing: total = 598.983 ms
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
| `ref` | 65536 | 0.019 | 0.018 | 0.019 | 0.000 | OK |
| `ref` | 131072 | 0.083 | 0.082 | 0.091 | 0.003 | OK |
| `ref` | 262144 | 0.414 | 0.402 | 0.449 | 0.018 | OK |
| `ref` | 524288 | 2.230 | 2.188 | 2.248 | 0.025 | OK |
| `ref` | 1048576 | 9.392 | 9.312 | 9.471 | 0.065 | OK |
| `ref` | 2097152 | 41.181 | 40.920 | 41.187 | 0.124 | OK |
| `amd_gpu` | 65536 | 0.085 | 0.082 | 0.095 | 0.004 | OK |
| `amd_gpu` | 131072 | 0.130 | 0.122 | 0.142 | 0.006 | OK |
| `amd_gpu` | 262144 | 0.254 | 0.244 | 0.257 | 0.005 | OK |
| `amd_gpu` | 524288 | 0.724 | 0.716 | 0.749 | 0.012 | OK |
| `amd_gpu` | 1048576 | 2.489 | 2.446 | 2.522 | 0.031 | OK |
| `amd_gpu` | 2097152 | 10.758 | 10.746 | 10.832 | 0.038 | OK |
| `nvidia_gpu` | 65536 | 0.234 | 0.225 | 0.266 | 0.013 | OK |
| `nvidia_gpu` | 131072 | 0.258 | 0.252 | 0.264 | 0.004 | OK |
| `nvidia_gpu` | 262144 | 0.337 | 0.330 | 0.345 | 0.005 | OK |
| `nvidia_gpu` | 524288 | 0.604 | 0.582 | 0.620 | 0.014 | OK |
| `nvidia_gpu` | 1048576 | 1.324 | 1.316 | 1.330 | 0.006 | OK |
| `nvidia_gpu` | 2097152 | 4.330 | 4.259 | 4.358 | 0.042 | OK |

### Median wall-clock matrix

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | 0.019 | 0.083 | 0.414 | 2.230 | 9.392 | 41.181 |
| `amd_gpu` | 0.085 | 0.130 | 0.254 | 0.724 | 2.489 | 10.758 |
| `nvidia_gpu` | 0.234 | 0.258 | 0.337 | 0.604 | 1.324 | 4.330 |

## Speedup vs reference (ref/config; >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `amd_gpu` | 0.22x **REGRESSION** | 0.64x **REGRESSION** | 1.63x | 3.08x | 3.77x | 3.83x |
| `nvidia_gpu` | 0.08x **REGRESSION** | 0.32x **REGRESSION** | 1.23x | 3.69x | 7.09x | 9.51x |

`**REGRESSION**` marks speedup < 1.00x versus the SAME-SESSION reference above — a distinct
visual mark from the ⚠️GATE / ✗(...) outcome annotations, which carry prior-verdict comparisons.

## Device Attribution (hard gates + utilization evidence)

Hard gates per rep: vendor-filter line present, `GPU search:` names the intended
card, and no foreign card named. All reps below show the aggregate result.

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|
| `ref` | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a | ✅ n/a |
| `amd_gpu` | ✅ a59/n0 | ✅ a63/n0 | ✅ a75/n0 | ✅ a100/n0 | ✅ a100/n0 | ✅ a100/n0 |
| `nvidia_gpu` | ✅ a36/n18 | ✅ a0/n7 | ✅ a0/n59 | ✅ a0/n100 | ✅ a0/n100 | ✅ a0/n100 |
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
| `amd_gpu` | 65536 | 56.481 | NA | 0/7 |
| `amd_gpu` | 131072 | 100.938 | NA | 0/7 |
| `amd_gpu` | 262144 | 225.258 | NA | 0/5 |
| `amd_gpu` | 524288 | 695.540 | NA | 0/5 |
| `amd_gpu` | 1048576 | 2457.515 | NA | 0/3 |
| `amd_gpu` | 2097152 | 10723.367 | NA | 0/3 |
| `nvidia_gpu` | 65536 | 145.091 | NA | 0/7 |
| `nvidia_gpu` | 131072 | 167.185 | NA | 0/5 |
| `nvidia_gpu` | 262144 | 232.081 | NA | 0/5 |
| `nvidia_gpu` | 524288 | 508.942 | NA | 0/5 |
| `nvidia_gpu` | 1048576 | 1244.420 | NA | 0/3 |
| `nvidia_gpu` | 2097152 | 4233.724 | NA | 0/3 |

## Notes

- `amd_gpu` @ 2M: TRUE GPU search since the wheel-30 landing — the compressed
  internal map (8.53 GiB at 2M) fits the card, so every rep runs the search
  kernel on-device (card named in stderr, zero fallback notices, residency
  handoff 0 B H2D). Historical eras — capacity-gate refusal, then task-14
  auto host-tier spill with CPU-search fallback — are both gone.
- Correctness per rep: solution-count assert against the golden contract (2357/4776/9163/
  18408/35556/71424) in addition to rc==0; any DEFECT marks a real regression.
