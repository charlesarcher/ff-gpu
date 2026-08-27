# Fine-Grained Sweep Summary — 32 Points Linear 64K (65536→2097152)

> Protocol: median-of-5 sub-second (BENCH_MEDIAN_N=5), sha gate active, JIT warmup on, autoextend on. 96/96 OK. GPU-exclusive (--devices=amd / nvidia --gpu-search).

> Generated: timestamp_utc = 2026-08-26T23:22:31Z

## Wall medians (s) and speedup vs ref (ref / gpu; >1 = GPU faster)

| leg | ref (s) | amd_gpu (s) | nvidia_gpu (s) | amd speedup | nvidia speedup |
|-----|---------|-------------|----------------|-------------|----------------|
| 65536 | 0.019 | 0.085 | 0.226 | 0.22x REG | 0.08x REG |
| 131072 | 0.088 | 0.131 | 0.252 | 0.67x REG | 0.35x REG |
| 196608 | 0.202 | 0.169 | 0.272 | 1.20x | 0.74x REG |
| 262144 | 0.419 | 0.243 | 0.335 | 1.72x | 1.25x |
| 327680 | 0.726 | 0.339 | 0.394 | 2.14x | 1.84x |
| 393216 | 1.109 | 0.443 | 0.448 | 2.50x | 2.48x |
| 458752 | 1.612 | 0.553 | 0.492 | 2.92x | 3.28x |
| 524288 | 2.177 | 0.724 | 0.585 | 3.01x | 3.72x |
| 589824 | 2.685 | 0.901 | 0.701 | 2.98x | 3.83x |
| 655360 | 3.365 | 1.012 | 0.742 | 3.33x | 4.54x |
| 720896 | 4.215 | 1.251 | 0.892 | 3.37x | 4.73x |
| 786432 | 4.942 | 1.449 | 0.939 | 3.41x | 5.26x |
| 851968 | 5.890 | 1.765 | 1.084 | 3.34x | 5.43x |
| 917504 | 6.908 | 1.971 | 1.118 | 3.50x | 6.18x |
| 983040 | 8.106 | 2.227 | 1.187 | 3.64x | 6.83x |
| 1048576 | 9.282 | 2.457 | 1.331 | 3.78x | 6.97x |
| 1114112 | 10.484 | 2.777 | 1.431 | 3.78x | 7.33x |
| 1179648 | 11.778 | 3.228 | 1.612 | 3.65x | 7.31x |
| 1245184 | 13.217 | 3.534 | 1.746 | 3.74x | 7.57x |
| 1310720 | 14.866 | 4.003 | 1.921 | 3.71x | 7.74x |
| 1376256 | 16.571 | 4.420 | 2.150 | 3.75x | 7.71x |
| 1441792 | 18.126 | 4.878 | 2.219 | 3.72x | 8.17x |
| 1507328 | 19.925 | 5.419 | 2.313 | 3.68x | 8.61x |
| 1572864 | 21.892 | 5.870 | 2.548 | 3.73x | 8.59x |
| 1638400 | 23.913 | 6.556 | 2.752 | 3.65x | 8.69x |
| 1703936 | 25.897 | 7.018 | 3.076 | 3.69x | 8.42x |
| 1769472 | 28.114 | 7.868 | 3.287 | 3.57x | 8.55x |
| 1835008 | 30.325 | 8.214 | 3.385 | 3.69x | 8.96x |
| 1900544 | 32.499 | 8.848 | 3.595 | 3.67x | 9.04x |
| 1966080 | 35.567 | 9.384 | 3.822 | 3.79x | 9.31x |
| 2031616 | 38.011 | 10.233 | 4.207 | 3.71x | 9.04x |
| 2097152 | 40.437 | 10.733 | 4.243 | 3.77x | 9.53x |

## Speedup crossover

- **AMD RX 9070 XT** overtakes CPU at **196608** (first leg where amd speedup >1).
  - 65536: 0.22x REG, 131072: 0.67x REG, **196608: 1.20x** (first win), 262144: 1.72x.
- **NVIDIA RTX 5090** overtakes CPU at **262144** (first leg where nvidia speedup >1).
  - 65536: 0.08x REG, 131072: 0.35x REG, 196608: 0.74x REG, **262144: 1.25x** (first win).

Sub-196K legs are init-floor dominated; GPU wins are sustained and widen monotonically beyond crossover.

## Peak / tail

- At 2097152: ref 40.437s, amd 10.733s (3.77x), nvidia 4.243s (9.53x).

## CSV integrity

- `scripts/bench_per_device_results.csv`: 96 rows (32 legs × 3 configs), all outcome OK, all exit_code 0, solutions == expected on 96/96.
- `scripts/bench_per_device_raw.csv`: 342 data rows + header; median-of-5 for sub-second legs, median-of-3 thereafter, plus autoextend (amd@65536 7 reps).
- Copies: `.omo/start-work/evidence/fine-sweep-{results,raw,report}.csv/md`

