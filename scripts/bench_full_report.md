# FF-GPU Full Benchmark Report

## Configurations

| # | Config | Description |
|---|--------|-------------|
| 1 | original | Reference ff_seg (31 threads) |
| 2 | gpu+cpu_5090 | GPU sieve + CPU search, RTX 5090 only |
| 3 | gpu+cpu_9070 | GPU sieve + CPU search, RX 9070 XT only |
| 4 | gpu_all_5090 | GPU sieve + GPU search, RTX 5090 only |
| 5 | gpu_all_9070 | GPU sieve + GPU search, RX 9070 XT only |

## Wall-Clock Time (seconds)


## Speedup vs Reference (ratio >1 = faster)

| Config | 65K | 131K | 262K | 524K | 1M | 2M |
|--------|-----|------|------|------|----|-----|

## Correctness Summary

