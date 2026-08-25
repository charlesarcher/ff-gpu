# Benchmark Methodology

This document defines the benchmark contract for ff-gpu performance claims.
The implementing script is `scripts/bench_per_device.sh`; its outputs are
`bench_per_device_results.csv` (summary), `bench_per_device_raw.csv`
(every timed rep), and `scripts/bench_per_device_report.md`.

## What is compared

| Config | Binary + flags | Meaning |
|---|---|---|
| `ref` | `reference/ff_seg 5 <leg>` | The untouched CPU reference implementation |
| `amd_gpu` | `build/ff_sieve --devices=amd --gpu-search 5 <leg>` | Full GPU enablement on the RX 9070 XT: **both** the sieve kernel and the Freudenthal search kernel run on that card |
| `nvidia_gpu` | `build/ff_sieve --devices=nvidia --gpu-search 5 <leg>` | Full GPU enablement on the RTX 5090: both kernels on that card |

Legs: the six golden legs `65536 … 2097152` (same work as the correctness
contract in GPU_PLAN §9).

## Map format: internal vs canonical

Since the wheel-30 landing, two byte layouts exist and both are load-bearing:

- **Internal (wheel-30)** — `internalMapBytes = ceil(span/30)` bytes; the
  layout the GPU sieve produces and the search kernel consumes **on-device**.
  It is what makes amd@2M fit its card: at 2097152 the internal map is
  9162596899 B = 8.53 GiB.
- **Canonical** — `ceil(span/16)` bytes; the hostMap / `--dump-map` format
  that defines the sha256 golden-map contract. It is re-materialized from the
  internal map at every non-GPU boundary (dump, host consumption), which is
  why dump-map hashes are unchanged vs the pre-wheel era.

Density consequence: internal is **1.875×** denser than canonical
(17179869185 B → 9162596899 B at 2M).

## Measurement protocol

1. **Environment snapshot** — one per sweep, written to `run/bench_env.txt`:
   git HEAD (+dirty count), kernel, CPU model, timestamp, rep/leg settings,
   and the verbatim `ff_sieve --list-devices` output (enumeration + PCI bus
   proof for both cards).
2. **Preflight** — binaries exist; warning if any process already holds an
   NVIDIA compute context (results would be polluted).
3. **Warmup** — one *untimed* rep per config at the smallest leg (GPU clock
   ramp, page cache). Never reported. Additionally — since the task-7
   re-baseline — one extra *untimed* JIT warmup before EVERY config×leg
   block (`BENCH_JIT_WARMUP=1`, default ON).
4. **Timed reps** — default 3 per config×leg (`BENCH_REPS`). Since the
   task-7 re-baseline, any cell whose FIRST rep's wall clock is < 1.000 s
   (sub-second heuristic) is automatically extended to median-of-N with
   N≥5 (`BENCH_MEDIAN_N=5`, default ON). Each rep runs in a fresh scratch
   cwd with `timeout 600`. Wall clock = full process time (spawn → exit),
   captured with `date +%s%N` around the whole invocation.
4b. **Auto-extend on spread** — while `(max−min)/median > 0.15`, up to 2
   extra reps are run per cell (`BENCH_AUTOEXTEND=1`, default ON).
5. **Aggregation** — summary CSV carries median/min/max/pstdev of the reps.
   The report tables quote the median; raw CSV preserves every rep so any
   statistic can be recomputed.
6. **Correctness inside timing** — a rep only counts as OK when rc==0 *and*
   the solution count matches the golden contract
   (2357/4776/9163/18408/35556/71424). A config×leg whose reps disagree or
   mismatch is marked `DEFECT`, never silently dropped.
7. **Per-rep stdout sha256 gate** — ACTIVE: each rep's stdout is normalized
   with `sed -E 's/(Prime|Freudenthal) time: [0-9]+/\1 time: N/'` BEFORE
   hashing and compared against the identically-normalized golden stdout;
   any mismatch marks the cell `DEFECT`. Timing-digit jitter passes;
   content corruption fires (fault-injection proven,
   `.omo/start-work/evidence/rebaseline-dual-report.md`).

## Frozen reference (task-7 re-baseline, 2026-08-25)

The sweep of 2026-08-25T16:50:49Z — run under the ACTIVATED protocol above
(`median_n=5 jit_warmup=1 autoextend=1`, sha256 gate live, 18/18 outcome=OK,
86/86 reps sha_ok) — is **the frozen ±3 % band reference going forward**.
All future wall-median comparisons quote bands against
`scripts/bench_per_device_results.csv` at this freeze; the old-vs-new dual
report (Gate-1 `d01d9dc` numbers side-by-side, zero >±10 % wall movers) is
archived at `.omo/start-work/evidence/rebaseline-dual-report.md`.

Environment state at freeze: `nvidia-persistenced` enablement was attempted
passwordless (`sudo -n`) and **BLOCKED** (password required); persistence
mode remained `Disabled` before and after. The freeze therefore carries no
persistenced effect — if persistenced is enabled later, that is a protocol
change and demands a fresh re-baseline before bands are quoted.

## Device attribution (making sure we bench the intended GPU)

Every GPU-config rep must pass hard gates derived from the binary's own
stderr:

1. The vendor filter engaged: `[ff_sieve] device filter: --devices=<v> kept
   1 of 2 logical device(s)`.
2. The search phase names exactly the intended card:
   `[ff_sieve] GPU search: device[0] AMD Radeon RX 9070 XT …` (resp. RTX 5090).
3. No `GPU search:` line names a foreign card.
4. Advisory evidence: a background sampler polls utilization of both cards
   (`rocm-smi --showuse`, `nvidia-smi --query-gpu=utilization.gpu`) during
   each rep; maxima are recorded per rep. Sub-second legs can complete
   between samples, so utilization is corroborating evidence — gates 1–3
   are authoritative.

A config×leg where all reps fail rc!=0 with `GATE FAIL` in stderr is marked
`EXPECTED_GATE_REFUSAL` (a documented capacity limit, not a defect). Anything
else that is not OK is a regression.

> **Erratum history for `amd_gpu @ 2097152`** (three eras; only the last is
> current):
> 1. *Pre-task-14*: this cell refused with a capacity GATE FAIL (canonical
>    map exceeded the card's default-budget backing).
> 2. *Task 14 → pre-wheel*: the aggregate capacity gate auto-enabled the host
>    overflow tier, so the cell completed by default (rc=0, byte-identical)
>    but with GPU search falling back to CPU via the documented capacity
>    notice — a completion-only cell.
> 3. *Wheel-30 landing (current truth)*: the compressed internal map
>    (9162596899 B = 8.53 GiB at 2M) fits the card's participation budget,
>    so this cell runs **true GPU search**: card named in stderr, zero
>    fallback notices, residency handoff 0 B H2D, median wall **12.483 s**
>    (bar ≤38 s). `EXPECTED_GATE_REFUSAL` remains in the harness as a generic
>    outcome label only; no current sweep cell is expected to hit it.

## Known limitations

- Config-major execution order (all legs × reps of one config before the
  next) keeps each block homogeneous but means thermal/clock state differs
  slightly across configs. Warmup + median mitigate this; interleave if you
  need publication-grade rigor.
- Wall clock includes process startup and device init (~0.25 s), which
  dominates the smallest leg; treat 65K numbers as end-to-end latency, not
  kernel throughput.
- Utilization sampling cadence ≈1 s (tool spawn cost), by design non-invasive.

## Reproducing

```bash
cmake --build --preset dev            # fresh build first
./scripts/bench_per_device.sh         # full sweep (~10–15 min under the activated protocol)
BENCH_LEGS="65536" BENCH_REPS=2 ./scripts/bench_per_device.sh   # quick check
```

Legacy single-warmup / median-of-3 / no-autoextend behavior per knob:
`BENCH_JIT_WARMUP=0 BENCH_MEDIAN_N=0 BENCH_AUTOEXTEND=0`.
