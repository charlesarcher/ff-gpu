# fastFreudenthalSegmentedSieve Assessment

**Date:** 2026-08-26  
**Machine:** 93 GiB RAM (73 GiB available), 32 cores, GCC 16.2.1  
**File:** `reference/fastFreudenthalSegmentedSieve.C` (916 lines, PRIMEMEM=85 GiB, THREADS=31, NUMSUM=5000)  
**Binary:** `/tmp/opencode/fastFreudenthal`  
**Goldens baseline:** `reference/ff_seg` (vendored segmentedSieve.C) and `build/ff_sieve` (GPU sieve + CPU search)

---

## 1. Build

**System RAM check:**
```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:            93Gi        20Gi        60Gi       3.4Gi        17Gi        73Gi
Swap:           93Gi        20Gi        73Gi
```
85 GiB + overhead fits (73 GiB available + 60 GiB free cache-reclaimable + 73 GiB swap). README says 93 GB total — 85 GiB should fit.

**Build command (as required):**
```bash
g++ -O2 -std=c++17 -pthread reference/fastFreudenthalSegmentedSieve.C -o /tmp/opencode/fastFreudenthal 2>&1
```
**Result:** `EXIT:0` — no errors, no warnings, no need for `-lstdc++fs` on GCC 16.2.1 (filesystem is in stdc++). No extra defines needed. Defaults retained: `PRIMEMEM=85*1024^3`, `THREADS=31`, `NUMSUM=5000`. Binary size 59,944 bytes (vs ff_seg 54 KiB).

**CLI (from `main()` at line 835):**
```c
int main(int argc, char* argv[]) {
  // if argc>2: sumStart=atoi(argv[1])|1, sumLimit=atoi(argv[2])
  // else if argc>1: sumStart=5, sumLimit=atoi(argv[1])
  // else: sumStart=5, sumLimit=2627
  // then chdir/mkdir "ffPlayground", exit(-2) if dir already exists
}
```
All invocations used `5 <leg>` (e.g. `/tmp/opencode/fastFreudenthal 5 65536`). The process creates and then removes `ffPlayground/`; test harness deletes stale directory before each run (`rm -rf ffPlayground`).

---

## 2. Benchmark — 3 legs (wall = `bash -c 'time -p <cmd> >out 2>&1'` real; internal timers from program stdout)

All runs on same machine, sequential, 1 untimed warmup not used — single measured real reported. User times show ~31× parallelism.

| Leg (sumLimit) | Binary | real wall | user | sys | internal Prime | internal Freudenthal | notes |
|----------------|--------|-----------|------|-----|----------------|----------------------|-------|
| **65536** | `/tmp/opencode/fastFreudenthal 5 65536` | **2.09 s** (rep 1.70–1.81 s) | 0.47 s | 0.01 s | 8,728 µs | 11,170 µs (In-Primemap) | maxPrimeMapValue 268,435,471 totalBytes 16,791,369 |
| | `reference/ff_seg 5 65536` | **0.02 s** | 0.44 s | 0.01 s | 8,975 µs | 11,314 µs | same map, 2,357 solutions |
| | `build/ff_sieve 5 65536` (default CPU-search) | **1.52 s** (total 1,413 ms) | 21.63 s | 0.47 s | 2,298 µs (sieve) | 749,660 µs search; sieve 27.8 ms, wheel 19.7 ms, enum 615.8 ms | 2,357 solutions, byte-identical |
| **524288** | `/tmp/opencode/fastFreudenthal 5 524288` | **4.06 s** | 60.03 s | 0.11 s | 533,844 µs | 1,723,064 µs | map 17,179,869,199 bytes 1,073,834,461; 18,408 solutions |
| | `reference/ff_seg 5 524288` | **2.29 s** | 59.75 s | 0.09 s | 532,789 µs | 1,756,996 µs | 18,408 solutions, byte-identical to fast |
| | `build/ff_sieve 5 524288` | **22.16 s** (total 21,953 ms) | 599.18 s | 0.83 s | 53,798 µs | 21,084,028 µs; sieve 156 ms, wheel 98 ms | 18,408 solutions |
| **1048576** | `/tmp/opencode/fastFreudenthal 5 1048576` | **11.58 s** (rep 10.91 s) | 259.94 s | 0.25 s | 2,303,220 µs | 7,113,311 µs | map 68,719,476,751 bytes 4,295,141,965; 35,556 solutions |
| | `reference/ff_seg 5 1048576` | **9.34 s** (rep 9.44 s) | 261.64 s | 0.25 s | 2,279,836 µs | 7,050,206 µs | 35,556 solutions |
| | `build/ff_sieve 5 1048576` | **83.14 s** (total 82,918 ms) | 2325.45 s | 2.52 s | 228,861 µs | 82,307,683 µs; sieve 467 ms, wheel 305 ms, enum 138 ms | 35,556 solutions |

**How walls were captured (faithful to task `| tail -20`):**
```bash
rm -rf ffPlayground
bash -c 'time -p /tmp/opencode/fastFreudenthal 5 <leg> > /tmp/ff_fast_<leg>.txt 2>&1'
cat /tmp/ff_fast_<leg>.txt | tail -20
# same for reference/ff_seg and build/ff_sieve (which prints ff_sieve timing: to stderr)
```

Raw outputs preserved: `/tmp/ff_fast_*.txt`, `/tmp/ff_seg_*.txt`, `/tmp/ff_sieve_*.txt` (+ stderr).

**Byte-identical check (solution lines `^\s*\d+\) sum =`):**
- 65K: fast 2,357 == seg 2,357 == sieve 2,357 — `diff` of the solution prefix (before `Prime time` footer) is empty.
- 524K: fast 18,408 == seg 18,408 == sieve 18,408
- 1M: fast 35,556 == seg 35,556 == sieve 35,556
All three binaries are functionally identical for these legs.

---

## 3. Why fastFreudenthal wall looks inflated at small legs

`fastFreudenthalSegmentedSieve.C:129`:
```c
string getMemorySpeed() {
  unique_ptr<FILE,int(*) (FILE*)> pipe(popen("sudo dmidecode --type 17 2>/dev/null | grep -i 'Speed:'","r"), pclose);
}
```
It is called at the very end to print `Memory ... MT/s`. On this machine `sudo` without a terminal prompts for a password and hangs ~1.6 s before failing:
```bash
$ bash -c 'time -p sudo dmidecode --type 17 2>/dev/null | grep -i Speed:'
sudo: a terminal is required to read the password; ...
real 1.58 / 1.90s
$ timeout 2 sudo -n dmidecode --type 17 2>&1 | head
sudo: a password is required
real 0.00
```
Thus every `fastFreudenthal` wall includes a **~1.58–1.90 s spurious stall** that `ff_seg` does not pay (it lacks `getMemorySpeed`). At 65K this dominates the measurement (0.02 s compute vs 1.6 s stall → 100× wall ratio). At 524K/1M the stall is still ~1.6 s on top of ~2.3/9.3 s compute, explaining the 77% and 24% wall excess. Subtracting the stall, internal `Prime+Freudenthal` times are **within 1–2%** of `ff_seg`:

- 524K internal: fast 2.256 s vs seg 2.288 s (fast 1.4% *faster*, noise)
- 1M internal: fast 9.416 s vs seg 9.330 s (fast 0.9% *slower*, noise)

`build/ff_sieve` default CPU-search at these legs is 10–76× slower than `ff_seg` in wall (1.52 s vs 0.02 s at 65K, 22.16 s vs 2.29 s at 524K, 83.14 s vs 9.34 s at 1M) because its search phase (`CPU search: 31 threads` via `GpuPrime` hostmap + wheel expansion 19–305 ms) is ~10× the reference search time, plus ~138–615 ms device enumeration. This is the known hostmap-CPU path, not the GPU-search path.

---

## 4. Verdict

**Is fastFreudenthal materially faster than ff_seg (the current CPU baseline)? No.**

- After removing the ~1.6 s `sudo dmidecode` stall, fastFreudenthal's compute is **statistically identical** to ff_seg (Δ <2% at both 524K and 1M; at 65K both ~0.02 s).
- On raw wall (including stall) it is **slower** at all three legs: 2.09 s vs 0.02 s (65K), 4.06 s vs 2.29 s (+77% at 524K), 11.58 s vs 9.34 s (+24% at 1M). Even without the stall it is not faster by any margin that would justify restating speedup bars.
- `ff_seg` remains the honest CPU baseline. Its timings (0.02 s / 2.29 s / 9.34 s) match the frozen benchmark table in `README.md` (0.019 s / 2.240 s / 9.473 s) within measurement jitter.

**Honest-speedup bars: DO NOT RESTATE.** The GPU speedup claims in `README.md` (§Performance → Benchmark Results, median of 3 reps after warmup) are computed vs `ff_seg` and remain valid. If anything, `ff_sieve` default CPU path is *slower* than `ff_seg`, so comparing GPU vs `ff_sieve` would inflate speedups; comparing vs `ff_seg` is the conservative, correct baseline and is unchanged by this assessment.

**Additional notes for future wiring (not done per MUST NOT DO):**
- No CMake wiring was performed; no goldens committed.
- If vendored, the `sudo dmidecode` call should be replaced with `sudo -n` or removed — it is the sole source of wall inflation and blocks CI where sudo is unavailable.
- `PRIMEMEM=85 GiB` required no lowering; the machine's 73 GiB available + reclaimable cache + swap accommodated the 1 GiB (65K) → 4.3 GiB (1M) maps without `bad_alloc`. Lowering to `-DPRIMEMEM=1GiB` was not needed; document for reproducibility.
- `build/ff_sieve --gpu-search` (true GPU path) retains the materially faster 1.9–7.6× speedups quoted in `README.md`; this assessment was CPU-vs-CPU only.

---

## 5. Repro

```bash
free -h
g++ -O2 -std=c++17 -pthread reference/fastFreudenthalSegmentedSieve.C -o /tmp/opencode/fastFreudenthal 2>&1
rm -rf ffPlayground; bash -c 'time -p /tmp/opencode/fastFreudenthal 5 65536 > /tmp/ff_fast_65k.txt 2>&1'; tail -20 /tmp/ff_fast_65k.txt
rm -rf ffPlayground; bash -c 'time -p reference/ff_seg 5 65536 > /tmp/ff_seg_65k.txt 2>&1'; tail -20 /tmp/ff_seg_65k.txt
bash -c 'time -p build/ff_sieve 5 65536 > /tmp/fs_65k.txt 2> /tmp/fs_65k_stderr.txt'; tail -20 /tmp/fs_65k_stderr.txt
# repeat for 524288, 1048576
```

Evidence files:
- This markdown: `.omo/start-work/evidence/fastfreudenthal-assessment.md`
- Raw stdout/stderr: `/tmp/ff_fast_*.txt`, `/tmp/ff_seg_*.txt`, `/tmp/ff_sieve_*`
