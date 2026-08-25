# F1 — Replay-order induction proof: compositeMask bit-walk ≡ compositePower2[64] table walk

Date: 2026-08-25 · Tree HEAD at start: `9f51a68` (`feat(search): overlap canonical expansion behind ensureCanonical`)
Product change: `source/m4/gpu_search_kernel.h` ONLY — per-lane `uint32_t compositePower2[MAX_COMP_MAX_POWER2]` (256 B/lane scratch) replaced by a single `uint32_t compositeMask` (4 B/lane).

**Claim.** For every odd `sum` in the supported domain, the Phase-3 replay loop after the F1 diet invokes
`dev_AllButOneProductOfTermPairsOfSumOfFactorPairsHasSingleFactorPair` on exactly the same `power2`
arguments **in exactly the same order** as before the diet. Since every downstream effect of Phase 3
(`numValid`, `termsFound`, `termA`, `termB`) is a pure function of that call sequence plus state that
the two versions hold identically, all emitted records are bit-identical, hence stdout is byte-exact.

---

## 0. The two implementations (quoted from the tree)

**Before** (HEAD `9f51a68`, gpu_search_kernel.h):

```c
// Phase 1 write site:
if (numComposite < MAX_COMP_MAX_POWER2)
    compositePower2[numComposite++] = (uint32_t)power2;

// Phase 3 read site:
int32_t nc = (int32_t)numComposite - 1;
while (nc >= 0) {
    int32_t idx = nc;
    --nc;
    uint64_t evenTerm = ((uint64_t)1 << compositePower2[idx]);
    uint64_t oddTerm = sum - evenTerm;
    if ((numValid += dev_AllButOne...(compositePower2[idx], oddTerm, sum, ...)) > 1)
        break;
    if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
}
```

**After** (this change):

```c
// Phase 1 write site:
compositeMask |= (1u << power2);

// Phase 3 read site:
uint32_t pending = compositeMask;
while (pending) {
    const uint32_t idx = 31u - (uint32_t)__clz(pending);   // highest set bit
    pending &= ~(1u << idx);
    uint64_t evenTerm = ((uint64_t)1 << idx);
    uint64_t oddTerm = sum - evenTerm;
    if ((numValid += dev_AllButOne...(idx, oddTerm, sum, ...)) > 1)
        break;
    if (!termsFound && numValid == 1) { termA = evenTerm; termB = oddTerm; termsFound = true; }
}
```

Phase 1's loop (both versions):

```c
for (uint64_t evenTerm = 4, power2 = 2; evenTerm < sum - 2; evenTerm <<= 1, ++power2) {
    uint64_t oddTerm = sum - evenTerm;
    if (dev_IsPrime(oddTerm, ...)) { ... }        // prime branch: no recording
    else               { /* record power2 */ }     // composite branch: recording
}
```

## 1. Lemma 1 (one-write-per-power, strictly ascending)

*Phase 1 visits each value of `power2` at most once, and the values at which it records are strictly
increasing.*

**Proof.** The loop header initializes `(evenTerm, power2) = (4, 2)` and each iteration executes
exactly `evenTerm <<= 1; ++power2`. So `power2` takes the values 2, 3, 4, … monotonically — no value
repeats. The composite branch executes at most once per iteration (it is one arm of an if/else).
Hence the table receives at most one write per distinct `power2`, in increasing `power2` order, and
`compositeMask` receives at most one OR per distinct `power2` (OR is idempotent, so even a repeat
would be harmless). ∎

**Corollary 1.1 (set representation).** After Phase 1, for every p ≥ 2:
bit `p` of `compositeMask` is set ⇔ `p` occurs somewhere in `compositePower2[0 .. numComposite-1]`.
The table and the mask represent the *same set* S(sum) = { p ≥ 2 : sum − 2^p is composite, 2^p < sum−2 }.

**Corollary 1.2 (strict ascent).** `compositePower2` is strictly increasing:
`compositePower2[0] < compositePower2[1] < … < compositePower2[numComposite−1]`.

## 2. Lemma 2 (domain bound: bits ≥ 32 never matter)

*On every input where GPU search can produce correct output, all recorded powers satisfy p ≤ 31, so
the uint32 mask loses nothing.*

**Proof.** A recorded p satisfies 2^p < sum − 2, i.e., sum > 2^p + 2. For p ≥ 32 this needs
sum > 2^32 + 2 ≈ 4.29×10⁹. But `GpuRecord.sum` is `uint32_t` (gpu_search_kernel.h, GpuRecord decl)
and the kernel writes `rec.sum = (uint32_t)sum`; emission prints `rec.sum` verbatim
(gpu_search_emission.cpp `FormatGpuSearchResult`). Any sum ≥ 2³² therefore already printed a
truncated, wrong value **before** this change — such inputs are outside the domain on which the
GPU-search path was ever byte-correct. On the supported domain (sum < 2³²) we have p ≤ 31, inside
uint32. The production contract is far tighter still: goldens cover legs ≤ 2097152, giving p ≤ 20.
∎

**Remark (retired cap was dead headroom).** The old `numComposite < MAX_COMP_MAX_POWER2 (=64)` guard
never fired on any terminating input: the loop terminates only when `evenTerm ≥ sum − 2` with
`evenTerm = 2^power2` fitting uint64 (for sum − 2 > 2^63 the shift `evenTerm <<= 1` wraps to 0 and
the *old* loop spins forever — a pre-existing property of the domain, unchanged here). Terminating
inputs therefore have power2 ≤ 62, i.e., at most 61 recorded entries < 64. Removing the guard removes
only dead headroom; Corollary 1.1 holds unconditionally on the real domain.

## 3. Theorem (replay-order equivalence)

Let the old walk be W_old and the new walk W_new (code in §0). Then for every sum in the supported
domain, W_old and W_new visit the same sequence of powers p₁ > p₂ > … > p_k (the elements of S(sum)
in descending order), and either both complete the sequence or both break after the same element.

**Loop invariant.** At the start of each iteration of both loops:

> (I1) The multiset of already-visited powers is identical in W_old and W_new, and equals the k
> largest elements of S(sum) for some k ≥ 0.
> (I2) W_old's remaining elements are exactly `compositePower2[0 .. nc]` (indices ≤ nc unvisited);
> W_new's remaining elements are exactly the set bits of `pending`.
> (I3) By Corollaries 1.1/1.2, both remainders equal S(sum) minus its k largest elements — the same
> finite non-empty set in both walkers (loop guards `nc >= 0` / `pending != 0` agree precisely
> because both remainders are empty or non-empty together).

**Base case (k = 0).** Before the first iteration: nothing visited; `nc = numComposite − 1` exposes
the whole table, `pending = compositeMask` the whole mask; by Corollary 1.1 both remainders equal
S(sum). (I1)–(I3) hold.

**Inductive step.** Assume (I1)–(I3) with both remainders non-empty. By (I3)/Corollary 1.2 the
remainder has a unique maximum p*. 
— W_old reads index `idx = nc`; since the table is strictly ascending (Corollary 1.2) and indices
above nc are exhausted, `compositePower2[nc]` is the maximum of W_old's remainder, i.e. p*.
It then decrements `--nc`, shrinking the remainder to "all elements below p*". 
— W_new computes `idx = 31 − __clz(pending)`, the position of the highest set bit, i.e. the maximum
of W_new's remainder, which by (I3) is the same p*. It clears that bit
(`pending &= ~(1u << idx)` — valid for every idx < 32, including 31), shrinking the remainder to
"all elements below p*". 
Both then execute the identical call
`dev_AllButOne...(p*, sum − 2^{p*}, sum, primeMap, maxPrimeMapValue, smallPrimes, smallPrimeCount,
primeIndex)` on identical device state (all other per-sum locals are version-independent and
untouched between iterations), so `numValid`'s increment and the `> 1` break decision are identical.
If neither breaks, the `!termsFound && numValid == 1` term-capture branch sees identical inputs and
executes identically. Thus the (k+1)-st visited element is p* in both walkers and (I1)–(I3) hold
with k+1.

**Termination/exit parity.** Each iteration consumes exactly one element, remainders stay equal and
finite, so both loops perform exactly min(|S|, first-break-position) iterations and stop together:
W_new's guard `pending != 0` fails exactly when the shared remainder is empty, matching W_old's
`nc >= 0`; the `break` fires on the same call in both. ∎

**Corollary (byte-exact stdout).** Phases 2/4 and the emission gate do not read the retired state;
`numValid`, `termsFound`, `termA`, `termB` evolve identically (Theorem); record writes go to the
same sum-indexed slot with the same payload; therefore `GpuSearchEmit` prints byte-identical lines.
The hard gate for this corollary is `ctest --preset dev -R m4_order` (emission vs `ff_seg` goldens,
byte-exact) plus `verify.sh --all-legs --gpu-search` on both devices.

## 4. Why the m4_kernel_unit host mirror needs no edit

`test/source/m4_kernel_unit.cpp` keeps its own host-side `uint32_t compositePower2[64]` mirror of
Phases 1/3. That mirror encodes the *semantics* (which powers replay, in which order), not the
storage layout; by the Theorem the kernel's semantics are unchanged, so kernel-vs-mirror agreement
remains the same tautology-strength check as before. Changing the mirror would only re-couple the
test to the new storage, weakening it.

## 5. Review against the actual kernel (not prose-only)

Checked line-by-line against the post-edit file (see `git show` of the F1 commit):
- Phase-1 loop header unchanged — monotone `(evenTerm, power2)` advance intact (Lemma 1 premise).
- Composite branch is the only writer; prime branch writes nothing (§0 excerpt matches file).
- New walk consumes the bit it visits (`pending &= ~(1u << idx)` AFTER computing idx from the SAME
  `pending` value) — no off-by-one between the bit tested and the bit cleared.
- `__clz` guarded by `while (pending)` — never called with 0 (its undefined-input case).
- `1u << idx` with idx ∈ [0,31]: in-range by construction of the mask; shift is well-defined.
- Break/capture branches textually identical to the old ones except for the index source.
- No other reader of `compositePower2`/`numComposite` exists in the file (grep-clean); the macro
  `MAX_COMP_MAX_POWER2` had no references outside this file and was removed with its table.
