# The Performance-Integration Niche — `tribble-clustering`

**Framing (your steer):** the contribution is *bringing the performance approaches
together* into one exact VAT/iVAT pipeline — not the individual primitives, each of
which has prior art. This document pins down (a) what the integrated system is, (b) the
four *separate* performance lines in the literature it unifies, (c) the genuinely
**unoccupied regime** it owns, and (d) the honesty flags and benchmarks needed to
defend it.
**Date:** 2026-07-10. Companions: `docs/bibliography.md`, `docs/novelty-review.md`.

---

## TL;DR — the one-sentence niche

> **Every existing fast-VAT line optimizes *one* axis in isolation — parallelism
> (GPU eVAT), memory (kd-tree matrix-free VAT), sampling (clusiVAT), or JIT (Fast-VAT,
> VAT-only) — and the memory/tree lines only work on *Euclidean coordinate* data. No
> one has built a single *exact* engine that is simultaneously parallel, SIMD-
> vectorized, memory-lean (in-place, symmetry-exploiting), and precision-fused, and
> that runs on an *arbitrary precomputed dissimilarity matrix* (non-metric,
> non-Euclidean, high-dimensional) for both VAT *and* iVAT. That integrated
> exact-arbitrary-dissimilarity engine is your niche.**

---

## 1. What the integrated system actually is (the artifact you're claiming)

A single dense VAT+iVAT pipeline that composes, in one code path:

1. **Compact active-set dense Prim MST** (`pcvat._prim_mst_kernel_*`): O(n) workspace,
   swap-with-last removal, fused relax + next-min scan → **O(n²) time**, branch-light,
   contiguous memory. (`pvat`/`pqvat` provide heap-based and Numba fallbacks.)
2. **In-place, symmetry-exploiting, bit-masked cycle-following permutation**
   (`pvat.shuffle_ordered_column` + `_set_bit`/`_get_bit`): reorders the dissimilarity
   matrix *in place* by walking permutation cycles, tracking visited cells in a packed
   bitmask → **one n×n buffer instead of two** (the commit-#10 "saving a ton of
   memory").
3. **In-place lower-triangular iVAT recursion** (`pcvat._compute_ivat_kernel_*`):
   `D'[r,c]=max(D*[r,j],D'[j,c])` written to the lower triangle only, then a single
   parallel back-copy mirrors it → **~half the write traffic and no second iVAT
   buffer**.
4. **Fused float32/float64 kernels** (one Cython source, both precisions) → optional
   **2× memory / bandwidth** via float32 with a float64 accumulator for conditioning.
5. **OpenMP parallel + auto-vectorized** regions with `_PAR_THRESHOLD` gating and
   `guided` scheduling for the triangular load profile; exclusive-row write ownership
   for race-freedom.
6. Coupled to **IVATMeans** clustering (see `docs/novel-niche.md`).

The claim is the **co-design**: memory-frugality (in-place + symmetry + fused
precision) and speed (compact O(n²) Prim + SIMD + threads) *at the same time*, exactly,
on any dissimilarity matrix.

---

## 2. The four separate performance lines you unify (prior art)

| Line | Representative work | Optimizes | Key limitation you exploit |
|---|---|---|---|
| **JIT / CPU reimplementation** | **Fast-VAT** — Avinash & Lachheb 2025 (Cython+Numba, up to 50×) | single-thread JIT | **VAT only (no iVAT)**; no parallel/SIMD; no in-place memory work; no clustering |
| **Parallel / GPU** | **eVAT** — Meng & Yuan 2018 (edge-based, CUDA GPU) | parallelism | needs a **GPU**; not memory-focused; not arbitrary-dissimilarity-friendly |
| **Memory / matrix-free** | **kdT-VAT / TkdT-VAT / BB-VAT** — *Information Sciences* 2024 | memory (avoid full n×n via k-d tree EMST) | **requires Euclidean coordinates** for the tree; **degrades in high dimensions**; cannot take a precomputed non-metric dissimilarity |
| **Sampling / approximation** | **clusiVAT / sVAT / bigVAT** — Kumar et al. 2016; Hathaway et al. 2006 | scale via maximin sampling | **approximate** (samples, then extends); accuracy loss vs. exact |
| **Complexity floor** | **ef-iVAT** — Havens & Bezdek 2012 | O(n³)→O(n²) iVAT | algorithmic baseline everyone builds on (you implement it exactly) |

**The point:** these are *different papers optimizing different axes*, and the two that
attack memory/scale without sampling (kd-tree, 2024) **assume Euclidean coordinates**.
Your engine sits where all four axes meet **and** where coordinates are unavailable.

---

## 3. The unoccupied regime (why the integration is a real gap, not just engineering)

There is a concrete problem setting none of the above serves well:

> **Exact VAT/iVAT on a large, arbitrary, precomputed dissimilarity matrix**
> (non-Euclidean, non-metric, or where only pairwise dissimilarities exist — e.g.
> edit distance, DTW, kernel/graph dissimilarities, domain metrics), where you cannot
> afford two n×n buffers and want CPU-only parallel speed without a GPU.

- **kd-tree / matrix-free methods can't enter this regime** — no coordinates ⇒ no tree
  ⇒ they must fall back to the full matrix, at which point *memory* is the wall your
  in-place symmetric ops address.
- **clusiVAT enters it only approximately** (sampling), and its single-linkage
  extension is sensitive to bridges/noise.
- **Fast-VAT / eVAT don't address memory or iVAT** (and eVAT needs a GPU).

So your defensible, novel claim is not "we made VAT fast" (Fast-VAT did) but:
**"the first *exact* VAT+iVAT engine that is simultaneously parallel, vectorized, and
half-memory via in-place symmetric operations, and therefore the fastest exact option
in the arbitrary-dissimilarity / no-coordinates / CPU-only regime where the sampling
and tree-based accelerators do not apply."**

---

## 4. Honesty flags (fix these before a systems reviewer finds them)

1. **Priority-queue Prim is *not* a speedup for dense VAT graphs — it is asymptotically
   *worse*.** The VAT graph is complete (dense), so heap Prim is O(E log V)=O(n² log n)
   vs. the O(n²) dense-array/compact-active-set Prim. The README's *"priority-queue MST
   speedups (NAFIPS 2025/2026)"* framing is exposed here. **Recommendation:** make the
   MST headline the **compact active-set dense kernel** (`pcvat`), and if you keep the
   heap variant, justify it empirically (e.g. early-exit / cache behavior on real data)
   rather than asymptotically. Benchmark `pqvat`/`pvat` (heap) vs. `pcvat` (compact) to
   show which actually wins and when — that comparison is itself a publishable result.
2. **Fast-VAT (2025) is a direct concurrent competitor** on the "Cython+Numba VAT
   acceleration" headline. You **must** cite it and state the delta: iVAT (not just
   VAT), OpenMP+SIMD (not single-thread), in-place/fused-precision memory reduction,
   arbitrary dissimilarities, and the clustering front-end. Benchmark head-to-head.
3. **The memory story must be stated precisely.** In-place permutation takes peak memory
   from **~2·n²·b to ~1·n²·b** (b = bytes/element) — a **constant-factor (≈2×) win**,
   not an asymptotic one. The kd-tree 2024 line achieves **sub-quadratic** memory but
   only for Euclidean data. Own the distinction: you win the *constant* in the regime
   where they can't reduce the order at all.
4. **The individual techniques are classical.** Compact dense Prim, in-situ
   cycle-following permutation (Cate & Twigg, *Algorithm 513*, 1977; Catanzaro et al.
   2014 for the parallel/decomposition view), symmetry exploitation, and SIMD/OpenMP
   are all known. Claim the **composition + the regime + the measured envelope**, not
   the parts.
5. **`float32` accuracy.** You already accumulate the squared sum in `double` before
   storing `float32` (good). Quantify the effect on the MST/ordering and on downstream
   cluster labels vs. `float64` — reviewers will ask whether the 2× memory win perturbs
   the result.

---

## 5. The experiments that prove the integration claim

**Axes:** peak memory (RSS) and wall-clock vs. `n`, on both Euclidean and
**non-Euclidean / precomputed** dissimilarities, for VAT *and* iVAT.

**Contenders:**
- naive/reference VAT (SciPy/textbook) — the baseline everyone speeds up;
- **Fast-VAT** (Cython+Numba, VAT) — the concurrent CPU-JIT competitor;
- **eVAT/GPU** (Meng & Yuan) — if a GPU is available, the parallel competitor;
- **kdT-VAT/TkdT-VAT** (IS 2024) — Euclidean-only; **show it cannot run / must fall
  back on a non-metric dissimilarity** (this is your regime argument, made concrete);
- **clusiVAT** — sampled; report its **accuracy loss** vs. your exact output;
- your engine in all configs: `pcvat` C/OpenMP vs. Numba `pvat`; float32 vs. float64;
  **in-place vs. out-of-place** (to isolate the 2× memory win and any speed cost).

**Headline plots:**
- *Exactness–memory frontier*: exact methods only, peak memory vs. `n` — your in-place
  float32 line should dominate on the full-matrix regime.
- *Speed vs. `n`* with thread scaling (1→N cores) and the `_PAR_THRESHOLD` crossover.
- *Regime table*: capability matrix (VAT? iVAT? non-Euclidean? memory-lean? no-GPU?
  exact?) with a ✓/✗ per method — your column is the only all-✓.
- *Accuracy*: your exact output vs. clusiVAT-sampled (ARI/NMI of resulting clusters,
  and RDI image difference) to quantify what sampling costs.

That capability matrix + the exactness–memory frontier *is* the contribution figure.

---

## 6. Drop-in positioning statement

> "Prior accelerations of VAT/iVAT each target a single axis: GPU parallelism (eVAT),
> sub-quadratic memory via k-d-tree EMST (kdT-VAT/TkdT-VAT), sampling (clusiVAT), or
> JIT reimplementation of VAT alone (Fast-VAT). The memory- and tree-based methods
> require Euclidean coordinates and degrade in high dimensions, while sampling
> sacrifices exactness. We present an integrated, *exact* VAT **and** iVAT engine that
> is simultaneously multi-threaded, SIMD-vectorized, and memory-frugal — reordering and
> transforming the dissimilarity matrix **in place** via symmetry-exploiting,
> bit-masked cycle-following permutation and lower-triangular iVAT recursion, over
> fused float32/float64 kernels. Because it operates on an arbitrary precomputed
> dissimilarity matrix, it is the fastest exact option precisely in the
> no-coordinates / non-metric / CPU-only regime where tree- and sampling-based
> accelerators do not apply."

Claim: **(1)** the co-designed exact engine (parallel + SIMD + in-place + fused
precision, VAT *and* iVAT); **(2)** ownership of the exact-arbitrary-dissimilarity
regime; **(3)** the measured time/memory/accuracy envelope against Fast-VAT, eVAT,
kd-tree VAT, and clusiVAT. Do **not** claim the primitives, and **retire or empirically
justify** the "priority-queue speedup" framing (§4.1).

---

## 7. New references (added to `docs/bibliography.md`, §5 Performance/Systems)
- **Fast-VAT** — Avinash & Lachheb (2025), arXiv:2507.15904 — Cython+Numba VAT, ~50×.
  PDF: `docs/sources/Avinash_Lachheb_2025_FastVAT_Cython_Numba.pdf`. **Concurrent competitor.**
- **eVAT / GPU** — Meng & Yuan (2018), *Int. J. Data Science and Analytics* — edge-based
  parallel VAT on CUDA.
- **Time/memory-scalable VAT** — *Information Sciences* 2024, art. S0020025524002378 —
  kdT-VAT / TkdT-VAT / BB-VAT, sub-quadratic memory via k-d-tree EMST (Euclidean).
  **Authors to verify** (Elsevier page paywalled; likely Rathore/Kumar et al.).
- **In-place permutation / transposition** — Cate & Twigg, *ACM Algorithm 513* (1977);
  Catanzaro et al., "A Decomposition for In-place Matrix Transposition," *PPoPP* 2014 —
  classical basis for the cycle-following permutation.
- **Prim (dense)** — already in bibliography [Prim]; note O(n²) dense-array optimality.
- **Havens & Bezdek 2012** [iVAT-fast] — the O(n²) iVAT floor you implement exactly.

---

## 8. Next steps
1. **Run the §5 capability matrix + exactness–memory frontier** — that single figure
   justifies the framing.
2. **Resolve the MST claim (§4.1):** benchmark compact-`pcvat` vs. heap-`pqvat`/`pvat`
   on dense data; reframe or empirically defend.
3. **Head-to-head vs. Fast-VAT** on identical datasets (VAT), then extend to iVAT where
   Fast-VAT can't follow.
4. **Non-Euclidean case study** (e.g. DTW or edit-distance dissimilarities) where
   kd-tree methods can't build a tree — the concrete proof of your regime.
5. Verify the IS-2024 authors and the eVAT author initials before submission.
