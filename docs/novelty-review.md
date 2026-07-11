# Novelty & Prior-Art Review — `tribble-clustering`

**Reviewer role:** independent expert review for PhD work.
**Scope:** the VAT/iVAT implementation (`pvat.py`, `pqvat.py`, `pcvat.pyx`), the
`IVATMeans` centroid-finding / clustering system (`ivatmeans.py`,
`pvat.get_ivat_levels`, `get_ivat_hierarchy`), and Fuzzy C-Means (`fcm.py`,
`cfcm.pyx`, `fuzzycmeans.py`).
**Date:** 2026-07-10. Companion file: `docs/bibliography.md`.

> **Bottom line.** FCM and VAT/iVAT are prior art and are implemented faithfully;
> your instinct is correct. The genuinely defensible contributions are (1) the
> **systems/performance** work (priority-queue & compact Prim MST, in-place
> bit-masked permutation, fused-precision C/OpenMP kernels), and (2) **`IVATMeans`
> as a deterministic, structure-aware seeding + automatic-`k` front-end for FCM**.
> The clustering *idea* underlying `IVATMeans`, however, is much closer to existing
> VAT-family work — especially **clusiVAT** (Kumar et al. 2016) — than it may
> appear, because the step that reads clusters off the iVAT off-diagonal is
> provably a form of **single-linkage / MST-cut clustering**. Novelty is real but
> **incremental**; it must be positioned explicitly against clusiVAT, aVAT/SpecVAT,
> and FCM++.

---

## 1. What the code actually does

### 1.1 VAT / iVAT (`pvat.py`, `pcvat.pyx`)
- `vat_prim_mst` runs **Prim's MST** seeded at the globally most-distant pair and
  returns the visiting order `p` (the VAT permutation) and the parent sequence `q`.
  This is precisely Bezdek & Hathaway's VAT reordering ([VAT], 2002).
- `compute_ivat` applies the recurrence
  `D'[r,c] = max(D*[r,j], D'[j,c])`, `j = argmin_{k<r} D*[r,k]` —
  the **exact O(n²) iVAT recursion of Havens & Bezdek** ([iVAT-fast], 2012), whose
  concept is Wang et al.'s path-based (minimax) transform ([iVAT], 2010).

**Assessment:** faithful, correct re-implementations of published algorithms. No
algorithmic novelty in VAT/iVAT themselves — nor is any claimed.

### 1.2 `IVATMeans` centroid finding (`get_ivat_levels`)
1. Take the **k=1 off-diagonal** of the iVAT matrix, `d = diag(D', 1)`.
2. **Sort** `d`; compute successive **differences**; the largest difference(s) set a
   **threshold** `peaks_threshold` (auto-`k` when `n_clusters = -1`, or the top
   `n_clusters-1` values when `k` is fixed).
3. Points whose diagonal value ≥ threshold are **cut points**; the VAT order is
   split into **contiguous segments**.
4. Each segment's **mean** becomes an **initial centroid**.
5. `IVATMeans.fit` then does a **nearest-centroid hard assignment**;
   `FuzzyCMeans`/`fuzzy_c_means` can consume these centroids as `initial_guess` for
   iterative FCM refinement.

`get_ivat_hierarchy` repeats the split at the top-`n_levels` gaps to build a
multi-resolution tree from a single iVAT computation.

### 1.3 Fuzzy C-Means (`fcm.py`, `cfcm.pyx`)
Standard Bezdek FCM alternating optimization ([Bezdek-FCM], 1981; [Dunn], 1973):
membership update `_get_weights`, center update `_get_v_ij`, ≤100 iterations,
`allclose` stopping. Default init = mean of random point pairs. Faithful; no
algorithmic novelty (none claimed).

---

## 2. The key theoretical observation (state this in the thesis before a reviewer does)

**The VAT ordering is a Prim MST traversal, so the ordered off-diagonal profile is
the sequence of MST edge weights, and cutting it is single-linkage clustering.**

- Gower & Ross ([MST↔SL], 1969) proved single-linkage clustering is fully
  determined by the MST.
- Zahn ([Zahn], 1971) formalized clustering by **cutting inconsistent/long MST
  edges** — exactly your "abrupt change" cut.
- Therefore steps 1–3 of §1.2, *before FCM refinement*, produce (approximately)
  **single-linkage clusters** — the same object clusiVAT ([clusiVAT-journal], 2016)
  explicitly extracts ("a relative of single linkage"). The 1-D "read clusters off
  an ordering-induced profile" pattern is also the OPTICS reachability-plot idea
  ([OPTICS], 1999).

**Two consequences:**
- **Do not claim** the cut step as new clustering theory; frame it as *"surfacing
  single-linkage structure through the iVAT ordering."* That framing is honest and
  still useful.
- **A subtlety that is genuinely yours to get right:** `get_ivat_levels` uses
  `diag(D', 1)` — the value between order-adjacent points `r` and `r-1` — whereas
  the true MST edge for the `r`-th vertex connects it to its Prim parent
  `q[r]` (= the `argmin`), which need not be `r-1`. Your own code flags this
  (`pvat.py` TODO: *"Get from the prim-mst sequence? jj = as_seq[r-1]"*). So the
  diagonal profile is a **proxy** for the merge-height sequence, not identical to
  it. Whether the proxy or the exact parent-edge profile gives better cuts is an
  **open, testable question** — and a legitimate, novel micro-contribution if you
  characterize it.

---

## 3. Prior-art map for the `IVATMeans` claim

| Ingredient in `IVATMeans` | Closest prior art | Verdict |
|---|---|---|
| VAT reorder via Prim MST | Bezdek & Hathaway 2002 [VAT] | Prior art |
| iVAT minimax recursion, O(n²) | Havens & Bezdek 2012 [iVAT-fast]; Wang et al. 2010 [iVAT] | Prior art |
| Auto-`k` from the reordered image/diagonal | aVAT (Wang et al. 2010 [iVAT]); DBE [DBE]; SpecVAT [SpecVAT/partition] | Prior art — **same goal** |
| Cut ordering → partition, then assign remaining points | **clusiVAT** [clusiVAT-journal] (SL cut + nearest-prototype) | **Very close prior art** |
| Cut at largest gap / longest edge | Zahn 1971 [Zahn]; MST-gap methods | Prior art |
| Segment-mean centroids seeding a partitional method | k-means++ [kmeans++]; FCM++ [FCM++] (different seeders) | Related; your seeder differs |
| **iVAT centroids → iterative FCM (soft) refinement** | — (clusiVAT stops at hard nearest-prototype) | **Plausibly novel combination** |
| Sorted-diagonal **max-difference** auto-`k` rule (parameter-free) | gap statistic [gap-statistic]; aVAT peak-counting | **Specific, defensible variant** |
| Single iVAT → multi-level hierarchy (`get_ivat_hierarchy`) | iVAT already encodes the SL dendrogram | Modest novelty (convenience) |

### Where the defensible novelty is
1. **Coupling iVAT-derived centroids to FCM.** clusiVAT does a single hard
   nearest-prototype extension; `IVATMeans` + `FuzzyCMeans` performs **iterative
   fuzzy refinement** from a **deterministic, geometry-aware seed**. This is a
   genuine, if incremental, methodological combination — and it addresses FCM's
   well-known initialization sensitivity with a principled, non-random seed.
2. **The parameter-free max-gap-in-sorted-diagonal auto-`k` rule**, applied to the
   *iVAT* (minimax) diagonal rather than a raw MST or an image. Concrete and
   easy to reproduce; contrast it against aVAT/DBE (image-based) and the gap
   statistic (resampling-based).
3. **Exact, full-data iVAT at scale.** clusiVAT/sVAT/bigVAT get to large `n` by
   *sampling*; your route is to make **exact** iVAT fast (priority-queue Prim,
   compact active-set O(n) memory, in-place bit-masked permutation, C/OpenMP
   float32/64). That is a **systems** contribution and probably your strongest,
   most quantifiable novelty — align it with the NAFIPS 2025/2026 work noted in the
   README.

---

## 4. Risks / weaknesses a reviewer will raise

1. **"This is clusiVAT with FCM instead of nearest-prototype."** Pre-empt it with a
   head-to-head experiment (see §5) and a crisp statement of the delta.
2. **Single-linkage's chaining / bridge sensitivity.** VAT/iVAT (hence `IVATMeans`)
   inherits SL's failure on noisy "bridge" points — the exact motivation for
   ConiVAT ([ConiVAT], 2020). Show behavior on bridged/noisy data and discuss.
3. **Threshold fragility.** The max-difference rule keys on a single largest gap in
   sorted values; with many near-equal gaps or heavy noise it can mis-count `k`.
   Report sensitivity vs. noise/overlap and compare to gap statistic / silhouette /
   aVAT.
4. **The diagonal-vs-parent-edge proxy (§2).** A reviewer familiar with iVAT may
   ask why `diag(D',1)` rather than the MST parent edges `q`. Answer it empirically.
5. **FCM refinement can migrate away from the recovered structure.** Once you hand
   centroids to FCM/k-means, the final partition may diverge from the iVAT cut.
   Quantify how often refinement helps vs. hurts vs. the raw cut.
6. **Correctness detail unrelated to novelty:** `fuzzy_c_means` default init draws
   `n*2` points via `np.random.choice(..., replace=False)`, which **raises if
   `2n > n_samples`** (small-data edge case). Worth hardening before benchmarking so
   it doesn't confound results.

---

## 5. Recommended experiments to substantiate novelty

Baselines (all real, in the bibliography):
- **clusiVAT** [clusiVAT-journal] — the must-beat comparison.
- **aVAT / SpecVAT** [iVAT], [SpecVAT/partition] — auto-`k` from the RDI.
- **Single-linkage + Zahn/gap cut** [MST↔SL], [Zahn], [gap-statistic] — to show what
  the iVAT front-end adds over plain MST clustering.
- **FCM with random init**, **FCM++** [FCM++], **k-means++** [kmeans++] — to isolate
  the value of iVAT seeding.
- **OPTICS** [OPTICS] — the analogous ordering-profile extractor.

Measurements:
- Clustering quality (ARI / NMI / purity) on synthetic (Gaussian blobs, varied
  separation; rings/moons; bridged clusters) and standard real sets.
- **Auto-`k` accuracy** vs. true `k` as a function of overlap and noise.
- **FCM convergence:** iterations-to-converge and final objective `J_m` from iVAT
  seed vs. random / FCM++ seed (the seeding claim).
- **Ablations:** raw iVAT cut vs. +FCM refinement; `diag(D',1)` vs. MST-parent-edge
  profile; max-gap rule vs. top-`k` peaks.
- **Scaling** (your strength): exact-iVAT wall-clock vs. `n` for Numba vs. C/OpenMP
  vs. sampled clusiVAT, with the accuracy trade-off of sampling made explicit.

---

## 6. How to frame the contribution (honest and publishable)

> "We present `IVATMeans`, a deterministic, parameter-light front-end that uses the
> improved VAT (iVAT) minimax ordering to (i) estimate the number of clusters via a
> single max-gap rule on the ordered dissimilarity profile and (ii) produce
> geometry-aware initial centroids that seed Fuzzy C-Means. Unlike clusiVAT, which
> extends a single-linkage cut by a one-shot nearest-prototype rule, `IVATMeans`
> couples the iVAT structure to iterative fuzzy refinement, and unlike
> sampling-based VAT scaling (sVAT/bigVAT/clusiVAT) we compute **exact** iVAT and
> instead attack cost with a priority-queue Prim MST and fused-precision C/OpenMP
> kernels."

Claim the **combination + auto-`k` rule + exact-fast implementation**, not the
underlying VAT/iVAT/FCM/MST-cut primitives. Acknowledge the single-linkage
equivalence up front (§2) — it strengthens, rather than weakens, the write-up,
because it connects your method to 50 years of MST-clustering theory while leaving
the FCM coupling and the fast exact pipeline as your own.

---

## 7. Summary verdict

| Component | Novelty |
|---|---|
| Fuzzy C-Means (`fcm.py`, `cfcm.pyx`) | None (correct re-implementation) |
| VAT/iVAT algorithm (`pvat.py`, `pcvat.pyx`) | None algorithmically; **systems novelty** in the fast exact implementation |
| `IVATMeans` auto-`k` + centroid seeding | **Incremental**: novel *combination* (iVAT seed → FCM) + a specific parameter-free auto-`k` rule; overlaps heavily with clusiVAT / aVAT / SpecVAT |
| `get_ivat_hierarchy` multi-level extraction | Modest (convenience over the SL dendrogram iVAT already encodes) |
| Priority-queue / compact Prim MST, in-place bit-masked permutation, C-SIMD | **Strongest, most quantifiable contribution** — publishable on its own merits |

The novelty you suspected is present but narrow. Pin it down with the clusiVAT and
FCM-seeding comparisons above, own the single-linkage connection, and lead with the
exact-fast-iVAT systems results.
