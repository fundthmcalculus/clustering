# Summary Report — Exact, Parallel, and Divide-and-Conquer VAT/iVAT at Scale

**Author:** Scott Phillips
**Date:** 2026-07-11
**Package:** `tribble-clustering` (import `tribbleclustering`)
**Status:** Consolidated summary of preliminary results — for committee review, not a finished paper.

This report distills the project's two research threads and their prior-art
positioning into a single, citable document. It is a companion to — and a
condensation of — `white-paper.md` (the claim and its evidence),
`performance-report.md` (every number and figure), `docs/novelty-review.md`
(the independent prior-art review), and `docs/bibliography.md` (the full,
DOI-checked reference list). Where a claim rests on measured data, the source
PR and figure are named; where it rests on published work, an inline citation
`[Key]` resolves to the **References** section below.

All timings are from a single workstation (32-core Intel, 64 GB RAM, NVIDIA
RTX 4080 Laptop, 12 GB — consumer-grade, so FP64 ≈ 1/64 FP32). Every result
marked **exact** is bit-identical to the serial reference engine. Experiments
are reproducible under `experiments/` (`python -m experiments.<name>`) and
benchmarks under `benchmarks/`.

---

## 1. Background and the organizing observation

VAT (Visual Assessment of cluster Tendency) reorders a pairwise dissimilarity
matrix so that clusters appear as dark diagonal blocks. The original algorithm
[VAT] reorders points by the order a modified Prim's MST traversal [Prim] adds
them, seeded at the globally most-distant pair. iVAT [iVAT] replaces the raw
dissimilarities with a path-based (minimax) transform, and Havens & Bezdek
[iVAT-fast] give the exact O(n²) recurrence

> `D'[r,c] = max(D*[r,j], D'[j,c])`, where `j = argmin_{k<r} D*[r,k]`,

which is what `pvat.compute_ivat` and the `pcvat.pyx` C/OpenMP kernels compute.

The single observation that unifies everything below:

> **VAT's output depends only on the MST, not on how the MST is built.**

Because Prim's traversal only ever crosses MST edges, any MST builder — serial
Prim, parallel Borůvka, or a GPU kernel — followed by an O(n log n) traversal
from the max-dissimilarity seed, reproduces the *bit-identical* VAT ordering.
Equivalently, and classically, cutting the ordered iVAT profile is
single-linkage clustering: Gower & Ross [MST-SL] proved that all information for
single-linkage is contained in the MST, and Zahn [Zahn] formalized clustering by
cutting inconsistent (long) MST edges. This reduces "fast / parallel /
approximate VAT" to "fast / parallel / approximate **MST**," and it is the lens
for both threads.

---

## 2. Systems thread — exact iVAT, cheaper and faster

All results in this section are **exact** unless noted.

| Contribution | Result | Exact? | Source |
|---|---|---|---|
| In-place iVAT construction + permutation (3 matrices → 1) | max feasible `n` at 64 GB: **52k → 89k**; `n=64k` float64 iVAT now runs (98 GB infeasible → **32.85 GB / 25 s**) | ✅ | PR #17 / #18 |
| GPU Fuzzy-C-Means (data-resident) | **30–56×** vs 32-core CPU at n = 50k–500k | same fixed point (~1e-5, labels >99% identical) | PR #20 |
| GPU pairwise distances | **1.3–2.5×** only at high dimension + float32; **<1×** low-d / float64 (honest negative) | ✅ | PR #19 |
| GPU Borůvka MST (device-resident) | **~5×** vs serial Prim at n = 32000, growing with n | ✅ (VAT-order match 1.0) | PR #22 |
| On-device VAT front-end (distances → MST → order) | **4.8–6.6×** end-to-end, growing with n | ✅ | PR #23 |

### 2.1 Memory is the scaling wall, not compute

iVAT's exact recurrence originally held **three** simultaneous n×n matrices; two
in-place transforms — building iVAT over the VAT buffer, then an in-place
symmetric permutation — reduce this to **one**. On a 64 GB box this lifts the
feasible problem size from ≈52k to ≈89k points and turns `n=64000` float64 iVAT
from *impossible* into a 33 GB, 25-second run (`performance-report.md` §2,
`memory_reduction.png`). The reduction is a constant factor (3×), not a change of
asymptotic order — stated as a boundary, not hidden.

The in-place permutation technique itself is classical [InPlacePerm]; the
contribution is its *application to VAT reordering*. A methodological byproduct
worth flagging: the previously shipped in-place permutation was **silently
incorrect** — it coupled a cell and its mirror during cycle-following — and the
test suite missed it because the tests only checked permutation-*invariant*
quantities. It is fixed and now verified bit-identical to an independent
reference (PR #18).

### 2.2 Where the GPU wins, and where it honestly does not

Not every stage benefits from the GPU on consumer hardware:

- **FCM (clean win, 30–56×):** iterative and data-resident, so it amortizes the
  PCIe transfer; converges to the same fixed point as the CPU (PR #20).
- **On-device MST/ordering (4.8–6.6×):** exact, and the lead *grows* with n
  because GPU bandwidth absorbs the O(n² log n) work — but only when the matrix
  is already device-resident (PR #22 / #23).
- **One-shot pairwise distances (loses in VAT's common case):** the O(n²) result
  must transfer back over PCIe, and consumer FP64 is weak, so the GPU wins only
  at higher feature dimension with float32 (PR #19). `pairwise_distances(
  backend="auto")` routes to the GPU only where it actually wins.

---

## 3. Methods thread — divide-and-conquer VAT

### 3.1 The naive ↔ exact spectrum

Partition n points into N blocks, VAT each block's within-block sub-matrix, and
merge. Sub-VAT is O((n/N)²), so the work drops ~N× and blocks are embarrassingly
parallel (ideal-parallel speedup ≈ N²). The design space has three points:

- **naive** — concatenate block orders: fast (up to ~800× ideal-parallel at
  N=32) but approximate;
- **stitched** — join blocks with a light cross-block edge set: the middle;
- **Borůvka** — all cross-block edges: exact.

Naive block-decomposition's cluster quality **collapses as N grows**: each block
boundary manufactures a "pseudo-cluster" (a seam artifact, the same phenomenon
known as a processing-window artifact). Quality is entirely partition-dependent
(`performance-report.md` §6, `dc_vat_quality_heatmap.png`).

### 3.2 The decisive test — does it beat k-means, or *is* it k-means?

Gaussian blobs are the wrong data to argue for VAT, because k-means already
solves them. On adversarial (non-convex) data, against the controls a reviewer
would demand (k-means alone, exact single-linkage), the picture is clear
(`adversarial_eval.png`; ARI vs ground truth):

| dataset | k-means | single-linkage | exact-VAT | naive-block | **stitched** |
|---|---|---|---|---|---|
| two_moons | 0.27 | 1.00 | 1.00 | 0.39 | **1.00** |
| circles | 0.00 | 1.00 | 1.00 | 0.10 | **1.00** |
| aniso | 0.61 | 0.00 | 0.00 | 0.30 | 0.00 |
| bridged | **1.00** | 0.00 | 0.00 | 0.07 | 0.00 |

Two symmetric readings, both honest:

- On non-convex data (moons, circles) k-means fails while exact-VAT and the
  stitched decomposition recover the true clustering. The k-means partition
  *cuts through* the rings, yet the stitch reconnects across those cuts — so the
  method is **not** its k-means partition in disguise. Naive concatenation fails
  there, so the stitch does real work.
- On bridged / anisotropic-touching data, VAT and the stitch **faithfully
  inherit single-linkage's failures** — this is a true VAT approximation, better
  exactly where single-linkage is and worse where it chains. That inheritance is
  the known VAT/single-linkage weakness that ConiVAT [ConiVAT] was designed to
  address.

### 3.3 A principled, bounded, partition-robust stitch

The naive light stitch (random representatives, one cross-edge per block pair) is
fragile — ARI swings 0↔1 with the partition and N. An ablation on two-moons over
a partition × N grid shows the fix needs two ingredients **together**:
boundary-aware representatives (farthest-point sampling) **and** top-m cross-edges
per block pair (`principled_stitch_two_moons.png`):

| stitch variant | mean ARI | min | frac ≥ 0.9 |
|---|---|---|---|
| light (random, m=1) | 0.51 | 0.00 | 0.44 |
| top-m only (random, m=8) | 0.74 | 0.00 | 0.72 |
| fps only (fps, m=1) | 0.39 | 0.00 | 0.32 |
| **principled (fps + top-m=8)** | **1.00** | **1.00** | **1.00** |

The principled stitch is ARI = 1.00 across *every* partition — including
adversarial partitions that slice through the clusters — at **bounded O(N²r²)**
cost, without collapsing to the O(n²) exact merge. (On circles it reaches mean
0.96 with a single failing configuration — near-total, not absolute.)

### 3.4 Arbitrary / non-metric dissimilarity, and auto-k

Because VAT consumes a dissimilarity matrix rather than coordinates, it and the
coordinate-free stitch apply where k-means and kd-tree EMST methods cannot. The
stitch preserves exact single-linkage on genuinely non-metric inputs — fractional
p=0.5 Minkowski (which violates the triangle inequality 14% of the time), cosine,
and kNN-geodesic — agreeing with exact VAT in every case (`HARDENING_FINDINGS.md`).

For choosing k without supervision, both a max-gap rule on the ordered iVAT
diagonal and a silhouette-on-D sweep recover the true k and ARI = 1.0 exactly
where single-linkage is valid, and neither recovers k where VAT itself fails — so
auto-k is bounded by the validity of the dendrogram, not by the k-picker
(`GAPS_FINDINGS.md`). This connects the front-end to the auto-k lineage of aVAT
[iVAT], SpecVAT [SpecVAT], Zahn's inconsistent-edge cut [Zahn], and the gap
statistic [GapStat].

---

## 4. Prior-art positioning and honest limits

The genuinely defensible contributions are a **combination**, not a new primitive
— VAT [VAT], iVAT [iVAT; iVAT-fast], FCM [Dunn; Bezdek-FCM], and the MST /
single-linkage equivalence [MST-SL; Zahn] are all faithfully re-implemented prior
art, and none is claimed as novel.

- **Not "the first fast / GPU VAT."** eVAT [eVAT] already gives an exact GPU
  VAT/iVAT, and Fast-VAT [Fast-VAT] already gives a Numba+Cython CPU VAT (up to
  50×). The prior-art-distinct contribution is the *specific intersection*:
  **exact** ∧ **parallel** ∧ on an **arbitrary (non-metric) dense** dissimilarity
  matrix kept **device-resident** ∧ via explicit device-side **Borůvka**, with a
  characterized divide-and-conquer approximation spectrum. Every close competitor
  drops at least one of those (`docs/novelty-review.md` §8).
- **The MST framing is self-citation, not external prior art.** The
  parallel / priority-queue MST VAT is the author's own NAFIPS 2025/2026 work
  (this repo's `pvat.py` / `pqvat.py`); a recurring web-search "pVAT / six orders
  of magnitude" claim was verified to be an ungrounded confabulation with no
  locatable primary source (`docs/novelty-review.md` §8, flag 2).
- **`IVATMeans` overlaps clusiVAT [clusiVAT].** clusiVAT samples the data,
  iVAT-images it to estimate k, cuts to form single-linkage clusters, and extends
  labels by nearest-prototype. `IVATMeans` shares that skeleton; its deltas are
  (a) iterative **fuzzy** refinement instead of one hard nearest-prototype pass,
  (b) a parameter-free max-gap auto-k rule, and (c) **exact, non-sampled** iVAT.
  This must be argued head-to-head against clusiVAT, and the incremental nature
  owned rather than oversold.
- **Divide-and-conquer with cross-edge merge is classical** (DiSC-style
  distributed single-linkage, distance-decomposition EMST); the block-VAT recipe
  is a naive instance and the seam is a known window artifact. The novelty is the
  VAT-specific characterization, the boundary-aware + top-m bounded stitch, and
  the arbitrary-dissimilarity regime.
- **Boundaries stated up front:** everything inherits single-linkage's regime
  (wins on non-convex / arbitrary dissimilarity, fails faithfully on
  bridged / anisotropic-touching data); divide-and-conquer speedups are
  *ideal-parallel* (largest block), not measured concurrent wall-clock; GPU
  results are hardware-specific (consumer FP64 is weak); and the memory win is a
  constant factor (3×), not asymptotic.

Sub-quadratic-memory exact VAT via k-d trees (BB-VAT / kdT-VAT [ScalableVAT])
beats this work on memory but **requires Euclidean coordinates** — the exact
boundary of the arbitrary-dissimilarity regime claimed here.

---

## 5. The defensible claim, and what a full paper must still add

> A parallel, bounded-cost, partition-robust, auto-k divide-and-conquer engine
> for VAT/iVAT that operates on arbitrary — including non-metric — dissimilarity
> matrices, preserving the single-linkage structure that centroid methods cannot
> represent, and whose error is confined to where single-linkage itself is
> unreliable. It is anchored by an exact GPU-Borůvka realization (bit-identical
> to serial VAT) and an in-place engine that lifts the feasible problem size.

Required before the academic paper (`next-steps.md` §3): (i) re-run all timings
on a thermally-stable machine with fixed clocks, and on a datacenter GPU with
full-rate FP64 to separate the algorithm from this card's FP64 penalty; (ii)
multiple seeds with error bars rather than single-seed point estimates; (iii)
real non-metric domains (DTW, edit, graph/kernel distances) rather than synthetic
non-metric norms; (iv) head-to-head against eVAT [eVAT] and clusiVAT [clusiVAT]
on identical datasets; (v) a theoretical approximation-error bound for the stitch
as a function of (partition, representatives r, top-m); and (vi) an on-device
iVAT recurrence so the *full* pipeline, not just the ordering, stays on the GPU.

---

## References

Full annotated list, with DOIs and verification notes, in `docs/bibliography.md`.

- **[VAT]** J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment
  of (cluster) tendency," *Proc. IJCNN*, Honolulu, 2002, vol. 3, pp. 2225–2230.
  doi:10.1109/IJCNN.2002.1007487.
- **[iVAT]** L. Wang, U. T. V. Nguyen, J. C. Bezdek, C. A. Leckie, and K.
  Ramamohanarao, "iVAT and aVAT: Enhanced Visual Analysis for Cluster Tendency
  Assessment," *PAKDD 2010*, LNCS 6118, pp. 16–27, Springer.
  doi:10.1007/978-3-642-13657-3_5.
- **[iVAT-fast]** T. C. Havens and J. C. Bezdek, "An Efficient Formulation of the
  Improved Visual Assessment of Cluster Tendency (iVAT) Algorithm," *IEEE TKDE*,
  vol. 24, no. 5, pp. 813–822, 2012. doi:10.1109/TKDE.2011.33.
- **[SpecVAT]** L. Wang, C. Leckie, K. Ramamohanarao, and J. Bezdek,
  "Automatically Determining the Number of Clusters in Unlabeled Data Sets /
  Enhanced Visual Analysis for Cluster Tendency Assessment and Data
  Partitioning," *IEEE TKDE*, vol. 22, no. 3, pp. 335–350, 2009/2010.
  doi:10.1109/TKDE.2009.135.
- **[clusiVAT]** D. Kumar, J. C. Bezdek, M. Palaniswami, S. Rajasegarar, C.
  Leckie, and T. C. Havens, "A Hybrid Approach to Clustering in Big Data," *IEEE
  Trans. Cybernetics*, vol. 46, no. 10, pp. 2372–2385, 2016.
  doi:10.1109/TCYB.2015.2477416.
- **[VAT-survey]** D. Kumar and J. C. Bezdek, "Visual Approaches for Exploratory
  Data Analysis: A Survey of the VAT Family of Algorithms," *IEEE SMC Magazine*,
  vol. 6, no. 2, pp. 10–48, 2020. doi:10.1109/MSMC.2019.2961163.
- **[ConiVAT]** P. Rathore, J. C. Bezdek, et al., "ConiVAT: Cluster Tendency
  Assessment and Clustering with Partial Background Knowledge," arXiv:2008.09570,
  2020.
- **[Prim]** R. C. Prim, "Shortest connection networks and some generalizations,"
  *Bell System Technical Journal*, vol. 36, no. 6, pp. 1389–1401, 1957.
  doi:10.1002/j.1538-7305.1957.tb01515.x.
- **[MST-SL]** J. C. Gower and G. J. S. Ross, "Minimum Spanning Trees and Single
  Linkage Cluster Analysis," *J. Royal Statistical Society, Series C*, vol. 18,
  no. 1, pp. 54–64, 1969. doi:10.2307/2346439.
- **[Zahn]** C. T. Zahn, "Graph-Theoretical Methods for Detecting and Describing
  Gestalt Clusters," *IEEE Trans. Computers*, vol. C-20, no. 1, pp. 68–86, 1971.
  doi:10.1109/T-C.1971.223083.
- **[GapStat]** R. Tibshirani, G. Walther, and T. Hastie, "Estimating the number
  of clusters in a data set via the gap statistic," *J. Royal Statistical
  Society, Series B*, vol. 63, no. 2, pp. 411–423, 2001.
  doi:10.1111/1467-9868.00293.
- **[Dunn]** J. C. Dunn, "A Fuzzy Relative of the ISODATA Process and Its Use in
  Detecting Compact Well-Separated Clusters," *J. Cybernetics*, vol. 3, no. 3,
  pp. 32–57, 1973. doi:10.1080/01969727308546046.
- **[Bezdek-FCM]** J. C. Bezdek, *Pattern Recognition with Fuzzy Objective
  Function Algorithms*, Plenum Press, 1981. doi:10.1007/978-1-4757-0450-1.
- **[Fast-VAT]** MSR Avinash and I. Lachheb, "Fast-VAT: Accelerating Cluster
  Tendency Visualization using Cython and Numba," arXiv:2507.15904, 2025.
- **[eVAT]** T. Meng and B. Yuan, "Parallel edge-based visual assessment of
  cluster tendency on GPU," *Int. J. Data Science and Analytics*, 2018.
  doi:10.1007/s41060-018-0100-7.
- **[GPU-MST]** V. Vineet, P. Harish, S. Patidar, and P. J. Narayanan, "Fast
  Minimum Spanning Tree for Large Graphs on the GPU," *HPG '09*, pp. 167–171,
  2009. doi:10.1145/1572769.1572796.
- **[ScalableVAT]** "Time and memory scalable algorithms for clustering tendency
  assessment of big data" (BB-VAT / kdT-VAT / TkdT-VAT), *Information Sciences*,
  vol. 660, 120222, 2024. doi:10.1016/j.ins.2024.120222.
- **[InPlacePerm]** E. G. Cate and D. W. Twigg, "Algorithm 513: Analysis of
  In-Situ Transposition," *ACM TOMS*, vol. 3, no. 1, pp. 104–110, 1977.
  doi:10.1145/355719.355729; and B. Catanzaro, A. Keller, and M. Garland, "A
  Decomposition for In-place Matrix Transposition," *PPoPP 2014*, pp. 193–206.
  doi:10.1145/2555243.2555253.
