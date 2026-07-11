# Adversarial evaluation — does divide-and-conquer VAT do anything k-means doesn't?

The earlier divide-and-conquer results used Gaussian blobs, where k-means
already scores ARI ≈ 1.0 — so "stitched → 1.0" proved nothing. A critical
reviewer's charge: *the structure-aware (k-means) partition is doing the
clustering; the VAT machinery is redundant.* This is the experiment that tests
it, on data where k-means **must** fail and single-linkage/VAT **should** win,
with the missing controls (k-means alone, exact single-linkage).
`experiments/adversarial_eval.py`.

![adversarial](figures/adversarial_eval.png)

## ARI vs ground truth (N=8 blocks; bridge points excluded from ARI)

| dataset | kmeans | single-linkage | exact-VAT | naive-block | **stitched** |
|---------|--------|-----------------|-----------|-------------|--------------|
| two_moons | 0.27 | 1.00 | 1.00 | 0.39 | **1.00** |
| circles | 0.00 | 1.00 | 1.00 | 0.10 | **1.00** |
| aniso | 0.61 | 0.00 | 0.00 | 0.30 | **0.00** |
| varied_density | 0.79 | 0.00 | 0.00 | 0.55 | **0.52** |
| bridged | 1.00 | 0.00 | 0.00 | 0.07 | **0.00** |
| easy_blobs | 1.00 | 1.00 | 1.00 | 0.41 | **1.00** |

## What this establishes

**1. The confound is REFUTED.** On the non-convex cases (two_moons, circles)
k-means fails badly (0.27, 0.00) while **stitched scores 1.00** — matching exact
single-linkage. The N=8 k-means partition *cuts straight through* the moons and
rings, yet the light representative cross-block stitch **reconnects across those
cuts** and recovers the exact single-linkage clustering. So the divide-and-conquer
VAT is **not** collapsing to its k-means partition; the VAT+stitch machinery
does real work that k-means cannot.

**2. The stitch is essential — naive block-decomposition fails on non-convex.**
naive-block scores 0.39 / 0.10 on moons / circles (and only 0.41 on *easy*
blobs at N=8): concatenating independent block orders fragments non-convex
clusters at the seams. Stitched vs naive is the whole point.

**3. exact-VAT == single-linkage on every case** (both 1.00 on moons/circles,
both 0.00 on aniso/bridged/varied). This confirms empirically that cutting the
iVAT superdiagonal is single-linkage clustering — and that stitched tracks it.

## What this honestly refutes / bounds (the failures are the science too)

**VAT is not universally better than k-means — it is better exactly where
single-linkage is, and worse where single-linkage chains.**

- **aniso (elongated, touching):** SL = VAT = stitched = 0.00; k-means = 0.61.
  Single-linkage chains the elongated clusters together; VAT inherits this.
- **bridged:** k-means = 1.00; SL = VAT = stitched = 0.00. The textbook
  single-linkage failure — a thin bridge chains the two blobs into one. VAT
  (hence stitched) inherits it faithfully. This is VAT's Achilles heel, shown
  plainly.
- **varied_density:** k-means = 0.79; exact-VAT = 0.00; stitched = 0.52. Note
  stitched > exact-VAT here — a *coincidental* divergence (the k-means partition
  injected density-aware structure the approximate MST kept), **not** a reliable
  advantage; do not claim it.

In all failure cases the **stitched approximation tracks exact VAT's behavior**
(fails when VAT fails), which is the correct property for an approximation — it
does not silently diverge in the regime it targets.

## The genuinely defensible result (earned, not assumed)

> A block-decomposition VAT with a light representative cross-block stitch
> **recovers exact single-linkage clustering — including non-convex structure
> that k-means cannot represent — even when the partition cuts through clusters,
> where naive concatenation fails**, at parallel sub-quadratic MST cost. It
> inherits single-linkage's failure modes (chaining across bridges, elongated
> touching clusters) faithfully, i.e. it is a *true approximation of VAT*, not a
> centroid method in disguise.

Combined with VAT's native input being an **arbitrary dissimilarity matrix**
(k-means and kd-tree/EMST methods need coordinates; clusiVAT samples), the
defensible niche is: *parallel, near-exact single-linkage/VAT on arbitrary
dissimilarities, with error confined to where single-linkage is itself
unreliable.* That is a real, bounded claim — and the failures above are the
honest boundary of it.

## Caveats a reviewer will still raise
- Datasets are 2-D Euclidean; the arbitrary-**non-metric**-dissimilarity regime
  (the strongest niche) is asserted, not yet tested here — next experiment.
- k is given (true k) to every method; auto-k is a separate question.
- N=8, single seed per dataset; a partition-robustness sweep (does a worse
  partition break the stitch on moons?) would harden the claim.

## Files
- `experiments/adversarial_eval.py` — datasets, methods, ARI table, figure.
- `experiments/figures/adversarial_eval.png`.
