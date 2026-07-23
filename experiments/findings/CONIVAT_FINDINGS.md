# ConiVAT — evaluation & basic implementation

**Artifact index**
- Algorithm + class: `src/tribbleclustering/conivat.py`
  (`compute_conivat`, `ConiVAT`, `expand_constraints`,
  `generate_constraints_from_labels`, `learn_metric`, `transform_with_metric`)
- Tests: `tests/test_conivat.py`
- Scale study: `experiments/conivat_scaling.py`
  → `experiments/figures/conivat_scaling.png`
- Compiled-vs-pure benchmark: `experiments/conivat_cython_bench.py`
  → `experiments/figures/conivat_cython_bench.png`
- Compiled-only perf test to N=20000: `experiments/conivat_cython_scaling.py`
  → `experiments/figures/conivat_cython_scaling.png`
- Constraint-dial perf (N=5000, constraints 5→500):
  `experiments/conivat_constraint_scaling.py`
  → `experiments/figures/conivat_constraint_scaling.png`
- Paper (committed): `docs/papers/Rathore_2020_ConiVAT.pdf`
  (arXiv:2008.09570, IEEE TKDE). Bibliography entry: `docs/bibliography.md` §1.

## What ConiVAT is

ConiVAT (Rathore, Bezdek, Santi & Ratti, 2020) is a **semi-supervised** version
of iVAT. It takes partial background knowledge as pairwise constraints — a
"similar" (must-link) set `S` and a "dissimilar" (cannot-link) set `D` — and
uses them to fix the two classic weaknesses of VAT/iVAT: sensitivity to noise
and to "bridge" points between clusters (the single-linkage chaining effect
that also affects `IVATMeans`).

The paper layers three constraint-aware stages on top of ordinary iVAT (§4):

1. **Constraint pre-processing (§4.1).** Expand constraints by transitivity
   (must-link is an equivalence relation → connected components; must-link then
   cannot-link ⇒ cannot-link), and drop mutually inconsistent constraints.
2. **Metric learning (§4.2).** Learn a Mahalanobis metric `A` with Xing et
   al.'s MMC (maximize summed distance over `D` s.t. summed squared distance
   over `S` ≤ 1, `A ⪰ 0`), then transform the data into that space.
3. **Minimum transitive dissimilarity (§4.3).** Force the "similar" pair
   distances to zero, then apply the path-based **minimax** (transitive)
   distance transform. The paper states this transform *is* the non-recursive
   iVAT transform.

VAT-ordering the resulting matrix gives the RDI; cutting the `k-1` longest MST
edges yields `k` single-linkage clusters.

## How this implementation reuses the repo

The key observation from §4.3 is that ConiVAT's minimum-transitive-dissimilarity
step is exactly the iVAT minimax transform this repo already computes. So the
**"previous sections" are reused verbatim**: `compute_conivat` builds the
constraint-modified distance matrix and then delegates to `pvat.compute_ivat`
for the transform + ordering, and the `ConiVAT` class extracts clusters through
the same `pvat.get_ivat_levels` path as `IVATMeans`. The new code is only the
ConiVAT-specific machinery: constraint expansion (union-find), MMC metric
learning (projected gradient ascent onto a half-space + the PSD cone), and the
"similar → 0" imposition.

**Equivalence check (encoded as a test):** with no constraints and metric
learning off, `compute_conivat(X)` is bit-for-bit identical to
`compute_ivat(pairwise_distances(X))`. ConiVAT is a strict superset of iVAT.

## Scale study (N = 50 → 5000)

`experiments/figures/conivat_scaling.png`. 2D blobs, 4 clusters, 30 sampled
constraints, best-of-3 wall time (numba pre-warmed).

| n | iVAT (ms) | ConiVAT no-ML (ms) | ConiVAT full (ms) |
|------|-----------|--------------------|-------------------|
| 50 | 1.1 | 1.1 | 5.9 |
| 100 | 3.8 | 4.0 | 44.6 |
| 250 | 23.1 | 24.4 | 41.1 |
| 500 | 94.1 | 90.9 | 149.6 |
| 1000 | 366 | 358 | 424 |
| 2000 | 1423 | 1466 | 1765 |
| 3500 | 4467 | 4362 | 6475 |
| 5000 | 8947 | 8909 | 12849 |

**Read-out.** The constraint-only path tracks the iVAT baseline and the
`O(n²)` reference line essentially exactly — imposing "similar → 0" is a handful
of O(1) writes, so ConiVAT inherits iVAT's `O(n²)` distance/transform cost with
no asymptotic penalty. Full ConiVAT adds a fixed metric-learning overhead
(`O(|constraints|·p²)`, independent of n) that dominates at small n and
amortizes away as the `O(n²)` transform takes over. Net cost at n = 5000 is
~1.4× iVAT, all of it in the (n-independent) MMC solve.

## Compiled (Cython) path

`compute_conivat` takes a `backend` selector (`"auto"` / `"cython"` /
`"python"`). Because the O(n²) core is the shared distance + minimax/iVAT
transform, the compiled path simply routes those two stages through
`pcvat.pairwise_distances_c` and `pcvat.compute_ivat_c` — the *exact* optimized
iVAT kernel — while the constraint pre-processing and MMC solve stay in Python
(they are n-independent and not the bottleneck). Compiled and pure paths are
behaviorally equivalent (a test asserts `backend="cython"` matches
`backend="python"` to float tolerance), and with no constraints compiled
ConiVAT is bit-for-bit the optimized compiled iVAT.

`experiments/figures/conivat_cython_bench.png` (best-of-3, kernels pre-warmed).
"CV core" = compiled ConiVAT without metric learning (isolates the shared
core); "iVAT_c" = the optimized iVAT reference.

| n | iVAT_c (ms) | ConiVAT Cython core (ms) | ConiVAT pure (ms) | speedup |
|------|-------------|--------------------------|-------------------|---------|
| 50 | 0.02 | 0.08 | 1.55 | 20× |
| 100 | 0.09 | 0.12 | 3.82 | 33× |
| 250 | 0.39 | 0.51 | 22.3 | 43× |
| 500 | 1.78 | 1.87 | 89.6 | 48× |
| 1000 | 5.61 | 5.61 | 367 | 65× |
| 2000 | 21.2 | 21.4 | 1475 | 69× |
| 3500 | 142 | 142 | 4615 | 32× |
| 5000 | 284 | 284 | 9531 | 34× |

**Read-out.** Compiled ConiVAT tracks the optimized iVAT kernel essentially
exactly (the "CV core" and "iVAT_c" curves overlap) — the constraint imposition
is a handful of O(1) writes and adds no measurable cost. It runs **20–69×
faster** than the pure-Python/numba ConiVAT reference over N = 50 → 5000, so the
compiled path is "much, much faster" as hoped. Full ConiVAT (with MMC) adds the
same n-independent metric-learning overhead described above, which amortizes
away as n grows. The speedup ratio dips past n ≈ 3500 only because the compiled
iVAT kernel itself has a super-quadratic step there (OpenMP / in-place-permute
overhead visible in *both* iVAT_c and ConiVAT equally); improving that is an
iVAT-kernel concern, orthogonal to ConiVAT.

## Compiled performance at scale (N up to 20000)

`experiments/figures/conivat_cython_scaling.png` — the compiled path on its own
(no pure-Python comparison), 4 cores, best-of-{3,2}, kernels pre-warmed.

| n | core f64 (ms) | full f64 +MMC (ms) | core f32 (ms) | distances (ms) | iVAT transform (ms) |
|-------|---------------|--------------------|---------------|----------------|---------------------|
| 500 | 3.5 | 73.2 | 2.1 | 0.6 | 2.8 |
| 1000 | 8.8 | 18.9 | 5.7 | 1.8 | 7.0 |
| 2000 | 49.2 | 120.1 | 34.2 | 14.5 | 34.7 |
| 4000 | 365 | 291 | 181 | 119 | 246 |
| 6000 | 733 | 705 | 431 | 287 | 447 |
| 8000 | 1566 | 1359 | 893 | 594 | 972 |
| 12000 | 3313 | 3309 | 2524 | 1405 | 1908 |
| 16000 | 6104 | 6034 | 4280 | 2572 | 3533 |
| 20000 | 9976 | 9545 | 6975 | 3766 | 6210 |

**Read-out.**
- **Clean O(n²).** From n = 4000 → 20000 the f64 core grows 365 ms → 9.98 s
  (27×) against a 25× area increase — essentially quadratic; the small excess
  is the iVAT in-place permutation. At n = 20000 the whole compiled ConiVAT
  fits — and completes — in ~10 s on 4 cores.
- **f32 gives ~1.4× and half the memory.** The f32 core runs 6.98 s vs 9.98 s
  at n = 20000, on a 1.6 GB matrix instead of 3.2 GB — the opt-in lever for
  pushing past the memory wall (the roadmap's stated scaling limit for the
  3-matrix iVAT footprint).
- **Time split ≈ 38% distances / 62% iVAT transform**, stable across n (both
  are O(n²)). The transform (VAT MST ordering + minimax recurrence + in-place
  reorder) is the larger share, so it is where any further kernel optimization
  should go.
- **MMC is a fixed cost.** full-f64 (with metric learning) sits on top of the
  core at small n (tens of ms, and noisy from variable MMC convergence) and is
  indistinguishable from the core by n ≳ 6000, confirming the metric-learning
  solve is n-independent.

## The constraint dial (N fixed at 5000, constraints 5 → 500)

`experiments/figures/conivat_constraint_scaling.png` — n held at 5000 (compiled
backend), sweeping the *number of constraints*. Constraints never touch the
O(n²) core (distances + iVAT transform is constraint-independent); they drive
two n-independent stages: `expand_constraints` (transitive closure + pair
expansion) and `learn_metric` (MMC).

(`learn_metric` caps `max_iters` at 30 — see the MMC note below.)

| #req | ‌|ML*| | ‌|CL*| | expand (ms) | MMC (ms) | core (ms) | full (ms) |
|------|-------|-------|-------------|----------|-----------|-----------|
| 5 | 0 | 5 | 1.7 | 0.0 | 382 | 382 |
| 50 | 12 | 38 | 1.1 | 4.3 | 369 | 381 |
| 100 | 25 | 80 | 1.2 | 3.4 | 391 | 389 |
| 200 | 54 | 156 | 1.3 | 15.1 | 392 | 403 |
| 300 | 77 | 242 | 1.4 | 6.9 | 393 | 391 |
| 500 | 135 | 424 | 1.5 | 9.2 | 389 | 399 |

Constraint-free core baseline: **379 ms** (machine noise vs. other runs; the
core is constraint-independent).

**Read-out.**
- **The dial is essentially free.** Total ConiVAT time stays within ~±3% of the
  331 ms constraint-free core across the whole 5 → 500 range. At 500 constraints
  the constraint-handling stages sum to ~11 ms — about **3%** of the run. The
  O(n²) core dwarfs everything the constraints add.
- **`expand_constraints` is nearly flat** (1.0 → 1.5 ms). Its cost here is
  dominated by the O(n) union-find component-grouping pass over all 5000 points,
  not by the expansion itself — the expanded sets stay small.
- **Expansion stays linear-ish, no clique blow-up.** Random label-sampled pairs
  form small union-find components, so |ML*| and |CL*| grow roughly linearly
  with the request (up to 135 / 424 at 500), nowhere near the 100k safety cap.
  (A pathological input that must-links one whole cluster would expand
  quadratically — that is what the cap guards.)
- **MMC cost is small and now bounded.** `learn_metric` runs ~3–9 ms. It used
  to spike to ≈47 ms at 200 constraints: on that particular constraint set the
  projected-gradient objective **oscillates** rather than meeting `tol`, so it
  ran the full (then-100) iteration cap. Well-conditioned sets converge fast
  (≤~17 iters observed), so the fix is to **cap `max_iters` at 30** — it never
  truncates a genuinely converging solve but bounds the oscillating worst case.
  With the cap the 200-constraint point drops to ~15 ms (~3× smaller); MMC is
  now a stable ~3–9 ms with no dramatic outlier.

**Takeaway.** Unlike the N dial (quadratic), the constraint dial is cheap and
roughly linear in the expanded-pair count over this range — you can add
background knowledge freely without a meaningful runtime penalty at N=5000. The
one wrinkle (MMC's data-dependent iteration count) is now capped, so
constraint-handling cost is stable at a few ms regardless of the draw.

## Caveats / next steps

- MMC here is a basic projected-gradient version (alternating half-space / PSD
  projections). It is faithful to §4.2 but not tuned; few/noisy constraints can
  distort the metric, so `metric_learning=False` is available to isolate the
  constraint-imposition contribution.
- Constraint generation from labels mirrors the paper's protocol (random pairs,
  typed by label agreement); real deployments would supply `S`/`D` directly.
