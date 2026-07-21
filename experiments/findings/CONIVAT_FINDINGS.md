# ConiVAT — evaluation & basic implementation

**Artifact index**
- Algorithm + class: `src/tribbleclustering/conivat.py`
  (`compute_conivat`, `ConiVAT`, `expand_constraints`,
  `generate_constraints_from_labels`, `learn_metric`, `transform_with_metric`)
- Tests: `tests/test_conivat.py`
- Scale study: `experiments/conivat_scaling.py`
  → `experiments/figures/conivat_scaling.png`
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

## Caveats / next steps

- This is a **pure-Python/numpy reference** (no Cython twin), per the request.
  The natural follow-up is the compiled path, reusing `pcvat.compute_ivat_c`
  the same way this reuses `pvat.compute_ivat`.
- MMC here is a basic projected-gradient version (alternating half-space / PSD
  projections). It is faithful to §4.2 but not tuned; few/noisy constraints can
  distort the metric, so `metric_learning=False` is available to isolate the
  constraint-imposition contribution.
- Constraint generation from labels mirrors the paper's protocol (random pairs,
  typed by label agreement); real deployments would supply `S`/`D` directly.
