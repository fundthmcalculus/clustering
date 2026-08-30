# Design Notes — `IVATMeans` back ends and the minimax geometry

Why `IVATMeans` has a `refine=` parameter at all, and why `dissimilarity_power`
defaults to 2.0. These are the two design decisions in the package whose
rationale is not readable off the code, so the docstrings for
`tribbleclustering.IVATMeans` and `tribbleclustering.nerfcm` point here.

This is a design note for the library. The research framing it was extracted
from — the prior-art survey, the novelty argument, and the bibliography — lives
in the [`grad-school`](https://github.com/fundthmcalculus/grad-school) repo
under `ClusteringExperiments/docs/`.

---

## 1. Why `refine=` exists: the front end and back end use different geometries

Two facts, each well established on its own, meet inside `IVATMeans`.

**The front end lives in minimax/ultrametric space.** The iVAT recurrence

```
D'[r, c] = max(D*[r, j], D'[j, c])
```

computes the **minimax path distance** — equivalently the single-link distance,
equivalently the weight of the largest edge on the MST path between two points
(Havens & Bezdek 2012; Chehreghani 2019/2020). Pairwise minimax distances form
an **ultrametric**. This is exactly why VAT/iVAT is good at elongated,
non-convex, chained structure.

**Euclidean means are convex-only.** k-means and Euclidean FCM represent each
cluster by a mean and assign by Euclidean distance, which constrains them to
convex regions. The mean of a ring or an elongated filament is not in the
cluster, and is a meaningless prototype for it.

**The consequence.** The original back end represented each recovered cluster
by `np.mean(all_cities[cluster_ids], axis=0)` and assigned points by nearest
Euclidean centroid. So `IVATMeans` used a coordinate-free front end to *find*
clusters and a coordinate-bound back end to *represent* them. A cluster no
Euclidean prototype can stand for is one the estimator will mislabel even when
iVAT cut it correctly — the mean of a ring is in its hole.

This is a **bound on where the estimator applies, not a bug**: the iVAT cut
itself is unaffected, and the demonstration is two moons or concentric rings,
where iVAT separates them cleanly and the segment means land in empty space so
nearest-centroid re-merges them.

`refine=` is the knob that lets a caller stay in the front end's geometry:

| `refine=` | prototype | assignment | when |
|---|---|---|---|
| `"medoid"` (default) | cluster medoid — an actual data point | nearest medoid | keeps prototypes inside the cluster; safe general default |
| `"relational"` | none; NERFCM memberships on `D'` | relational, no coordinates | stays entirely in minimax space; returns soft memberships |
| `"euclidean"` | Euclidean mean | nearest centroid | the legacy behaviour, kept for reproducibility |

See GitHub issue #54 for the change, and
`tests/test_ivatmeans_refine.py` for the two-moons/rings reproduction.

---

## 2. Why `dissimilarity_power` defaults to 2.0

`refine="relational"` feeds the iVAT minimax matrix `D'` to NERFCM
(`tribbleclustering.nerfcm`) raised to the power `dissimilarity_power`. The
front-end cut is always taken on the raw `D'`, so this power is the
*refinement's* geometry and never the front end's.

**Every power is admissible.** `D'` is the subdominant ultrametric `u(D)`, and
ultrametrics have strict *p*-negative type for every `p >= 0` (Faver et al.,
"Roundness properties of ultrametric spaces," *Glasgow Math. J.*
56(3):519–535, 2014). So `u(D)` already satisfies the relational dual's
requirement at any power — the choice is a geometry, not a correctness fix.

**`p = 1` is the theoretical spine.** Chehreghani's minimax embedding is an
embedding whose *squared* distances equal the minimax distance, so it argues
the `p = 1` geometry.

**`p = 2` measures better, so it is the default.** On the sweep in GitHub issue
#95 (4 datasets × 4 seeds), `p = 2` lifts ARI(`labels_`) from 0.9474 to 0.9993,
and on 20-D data the soft memberships recover from exactly uniform
(crispness <= 0.05) to 0.43–0.59. The library defaults to the stronger measured
result and leaves the spine one keyword away.

**β-spread is inert here, and is not a selling point.** Because `u(D)` is
already realizable as a matrix of squared Euclidean distances, NERFCM's
β-spread correction — a safeguard for inputs *outside* that class — provably
never fires on `D'`. That is one less thing to defend about the composition,
not a feature. See GitHub issue #89 and
`tests/test_ivatmeans_refine.py::TestBetaSpread`.

---

## References

Only the entries the two decisions above rest on. The full bibliography moved
to `grad-school` (`ClusteringExperiments/docs/bibliography.md`).

- **Havens & Bezdek (2012)**, "An efficient formulation of the improved visual
  assessment of cluster tendency (iVAT) algorithm," *IEEE TKDE* 24(5):813–822.
  Committed copy: `docs/papers/Havens_Bezdek_2012_iVAT_efficient.pdf`.
- **Chehreghani (2019/2020)**, "Minimax distance representation learning /
  embedding," arXiv:1904.13223 / *Machine Learning*. Minimax = single-link path
  distance; Euclidean embedding such that squared distance = minimax.
- **Hathaway & Bezdek (1994)**, "NERF c-means: Non-Euclidean relational fuzzy
  clustering," *Pattern Recognition* 27(3):429–437. The `refine="relational"`
  back end.
- **Faver, Kosta, et al. (2014)**, "Roundness properties of ultrametric
  spaces," *Glasgow Math. J.* 56(3):519–535. Strict *p*-negative type of
  ultrametrics at every `p >= 0`.
- **Bien & Tibshirani (2011)**, "Hierarchical clustering with prototypes via
  minimax linkage," *JASA* 106(495):1075–1084. Medoid prototypes for non-convex
  clusters; the `refine="medoid"` default.
