# Finding Your Novel Niche — `tribble-clustering`

**Purpose:** go past "what's prior art" (see `docs/novelty-review.md`) and pin down a
*specific, defensible, currently-open* research contribution for the PhD.
**Date:** 2026-07-10. Companion files: `docs/novelty-review.md`, `docs/bibliography.md`.

---

## TL;DR — the one-sentence niche

> **There is no *fuzzy* member of the VAT clustering family, and the reason is a
> geometry bound nobody has lifted: iVAT recovers structure in a
> minimax/ultrametric (single-link) space, but every VAT-based partitioning method
> to date either stays crisp or, when it needs prototypes, summarises each cluster
> by a *Euclidean mean*, which bounds it to clusters a prototype can stand for.
> Close that loop: carry the iVAT minimax structure all the way into a *soft*
> partition, covering the case the prototype cannot.**

That is your niche. The rest of this document argues why it is open, why it is
defensible, and gives you three concrete instantiations ranked by risk/payoff, plus
the experiments and citations to back each.

---

## 1. The core insight (this is the thesis-defining tension)

**Read this section as an operating envelope, not as a bug report.** `IVATMeans` is a
contribution in its own right, and the grad-school document now presents it as one
(Ch. 3 §3.3.5). It is **initialization-free**: the partition comes off the deterministic
iVAT ordering, so the same data gives the same answer every run, where FCM has to be
started somewhere and its random starts give run-to-run variation with no guarantee of
the same partition twice. The reordered image is an artifact a person can **verify the
partition against**, which no prototype method offers. One `fit` returns **assignment and
membership together**. And cluster extraction is called at `n_levels=1` inside a routine
written for several levels, so the same call **begins to advise on tree structure** (no
hierarchy is exposed on the estimator; that part is unfinished, not claimed). Everything
below is about where the *back end* stops, which is a different question from whether the
method is good.

Two facts from the literature, each individually well established, meet inside your
own `IVATMeans` and bound what its back end can represent:

**Fact A — iVAT lives in minimax/ultrametric space.**
The iVAT recurrence `D'[r,c] = max(D*[r,j], D'[j,c])` computes the **minimax path
distance** = the **single-link distance** = the weight of the largest edge on the MST
path between two points (Havens & Bezdek 2012; Chehreghani 2019/2020). Pairwise
minimax distances form an **ultrametric**. This is precisely why VAT/iVAT is good at
**elongated, non-convex, chained** structure.

**Fact B — Euclidean means / FCM are convex-only.**
k-means and (Euclidean) FCM represent each cluster by a **mean** and assign by
Euclidean distance; they are "fundamentally constrained to create convex regions" and
are known to fail on non-convex/elongated clusters. The **mean of a ring or an
elongated filament is not in the cluster** and is a meaningless prototype.

**Where the bound bites (in `pvat.get_ivat_levels` → `ivatmeans.py`):**
`IVATMeans` uses iVAT (Fact A) to *find* clusters, then

1. represents each recovered cluster by `np.mean(all_cities[cluster_ids], axis=0)`,
   a **Euclidean centroid**, and
2. assigns every point by **nearest Euclidean centroid** (Fact B). The same centroids
   also serve as FCM seeds wherever FCM is run downstream.

So the front end and the back end have different reach, and that is the whole content
of the tension. The front end is where `IVATMeans` earns its place: exact on the whole
data at the scale the engine reaches, verifiable against the image it read the cuts off,
deterministic and unseeded, returning assignment and membership from one fit. The back
end is where it stops: a mean is a Euclidean prototype and nearest-centroid assigns by
Euclidean distance, so the reach of the method as a whole is the reach of the prototype,
and no better centroid heuristic lifts that, a ring having no good Euclidean prototype
at all. **That bound is real and demonstrable, and it is the opening**: a relational
method in the minimax space is the *complement* covering exactly the case a prototype
cannot, not a correction of something done wrong. (You can show the bound in one figure:
two moons / concentric rings, where iVAT cuts them perfectly but the segment *means* land
in empty space and nearest-centroid re-merges them.)

Nobody has covered that case because the VAT community has stayed on the *crisp/visual*
side (clusiVAT, aVAT, SpecVAT, ML-aVAT, kernel-iVAT) while the *fuzzy* community
(FCM++, relational FCM) never adopted the VAT/minimax ordering. **You sit exactly in
that unoccupied intersection**, and `IVATMeans` already reaches into it from the
coordinate side, as far as a Euclidean prototype can go.

---

## 2. Why each "obvious" niche is already taken (so you don't waste a chapter)

| Tempting claim | Why it's crowded / taken | Cite to differentiate |
|---|---|---|
| "iVAT ordering → cut → clusters" | **clusiVAT** already does sample→iVAT→SL cut→nearest-prototype | Kumar et al. 2016 |
| "Auto-`k` from the reordered image/diagonal" | aVAT, DBE/E-DBE, and **two 2023–24 papers**: ML-aVAT (also infers hierarchy!) and kernel-iVAT (adaptive extraction) | Wang 2010; Mittal et al. 2023; Zhang et al. 2024 |
| "Hierarchy from a single iVAT" | **ML-aVAT (2023)** explicitly infers sub-cluster hierarchy from the RDI; iVAT already encodes the SL dendrogram | Mittal et al. 2023 |
| "Max-gap / longest-edge cut" | classical MST clustering | Zahn 1971; Gower & Ross 1969 |
| "Better FCM seeding" | FCM++, MaxMin, many schemes | Stetco 2015; and others |
| "Minimax-space clustering" | Chehreghani embeds minimax into Euclidean and runs k-means/spectral | Chehreghani 2019/2020 |
| "Minimax prototypes" | Bien & Tibshirani's minimax-linkage medoids | Bien & Tibshirani 2011 |

**Read this table as a map, not a wall.** Every row is a *component* that exists in
isolation. Your contribution is the **specific composition none of them performed**:
a *fuzzy* VAT-family clustering that keeps the minimax geometry end-to-end. The
2023–24 papers (ML-aVAT, kernel-iVAT) matter most — they show the auto-`k`-from-image
lane is actively contested, so **do not stake your primary claim on auto-`k`.** Stake
it on the *fuzzy + geometry-consistent* axis.

---

## 3. Three concrete instantiations (ranked)

### Niche 1 (recommended) — "Fuzzy clusiVAT": relational fuzzy clustering on the iVAT dissimilarity itself
**Idea.** iVAT already produces a **dissimilarity matrix** `D'` (minimax/ultrametric).
Do not go back to feature vectors and means. Feed `D'` (and iVAT-derived auto-`k` +
ordering-based seeds) into a **relational fuzzy clustering** algorithm that operates
*directly on a dissimilarity matrix* and returns soft memberships — **NERFCM**
(Non-Euclidean Relational FCM, Hathaway & Bezdek 1994) or **FANNY** (Kaufman &
Rousseeuw). NERFCM even has the β-spread trick for exactly the non-Euclidean
dissimilarities iVAT produces.
**Why it's novel & defensible.** Every ingredient is published and trusted, but the
composition, *the first soft/fuzzy member of the VAT family, computed in the iVAT
minimax space with no Euclidean-mean step*, does not exist. It covers exactly the case
§1's prototype bound excludes, while keeping `IVATMeans`'s initialization-free front end.
**Risk:** low-medium. **Payoff:** high (clean "first fuzzy VAT clustering" story).
**Key comparisons:** clusiVAT (crisp), FCM/FCM++ (Euclidean), NERFCM on raw `D`
(no iVAT structure) — to isolate what the iVAT ordering/auto-`k` adds.

### Niche 2 (safe, incremental) — minimax-medoid prototypes for VAT-seeded prototype clustering
**Idea.** Keep a prototype-based method, but replace each segment's **Euclidean mean**
with its **minimax medoid** (the point minimizing the maximum within-segment distance
— exactly Bien & Tibshirani's minimax-linkage prototype). Now the prototype is always
a real object inside the (possibly non-convex) cluster.
**Why it's novel & defensible.** Bien & Tibshirani's prototypes are for *crisp
hierarchical* clustering; nobody uses them as **VAT-derived seeds for (possibilistic)
fuzzy** partitioning. Small, clean, low-risk contribution; good as a chapter section
or the "prototype" ablation of Niche 1.
**Risk:** low. **Payoff:** medium.

### Niche 3 (highest theory payoff) — soft cut / graded boundaries from the iVAT profile
**Idea.** Today the cut is a **hard threshold** on the sorted off-diagonal. Instead,
define a **fuzzy membership of the cut itself**: points whose off-diagonal (MST-edge /
minimax) profile sits near a boundary get **graded** assignment to the two adjacent
segments, with a principled (e.g. gap-significance / stability) confidence. This yields
a *soft* number-of-clusters and soft boundaries end to end — genuinely new for the
VAT family and tightly on-theme with fuzziness.
**Risk:** medium-high (needs a principled boundary model, not a heuristic). **Payoff:**
high (a real theoretical contribution, not just a pipeline).
**Contrast with:** the gap statistic (Tibshirani 2001), aVAT/ML-aVAT (crisp counts).

**Cross-cutting pillar — exact iVAT at scale (your engineering edge).**
All VAT-family scaling (sVAT/bigVAT/clusiVAT) relies on **sampling**. Your
priority-queue/compact Prim MST + in-place bit-masked permutation + fused-precision
C/OpenMP (README: NAFIPS 2025/26) computes **exact** iVAT fast. Frame this as: *"we
don't approximate the dissimilarity structure by sampling; we make the exact
computation affordable,"* and quantify the accuracy sacrificed by clusiVAT's sampling
vs. your exact route. This is the empirical backbone that lets any of Niches 1–3 run
on full data.

---

## 4. The single experiment that characterises the envelope

One figure will motivate the entire thesis. On **two-moons** and **concentric rings**
(canonical non-convex sets):

1. Show iVAT recovers the structure (clean cut points on the ordered profile).
2. Overlay the **segment means**: they land *between*/outside the true clusters.
3. Show Euclidean nearest-centroid (`IVATMeans`'s back end, or FCM on those seeds)
   re-merges or mislabels them. This is the prototype bound, measured.
4. Show your **Niche-1/2** variant (relational-fuzzy on `D'` / minimax medoids)
   preserves the correct soft partition, covering the case step 3 cannot.

Then quantify over many datasets (ARI/NMI, auto-`k` accuracy, and, for the fuzzy
claim, a fuzzy validity index such as the partition coefficient / Xie–Beni) with the
baselines in §3. That progression *is* the contribution narrative.

**This quantification is now a stated dissertation goal, not just a good idea.**
grad-school Goal **G9** (Ch. 7 §7.2) benchmarks `IVATMeans` head-to-head against FCM and
k-means on both halves: wall clock across a size ladder with the CPU and on-device
backends separated, and partition quality by ARI including the non-convex sets where the
envelope predicts `IVATMeans` loses. Two things to carry into that experiment. The
predicted loss is the refutation condition: if the Euclidean prototype costs nothing on
rings and moons, the whole argument for the relational method weakens. And the
determinism is asymmetric: FCM and k-means need restarts reported as a spread, while
`IVATMeans` has no spread over seeds, which is itself a result and should be printed
rather than left as a blank column.

---

## 5. Honest positioning statement (drop-in for an intro/abstract)

> "The VAT family assesses cluster tendency and, in clusiVAT, performs crisp
> single-linkage partitioning by imaging a sampled minimax (iVAT) dissimilarity and
> extending labels via a nearest-prototype rule. We observe that this and all
> prototype-based VAT variants summarise each cluster by a Euclidean mean, which
> bounds them to clusters a prototype can represent and so excludes precisely the
> non-convex structure iVAT is otherwise able to capture. We propose
> the first *fuzzy* member of the VAT family: cluster count and seeds are derived from
> the exact iVAT ordering, and a *soft* partition is computed **in the minimax
> dissimilarity space itself** via relational fuzzy clustering, never reverting to
> Euclidean means. Exact full-data iVAT is made tractable by a priority-queue MST and
> fused-precision parallel kernels, avoiding the sampling approximation of prior
> scalable VAT methods."

Claim: **(1)** the prototype bound, stated and quantified, **(2)** the first
fuzzy/relational VAT clustering covering the case it excludes, **(3)** the exact-fast
implementation enabling both on full data, `IVATMeans` included. Do **not** claim VAT, iVAT, FCM, MST-cut, minimax distances, or auto-`k`-from-image
as new — cite them and stand on them.

---

## 6. New references introduced by this analysis
(Full entries appended to `docs/bibliography.md`.)
- **NERFCM** — Hathaway & Bezdek (1994), *Pattern Recognition* — relational fuzzy
  clustering directly from a (non-Euclidean) dissimilarity matrix. **Core enabler of
  Niche 1.**
- **FANNY** — Kaufman & Rousseeuw (1990) — relational fuzzy clustering; ≈ RFCM at m=2.
- **Minimax linkage prototypes** — Bien & Tibshirani (2011), *JASA* — object
  (medoid) prototypes for non-convex clusters. **Core enabler of Niche 2.**
  PDF: `docs/sources/Bien_Tibshirani_2011_Minimax_Linkage_Prototypes.pdf`.
- **Minimax distance representation learning / embedding** — Chehreghani (2019/2020),
  arXiv:1904.13223 / *Machine Learning* — minimax = single-link path distance;
  Euclidean embedding s.t. squared distance = minimax. **Theoretical spine of §1.**
  PDF: `docs/sources/Chehreghani_2019_Minimax_Representation_Learning.pdf`.
- **ML-aVAT** — Mittal, Laxman & Kumar (2023), *Big Data Research* 34:100413 — 2-stage
  ML auto-`k` **and hierarchy** from the RDI. **The frontier your auto-`k` must not
  compete with head-on.**
- **Kernel-based iVAT with adaptive cluster extraction** — Zhang, Zhu, Cao et al.
  (2024), *Knowledge and Information Systems* 66:7057–7076 — adaptive RDI cluster
  extraction. Current frontier of crisp iVAT clustering.

---

## 7. What to do next (order of operations)
1. **Reproduce the §4 figure** on two-moons / rings with the current `IVATMeans`. If the
   prototype bound shows up (it will), you have your motivating result, and it doubles as
   the quality half of Goal G9.
2. **Prototype Niche 1** (NERFCM on the iVAT `D'`, seeded by iVAT segments). This is the
   smallest step to a defensible "first fuzzy VAT clustering" claim.
3. **Add Niche 2** (minimax medoids) as the prototype ablation.
4. **Benchmark** vs. clusiVAT, FCM++, NERFCM-on-raw-`D`, OPTICS; report ARI/NMI, auto-`k`
   accuracy, and a fuzzy validity index; add the exact-vs-sampled scaling curves.
5. Keep Niche 3 (soft cut) as the theory-heavy stretch chapter if time allows.
6. **Before submission:** re-verify the 1994 NERFCM and 2011 JASA page numbers and the
   Wang-et-al. TKDE 2009/2010 pairing (flagged in `docs/bibliography.md`).
