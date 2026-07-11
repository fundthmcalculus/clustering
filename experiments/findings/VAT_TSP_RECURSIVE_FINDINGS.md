# Recursive IVAT-clustered TSP: in-cluster + cluster-to-cluster ordering (n=1000)

Follows up `VAT_TSP_RESLICE_FINDINGS.md`. Replaces the ad-hoc "largest-gap
blocking" with **IVATMeans' own cluster detection, applied recursively**, and
separates the two ordering problems the hierarchy poses. Source:
`experiments/vat_tsp_recursive.py`. n=1000, 12 gaussian blobs (blob size ~83),
mean over 5 seeds, LKH (`elkai`) reference.

## Method

At each node with > s points, run IVAT on the sub-block and ask
`get_ivat_levels(n_clusters = round(m/s))` for its sub-clusters — the same
abrupt-change-on-the-iVAT-superdiagonal detector IVATMeans uses, in K-cluster
mode so the children land near the target leaf size s (the `n_clusters=-1` mode
over-fragments homogeneous blobs — it gave 693 leaves at s=16). Recurse until
leaves ≤ s; solve each leaf's **in-cluster** TSP with LKH. Bottom-up, optimise
the **cluster-to-cluster** ordering at every level (order the child arcs by an
endpoint TSP + per-arc orientation DP), then a final unified-GPU 2-opt polish.

## Result — % over LKH, mean over 5 seeds

| s | leaves | in-cluster only | + cluster-to-cluster | + GPU 2-opt | time |
|-----|--------|-----------------|----------------------|-------------|------|
| 16 | 288 | 60.7% | 26.1% | 3.9% | 0.19 s |
| 32 | 226 | 49.2% | 28.9% | 4.1% | 0.24 s |
| 64 | 135 | 35.7% | 23.3% | 3.6% | 0.50 s |
| 128 | 14 | 22.2% | **9.8%** | 2.5% | 0.87 s |
| 256 | 8 | 18.1% | 11.7% | **2.0%** | 4.26 s |

("in-cluster only" = leaves solved by LKH but kept in VAT order between clusters;
"+ cluster-to-cluster" = the recursive arc-ordering stitch; "+ GPU 2-opt" = the
resident-matrix 2-opt from `vat_tsp_reslice`.)

## Findings

1. **The cluster-to-cluster ordering is the big lever.** Keeping leaves in VAT
   order between clusters leaves 18–61% on the table; recursively ordering and
   orienting the cluster arcs cuts that roughly in half to two-thirds (e.g. at
   s=128, 22.2% → 9.8%). Solving clusters well is not enough — how they are
   strung together dominates.

2. **2-opt then closes most of the remaining gap**, landing every s at **2–4%
   over LKH**. The recursive cluster tour is a strong 2-opt initialiser (from it,
   the resident-GPU 2-opt reaches 2–4%, vs 6.5% from the raw VAT tour in the
   reslice study).

3. **Leaf size s: bigger is better for quality, up to a time cost.** Quality
   improves as s grows and jumps once **s ≥ the natural cluster size (~83 here)**
   — at s≥128 the leaves *are* whole blobs, so there are few seams to stitch
   (cluster-to-cluster drops to ~10%). Below the cluster size the recursion
   fragments blobs (288 leaves at s=16) and adds seams. Cost grows with s (bigger
   per-leaf LKH): s=256 is 4.3 s vs 0.5 s at s=64.

4. **Practical pick.** The requested target **s=64** gives 3.6% over LKH in 0.5 s
   — a good speed/quality balance; pushing to **s≈128** (≈ the cluster size)
   reaches 2.5% at 0.9 s. Setting s below the natural cluster size is
   counterproductive.

![recursive](../figures/vat_tsp_recursive.png)
(A: each stage's contribution vs s. B: time vs s. C: the recursive+2-opt tour on
the 12-blob instance, s=256, +2.0% over LKH.)

Note: on **uniform** (unclustered) data the whole approach is s-insensitive — IVAT
finds no real structure, so the recursion just bisects and every s gives the same
tour (2-opt still reaches ~1.4%). The recursive-clustering value is specific to
data that actually has cluster structure.

## Files
- `experiments/vat_tsp_recursive.py` — `recursive_route`, `_ivat_split`
  (leverages `get_ivat_levels`), multi-seed sweep.
- `experiments/figures/vat_tsp_recursive.png`.
