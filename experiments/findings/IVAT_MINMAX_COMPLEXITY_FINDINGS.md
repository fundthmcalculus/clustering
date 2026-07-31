# Can the iVAT min-max go O(n²) → O(n log n)?

**Short answer:** the *full iVAT image* cannot (it has n² pixels — that's a
hard output lower bound), but the **min-max / MST machinery that actually
drives clustering and auto-k can**, and it does so exactly and in O(n) memory —
provided the data is a genuine low-dimensional geometric embedding. A priority
queue alone is not the lever; geometric edge-sparsification is.

**Artifact index**
- Evaluation + prototype: `experiments/ivat_minimax_complexity.py`
- Figure: `experiments/figures/ivat_minimax_complexity.png`
- Builds on: `experiments/boruvka_vat.py` (`vat_order_from_mst`), the kdT-VAT
  prior art in `docs/bibliography.md` §5, and `pcvat.vat_prim_mst_c` (the dense
  O(n²) reference).

## The two costs have different lower bounds

iVAT bundles two things people call "the min-max":

1. **The reordered image** `I(D'*)` — an n×n matrix of all-pairs minimax path
   values. Producing it is **O(n²) by output size alone**; the Havens–Bezdek
   recurrence `D'[r,c] = max(D*[r,j], D'[j,c])` already meets that bound. No
   data structure beats writing n² pixels. If you need the picture, O(n²) is
   final.

2. **The MST / ordering machinery** — VAT's modified-Prim ordering plus the
   minimax path values that feed single-linkage clustering and the auto-k gap
   rule (`get_ivat_levels`). This is O(n²) *here* only because Prim runs on the
   **dense complete graph** and relaxes all n neighbours of every vertex.

## Why a priority queue alone is not enough

Prim with a binary heap is O(E log V). The current `vat_prim_mst` already uses
a heap — but on the complete graph **E = Θ(n²)**, so the heap does not lower the
order (and strictly, `n² log n` is worse than the compiled dense scan the repo
ships). The quadratic term is the **n² candidate edges**, not the queue. To win
you must stop enumerating all n² edges.

## What does work: geometric sparsification + PQ-Prim on the tree

For points in a low-dimensional metric space the **exact Euclidean MST is a
subgraph of the Delaunay triangulation** (2-D: O(n) edges, built in O(n log n));
a k-d / cover tree (dual-tree Borůvka) gives the same for higher, modest
dimension. The pipeline:

1. Delaunay → O(n) candidate edges — **O(n log n)**.
2. Union-find MST (Kruskal/Borůvka) over those edges — **O(n α(n))**.
3. Priority-queue Prim over the **tree** (n−1 edges) for the VAT order + the
   per-vertex connect weight = the single-linkage cut magnitude — **O(n log n)**.

This produces, exactly and without ever forming the n×n matrix: the VAT
ordering, the k-clustering (cut the k−1 heaviest MST edges), and the 1-D
cut-magnitude profile that auto-k reads. **O(n log n) time, O(n) memory.**

## Verification (exact, not approximate)

`verify()` in the script, 2-D blobs:

| n | dense MST weight | sparse EMST weight | match | cut-clustering ARI vs truth |
|------|------------------|--------------------|-------|-----------------------------|
| 500 | 182.7962 | 182.7962 | ✓ | 1.000 |
| 2000 | 322.5379 | 322.5379 | ✓ | 1.000 |
| 5000 | 482.2026 | 482.2026 | ✓ | 1.000 |

The Delaunay EMST is bit-for-bit the dense MST, and single-linkage labels from
cutting it match the dense iVAT clustering exactly (ARI = 1.0 to each other and
to ground truth).

## Scaling (VAT-order production time)

| n | dense Prim, O(n²) | geometric, O(n log n) |
|--------|-------------------|-----------------------|
| 2000 | 22 ms | 21 ms |
| 4000 | 146 ms | 44 ms |
| 8000 | 577 ms | 91 ms |
| 16000 | 3319 ms | 196 ms |
| 32000 | — (3.2 GB matrix) | 563 ms |
| 64000 | — | 1251 ms |
| 128000 | — | 2857 ms |
| 200000 | — | 4376 ms |

- Crossover at n ≈ 2000; by n = 16000 the geometric route is **~17× faster**.
- The dense path **stops at n ≈ 16000** on this 15 GB box — the n×n f64 matrix
  is 3.2 GB at 20k and crosses 15 GB near n ≈ 43k. The geometric route's edge
  list is ~15 MB at n = 200000, so it runs a scale the dense matrix physically
  cannot reach, in 4.4 s.
- The measured geometric curve sits slightly above the pure `n log n` slope —
  that's the constant from a pure-Python `heapq` traversal and the Delaunay
  build, not a change of order; it is decisively sub-quadratic (a true O(n²)
  would be ~150× the n=16000 time at n=200000; we see ~22×).

## Caveats / where O(n²) still stands

- **Geometric embedding required.** Delaunay/k-d-tree tricks need actual
  coordinates. On an *arbitrary precomputed dissimilarity matrix* there is no
  triangulation to exploit and the min-max stays O(n²). This is exactly the
  boundary the kdT-VAT line (`docs/bibliography.md` §5) draws.
- **Dimension.** Delaunay is O(n) edges only in low d; it blows up combinatorially
  in high d. A dual-tree Borůvka on a k-d/cover tree extends the idea to modest
  d but degrades toward O(n²) as d grows (curse of dimensionality). For genuinely
  high-d data, an approximate-kNN candidate graph trades exactness for scale.
- **The image is still O(n²).** This buys the *clustering* and *auto-k*, not the
  2-D RDI. For visual assessment at large n the existing sampled/divide-and-conquer
  routes (sVAT/clusiVAT, `experiments/dc_vat_scaling.py`) remain the tool.

## Status — recorded note, not being pursued

Kept as a local note only. Not productionized: the iVAT core is under active
development directly, so this is left as a reference point rather than a
roadmap item. (For the record, the exact geometric route would be a dual-tree
Borůvka EMST → PQ-Prim order + cut magnitudes → `get_ivat_levels`, with the
dense kernel as fallback — but that is out of scope here.)
