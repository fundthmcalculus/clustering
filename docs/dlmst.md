# Certificate-and-filter MST for VAT/IVAT (`dlmst`)

`vat_order_dlmst`, `compute_vat_dlmst` and `compute_ivat_dlmst` return the
same VAT order, VAT image and IVAT image as `vat_prim_mst` / `compute_vat` /
`compute_ivat`. They build the MST another way, and can work straight from
points without an `n x n` distance matrix.

```python
import tribbleclustering as tc

p, q, w = tc.vat_order_dlmst(X, from_points=True)          # O(n) memory
ivat, argmin_seq, p = tc.compute_ivat_dlmst(X, from_points=True)  # output only
vat, p = tc.compute_vat_dlmst(D)                            # from a matrix
```

## Where it comes from

Khanna, Putterman & Song, *A Deterministic Linear Time Algorithm for Minimum
Spanning Tree* (arXiv:2611.31187, 2026), Algorithm 2: Borůvka steps, a
certificate forest `F`, drop every edge `F` supports (cycle property),
recurse on `F ∪ U(F)`. The paper's own certificate procedure finds its
schedule by enumerating all schedules against all execution histories. That
bounds comparisons, not time, and cannot be run. `dlmst` keeps the runnable
skeleton and substitutes a practical deterministic certificate. The paper
analysis lives in `grad-school` (`ClusteringExperiments/docs/linear-time-mst-2611.31187.md`).

**No asymptotic gain is possible for VAT.** VAT works on the complete graph,
where array Prim is already `O(n^2) = O(m)`, which is linear. The gains below
come from structure: every `O(n^2)` pass is independent per row, whereas
Prim's round loop is serial, and the filter and IVAT need only the MST.

## Algorithm

| Pass | What | Reads |
|---|---|---|
| 1 | `k` lightest edges per vertex (k=4) + VAT seed; `F0 = MSF` of them | n² |
| 2 | one contraction pass: the lightest edge between every pair of `F0` components (≤256; Borůvka merges above that) → spanning `F` | n²/2 |
| 3 | filter: in `F`'s Kruskal-reconstruction-tree leaf order, `F`-minimax(i, j) = max(g[i..j−1]), so each row is one running max, which is one comparison per pair | n²/2 |
| — | Kruskal on `F ∪ U(F)` (U(F) is typically 0–10 edges) | O(n log n) |
| — | Prim on the MST alone gives the VAT order (exact for distinct weights, by the cut property) | O(n log n) |
| IVAT | each row is a running max over the MST's KRT leaf order, row-parallel; no VAT matrix | n² writes |

With tied weights the tree is a valid MST but may differ from Prim's
tie-break, so the VAT order can differ between equal-weight options.

## When to use which

Measured on 4 CPUs, d=10, 8 Gaussian blobs, end to end from points
(`benchmarks/mst_pareto.py`; subprocess-isolated peak RSS).

| Goal | Use | dlmst | existing best |
|---|---|---|---|
| VAT order, n=24 000 f64 | `vat_order_dlmst(X, from_points=True)` | **1.50 s / 6 MB** | Prim on matrix: 3.97 s / 4.61 GB |
| IVAT image, n=16 000 f64 | `compute_ivat_dlmst(X, from_points=True)` | **1.19 s / 2.05 GB** (the output only) | `compute_ivat_c(inplace=True)`: 4.05 s / 2.05 GB |
| MST stage given `D`, n=24 000 f64 | `vat_order_dlmst(D)` | **0.66 s** | `vat_prim_mst_c`: 1.45 s |

The dense-matrix path wins only through threads: single-threaded it is about
1.5× slower than Prim, because it makes ~2 passes over the matrix against
Prim's ½ pass. See `benchmarks/README.md` for the full tables and the Pareto
plots.
