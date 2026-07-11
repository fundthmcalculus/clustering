# Optimizing the reverse-delete spike — what worked, with numbers

**Date:** 2026-07-11 · `experiments/reverse_delete_opt.py`
**Context:** follow-up to the adversarial review of PR #46. Each proposed
optimization was implemented and benchmarked against the baseline
(`reverse_delete_tsp.py`). Reproduce: `python -m experiments.reverse_delete_opt`.

All timings are means over 3–10 seeds on uniform-random Euclidean instances,
pure-Python (dict-of-sets) graphs, single core.

## OPT 1 — Sparsify before reverse-delete (biggest win) ✅

Run reverse-delete on a k-NN (k=10) or Delaunay candidate graph instead of the
complete graph. For 2-D Euclidean points the Delaunay triangulation provably
contains the MST, so the `m=1` result is unchanged — verified equal to Kruskal.

| n | full graph | k-NN | Delaunay | k-NN speedup | Delaunay speedup | edges full→k-NN | MST correct? |
|----|----|----|----|----|----|----|----|
| 60 | 0.024 s | 0.0034 s | 0.0014 s | 7× | 17× | 1770→353 | ✓ all three |
| 120 | 0.336 s | 0.0146 s | 0.0052 s | 23× | 65× | 7140→702 | ✓ |
| 200 | 2.200 s | 0.0422 s | 0.0143 s | **52×** | **154×** | 19900→1187 | ✓ |

**Takeaway:** never reverse-delete the dense graph. Sparsify first — 154× at
n=200 with the identical MST. (For non-metric dissimilarity, where Delaunay does
not apply, use a top-k-per-row candidate graph.)

## OPT 2 — Don't reverse-delete for `m=1` at all ✅

`m=1` reproduces the package's O(n²) Prim MST at O(n⁴) cost. Straight swap:

| n | reverse-delete | Prim (`vat_prim_mst`) | Kruskal | speedup (rd/Prim) | weight match |
|----|----|----|----|----|----|
| 100 | 0.177 s | 0.0054 s | 0.0036 s | 33× | ✓ |
| 200 | 2.234 s | 0.0046 s | 0.0148 s | **487×** | ✓ |
| 400 | *intractable* | 0.017 s | 0.072 s | — | ✓ |
| 800 | *intractable* | 0.066 s | 0.468 s | — | ✓ |

**Takeaway:** keep reverse-delete `m=1` as the conceptual dual only; route real
MST work through Prim. Reverse-delete is intractable by n=400 where Prim is 17 ms.

## OPT 3 — Cheaper connectivity test ✅

Replace the single-source DFS reachability probe with a **bidirectional** BFS
(expand the smaller frontier, stop when they meet). Identical results, growing
speedup (measured on the `m=1` k-NN reverse-delete path):

| n | single DFS | bidirectional | speedup | same result |
|----|----|----|----|----|
| 120 | 0.0187 s | 0.0032 s | 5.8× | ✓ |
| 200 | 0.0568 s | 0.0064 s | 8.8× | ✓ |
| 300 | 0.1557 s | 0.0112 s | **13.9×** | ✓ |

**Takeaway:** a free, exact constant-factor win; stacks with OPT 1. (A numba/CSR
rewrite would add more, but the dynamic graph makes that awkward — deferred.)

## OPT 4 — Reverse-delete 2-core / k-NN as a candidate list for local search ⚠️

Neighbour-list 2-opt (only reconnect a city to a candidate neighbour, both
anchors, sorted-break) vs full O(n²) 2-opt, both from the same NN start:

| n | full 2-opt len | full s | cand 2-opt len | cand s | len ratio | speedup |
|----|----|----|----|----|----|----|
| 100 | 836.9 | 0.014 s | 946.5 | 0.001 s | 1.131 | 28× |
| 200 | 1159.7 | 0.058 s | 1275.5 | 0.002 s | 1.100 | 32× |
| 400 | 1606.9 | 0.321 s | 1735.5 | 0.005 s | 1.080 | 64× |
| 800 | 2277.0 | 1.321 s | 2413.1 | 0.019 s | 1.060 | **68×** |

**Takeaway:** the candidate list delivers a 28–68× local-search speedup, but a
*minimal* candidate 2-opt still trails full 2-opt by 6–13% (gap narrows with n).
Closing it needs a richer move-set (Or-opt / Lin-Kernighan) — i.e. feed the
candidate list into the repo's LK solver (**PR #45**) rather than roll a bespoke
2-opt. The candidate structure is the asset; the move-set governs the residual.

## OPT 5 — Principled degree-2 instead of a greedy that never converges ✅

The subtractive `m=2` greedy stalls (0% convergence for n≥20). Compared against
the **additive** dual (greedy-edge tour: add cheapest edge keeping degree ≤2 and
no premature subtour — always converges) and the **exact min-weight 2-factor**
(Tutte→blossom matching, n≤60). Tour length as a ratio to the per-instance best:

| n | RD `m=2` conv% | RD+2opt | greedy-edge | greedy-edge+2opt | 2-factor #subtours | 2-factor+patch+2opt |
|----|----|----|----|----|----|----|
| 20 | 0% | 1.024 | 1.142 | 1.017 | 2.2 | **1.007** |
| 50 | 0% | 1.015 | 1.139 | 1.022 | 9.4 | 1.024 |
| 100 | 0% | 1.024 | 1.119 | **1.003** | n/a | n/a |

**Takeaways:**
- The **additive greedy-edge** converges to a tour 100% of the time at
  O(n² log n) — the right constructor where the subtractive `m=2` greedy fails.
- The **exact min-weight 2-factor** confirms *minimum-weight-degree-2 ≠ a tour*:
  it fragments into subtours (2→9 as n grows), so it needs patching anyway, and
  blossom matching is expensive (n≤60 here). It is a good lower-bound object, not
  a tour builder.
- After a 2-opt polish everything lands within ~2% of best — reinforcing the
  review's core point that **2-opt dominates the starting construction**.

## OPT 6 — Assert the invariant, not the accident ✅

The duality test asserted exact **edge-set** equality. Under tied distances the
MST is non-unique, so that can diverge between algorithms while the **weight** is
always identical. On a 5×5 unit grid (many ties) both still agreed on edges here
(aligned tie-breaks), so the risk is latent, not observed — but the regression
test now asserts **weight equality** as the guaranteed invariant
(`test_duality_weight_is_the_invariant_under_ties`).

## Bottom line

| optimization | verdict | headline metric |
|----|----|----|
| 1 sparsify first | **adopt** | 154× (Delaunay), identical MST @ n=200 |
| 2 Prim for m=1 | **adopt** | 487× @ n=200; scales where rd can't |
| 3 bidirectional connectivity | **adopt** | 13.9× @ n=300, exact |
| 4 candidate local search | partial | 68× faster, 6–13% quality gap (use LK to close) |
| 5 additive greedy / 2-factor | **adopt greedy-edge** | 100% convergence, 1.003 @ n=100 post-2opt |
| 6 weight-invariant test | **adopt** | fixed |

Combined, OPTs 1+2+3 turn the O(n⁴)/O(n²)-memory dense reverse-delete into an
O(n log n)-candidate + O(n²) Prim pipeline for `m=1`; OPT 5 gives a converging
`m=2` constructor; and OPT 4 shows the pruned candidate set is the right input to
the existing LK solver rather than a bespoke local search. Consistent with the
review, none of this makes reverse-delete a *better* TSP method than what the
repo already has (#44/#45) — its defensible role remains the MST↔TSP framing and
a sparsifier for arbitrary/non-metric dissimilarity.

## Files
- `experiments/reverse_delete_opt.py` — all six optimizations + benchmark suite.
- Baseline & framing: `REVERSE_DELETE_TSP_FINDINGS.md`.
