# OPT 4 "done right": reverse-delete candidates → the #45 Lin-Kernighan solver

**Date:** 2026-07-11 · `experiments/reverse_delete_lk.py`
**Context:** the review's OPT 4 said the reverse-delete 2-core should be used as
a *candidate list* for a real local search (the LK solver in PR #45), not as a
bespoke 2-opt. This wires that up and measures whether it helps.

## The wiring (one hook + one fix)

#45's `lin_kernighan` builds its candidate neighbour lists internally from k-NN
with no injection point. The change proposed (and vendored here with
attribution until #45 merges) is a single parameter:

```python
def lin_kernighan(distances, candidates=None, ...):
    if candidates is not None:
        neigh = _sort_candidates(distances, candidates)   # external list
    else:
        neigh = _build_neighbor_lists(distances, k)       # existing k-NN
```

`neigh[t2]` is only ever iterated, so a ragged per-city list drops in for the
fixed-width k-NN array with no other change to `_lk_step`/`_optimize`.

**Bug found while wiring:** the pure LK gain test uses `_EPS = 1e-9`, so a
**float32** distance matrix (≈1e-5 precision — exactly what `distance_matrix`
returns) makes LK chase rounding noise and *never terminate*. #45's compiled
kernel already accumulates gain in `double` for this reason; the pure path
should cast too. Fixed here by forcing float64 inside `lin_kernighan` — worth
upstreaming to #45's `lk.py`.

## Does a reverse-delete candidate set beat plain k-NN? No.

Same LK engine (`n_starts=3, max_depth=5`), only the candidate list differs.
Tour length as a ratio to the per-instance best; time = mean s/instance.

| n | full 2-opt | LK · k-NN(8) | LK · reverse-delete 2-core | LK · Delaunay |
|----|----|----|----|----|
| 50 | **1.003** (0.003 s) | 1.054 (0.004 s) | 1.076 (0.004 s) | 1.057 (0.004 s) |
| 100 | **1.000** (0.014 s) | 1.078 (0.009 s) | 1.078 (0.009 s) | 1.078 (0.009 s) |
| 200 | **1.000** (0.061 s) | 1.079 (0.023 s) | 1.099 (0.022 s) | 1.078 (0.023 s) |

- The reverse-delete 2-core (avg width ≈5.1 after ∪ k-NN(5)) is a **thinner**
  candidate set than k-NN(8) and gives **equal-or-slightly-worse** LK tours. It
  never wins. The candidate *source* is not the lever — exactly what the review
  predicted from the ablation.

## And LK vs full 2-opt on these instances?

Full exhaustive-neighbourhood 2-opt is the best-quality method here; #45's pure
**breadth-1** LK trails it by 6–8%, even when tuned deeper/wider:

| n | full 2-opt | LK d5/k8/s3 | LK d10/k12/s5 |
|----|----|----|----|
| 100 | **1.000** (0.014 s) | 1.078 (0.009 s) | 1.074 (0.025 s) |
| 200 | **1.000** (0.062 s) | 1.079 (0.024 s) | 1.062 (0.064 s) |

Deepening/widening LK helps marginally (1.079 → 1.062 at n=200) but does not
close the gap, at higher cost. LK's advantage on these small uniform instances
is **speed/scaling** (O(candidate) per move vs 2-opt's O(n²)), not quality — the
quality case for LK needs the compiled kernel and larger n.

## Takeaways

1. **The wiring is done and correct** — LK now accepts any external candidate
   list (`candidates=`), plus a float64 fix that #45's pure path needs anyway.
2. **Reverse-delete / Delaunay candidates do not beat plain k-NN** for LK; the
   reverse-delete set is thinner and marginally worse. This *confirms* the
   review: reverse-delete's sparsification is not a useful candidate-generation
   advantage over the k-NN that LK already builds for free.
3. **Net:** there is no configuration in which the reverse-delete machinery
   improves the tour over what #45 (or plain 2-opt) already does. Its value
   remains the MST↔TSP framing and non-metric sparsification — not TSP quality.

## Files
- `experiments/reverse_delete_lk.py` — vendored LK + `candidates=` hook + benchmark.
- Companions: `REVERSE_DELETE_TSP_FINDINGS.md`, `REVERSE_DELETE_OPT_FINDINGS.md`.
