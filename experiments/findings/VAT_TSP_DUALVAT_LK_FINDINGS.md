# A real LK step + dual-VAT construction (n=1000)

Two requested experiments. n=1000, 12 gaussian blobs, mean over 3 seeds, LKH
(`elkai`) reference. Source: `experiments/vat_tsp_dualvat_lk.py`.

## Update — min-non-zero-edge seeding, on TSPLIB reference data (pr1002)

Studied **six initialisation points** for the dual-source fronts, on the
repeatable TSPLIB instance nearest n=1000 (**pr1002**, dim 1002;
`nearest_euc_instance`). Reference = the **published optimum 259045** from the
submodule's `solutions` file (`optimal_length`) — no LKH; we care about *time to
near-optimal*.

Two families of initialisation. **Edge-distance rules** (min/max/mean edge,
MST-gap) pick a *pair by their distance*; **placement rules** put the two seeds
in *different dense regions*.

| init | family | \|C1\| | \|C2\| | + neighbour-LK | + full 2-opt |
|------|--------|------|------|----------------|--------------|
| min-non-zero edge | edge | 74 | 928 | +6.3% | +7.8% |
| max edge | edge | 6 | 996 | +7.3% | +7.1% |
| mean-distance pair | edge | 1000 | 2 | +8.6% | +7.5% |
| longest-MST-edge | edge | 6 | 996 | +7.7% | +7.9% |
| PCA principal axis | placement | 6 | 996 | **+5.9%** | +7.1% |
| random pair | baseline | 332 | 670 | +6.6% | +7.1% |
| density-peak + farthest | placement | 670 | 332 | +7.7% | +7.6% |
| two density peaks | placement | 670 | 332 | **+5.7%** | +7.5% |
| balanced-MST cut | placement | 338 | 664 | +8.5% | +8.6% |

(all built + polished in **under half a second**; reference = published optimum.)

- **Balance comes from seed *placement*, not seed *distance*.** Every
  edge-distance rule gave a degenerate split on this connected cloud — a tiny
  pocket + "the rest" (6/996; 74/928; and mean-distance the *worst* at 1000/2).
  Balance is governed by which seed sits nearer the dense bulk (winner-take-most),
  so a mid-range gap does **not** delay the merge. The **placement** rules —
  seeding two different dense regions (density-peak + farthest, two density
  peaks) or cutting the MST for balance — give clean, spatially-coherent **~2:1
  bipartitions (670/332, 338/664)**: a left/right or vertical split of pr1002.
  (`two_dpeaks` and `dpeak_far` even land the seeds in opposite halves.)
- **…but the final tour is unaffected.** Every initialisation — balanced or
  degenerate — polishes to a tight **+5.7% to +8.6% over the optimum**. Best:
  two-density-peaks (+5.7%) and PCA (+5.9%) with the neighbour-LK. So seed choice
  matters for the **clustering interpretation** (use a placement rule for a
  meaningful balanced 2-way split) but is a wash for **tour quality** — the
  construction + local search converge regardless.
- **The neighbour-list LK works well on this real instance** (~6-8%), unlike on
  the blob tours below (+22%): pr1002's fairly uniform layout has few long "jump"
  edges, so the neighbour-list moves suffice. Repeatable reference data gives the
  trustworthy picture.

![init study](../figures/vat_tsp_dualvat_seed.png)
(Five clustering images by initialisation + a quality bar chart; all reach
~6-8% over the published optimum after polish.)

## 2. Dual-VAT on synthetic blobs (original max-edge study — kept for the
## balanced-split illustration and the LK-vs-full-2opt finding)

- **2.1** pick the largest-dissimilarity pair (i0, j0);
- **2.2** seed cluster 1 at i0 (pq-1), cluster 2 at j0 (pq-2);
- **2.3** grow both single-linkage (Prim) fronts at once — at each step take the
  globally smallest frontier edge and let that city join whichever front reached
  it; each city ends in exactly one cluster (a dual-source MST partition);
- **2.4** each cluster's *assignment order* is its VAT path (Prim insertion
  order); join the two paths into one closed tour by the **optimal conjunction**
  — exhaustive over the 4 endpoint pairings/orientations (the seed pair from 2.1
  is one fixed junction; the other junction is optimised).

The dual-source partition is a clean spatial 2-way split — each blob goes wholly
to the nearer seed, cut along the max-dissimilarity edge (see figure panel A,
the requested clustering image).

**As a TSP suggestion** (mean % over LKH):

| stage | over LKH |
|-------|----------|
| dual-VAT raw closed tour | +76.6% |
| dual-VAT + neighbour-LK | +22.3% |
| **dual-VAT + full 2-opt** | **+5.3%** |
| (compare) NN + full 2-opt | +3.7% |

The raw dual-VAT tour is a VAT-quality path pair (~+77%, the usual VAT
closed-tour cost); polished with a full best-improvement 2-opt it reaches +5.3%
over LKH — a **sound construction**, competitive with (slightly behind) nearest-
neighbour as a 2-opt starting point. The two-junction "optimal conjunction"
replaces the single long wrap edge a single-VAT closed tour would carry.

![dual-vat](../figures/vat_tsp_dualvat_lk.png)
(A: dual-source clustering. B: raw dual-VAT tour, +68%. C: after full 2-opt,
+4.1% over LKH.)

## 1. The LK step — and why the candidate list is the catch

Implemented an LK-family local search (`lk_search`): full neighbour-list 2-opt
(best improvement — my earlier `neighbor_two_opt` skipped the j<i half of the
neighbourhood, the bug that stalled it at 16-23%) plus Or-opt (relocate segments
of length 1-3), run to convergence.

**Finding: on VAT-structured tours the neighbour-list LK converges to markedly
weaker optima than a full O(n²) 2-opt.**

| start | + neighbour-LK | + full best-improvement 2-opt |
|-------|----------------|-------------------------------|
| dual-VAT | +22.3% | +5.3% |
| nearest-neighbour | +11.6% | +3.7% |

The reason is structural: VAT/dual-VAT tours carry a few long "jump" edges (where
Prim leapt to a far branch). Repairing one needs a move whose *new* edges connect
cities that are **not** in each other's k-nearest-neighbour list — exactly the
moves a candidate-list search never tries. The full 2-opt scans all pairs and
finds them, reaching +3.7-5.3%; the neighbour-list LK (k=16) cannot and stalls.
Neighbour-list local search is the right tool on *already-good* tours (few long
edges) and for scaling (O(n·k)); it is a poor finisher for VAT-jump tours, where
the O(n²) 2-opt — which does not scale — is the effective local optimiser.

**Takeaway.** (a) Dual-VAT is a valid, clean 2-way construction that polishes to
~+5% over LKH. (b) A true LK win at scale needs the *variable-depth sequential*
LK move (whose gain chain reaches beyond the immediate neighbour list), not the
fixed 2-opt+Or-opt neighbourhood implemented here — that is the remaining lever
for closing the gap at scale.

## Files
- `experiments/vat_tsp_dualvat_lk.py` — `dual_vat(seed_mode='min'|'max')` /
  `dual_vat_tour`, `lk_search`; runs on TSPLIB reference data
  (`nearest_euc_instance`).
- `experiments/figures/vat_tsp_dualvat_seed.png` (min-vs-max seed on pr1002),
  `vat_tsp_dualvat_lk.png` (original blob study).
