# Scale run: multi-start NN + 2-opt/3-opt (take-best), n=2000 → 18k

The validated scalable pipeline (from the m-sweep): for S spread starts, build a
nearest-neighbour tour, polish with neighbour-list 2-opt* then 3-opt* to
convergence, **keep the shortest** (take-best). GPU-built distance matrix / kNN,
nearest-size EUC_2D TSPLIB, % over published optimum. Source:
`experiments/vat_tsp_scale.py`.

## Results (8 starts)

| instance | n | raw NN (best) | polished mean | polished worst | **take-best** | time |
|----------|------|---------------|---------------|----------------|---------------|------|
| d2103 | 2 103 | +8.8% | +6.18% | +9.35% | **+2.31%** | 0.07 s |
| fnl4461 | 4 461 | +23.0% | +4.86% | +7.12% | **+3.59%** | 0.28 s |
| d18512 | 18 512 | +22.0% | +4.66% | +5.14% | **+4.09%** | 3.82 s |

![d18512](../figures/vat_tsp_scale_d18512.png)
![summary](../figures/vat_tsp_scale_summary.png)

## Findings

1. **Take-best matters a lot.** On d2103 the best of 8 starts (+2.31%) is nearly
   3× better than the per-start mean (+6.18%) — multi-start diversity, not m
   tuning, is where the gain is (as the sweep predicted). The effect shrinks as n
   grows (d18512 best +4.09% vs mean +4.66%) because per-start variance narrows.
2. **It scales cleanly on the GPU pipeline**: +4.09% at **n=18 512 in 3.82 s** for
   all 8 starts (construction + full 2-opt + full 3-opt to convergence, each
   O(n·k)). No all-pairs kernel, no uncrossing pre-pass — the neighbour-list
   operators are enough because the NN construction has no long seams to miss.
3. **End-to-end it's ~+2–4% over the published optimum across 2k–18k**, from raw
   NN tours of +9–23%, and far below the raw VAT insertion order (+55…+94% in
   earlier runs). Good, honest local-search quality (not LKH-level, but produced
   in seconds and fully scalable).

## Verdict

`multi-start NN → neighbour-list 2-opt* → 3-opt* → take-best` is the recommended
scalable route: **~+4% at n=18k in under 4 s**, simple, and it sidesteps every
sharp edge we found (VAT-insertion-order seams, k-NN quality cap, one-move/pass
GPU 2-opt). More starts trade linearly more time for a better best (especially at
smaller n where variance is larger).

## Canonical solver

This pipeline is now the **default entry point**:
`experiments/vat_tsp_solve.py::solve_tsp(coords, n_starts=8)` (GPU, NumPy
fallback) — CLI `python -m experiments.vat_tsp_solve <n> [--starts S] [--plot]`.

## Files
- `experiments/vat_tsp_scale.py` (scale study), `experiments/vat_tsp_solve.py`
  (canonical solver).
- `experiments/figures/vat_tsp_scale_d2103.png`, `_fnl4461.png`, `_d18512.png`,
  `_summary.png`.
