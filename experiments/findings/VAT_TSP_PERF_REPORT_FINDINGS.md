# Dual-VAT + LK TSP — performance report (quality & time, fp32)

Sweep 50 → 50000 target cities on the GB10, **fp32**, distances resident on the
device. Method: **GPU dual-VAT build (dual-source Prim on the resident matrix) →
neighbour-list LK polish** (2-opt + Or-opt, candidates from the resident-matrix
kNN). Reference = the **published TSPLIB optimum** (`solutions` file — no LKH).
Each target resolves to its nearest-size coordinate instance (EUC_2D, plus
CEIL_2D `pla*` for the large end). Source: `experiments/vat_tsp_perf_report.py`.

## Results

| instance | n | ewt | optimum | raw % | **final %** | build s | polish s | **total s** |
|----------|------|------|---------|-------|-------------|---------|----------|-------------|
| eil51 | 51 | EUC | 426 | +84% | +7.5% | 0.03 | 0.06\* | 0.09 |
| kroA100 | 100 | EUC | 21 282 | +59% | **+2.0%** | 0.01 | 0.00 | 0.01 |
| kroA200 | 200 | EUC | 29 368 | +75% | +5.7% | 0.02 | 0.00 | 0.02 |
| d493 | 493 | EUC | 35 002 | +107% | +6.0% | 0.06 | 0.00 | 0.06 |
| dsj1000 | 1 000 | CEIL | 18 660 188 | +148% | +10.4% | 0.14 | 0.00 | 0.14 |
| d2103 | 2 103 | EUC | 80 450 | +71% | +13.3% | 0.30 | 0.01 | 0.30 |
| fnl4461 | 4 461 | EUC | 182 566 | +240% | +4.9% | 0.63 | 0.07 | 0.70 |
| rl11849 | 11 849 | EUC | 923 288 | +231% | +20.0% | 1.69 | 0.30 | 1.99 |
| d18512 | 18 512 | EUC | 645 238 | +462% | +13.1% | 2.65 | 1.03 | 3.68 |
| **pla33810** | **33 810** | CEIL | 66 048 945 | +244% | +27.4% | 5.06 | 1.78 | **6.84** |

\* first call includes numba JIT compilation (one-time).

![report](../figures/vat_tsp_perf_report.png)

## GPU dual-VAT build (the change)

The dual-source Prim growth now runs on the device (cupy over the resident fp32
matrix): `best0/best1` (each vertex's distance to the two fronts) and the labels
stay on the GPU, every round is min / argmin / masked-relax, and only the winning
index crosses to the host per round. Effects:

- **Removes the host O(n²) wall.** The earlier host build was 4.07 s at n=18 512
  (plus a full n×n host copy); the GPU build is **2.65 s** there (~1.5×) and needs
  no host matrix. This is what lets the sweep **reach n = 33 810 (pla33810)** —
  the nearest coordinate instance to the 50k target — in **6.84 s total**.
- Below ~15k the GPU build is not faster than host (per-round sync + kernel
  launch overhead dominates), but those sizes are already sub-second.
- (Verified: the GPU build's partition is identical to the host build for the
  same seeds; a subtle view-aliasing bug — `Dg[i0]` is a view, so the front
  arrays must be copied before mutation — was fixed.)

## Quality

- The LK polish takes the raw dual-VAT tour (+59 … +462% over optimum) to **+2.0
  … +27.4%** — a fast approximate solver.
- **Instance-dependent, not monotone in n**: structured instances polish well
  (kroA100 +2.0%, fnl4461 +4.9%), hard near-uniform / VLSI ones worse (rl11849
  +20%, pla33810 +27.4%). The **neighbour-list LK is the quality ceiling**, not
  the construction — its fixed candidate list can't repair the long "jump" edges
  on hard tours.

## Time (fp32, GB10)

- **Total ≤ 6.84 s to n = 33 810**; sub-second through n ≈ 5000.
- Both stages scale ~O(n²) here (the build's dual-Prim rounds and the polish's
  kNN + 2-opt); build dominates. fp32 halves the resident matrix vs f64 at no
  measured quality cost.

## Limits / levers remaining

- **Beyond ~34k**: pla85900 (86k) is the only larger coordinate instance; the
  build handles it but the Or-opt array-splice in the polish is O(n) per move and
  gets slow — a linked-list Or-opt would fix that.
- **Beating ~5-27%**: a true variable-depth sequential LK (gain chain beyond the
  fixed neighbour list) is the outstanding quality lever.

## Files
- `experiments/vat_tsp_perf_report.py`, `experiments/figures/vat_tsp_perf_report.png`.
- GPU build: `vat_tsp_dualvat_lk.dual_vat_device` / `dual_vat_tour_device`.
