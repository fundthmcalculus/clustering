# VAT→TSP on the DGX Spark — consolidated thread summary

**Author:** Scott Phillips (with Claude) **· Date:** 2026-07-11
**Branch / PR:** `claude/vat-aco-hot-start-wv5sw2` → **PR #44**
**Hardware:** NVIDIA **GB10 DGX Spark** (aarch64, ~128 GB coherent unified
memory, CUDA 13, `cupy-cuda13x`).

This is the index + narrative for the second VAT↔TSP work stream: taking the
VAT/iVAT machinery onto the GPU, building a **dual-VAT** TSP construction, and
studying local-search polishes (2-opt+Or-opt, variable-depth LK, and an
**intersection-driven uncrossing** pre-pass). Each section links its
per-experiment findings doc and figure. The honest **novelty verdict** is in
§Novelty (spoiler: nothing here is algorithmically new).

**Standing conventions used throughout** (per project memory):
- Repeatable data only — every size-*n* test picks the **nearest-size EUC_2D
  TSPLIB instance**, never random points.
- Quality is measured as **% over the published optimum** (the instance's
  reference tour / optimal length), not vs LKH wall-time.
- **fp32** unless stated; GPU-VAT dtype policy is f32 default, f16 opt-in,
  f64→f32 with a warning.

---

## 1. GPU VAT/iVAT on unified memory  (`BORUVKA_VAT_FINDINGS.md`, `VAT_TSP_DGX_SCALE_FINDINGS.md`)

Ported VAT/iVAT to CuPy (`src/tribbleclustering/gpu.py`, `gpu_vat.py`; shipped via
PR #42): templated Borůvka MST + VAT ordering + iVAT image kernels, f16/f32/f64,
device-resident distance matrices, `IVATMeans(on_device=…, dtype=…)` auto-detect.

- **Transfer collapse.** On the GB10's coherent memory the host↔device copy that
  dominates discrete-GPU VAT effectively disappears; the device path wins from
  small *n* instead of only at large *n*.
- **Precision.** f32 is bit-faithful for ordering; f16 is near-exact at ~half the
  memory. Scaled to **n ≈ 200k (MST) / 100k (iVAT)** on-device.
- **Verdict:** a **systems/performance** contribution over prior-art algorithms
  (VAT = Bezdek–Hathaway 2002; iVAT = Havens–Bezdek 2012; Borůvka MST standard).

## 2. VAT-cluster-blocking scale study  (`VAT_TSP_DGX_SCALE_FINDINGS.md`)

The Part-2 cluster-blocking TSP (partition → per-block sub-TSP → endpoint/
orientation stitch → polish) run at DGX scale on real TSPLIB, capped at d18512
(pla33810 per-block LKH hung and was dropped from the blocking benchmark).
Confirms the paradigm scales; the local search dominates quality.

## 3. Recursive iVAT-clustered TSP  (`VAT_TSP_RECURSIVE_FINDINGS.md`)

`vat_tsp_recursive.py`: recursively iVAT-split into K≈round(m/s) clusters, order
within and across clusters, to **n = 20 000**. Needed clustered blobs to show a
split effect (uniform data has no structure to exploit) — an honest negative that
matches the prior-art finding "structureless data defeats the partitioner."

## 4. Warm-start / reslice studies  (`VAT_TSP_WARMSTART_FINDINGS.md`, `VAT_TSP_RESLICE_FINDINGS.md`)

Hot-start seed study (smallest-non-zero vs largest first edge), a **unified GPU
2-opt** (`gpu_two_opt`, best-improvement RawKernel), and a **largest-k reslicer**
to break the longest intersection lines. The reslice idea matured later into the
crossing pre-pass (§8).

## 5. Dual-VAT construction + seed studies  (`VAT_TSP_DUALVAT_LK_FINDINGS.md`)

`vat_tsp_dualvat_lk.py` — the core construction this stream is about:

- **Dual-source Prim partition.** From a seed pair, grow two fronts, assigning
  each city to one of two clusters (a graph-Voronoi bisection); turn each cluster
  into a VAT/Hamiltonian path; join the two into a closed tour. Also on the GPU
  (`dual_vat_device`, fixed a view-aliasing bug that corrupted `Dg`).
- **Seed studies.** min-non-zero / max / mean-distance / MST-gap / PCA / density
  -peak / balanced-MST seed pairs. **Findings:** tour quality is **nearly
  seed-independent**; seed *placement* (dense-region vs far apart), not the
  seed *distance*, is what drives partition **balance**. Seeding from the minimal
  non-zero edge is the simple default.
- **GPU build cost:** ~**2.65 s @ n=18512** (vs 4.07 s host); reaches
  pla33810 (~34k) in ~**6.84 s**.

## 6. MST-join mechanisms  (`VAT_TSP_JOIN_FINDINGS.md`)

How to close the two VAT paths into one tour:
- **endpoint** — best of the 4 end-orientations, O(1).
- **N×M GPU cycle-merge** — close each path into a sub-cycle, then take the best
  2-opt-across move over all N×M cross-edge pairs, evaluated as a device delta
  matrix (`join_nxm_device`). O(N·M).

**Finding:** the N×M merge improves the *raw* bridge every time and wins the
*polished* tour on hard/large balanced instances — standout **rl11849
+23.7% → +16.9%** — for a few ms. Endpoint stays the right default when a cluster
is tiny. Neither changes the headline that the **local search**, not the join, is
the dominant quality lever. (Merge = standard subtour patching.)

## 7. Local search: 2-opt+Or-opt vs variable-depth LK  (`VAT_TSP_LK_FINDINGS.md`)

`lk_search` (neighbour-list 2-opt + Or-opt(1,2,3)) is the strong baseline.
`lk_search_vd` (variable-depth sequential LK, reverse-suffix gain chain) was
added on request.

**Honest result:** standalone LK-vd is **weak** (+28…+132% over optimum) vs
2-opt+Or-opt (+4.9…+13.3%) — its anchor-based single-direction reverse-suffix
chain with no backtracking is a far narrower neighbourhood. As a **post-2-opt
refinement** it shaves a little (d2103 +13.3→+12.2%). A competitive LK needs the
full LKH machinery (both directions, backtracking, don't-look bits) — which the
`elkai`/LKH binding already provides. **Not a substitute for LKH.**

## 8. Intersection-driven uncrossing pre-pass  (`VAT_TSP_CROSS_FINDINGS.md`) — the useful result

`vat_tsp_cross.py` + `vat_tsp_cross_sweep.py`. Attack the tour's crossings
directly (a 2-D euclidean optimum has none): take the longest edge, find every
edge that **geometrically intersects** it (GPU-vectorised orientation test, one
long edge vs all *n*), apply the uncrossing 2-opt (optionally an Or-opt(1)
relocation competes), repeat over the top-k longest until none cross; then hand to
2-opt+Or-opt.

- **`uncross → 2-opt+Or-opt` beats plain 2-opt+Or-opt** on 4/5 instances, biggest
  where 2-opt was stuck: **d2103 +13.3% → +5.8%** (halved), d493 6.0→4.3%,
  pr1002 7.0→5.9%, kroA200 5.7→5.0%. (fnl4461 the one regression, a basin effect.)
- **Top-k sweep (8/16/32/64) × 2-opt vs 2-opt+Or-opt, n=200…11 849:** the
  dominant effect is **top-k-independent** — *any* uncrossing pre-pass rescues the
  stuck instances (**rl11849 +20.0% → ~7%, a 13-pt gain**; d2103 → ~6%). More
  top-k removes more raw crossings and costs proportionally more time, but the
  polish washes the final-quality difference out; **Or-opt(1) is a wash** vs
  2-opt-only. **Sweet spot: top-16, 2-opt-only** (<0.8 s through n≈12k).

## Consolidated headline numbers

| study | metric | result |
|---|---|---|
| GPU VAT (unified mem) | on-device scale | MST n≈200k, iVAT n≈100k; transfer collapse; f32 exact / f16 near-exact |
| GPU dual-VAT build | wall-time | 2.65 s @ n=18512; ~6.84 s @ pla33810 (~34k) |
| dual-VAT seeds | quality sensitivity | tour quality ~seed-independent; placement drives *balance* |
| N×M join (vs endpoint) | hard instance | rl11849 polished +23.7% → +16.9% |
| variable-depth LK | vs 2-opt+Or-opt | worse standalone (+28…+132%); tiny post-2-opt refinement |
| **uncross → 2-opt+Or-opt** | **stuck instances** | **d2103 +13.3%→+5.8%; rl11849 +20.0%→~7%** |
| uncross top-k sweep | best config | top-16, 2-opt-only; Or-opt & higher-k are washes |

## Cross-cutting conclusions

1. **The local search is the quality lever**, not the construction, the seed, or
   the join — every study here re-confirms Johnson–McGeoch 1997.
2. **The one genuinely useful new lever is the uncrossing pre-pass**: cheap,
   GPU-friendly, and it rescues exactly the large/hard instances where plain
   2-opt gets stuck (rl11849 +20%→+7%). Recommended default:
   **dual-VAT (min-seed) → top-16 uncross → 2-opt+Or-opt.**
3. **Honest negatives kept:** VAT is a poor *closed-tour* start; variable-depth LK
   (this restricted form) underperforms; structureless data defeats the
   partitioner; the GPU wins are hardware-specific.

## Novelty (see `docs/vat-tsp-session2-novelty.md`, and `docs/vat-tsp-prior-art.md`)

**Nothing in this stream is algorithmically novel**, and the theory *corroborates*
our own findings:
- **Uncrossing IS 2-opt** (Croes 1958; move by Flood 1956), formalised as
  tour-untangling (van Leeuwen–Schoone 1981) and **proven a poor standalone
  heuristic** — Ω(n) worst / Ω(√n) average (COCOA 2023) — matching our measured
  +37…+56% standalone. Or-opt = Or 1976; longest-edge ordering = Bentley 1992;
  GPU 2-opt exists since 2011.
- **The dual-VAT partitioner is not novel** — clustering-based (incl. recursive
  -bisection and MST/single-linkage) partitioning for D&C TSP is established
  (deep-clustering, Comput. Oper. Res. 2024; MST-clustering, TKDE 2009; Karp 1977;
  single-linkage ≡ MST-cut). "VAT as the partitioner" is a relabelling of a known
  method. The stitch is Guttmann-Beck 2000; the N×M merge is subtour patching.
- **Variable-depth LK** = Lin–Kernighan 1973 / LKH.
- **The GPU/unified-memory VAT–iVAT build** is the only piece worth a write-up,
  and it is a **systems/performance** paper, not an algorithmic claim.

## Reproduction

```bash
python -m experiments.vat_tsp_dualvat_lk      # dual-VAT construction + seeds + LK
python -m experiments.vat_tsp_join            # endpoint vs N×M GPU join
python -m experiments.vat_tsp_lk              # 2-opt+Or-opt vs variable-depth LK
python -m experiments.vat_tsp_cross           # uncrossing pre-pass (+ tour figure)
python -m experiments.vat_tsp_cross_sweep     # top-k / Or-opt sweep (quality & time)
python -m experiments.vat_tsp_perf_report     # 50→50000 GPU build + LK, fp32
```
All write PNGs into `experiments/figures/vat_tsp_*.png`; each has a matching
`experiments/findings/VAT_TSP_*_FINDINGS.md`.

## Per-experiment index

| findings doc | code | figure(s) |
|---|---|---|
| `VAT_TSP_DGX_SCALE_FINDINGS.md` | `vat_tsp_tsplib.py` | `vat_tsp_dgx_scale.png`, `vat_tsp_tsplib.png` |
| `VAT_TSP_WARMSTART_FINDINGS.md` | (warm-start study) | `vat_tsp_warmstart.png` |
| `VAT_TSP_RESLICE_FINDINGS.md` | `vat_tsp_reslice.py` | `vat_tsp_reslice.png` |
| `VAT_TSP_RECURSIVE_FINDINGS.md` | `vat_tsp_recursive.py` | `vat_tsp_recursive.png`, `_scale.png` |
| `VAT_TSP_DUALVAT_LK_FINDINGS.md` | `vat_tsp_dualvat_lk.py` | `vat_tsp_dualvat_lk.png`, `_seed.png` |
| `VAT_TSP_JOIN_FINDINGS.md` | `vat_tsp_join.py` | `vat_tsp_join.png` |
| `VAT_TSP_LK_FINDINGS.md` | `vat_tsp_lk.py` | `vat_tsp_lk.png` |
| `VAT_TSP_CROSS_FINDINGS.md` | `vat_tsp_cross.py`, `vat_tsp_cross_sweep.py` | `vat_tsp_cross.png`, `_tour.png`, `_sweep.png` |
| `VAT_TSP_PERF_REPORT_FINDINGS.md` | `vat_tsp_perf_report.py` | `vat_tsp_perf_report.png` |
