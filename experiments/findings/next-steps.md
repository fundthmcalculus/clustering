# Roadmap to the Paper — Findings, Loose Ends, and Next Steps

**Author:** Scott Phillips **· Date:** 2026-07-11
**Purpose:** a durable record of where this work stands, what is proven vs. still
open, and what must happen before the academic paper. Companions:
`white-paper.md` (the claim + evidence), `performance-report.md` (all numbers +
graphs), `docs/novelty-review.md` / `docs/bibliography.md` (prior art).

> **DGX / dual-VAT / uncrossing stream:** see
> `VAT_TSP_DGX_THREAD_SUMMARY.md` for the consolidated index + narrative of the
> GPU VAT, dual-VAT construction, joins, LK, and uncrossing-pre-pass work, and
> `docs/vat-tsp-session2-novelty.md` for its (negative) novelty verdict.

> **Default TSP solver (use this):** `experiments/vat_tsp_solve.py` —
> `solve_tsp(coords, n_starts=8)` runs the validated pipeline (multi-start
> nearest-neighbour → neighbour-list 2-opt* → 3-opt* → take-best), ~+2–4% over
> optimum from n=2k–18k in seconds (GPU, NumPy fallback). CLI:
> `python -m experiments.vat_tsp_solve <n> [--starts S] [--plot]`. Evidence &
> the dead ends it avoids: `VAT_TSP_SCALE_FINDINGS.md`, `VAT_TSP_MPRIM_SWEEP_FINDINGS.md`,
> `VAT_TSP_KOPT_FINDINGS.md`.

---

## 0. The defensible claim (carry this forward verbatim)

> A parallel, bounded-cost, partition-robust, auto-k divide-and-conquer engine
> for VAT/iVAT that operates on arbitrary — including **non-metric** —
> dissimilarity matrices, preserving the single-linkage structure that centroid
> methods cannot represent, and whose error is confined to where single-linkage
> itself is unreliable. Anchored by an exact GPU-Borůvka realization
> (bit-identical to serial VAT) and an in-place engine that lifts the feasible
> problem size.

**Organizing observation:** VAT's output depends only on the MST, not how it is
built ⇒ fast/parallel/approximate VAT = fast/parallel/approximate MST.

**Boundaries to state up front (not hide):**
- Everything inherits single-linkage's regime — wins on non-convex / arbitrary
  dissimilarity, **fails on bridged / anisotropic-touching data** (faithfully).
- Divide-and-conquer speedups reported are *ideal-parallel* (largest block).
- GPU results are hardware-specific (consumer FP64 is weak); memory wins are
  constant-factor (3×), not asymptotic.
- **Not "the first GPU VAT"** — eVAT (Meng & Yuan 2018) exists; the parallel/PQ-MST
  framing is the author's own NAFIPS 2025/2026 work (self-citation, not prior art).

---

## 1. What is proven (with evidence)

| Result | Status | Evidence |
|---|---|---|
| In-place iVAT 3→1 matrices; n=64k f64 now runs; max n 52k→89k | exact, tested | PR #17/#18, `performance-report.md` §2 |
| Silent correctness bug in old in-place permutation, fixed | fixed | PR #18, `HARDENING`/commit |
| GPU FCM 30–56× | same fixed point | PR #20, `benchmarks/gpu_fcm.md` |
| GPU pairwise wins only high-d/f32 (honest negative elsewhere) | exact | PR #19, `benchmarks/gpu_pairwise.md` |
| Exact GPU-Borůvka MST ~5× (grows with n); on-device front-end 4.8–6.6× | exact | PR #22/#23, `experiments/findings/BORUVKA_VAT_FINDINGS.md` |
| Divide-and-conquer spectrum (naive ~N² but quality collapses) | measured | PR #25, `experiments/findings/DC_VAT_SCALING_FINDINGS.md` |
| VAT beats k-means on non-convex; stitch preserves it; both inherit SL failures | measured, controlled | `experiments/findings/ADVERSARIAL_EVAL_FINDINGS.md` |
| Works on non-metric D (fractional-p0.5, cosine, geodesic) = exact SL | measured | `experiments/findings/HARDENING_FINDINGS.md` |
| Principled stitch (fps + top-m) robust across partition×N; ablation | measured | `experiments/findings/GAPS_FINDINGS.md` |
| Auto-k recovers k where SL valid; bounded by dendrogram validity | measured | `experiments/findings/GAPS_FINDINGS.md` |

---

## 2. Engineering loose ends (repo hygiene, pre-merge)

- [ ] **Regression test for the fixed correctness bug.** The old in-place
      permutation was silently wrong; the suite missed it because it only checked
      *permutation-invariant* quantities. Add a committed test asserting
      `compute_ivat_c(inplace=True) == inplace=False` **full-matrix** equality
      (and numba `compute_ivat`), so it can never regress. *(Highest-priority
      quality gap.)*
- [ ] **Decide merge order.** Memory PRs #16→#17→#18 are exact, tested,
      dependency-free, highest-value/lowest-risk → land first. GPU PRs #19→#24
      add the CuPy dependency (degrade cleanly without it). Experiment PRs #22,
      #25 are research spikes → likely keep unmerged or move to a
      `docs/experiments` area rather than the package.
- [ ] **Update README / docs** for the new public surface if GPU PRs land:
      `tribbleclustering.gpu`, `FuzzyCMeans(use_gpu=…)`,
      `IVATMeans(distance_backend=…, on_device=…)`.
- [ ] **`pyproject.toml` gained a `[gpu]` extra but `uv.lock` wasn't regenerated**
      — `uv sync` won't pull CuPy. Regenerate or document `pip install .[gpu]`.
- [ ] **Wire silhouette-on-D auto-k into `IVATMeans`** (currently only in the
      experiment). Recommended default with the max-gap rule as fallback.
- [ ] **`cfcm` (Cython FCM) is dead weight** — the profiling (`PROFILING_RESULTS.md`)
      shows it loses to NumPy/BLAS for medium+; consider deprecating/removing.
- [ ] **`white-paper.md` / `performance-report.md` live on the spike branch**, not
      `main`. If the spike PR won't merge, move them to `main` with figure links
      repointed to pinned raw-GitHub URLs so they render independently.

---

## 3. Scientific next steps (required for the paper)

- [ ] **Re-run all timings on a thermally-stable machine** with fixed clocks.
      End-of-session laptop runs were throttled ~3×; the report plots the earlier
      clean numbers, but the paper needs a controlled sweep.
- [ ] **Datacenter GPU (full-rate FP64).** Will shift the pairwise and Borůvka
      numbers upward and separate the algorithm from this card's FP64 penalty.
- [ ] **Multiple seeds + error bars.** Most experiments use a single seed;
      robustness/ARI claims need distributions, not point estimates.
- [ ] **Real non-metric domains**, not synthetic norms: DTW (time series), edit
      distance (strings), graph/kernel dissimilarities. This is the *core niche*
      claim (arbitrary dissimilarity) — demonstrate it on genuine non-coordinate
      data where k-means / kd-tree cannot apply at all.
- [ ] **Head-to-head vs eVAT and clusiVAT** on identical datasets (exactness,
      speed, memory) — the two comparisons a reviewer will demand first.
- [ ] **Approximation-error bound for the stitch** as a function of (partition,
      representatives r, top-m). Currently empirical only; a theoretical bound
      (or a characterized failure condition) would elevate it from heuristic.
- [ ] **On-device iVAT recurrence** so the *full* pipeline — not just the ordering
      — stays on the GPU (removes the D→H transfer that currently caps the
      on-device fit at ~parity). This is the piece that would make an end-to-end
      GPU iVAT a real full-fit speedup.
- [ ] **Larger / higher-dimensional scale** (toward the 64 GB and beyond via the
      1-matrix engine) and non-blob structured data for the timing story.
- [ ] **Circles asterisk:** principled stitch is mean 0.96 / one failing config on
      circles (frac≥0.9 = 0.96) — chase the single failure; is it a k-cut issue
      or a genuine stitch miss?

---

## 4. Open questions / reviewer risks

- Is the `(min,max)`-closure ≡ single-linkage-cophenetic framing worth a formal
  statement, or is it too close to Gower–Ross (1969) folklore? (Decide framing.)
- The iVAT-superdiagonal cut is a *proxy* for the exact merge-height sequence
  (the `pvat.py` TODO). Quantify proxy-vs-exact-parent-edge cut quality — a small
  but genuinely novel micro-result.
- Does a *principled* (not single-cheapest) bounded stitch have a provable
  approximation guarantee, or only empirical robustness? (Ties to §3 bound.)
- Auto-k ceiling: can any internal index detect that VAT is the *wrong model*
  (aniso/bridged) and abstain, rather than returning a confident wrong k?

---

## 5. Artifact index (so nothing is lost)

**PRs (GitHub `fundthmcalculus/clustering`):**
- #16 benchmark harness · #17 iVAT 3→2 · #18 iVAT 2→1 + bug fix
- #19 GPU pairwise · #20 GPU FCM · #21 IVATMeans distance routing
- #23 on-device VAT front-end (+ novelty §8) · #24 IVATMeans on_device
- #22 Borůvka spike · #25 divide-and-conquer spike (holds white-paper +
  performance-report + all D&C experiments)

**Docs:** `white-paper.md`, `performance-report.md`, `docs/novelty-review.md`
(§8 = GPU/Borůvka prior art), `docs/vat-tsp-prior-art.md` (VAT↔TSP prior
art/novelty/benchmarks), `docs/bibliography.md` (§6 = VAT↔TSP refs).
`popmusic-spacefilling.md` (repo root) = the actionable plan to close the Part-2
benchmarking gap (POPMUSIC + space-filling baselines, LKH-binary scale harness,
stronger local search) — for a follow-up agent on a faster/unrestricted machine.

**Experiment code (`experiments/`):** `boruvka_vat.py`,
`boruvka_gpu.py`, `blockwise_vat.py`, `stitched_vat.py`, `dc_vat_scaling.py`,
`adversarial_eval.py`, `hardening_eval.py`, `principled_stitch.py`,
`autok_eval.py`, `vat_tsp.py`, `vat_tsp_warmstart.py`, `vat_tsp_benchmark.py`,
`perf_report_figs.py` — each with a matching `*_FINDINGS.md` under
`experiments/findings/`. Figures in `experiments/figures/`; these reports also
live in `experiments/findings/`.
(`vat_tsp.py`: VAT ordering as a TSP/seriation objective + ACO hot start —
`VAT_TSP_FINDINGS.md`. `vat_tsp_warmstart.py`: the follow-up "real result" —
open-path formulation, real construction baselines, harder/non-blob instances,
and a non-metric study — `VAT_TSP_WARMSTART_FINDINGS.md`. `vat_tsp_benchmark.py`:
Lin-Kernighan (LKH via elkai) baseline + a VAT-cluster-blocking strategy that
beats flat 2-opt at scale — `VAT_TSP_BENCHMARK_FINDINGS.md`.)

**Reproduce:** `python -m experiments.<name>` (each script regenerates its
figures); `python -m benchmarks.scale_bench` for the CPU baseline sweep.

**Key references:** Bezdek & Hathaway 2002 (VAT); Havens & Bezdek 2012 (iVAT);
Gower & Ross 1969 (MST ≡ single-linkage); Meng & Yuan 2018 (eVAT — the exact-
GPU-VAT overlap); Kumar et al. 2016 (clusiVAT); Vineet et al. 2009 (GPU Borůvka);
Jin et al. (DiSC, distributed single-linkage). Full list in `docs/bibliography.md`.
