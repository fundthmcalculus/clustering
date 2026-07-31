# Plan — POPMUSIC + space-filling baselines, strengthened local search, and a scale benchmark for the VAT↔TSP work

**Audience:** a follow-up agent running on a **much faster machine with
unrestricted network** (can build LKH/Concorde from source, run LKH at n=100k, do
many seeds). This session could not: the egress policy blocks scholarly hosts and
`elkai`'s flat LKH took ~7–15 min at n=5000 (no time/trials cap), so `n=5000` here
used a flat-VAT+2-opt reference instead of LKH.

**Why this exists.** The prior-art review (`docs/vat-tsp-prior-art.md`, biblio §6)
found the Part-2 cluster-blocking solver is only benchmarked against a *weak* flat
2-opt baseline. To be credible it must be measured against the real competitors —
**POPMUSIC** and a **space-filling-curve (SFC)** heuristic — with **%-over-LKH/
optimum AND runtime at n = 1k/5k/10k/100k**, following the DIMACS / Johnson-McGeoch
protocol. Two other gaps: our flat 2-opt (~7–13% over LKH) is worse than the
canonical ~5%, and the stitch is essentially the CTSP decomposition of
Guttmann-Beck et al. (2000), so it should be compared to a real CTSP solve.

## Context / where things are
- **Code:** `experiments/vat_tsp_benchmark.py` (constructions, `lkh_tour` via
  `elkai`, cluster-blocking: `vat_blocks`, `maximin` via `stitched_vat`,
  `solve_blocks`, `stitch`, `_orient_cycle`, `_solve_block`). Reuse
  `experiments/vat_tsp_warmstart.py` (`nn_order`, `greedy_edge_order`,
  `mst_dfs_order`, `two_opt_path`, `or_opt_path`, `local_search`) and
  `experiments/vat_tsp.py` (closed `two_opt`).
- **Findings:** `experiments/findings/VAT_TSP{,_WARMSTART,_BENCHMARK}_FINDINGS.md`.
- **Prior art / targets:** `docs/vat-tsp-prior-art.md` §7 (benchmarks to match),
  `docs/bibliography.md` §6.
- **Conventions:** black (line 88), flake8 120 on `src tests` only (experiments
  are NOT linted, but keep them black-clean), figures are `*.png`-git-ignored so
  `git add -f`. `elkai`/deps are the optional `[experiments]` extra. Integer
  distance matrices everywhere (what LKH consumes) so all methods share one
  objective. Develop on the branch `claude/vat-aco-hot-start-wv5sw2` (PR #44).

## Reference solvers to obtain/build (do this first)
1. **LKH binary** (Keld Helsgaun, `http://akira.ruc.dk/~keld/research/LKH/` and
   LKH-3 at `.../LKH-3/`). Build with `make`. Wrap via subprocess: write a
   TSPLIB `.tsp` (EUC_2D for coordinates, or `EDGE_WEIGHT_TYPE: EXPLICIT` +
   `FULL_MATRIX` for arbitrary/integer D) + a `.par` file, run, parse the output
   tour. This replaces `elkai` for anything n≥5k because it exposes:
   - `INITIAL_TOUR_ALGORITHM = POPMUSIC` (the POPMUSIC baseline, for free),
   - `TIME_LIMIT`, `RUNS`, `MAX_TRIALS` (fixes the "no cap → 15 min" problem),
   - `CANDIDATE_SET_TYPE`, and LKH-3 clustered-TSP constraints (for task 6).
   Keep `elkai` as the fallback when the binary is absent (guarded, like the
   current `_HAS_LKH`).
2. **Concorde** (`https://www.math.uwaterloo.ca/tsp/concorde.html`) — optional,
   for *exact* optima on small/medium instances so gaps are "% over optimum," not
   just "% over LKH." Where Concorde is impractical, use LKH-with-time-limit and
   report "% over best-found."
3. **TSPLIB** instances + published optima (`.../tsp/`); also uniform-random in
   the unit square (the neural-solver convention).

## Tasks (ordered; each has a Definition of Done)

### T1 — Space-filling-curve baseline (quick win)
Implement a Hilbert-curve order (O(n log n)): map 2-D points to Hilbert indices,
sort. Add it (a) as a construction start in `constructions()`, and (b) as a
blocking partitioner (contiguous SFC segments) alongside `vat`/`maximin`.
Cite Platzman-Bartholdi 1989 / Bartholdi-Platzman 1982.
**DoD:** SFC tour is ~15–25% over optimum on uniform-random; available as both a
warm start and a partition method; appears as a baseline column in T4/T5.

### T2 — POPMUSIC baseline
Via the LKH binary: `INITIAL_TOUR_ALGORITHM = POPMUSIC`, report both the POPMUSIC
initial tour quality and POPMUSIC+LKH. (Optionally also implement standalone
POPMUSIC per Taillard-Helsgaun 2019, but the LKH-provided one is the credible
baseline.)
**DoD:** POPMUSIC gap (low single-digit %) reproduced at n=1k…100k with runtime;
it is a first-class baseline column, and the strongest competitor for Part 2.

### T3 — Strengthen the flat local search to canonical quality
Current 2-opt/Or-opt is ~7–13% over LKH vs the canonical ~5% (2-opt) / ~4%
(Or-opt) over Held-Karp. Add: **k-nearest neighbour candidate lists**,
**don't-look bits**, **bidirectional Or-opt** (both segment orientations, s=1..3),
and iterate 2-opt+Or-opt to a joint local optimum. See Bentley 1992, Johnson-
McGeoch 1997.
**DoD:** flat 2-opt ≤ ~6% and 2-opt+Or-opt ≤ ~5% over LKH on uniform n=1k; the
Part-1 table is re-run and the "worse than canonical" note in
`VAT_TSP_BENCHMARK_FINDINGS.md` is updated.

### T4 — Scale benchmark (the core deliverable)
For n ∈ {1k, 5k, 10k, 100k}, on **uniform-random** and a spread of **real TSPLIB/
VLSI** instances, ≥5 seeds each:
- Baselines: random+2opt, NN+2opt, greedy+2opt, **SFC** (T1), **POPMUSIC** (T2),
  flat VAT+2opt.
- Method: VAT-cluster-blocking (vat / maximin / SFC / k-means / grid partitions)
  + optimized stitch + polish.
- Reference: LKH binary with a fixed `TIME_LIMIT` (and Concorde optima where
  tractable).
- Report **% over LKH/optimum** (mean ± std) **and wall-clock**, as a **Pareto
  (quality vs time)** table + figure; Wilcoxon/Friedman significance across the
  instance set. Report the Held-Karp lower bound (LKH 1-tree) too.
**DoD:** a DIMACS-style results table + Pareto figure at all four sizes; blocking
beats SFC and flat 2-opt, and its position vs POPMUSIC is stated honestly
(match/approach or lose — report it either way).

### T5 — Ablations (reviewer-required)
Isolate each component's contribution: partition method (VAT single-linkage vs
maximin vs k-means vs grid vs SFC); number/size of blocks; **orientation DP
on/off**; **global polish on/off**; block sub-solver (LKH vs 2-opt); metric vs
non-metric D.
**DoD:** an ablation table showing each component's marginal effect on gap and
t_par.

### T6 — CTSP-via-LKH-3 comparison (closest-prior-art delta)
Our stitch ≈ the CTSP decomposition of Guttmann-Beck et al. (2000). On the *same*
VAT blocks, solve the (free) CTSP with LKH-3's clustered constraints and compare
to our cheap endpoint-TSP + orientation-DP stitch: quantify how much the heuristic
stitch loses vs an optimized CTSP solve.
**DoD:** a table quantifying stitch-vs-CTSP-optimal gap; frames the exact delta to
Guttmann-Beck in `docs/vat-tsp-prior-art.md` §4.

### T7 — Position vs neural D&C (cite, don't run)
Tabulate published gaps for H-TSP (~3.4% @10k) and GLOP (~5% @100k) at matching n;
state where VAT-blocking sits. No need to run neural solvers.

## Deliverables
- Extend `experiments/vat_tsp_benchmark.py` and/or add `experiments/vat_tsp_scale.py`
  (LKH-binary wrapper, SFC, candidate-list local search, scale harness).
- `experiments/findings/VAT_TSP_SCALE_FINDINGS.md` (results, Pareto figure,
  ablations, honest verdict), figures in `experiments/figures/`.
- Update `docs/vat-tsp-prior-art.md` §7/§4 with measured numbers; add any new refs
  to `docs/bibliography.md` §6; update `next-steps.md`.
- Update PR #44 description; keep black/flake8(`src tests`)/pytest green.

## Definitions & gotchas
- `%gap = 100·(L − L_ref)/L_ref`. Held-Karp bound = LKH 1-tree lower bound.
- **`elkai` has no time/trials cap** — do not use it above n≈2k; use the LKH binary
  with `TIME_LIMIT`. This is the single biggest reason this session stopped at a
  flat-VAT+2-opt reference for n=5000.
- LKH needs **integer** distances; scale coordinates before `nint`. For non-metric
  D, use `EDGE_WEIGHT_TYPE: EXPLICIT` (full matrix).
- VAT single-linkage blocks are **unbalanced** (one fat block); cap per-block LKH
  by size or force balanced blocks for the parallel-timing story (see `_solve_block`
  `lkh_cap`). Report t_par = max single-block solve + stitch (+ polish).
- Keep the compiled/pure-Python and f32/f64 conventions if you touch `src/`
  (you shouldn't need to — this is all `experiments/`).

## Priority
T1 + T3 are fast, high-value. T2 + T4 are the core (need the LKH binary). T5 + T6
are for a paper-grade result. T7 is a 30-minute table.
