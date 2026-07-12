# Prior-Art & Novelty Review — the DGX/dual-VAT/uncrossing thread

**Reviewer role:** independent prior-art / novelty assessment for the *second*
VAT↔TSP work stream (GPU VAT on unified memory, the **dual-VAT** tour
construction + joins, the **intersection-driven uncrossing** local search, and
variable-depth LK). Companion to `docs/vat-tsp-prior-art.md` (which covered the
first stream: seriation↔TSP, warm-start, MST-seeded ACO, cluster-blocking).
**Date:** 2026-07-11.

**Method / constraint (unchanged).** Prior art gathered by web search + page
fetch. Scholarly hosts (Springer, IEEE, ScienceDirect) still 403 the proxy, so
paywalled items are verified via search-result/abstract metadata and cited by
DOI/stable URL; a few conference chapters are **[metadata-only]**.

> **Bottom line.** **Nothing in this stream is algorithmically novel.** The
> uncrossing local search is the *founding intuition of 2-opt itself* (Flood
> 1956; Croes 1958), formalized as tour-untangling (van Leeuwen–Schoone 1981) and
> recently proven to be a *poor* stand-alone heuristic — which is exactly what our
> experiments independently measured. The dual-VAT construction is a
> cluster-first / divide-and-conquer TSP (Karp 1977; CTSP endpoint-stitch,
> Guttmann-Beck 2000) with a VAT partitioner and a standard sub-cycle merge. The
> GPU work is a systems/performance contribution over prior-art algorithms. The
> defensible contributions are, as before, **compositional + empirical +
> engineering** — and two of our own honest findings are *corroborated* by the
> theory, not contradicted.

---

## 1. Intersection-driven uncrossing local search (`vat_tsp_cross*.py`)

**What we built.** Take the top-k longest tour edges; for each, use an explicit
GPU-vectorised **geometric segment-intersection test** to find every edge that
crosses it; apply the 2-opt reversal that removes the crossing (optionally let an
Or-opt(1) relocation compete); loop until no top-k edge crosses; then hand off to
full neighbour 2-opt+Or-opt. Marketed internally as a cheap "break the largest
intersection lines first" pre-pass.

**Prior art — this is textbook, and old.**
- **[Flood1956]** M. M. Flood, "The Traveling-Salesman Problem," *Oper. Res.*
  4(1):61–75, doi:10.1287/opre.4.1.61 — first suggests the edge-swap move.
- **[Croes1958]** G. A. Croes, "A Method for Solving Traveling-Salesman
  Problems," *Oper. Res.* 6(6):791–812, doi:10.1287/opre.6.6.791 — **2-opt**. Its
  stated intuition is verbatim ours: *"take a route that crosses over itself and
  reorder it so that it does not."* Removing crossings **is** 2-opt.
- **[vLS1981]** J. van Leeuwen & A. A. Schoone, "Untangling a Traveling Salesman
  Tour in the Plane," *Proc. 7th Workshop on Graph-Theoretic Concepts in CS (WG
  '81)*, pp. 87–98 (also Utrecht tech. rep. RUU-CS-80-11). — the **uncrossing-only
  restriction of 2-opt**: O(n³) uncrossing flips suffice to reach a crossing-free
  tour, with Ω(n²) worst-case examples. This *is* our "crossing_repair" move, named
  and analysed 45 years ago.
- **[Untangle2023]** "Approximation Ineffectiveness of a Tour-Untangling
  Heuristic," *COCOA 2023*, LNCS, doi:10.1007/978-3-031-49815-2_1 [metadata-only]
  — the pure tour-untangling heuristic has **worst-case ratio Ω(n)** and
  **average-case Ω(√n)** for Euclidean TSP.
- **X-opt / efficient 2-opt variants:** "Performance of efficient variants of the
  2-Opt heuristic…," *Discrete Applied Mathematics* 2025,
  S0166218X2500294X — analyses polynomial-time 2-opt variants including one
  restricted to removing intersecting edge pairs.
- **Or-opt** competing move: **[Or1976]** I. Or, PhD thesis, Northwestern, 1976 —
  segment relocation of 1–3 cities; a subset of 3-opt.
- **Longest-edge / edge-importance move ordering** and **neighbour lists +
  don't-look bits** as 2-opt speedups are standard (Bentley 1992, already in
  `docs/bibliography.md`; Johnson–McGeoch 1997).
- **GPU 2-opt** exists: "A Highly-Parallel TSP Solver for a GPU Computing
  Platform," EvoApplications 2011, doi:10.1007/978-3-642-18466-6_31 (~24×);
  "CUDA Accelerated 2-OPT…," IntechOpen (doubly-linked-list tour, O(N) memory,
  >10k cities). GPU segment-intersection kernels are also standard
  (e.g. CUDA ray/segment–triangle intersection, arXiv:2209.02878).

**Verdict — not novel.** Every ingredient is prior art: the move (Croes 1958),
the uncrossing restriction (van Leeuwen–Schoone 1981), the Or-opt alternative (Or
1976), longest-edge move ordering (Bentley 1992), and GPU 2-opt (2011+). Crucially
our **own headline finding — that standalone uncrossing is weak (+37…+56% over
optimum) but the polish rescues it — is exactly what [Untangle2023] proves**
(Ω(n)/Ω(√n)). The sweep's other conclusions (top-32 ≈ top-16 after the polish;
Or-opt(1) is a wash) are useful *engineering* results but not new theory.

*The only micro-delta* (and it is minor): most treatments detect improving 2-opt
moves via the **metric Δ-test** and regard "no crossings" as an emergent property;
we run an **explicit geometric intersection test restricted to the top-k longest
edges, on the GPU, as a construction-basin-escape pre-pass.** That is an
implementation/engineering choice we did not find named verbatim — but it is
fully subsumed by the uncrossing-2-opt and edge-importance-ordering literature and
should be presented as such, not as a contribution.

---

## 2. Dual-VAT construction + sub-cycle joins (`vat_tsp_dualvat_lk.py`, `vat_tsp_join.py`)

**What we built.** A dual-source Prim partition (grow two fronts from a seed pair
→ two clusters), turn each cluster into a VAT/Hamiltonian path, then join the two
paths into one tour (endpoint join, or a GPU **N×M cycle-merge** picking the best
cross-edge 2-opt-across move).

**Prior art.**
- **Cluster-first / divide-and-conquer TSP:** **[Karp1977]** deterministic
  geometric bisection (*Math. Oper. Res.* 2(3):209–224, doi:10.1287/moor.2.3.209);
  Valenzuela & Jones "Evolutionary Divide and Conquer" for TSP; the general
  "partition → solve sub-problems as **open-loop TSPs with fixed endpoints** →
  merge via a **cluster-level TSP**" template (surveyed e.g. in the Santa Claus
  Challenge 2020 write-up, *Front. Robot. AI* 8:689908). Recent large-scale
  instances: **DualOpt** (arXiv:2501.08565) — note the name collision, but its
  "dual" = grid-D&C + path-D&C with a *neural* solver, not a two-source MST.
- **The endpoint / orientation stitch is Guttmann-Beck et al. (2000)** and
  Anily-Bramel-Hertz (1999) (both in `docs/vat-tsp-prior-art.md` §4) — decompose
  into cluster-ordering + per-cluster entry/exit endpoints + intra-cluster
  Hamiltonian paths. Our endpoint join is a heuristic instance of exactly this.
- **The N×M cycle-merge** — close each cluster into a sub-cycle, then remove one
  edge from each and reconnect crosswise for the best gain — is **subtour patching
  / tour merging** (Karp subtour merge; Cook–Seymour tour merging, in the prior
  doc). Evaluating all N×M cross pairs is a 2-opt-across-two-cycles brute force.
- **Two-source Prim as a partition** is a graph-**Voronoi** bisection (region
  growing from two seeds); a special case (k=2) of MST-forest / single-linkage
  partitioning, which is the standard VAT clustering front-end.

### Is the *partitioner itself* novel? — No.

This is the piece most tempting to claim, so it deserves its own honest verdict.
**Clustering-based partitioning for divide-and-conquer TSP is a mature, active
paradigm**, and the VAT/single-linkage cut sits squarely inside it:

- **Cluster-first-route-second for plain TSP is standard** — cluster the nodes →
  sub-TSP per cluster → reconnect — routinely done with k-means (e.g. Firefly +
  k-means TSP, IEEE 2019) and, most tellingly, **recursive two-group bisection**:
  **"Deep clustering of the traveling salesman problem to parallelize its
  solution," *Computers & Oper. Res.* 2024 (S0305054824000200)** partitions
  "iteratively by two-group partitioning of every cluster" — conceptually the same
  move as our dual-source (k=2) split. Also **hyper-tour / cluster-first on sparse
  heatmap graphs** for large TSP (arXiv:2510.20169, 2025).
- **MST-based clustering is textbook** (Zahn 1971; divide-and-conquer MST
  clustering, IEEE TKDE 2009, doi:10.1109/TKDE.2009.37), and the identity
  **single-linkage ≡ MST minus the heaviest edges** is standard. So "use a
  single-linkage cut to partition a TSP" is an *instance* of MST-clustering-for-TSP,
  not a new mechanism.
- **VAT's cluster structure literally is single-linkage/MST-based.** Substituting
  "VAT/iVAT" for "single-linkage clustering" is therefore a **relabelling of a
  known partitioner**, not a new algorithm. The iVAT ordering adds a seriation
  view but does not change *what the cut produces*.

**Verdict — the partitioner is not novel.** It is a substitution within a crowded
paradigm. The paradigm is textbook D&C CTSP; the stitch is Guttmann-Beck (2000);
the merge is subtour patching. What is *unoccupied* is narrow, **compositional and
engineering** — not algorithmic: (a) naming VAT/iVAT seriation as the front-end (a
relabelling); (b) the specific **k=2 dual-source-Prim** realisation (a known
graph-Voronoi bisection primitive); (c) a fully **GPU-resident** build (dual-source
Prim + N×M merge on the device); (d) the **empirical** characterisation (the N×M
merge helps hard instances, e.g. rl11849; tour quality is nearly seed-independent;
placement, not distance, drives partition balance). Position against the 2024 deep
-clustering bisection, MST-clustering (TKDE 2009), Karp (1977), and Guttmann-Beck
(2000); claim only the GPU build + measurements, and describe the partitioner as
applied prior art.

*(This corrects the first draft of this review and `vat-tsp-prior-art.md` §4,
which framed the "VAT partitioner" as a defensible delta. On a targeted search it
is thinner than that: a known clustering paradigm with a VAT-flavoured cut.)*

---

## 3. GPU VAT/iVAT on unified memory (DGX Spark) — `gpu.py`, `gpu_vat.py`

**Verdict — systems/performance, not algorithmic.** VAT (Bezdek–Hathaway 2002)
and iVAT (Havens–Bezdek 2012) are prior art and already assessed in
`docs/novelty-review.md`; Borůvka/Prim MST is standard. The contribution is a
fast, dtype-flexible (f16/f32/f64), unified-memory CuPy implementation and the
transfer-collapse observation on the GB10 coherent memory — a legitimate
engineering/perf result, not a new algorithm. GPU MST and GPU VAT-style pairwise
work exist in the literature; the novelty is the platform-specific implementation
and the measured scaling (n≈200k MST / 100k iVAT), which belongs in a
performance/systems venue, not as an algorithmic claim.

---

## 4. Variable-depth Lin-Kernighan (`lk_search_vd`)

**Verdict — not novel (and it underperformed).** Variable-depth sequential search
*is* Lin-Kernighan (**[LK73]**, in the prior doc); LKH (Helsgaun) is the reference
implementation. Our restricted reverse-suffix-from-anchor form is a weaker subset
and our findings say so. No claim.

---

## 5. Consolidated novelty table

| Item (this stream) | Verdict | Closest prior art |
|---|---|---|
| Uncrossing pre-pass on top-k longest edges (2-opt/Or-opt) | **not novel** | Flood 1956; **Croes 1958 (2-opt = uncrossing)**; **van Leeuwen–Schoone 1981 (tour-untangling)**; Untangle-ineffectiveness COCOA 2023; Or 1976; Bentley 1992 |
| Explicit GPU geometric crossing detection driving the moves | engineering micro-delta | GPU 2-opt (2011+); GPU segment-intersection kernels; subsumed by uncrossing-2-opt |
| VAT/single-linkage cut as the TSP **partitioner** | **not novel** (substitution in a crowded paradigm) | **Deep-clustering bisection 2024**; MST-clustering TKDE 2009; Karp 1977; single-linkage ≡ MST-cut |
| Dual-VAT composition (dual-source Prim → 2 VAT paths → join) | **compositional/engineering** | Karp 1977; **Guttmann-Beck 2000 (endpoint stitch)**; cluster-first-route-second; DualOpt 2025 (name only) |
| N×M GPU cycle-merge join | not novel | subtour patching / tour merging (Karp; Cook–Seymour) |
| GPU VAT/iVAT on unified memory | **systems/perf, not algorithmic** | Bezdek–Hathaway 2002; Havens–Bezdek 2012; GPU MST |
| Variable-depth LK | not novel | Lin–Kernighan 1973; LKH |

**Where the theory backs our own honest findings.** (1) Standalone uncrossing is a
poor approximation — proven Ω(n)/Ω(√n) (COCOA 2023), measured +37…+56% here.
(2) LKH-class quality needs the full LK machinery — our restricted variable-depth
LK underperforming is consistent with why LKH is a large codebase. (3) A good
construction is not automatically a good warm start and the local search dominates
quality — Johnson–McGeoch 1997, and every sweep here (the polish, not the top-k or
the join, sets the floor).

## 6. If any of this were to be written up

Only **one** framing survives a targeted search, and it is a *systems* paper:
1. **A GPU/unified-memory VAT–iVAT systems paper** — the scaling and
   transfer-collapse results on the GB10 coherent memory, positioned squarely as
   engineering over prior-art algorithms.

The **partitioner is not a contribution** — clustering-based (incl. recursive
-bisection and MST/single-linkage) partitioning for D&C TSP is established
(deep-clustering 2024; MST-clustering TKDE 2009; Karp 1977). A "VAT/single-linkage
as a TSP partitioner" paper would need to *beat* POPMUSIC and space-filling
**and** show the VAT cut beats k-means / recursive-bisection partitioners in an
ablation — otherwise it is a substitution, not a result. The uncrossing pre-pass,
the dual-source merge, and variable-depth LK are likewise **not** independently
novel; describe them as applied prior art with an empirical study, never as
contributions.

## References
New this review (others in `docs/bibliography.md`): Flood 1956
(doi:10.1287/opre.4.1.61); Croes 1958 (doi:10.1287/opre.6.6.791); van
Leeuwen–Schoone 1981 (WG '81, pp. 87–98; RUU-CS-80-11); Tour-untangling
ineffectiveness (COCOA 2023, doi:10.1007/978-3-031-49815-2_1) [metadata-only];
"Performance of efficient variants of the 2-Opt heuristic" (DAM 2025,
S0166218X2500294X); Or 1976 (thesis); Highly-Parallel GPU TSP
(doi:10.1007/978-3-642-18466-6_31); DualOpt 2025 (arXiv:2501.08565); Deep
clustering of the TSP (*Comput. Oper. Res.* 2024, S0305054824000200); D&C
MST-based clustering (IEEE TKDE 2009, doi:10.1109/TKDE.2009.37); hyper-tour
cluster-first large-scale TSP (arXiv:2510.20169). PDFs not committed (egress
blocks scholarly hosts).
