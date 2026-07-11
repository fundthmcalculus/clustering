# Prior-Art & Novelty Review — the VAT ↔ TSP thread

**Reviewer role:** independent prior-art / novelty assessment for the research in
`experiments/vat_tsp*.py` and `experiments/findings/VAT_TSP*_FINDINGS.md`.
**Date:** 2026-07-11. Companion: `docs/bibliography.md` (§6, added with this review).

**Scope.** Three claims from the VAT↔TSP experiments: (1) the VAT/MST ordering is
a minimum-Hamiltonian-path (seriation-TSP) solution and a *warm start* for TSP;
(2) seeding an Ant System's pheromone from the VAT/MST ordering is a useful
"hot start"; (3) a VAT-cluster-blocking divide-and-conquer TSP solver — find
blocks, solve each, and optimize the block-to-block connections.

**Method & a hard constraint.** Prior art was gathered by parallel web search and
page-fetching. **PDFs could not be downloaded/committed:** this session's egress
policy denies scholarly hosts (arXiv, IEEE, Springer, Elsevier all return 403 to
the proxy — see the recorded `connect_rejected` for `arxiv.org:443`). Every
reference below is therefore cited by DOI / stable URL and verified via
search-result and abstract metadata (the same "verified via metadata only"
standard the existing `docs/bibliography.md` uses for paywalled entries). Items
that could not be fully verified (mostly IEEE conference papers behind the same
403) are flagged **[unverified DOI]**. Open-access PDFs are noted so they can be
retrieved in an environment without the egress restriction and committed to
`docs/papers/`.

> **Bottom line.** As with the FCM/VAT core (see `docs/novelty-review.md`), the
> primitives here are all prior art and are used faithfully. The seriation↔TSP
> identity is 50 years old (Lenstra 1974); LKH is the reference solver;
> heuristic-seeded ACO pheromone is standard (ACS 1997) and *MST*-seeded
> pheromone specifically already exists (Dai et al. 2009); and cluster-first /
> clustered-TSP with an endpoint-based stitch is a mature line (Chisman 1975;
> Guttmann-Beck et al. 2000). The defensible contributions are **narrow and
> compositional**, and two of the experiments' own honest findings (VAT is a poor
> *closed-tour* start; a shorter tour ≠ better clustering) are corroborated by
> that literature rather than contradicted by it.

---

## 1. Seriation as TSP, and VAT/MST orderings

**Lineage (all real, verified):**
- **[Lenstra1974]** J. K. Lenstra, "Clustering a Data Array and the Traveling-
  Salesman Problem," *Operations Research* 22(2):413–414, 1974.
  doi:10.1287/opre.22.2.413. — the original identity: reordering a data array to
  minimize the sum of adjacent-row dissimilarities *is* a TSP.
- **[HubertBaker1978]** L. J. Hubert & F. B. Baker, "Applications of Combinatorial
  Programming to Data Analysis: The TSP and Related Problems," *Psychometrika*
  43(1):81–91, 1978. doi:10.1007/BF02294091. OA-PDF available.
- **[ClimerZhang2006]** S. Climer & W. Zhang, "Rearrangement Clustering: Pitfalls,
  Remedies, and Applications," *JMLR* 7:919–943, 2006. Open access
  (jmlr.org/papers/volume7/climer06a). — identifies the tour-vs-path pitfall and
  the **dummy-city** reduction (open Hamiltonian path via a zero-distance node).
- **[Hahsler2008]** M. Hahsler, K. Hornik & C. Buchta, "Getting Things in Order:
  An Introduction to the R Package seriation," *J. Statistical Software*
  25(3):1–34, 2008. doi:10.18637/jss.v025.i03. Open access. — states an object
  order *is* a Hamiltonian path and its `"TSP"` method minimizes path length;
  formalizes the seriation loss/merit measures (anti-Robinson events, stress,
  path length, 2-SUM).
- Context: **[Liiv2010]** (seriation historical overview, *SADM* 3(2):70–91,
  doi:10.1002/sam.10071), **[Behrisch2016]** (matrix-reordering survey, *CGF*
  35(3), doi:10.1111/cgf.12935, OA on HAL), **[WilkinsonFriendly2009]** (history
  of the cluster heat map, *Am. Statistician* 63(2):179–184).
- Note: the **`seriation` R package already bundles both `VAT` and a `"TSP"`
  ordering method** — so VAT and seriation-TSP coexist in one framework, but as
  *separate* methods; nothing in that literature frames VAT's ordering itself as
  a TSP path.

**What is prior art.** The seriation↔TSP objective, the open-path/dummy-city
reduction, and evaluating orderings by path length / anti-Robinson criteria — all
established. VAT (Bezdek & Hathaway 2002) and iVAT (Havens & Bezdek 2012) as
reorderings are prior art (already in `docs/bibliography.md`).

**Defensible novelty (narrow).** No located source explicitly states that **VAT's
Prim-MST *visit order* is a seriation-TSP Hamiltonian path**, nor draws the
distinction between that visit order and the **double-tree shortcut tour**
(Rosenkrantz–Stearns–Lewis 1977; fast variant Deineko–Tiskin 2007) whose cost is
bounded by 2·MST. The experiments measure exactly this (VAT/MST ratio vs the 2×
bound) and extend it to **non-metric** dissimilarity, where the metric bound is
void — a regime the seriation-TSP literature (which assumes a metric) does not
treat. This is a genuine but *connective/observational* contribution, not a new
algorithm.

---

## 2. TSP baselines, and the "is VAT a good warm start?" question

**Solvers & local search (verified):** Lin–Kernighan **[LK73]** (*Oper. Res.*
21(2):498–516, doi:10.1287/opre.21.2.498); **LKH** Helsgaun **[Hel00]** (*EJOR*
126(1):106–130, doi:10.1016/S0377-2217(99)00284-2) and **LKH-2** **[Hel09]**
(*Math. Prog. Comp.* 1(2–3):119–163, doi:10.1007/s12532-009-0004-6); 2-opt
**[Croes58]**; 3-opt **[Lin65]**; Or-opt **[Or76]** (thesis).

**Constructions (verified):** double-tree/2-approx & NN bounds **[RSL77]** (*SIAM
J. Comput.* 6(3):563–581, doi:10.1137/0206041); Christofides 3/2 **[Chr76]**;
greedy-edge / engineering reference **[Bentley92]** (*ORSA J. Comput.*
4(4):387–411, doi:10.1287/ijoc.4.4.387); space-filling curve **[PB89]**
(*JACM* 36(4):719–737) and **[BartholdiPlatzman1982]** (*ORL* 1(4):121–125).

**Warm-start / decomposition (verified):** **POPMUSIC** Taillard & Helsgaun
**[TH19]** (*EJOR* 272(2):420–429, doi:10.1016/j.ejor.2018.06.039) — initial tour
+ candidate edges for LKH in near-linear time; **tour merging** Cook & Seymour
**[CS03]** (*INFORMS J. Comput.* 15(3):233–248); Held–Karp 1-tree bound
**[HK70]** (*Oper. Res.* 18(6):1138–1162).

**Benchmarks & conventions (verified):** TSPLIB **[Rein91]**; the 8th DIMACS TSP
Challenge and its companion analysis Johnson & McGeoch **[JM97]/[JM02]** (the
authoritative %-over-Held-Karp tables); Waterloo national/VLSI/Art-TSP sets.

**The result the warm-start claim must confront.** Johnson & McGeoch **[JM97]**
established that (a) a *better construction is not necessarily a better warm start*
— **greedy beats nearest-neighbour as a 2-/3-opt start** even though both are
mediocre constructions — and (b) LKH is **famously insensitive to the starting
tour** (its α-nearness candidate structure reconstructs regardless). Both are
directly relevant:

- The experiments' Part 1 finding — VAT's raw **closed-tour** init is the *worst*
  (a long wrap edge), yet after 2-opt it is mid-pack, and LKH beats everything —
  is exactly what this literature predicts. So "VAT is a good closed-tour warm
  start" is **not supportable**, and the experiments already say so.
- Any positive warm-start claim survives only where it is *cheap* (2-opt/Or-opt
  from a free ordering), **not** against LKH.

**Gap / novelty.** "MST informs TSP" is thoroughly established (RSL double-tree,
Christofides, LKH's α-nearness, Held–Karp 1-tree). Using a **VAT/iVAT ordering as
a construction/warm start**, evaluated against NN/greedy/double-tree under
2-opt+Or-opt and vs LKH, was not found in the literature — but the honest verdict
is that it is a *free byproduct* of a VAT clustering pipeline that is
**competitive, not superior**, to standard cheap constructions. Modest.

---

## 3. ACO and pheromone warm-starts

**Verified:** Ant System **[AntSystem96]** (Dorigo et al., *IEEE TSMC-B*
26(1):29–41, doi:10.1109/3477.484436); Ant Colony System **[ACS97]** (Dorigo &
Gambardella, *IEEE TEC* 1(1):53–66, doi:10.1109/4235.585892) — **already sets
τ₀ = 1/(n·L_NN)** from the nearest-neighbour tour, i.e. heuristic-scaled initial
pheromone; MAX-MIN **[MMAS00]** (Stützle & Hoos, *FGCS* 16(8):889–914); the Dorigo
& Stützle 2004 book.

**Closest competitor (verified):** **[DaiJi2009]** Q. Dai, J. Ji, C. Liu, "An
effective initialization strategy of pheromone for ant colony optimization,"
*Proc. BIC-TA 2009* — **initializes the ACO pheromone matrix from minimum-
spanning-tree information** and reports faster/better convergence on TSP. This is
the single closest prior art to the experiments' "seed pheromone from the
VAT/MST ordering." [unverified DOI — IEEE doc 5338067; identity confirmed via
Semantic Scholar and ResearchGate.] Related: **[NPI-ACS17]** non-uniform informed
pheromone [unverified DOI]; **[Stodola22]** clustering-guided adaptive ACO (*Swarm
& Evol. Comput.* 70:101056, doi:10.1016/j.swevo.2022.101056).

**Anytime methodology (verified):** López-Ibáñez & Stützle, "Automatically
improving the anytime behaviour…" *EJOR* 235(3):569–582, 2014
(doi:10.1016/j.ejor.2013.10.043); "ACO on a Budget of 1000" (ANTS 2014, LNCS
8667). These define the *solution-quality-over-time* / fixed-budget framing that a
"hot start leads throughout" claim should use.

**Gap / novelty.** A generic "warm-started ACO" claim will **not** survive review:
ACS already scales τ₀ by a construction heuristic, and Dai et al. 2009 already
seed from the MST. The only remaining niche is narrow: seeding specifically from a
**VAT/iVAT seriation ordering** (vs raw MST edge weights), for **open-path /
seriation** ACO, framed with an explicit **anytime** evaluation. Position against
[DaiJi2009] and [ACS97]; do not overclaim.

---

## 4. Clustered / divide-and-conquer TSP (the Part-2 core)

**Verified prior art:**
- **[Chisman1975]** J. A. Chisman, "The clustered traveling salesman problem,"
  *Computers & Oper. Res.* 2(2):115–119, doi:10.1016/0305-0548(75)90015-5. —
  origin of the **Clustered TSP (CTSP)**: partition into clusters, **each visited
  contiguously**; solved by adding a large constant to inter-cluster edges. This
  is exactly the "each block is a contiguous run" structure of the experiments.
- **[GuttmannBeck2000]** N. Guttmann-Beck, R. Hassin, S. Khuller, B. Raghavachari,
  "Approximation Algorithms with Bounded Performance Guarantees for the CTSP,"
  *Algorithmica* 28(4):422–437, 2000, doi:10.1007/s004530010045 (ratio 2.75). —
  **the closest prior art to the experiments' "stitch."** Decomposes CTSP into
  (a) a **cluster-ordering** problem and (b) choosing per-cluster **entry/exit end
  vertices**, connecting clusters by intra-cluster **Hamiltonian paths with fixed
  endpoints**. This is precisely the experiments' *endpoint-distance TSP over
  blocks + per-block orientation choice + block sub-TSP*.
- **[AnilyBramelHertz1999]** "A 5/3-approximation for the clustered TSP tour and
  path problems," *Oper. Res. Lett.* 24(1–2):29–35,
  doi:10.1016/S0167-6377(98)00046-7 — same "solve blocks, connect endpoints"
  template.
- **[Ding2007]** C. Ding, Y. Cheng, M. He, "Two-Level Genetic Algorithm for CTSP
  with Application in Large-Scale TSPs," *Tsinghua Sci. & Tech.* 12(4):459–465,
  doi:10.1016/S1007-0214(07)70068-8 — **closest on intent**: uses CTSP as a
  **divide-and-conquer for large plain TSP** (level 1 = sub-tour per cluster,
  level 2 = sequence the sub-tours). Same two-level shape; a GA replaces the
  experiments' endpoint-TSP + orientation DP.
- Cluster-first-route-second origins: **[FisherJaikumar1981]** (*Networks*
  11(2):109–124), **[GillettMiller1974]** sweep (*Oper. Res.* 22(2):340–349).
- Geometric divide-and-conquer: **[Karp1977]** probabilistic partitioning
  (*Math. Oper. Res.* 2(3):209–224, doi:10.1287/moor.2.3.209).
- Large-scale frontier: POPMUSIC **[TH19]**; **[Taillard2022]** linearithmic
  heuristic (*EJOR* 297(2):442–450); neural D&C — **Learning-to-Delegate**
  (NeurIPS 2021), **H-TSP** (AAAI 2023, ~3.4% gap at 10k), **GLOP** (AAAI 2024,
  "global partition + local construction," ~5% at 100k).
- Background: a VAT single-linkage cut **is** an MST-based partition (single-
  linkage ≡ Kruskal MST minus the k−1 heaviest edges); the clustering front-end
  is standard.
- GTSP (visit **one** node per cluster) is a *different* problem — cite the 2024
  GTSP survey to distinguish it from CTSP contiguity.

**Gap / novelty & the precise delta.** The paradigm ("cluster-first, route-second,
then order/connect blocks") is textbook, the contiguity constraint *is* CTSP, and
the **specific endpoint/orientation stitch is Guttmann-Beck et al. (2000)**. So
the Part-2 pipeline is **not novel as a concept.** The defensible, narrow deltas:

1. **The partitioner.** Using a **VAT/iVAT single-linkage cut** (a cluster-tendency
   seriation) to *define the TSP blocks* was not found in the CTSP / D&C-TSP
   literature. In classical CTSP the clustering is *given as part of the problem*;
   here it is *imposed as a solution heuristic* for a plain TSP — closer to Karp/
   Ding-style D&C, but with a VAT/single-linkage cut rather than a geometric grid
   or GA clusters.
2. **The stitch realization.** An explicit **endpoint-distance TSP for block
   sequencing + a cyclic 2-state DP over block orientations + a global 2-opt
   polish** is a concrete, lightweight heuristic instantiation of the Guttmann-
   Beck subproblem — engineering novelty, not a new result. (Guttmann-Beck use
   matching/enumeration with proven ratios; the experiments use heuristics with
   no guarantee.)
3. **The empirical finding** that this composition beats a flat 2-opt and reaches
   near-LKH quality far faster at scale, *and where it fails* (structureless/
   uniform data; VAT block imbalance limiting parallelism).

Frame novelty as **composition + the VAT partitioner**, cite Guttmann-Beck (2000),
Anily-Bramel-Hertz (1999), and Ding (2007) as nearest neighbours, and distinguish
CTSP (contiguity) from GTSP (one-per-cluster).

---

## 5. Consolidated gaps

- **No VAT/iVAT-as-TSP framing exists** — the connective observation (VAT visit
  order = seriation-TSP path; iVAT transform = seriating on the single-linkage
  cophenetic/ultrametric distance) is unoccupied but *observational*.
- **Non-metric regime is untreated** by seriation-TSP and the 2·MST bound — a real
  niche (VAT routinely runs on non-metric D; the experiments already probe it).
- **Warm-start-against-LKH is the hard open question** (LKH is start-insensitive
  [JM97]); the honest, supportable claim is warm start *for cheap local search*.
- **MST-seeded ACO already exists** (Dai 2009); only VAT-seriation seeding + open-
  path + anytime framing is arguably new.
- **The endpoint stitch is Guttmann-Beck (2000)**; only the VAT partitioner + the
  specific DP realization + the empirical scaling study are the delta.

## 6. Novelty — honest summary

| Claim | Verdict | Closest prior art / delta |
|---|---|---|
| VAT visit order = seriation-TSP path; vs 2·MST double-tree; non-metric | connective/observational; **incremental** | Lenstra 1974, Climer-Zhang 2006, RSL 1977; delta = VAT-specific + non-metric |
| VAT as a TSP warm start | **weak/none** for closed tours; free-but-competitive for cheap 2-opt | JM97 (greedy>NN; construction≠warm-start), Bentley 92 |
| VAT/MST-seeded ACO "hot start" | **narrow** | ACS 1997 (τ₀ from NN), **Dai 2009 (MST-seeded pheromone)** |
| VAT-cluster-blocking + endpoint/orientation stitch | **compositional/engineering** | **Guttmann-Beck 2000** (stitch), Ding 2007 (D&C-for-large-TSP), Chisman 1975 (CTSP) |

The strongest, most defensible contribution is **Part 2 as a composition**: a
VAT/single-linkage partitioner feeding a cheap CTSP-style endpoint+orientation
stitch and a polish, characterized empirically (beats flat 2-opt; scales; fails
on structureless data) — provided it is positioned against Guttmann-Beck and
benchmarked properly (§7).

## 7. Benchmarks to match or exceed

**Adopt the standard protocol** (DIMACS / Johnson-McGeoch): report **% over
optimum (Concorde) or over the Held–Karp lower bound**, plus **normalized
runtime**, on **both** random-uniform Euclidean instances *and* real TSPLIB/VLSI
instances, averaged over multiple seeds with significance tests
(Wilcoxon/Friedman). The experiments' current "% over LKH on synthetic families"
is a reasonable start but under-powered for a paper.

**Part 1 targets (canonical avg % over Held–Karp on uniform Euclidean; from
JM97/Bentley — match these):** space-filling ~25–40%; NN ~24–26%; greedy-edge
~14–16%; Christofides ~10%; **2-opt ~5%; Or-opt ~4–6%; 3-opt ~3%; LK ~1.5–2%;
LKH ~0–0.1%**. The experiments' VAT+2-opt at ~7–13% over LKH is *worse* than the
~5% canonical 2-opt figure — closing that gap needs neighbour lists, don't-look
bits, full (bidirectional, s≥1) Or-opt, and multiple passes.

**Part 2 competitors to match/exceed:** flat 2-opt (weak baseline — already
beaten); space-filling-curve (~15–25%); **POPMUSIC** (low single-digit % gap,
near-linear time) and **Taillard 2022** (linearithmic, tested to 10⁹);
neural D&C **H-TSP** (~3.4% @10k) and **GLOP** (~5% @100k). To be competitive the
blocking solver must report %-over-LKH **and runtime** at **n = 1k / 5k / 10k /
100k** on uniform + TSPLIB, and ideally beat space-filling and approach POPMUSIC.

**Standard instances at the experiments' sizes** (published optima, for a real —
not synthetic — test even though the current experiments use synthetic families):
eil51=426, berlin52=7542 (n≈50); rd400=15281, pcb442=50778, lin318=42029
(n≈400–500); fnl4461=182566, rl5915=565530 (n≈5000); pla85900 for a large anchor.

**Ablations a reviewer will require:** partition method (VAT single-linkage vs
maximin vs k-means vs grid vs space-filling); number/size of blocks; orientation
DP on/off; global polish on/off; block sub-solver (LKH vs 2-opt); metric vs
non-metric D.

## 8. What a reviewer will demand first

1. Benchmark against **POPMUSIC and a space-filling-curve baseline**, not just flat
   2-opt, and report **runtime vs n** to 10k–100k (the whole point of D&C).
2. State the **Guttmann-Beck (2000)** relationship explicitly and give the precise
   delta (VAT partitioner + heuristic DP realization + no approximation guarantee).
3. For the ACO hot start, compare against **Dai (2009) MST-seeded pheromone** and
   **ACS τ₀**, and use an **anytime (SQT) curve**, not a single final number.
4. Drop or heavily qualify any "VAT is a good TSP warm start" phrasing for closed
   tours (JM97 predicts, and the data confirms, it is not).
5. Multiple seeds + significance; % over optimum/Concorde on real instances.

---

## References
All entries above, with the extra bibliographic detail and OA-PDF links, are added
to `docs/bibliography.md` §6 (VAT↔TSP, seriation, and clustered TSP). PDFs are not
committed — the session egress policy blocks scholarly hosts (see the header note);
retrieve the OA-PDFs listed in the bibliography from an unrestricted environment
into `docs/papers/` when possible.
