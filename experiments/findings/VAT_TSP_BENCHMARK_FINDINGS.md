# VAT warm starts and VAT-cluster-blocking vs Lin-Kernighan (LKH) — findings

**Author:** Scott Phillips **· Date:** 2026-07-11
**Code:** `experiments/vat_tsp_benchmark.py` **· Figure:** `experiments/figures/vat_tsp_benchmark.png`
**Baseline:** LKH (Lin-Kernighan-Helsgaun) via `elkai` (optional: `pip install .[experiments]`)
**Builds on:** `VAT_TSP_WARMSTART_FINDINGS.md`

> There is no Lin-Kernighan implementation in this repo, so the real solver
> (`elkai` = LKH) is used as the baseline; the script guards the import and falls
> back to "best of our methods" if it is absent. Instances are self-contained
> synthetic families (no TSPLIB) at n ≈ 50 / 500 / 5000, so the cluster-blocking
> strategy has real structure to exploit. Everything runs on one integer distance
> matrix (what LKH consumes), so every method is scored on the same objective.

## Headline

1. **Closed-tour warm start: VAT is a weak *tour* constructor.** Its raw
   closed-tour init is poor (the open VAT path closed into a cycle adds one long
   wrap edge, worsening with n). After 2-opt it is competitive but not best —
   greedy-edge and MST double-tree tie or beat it, and LKH beats everyone. VAT's
   warm-start value is for the *open path* (the seriation objective, prior
   findings), not the closed tour.
2. **The real result — VAT-cluster-blocking.** Find blocks, solve each with LKH,
   then an **optimized block-to-block stitch + one global 2-opt polish**. This
   beats flat VAT+2-opt on every instance — reaching within **~5–9% of LKH at
   n≤500**, and at **n=5000 beating flat 2-opt (−1 to −4%) in seconds where flat
   LKH is impractical (7–15+ min)**. That is the divide-and-conquer payoff, and
   the direct answer to "find the blocking, then optimize the connections."

## 1. Warm start + 2-opt vs LKH (% over the LKH tour)

Closed-tour TSP; `init` = construction only, `+2opt` = after 2-opt. LKH ≈ optimal.

**n=50**

| method | init % | +2opt % |
|---|---|---|
| random (×3) | – | 4.2–6.7 |
| nearest-neighbour | 7–20 | 0.2–10.1 |
| greedy-edge | 5–18 | 2.5–5.7 |
| MST double-tree | 22–35 | 4.3–12.0 |
| VAT (free) | 15–57 | 0.2–10.8 |

**n=500** (blobs / uniform / moons)

| method | init % | +2opt % |
|---|---|---|
| random (×3) | – | 9.4 / 12.6 / 11.3 |
| nearest-neighbour | 27.9 / 27.5 / 29.8 | 6.1 / 8.8 / 10.7 |
| greedy-edge | 12.0 / 20.0 / 23.1 | **4.3 / 6.0 / 10.4** |
| MST double-tree | 33.5 / 34.5 / 32.2 | 8.2 / 9.0 / 7.5 |
| VAT (free) | **111.8 / 138.4 / 203.4** | 7.6 / 10.9 / 13.1 |

Reads: (i) **LKH wins outright** — every cheap method sits 4–13% above it after
2-opt. (ii) **VAT's raw closed-tour init is by far the worst** (90–200% over LKH
at n=500): closing its open path adds one huge return edge, and the effect grows
with n (15–57% at n=50 → 90–200% at n=500). (iii) **After 2-opt VAT is
mid-pack** — greedy-edge is generally the best cheap construction; VAT no longer
stands out. So for *closed-tour* TSP, VAT is not a compelling warm start; its
warm-start value is the open-path/seriation case (prior findings).

## 2. Cluster-blocking + optimized block-to-block connections

"Cluster-first, route-second": find B blocks, solve each block's sub-TSP (LKH for
blocks ≤ 700, else NN+2-opt), then stitch. Three stitch levels:
**naive** (block-order concatenation), **opt** (order blocks by a TSP over their
endpoints + choose each block's orientation by a cyclic DP), **opt+polish** (one
global 2-opt). Blocking by VAT single-linkage cut or balanced maximin.
`t_par` = max single-block solve + stitch (+ polish) — the parallel-proxy
wall-clock, since blocks are independent.

**n=500, % over LKH**

| strategy | blobs | uniform | moons |
|---|---|---|---|
| flat VAT + 2-opt | 7.6 | 10.9 | 13.1 |
| vat block **naive** | 21.3 | 51.6 | 30.7 |
| vat block **opt** | 9.7 | 14.4 | 19.6 |
| vat block **opt+polish** | **5.3** | **4.5** | **8.9** |
| maximin block opt+polish | 4.9 | 10.8 | 10.8 |

Three clean findings:

- **The block-to-block optimization is essential.** Naive concatenation is 3–5×
  worse than the optimized stitch (uniform 51.6% → 14.4%; blobs 21.3% → 9.7%) —
  the same seam problem as naive block-VAT, fixed the same way (order + connect).
- **opt+polish beats flat VAT+2-opt on every instance** (blobs 5.3 vs 7.6,
  uniform 4.5 vs 10.9, moons 8.9 vs 13.1). Solving each small block with LKH gives
  near-optimal *intra-block* structure that a flat 2-opt never finds; the polish
  then only has to fix the seams. This is a genuine win, not just a tie.
- **VAT vs balanced blocking is a quality/parallelism trade.** VAT single-linkage
  blocks are unbalanced (one fat block + singletons), which gives strong quality
  but a larger max-block solve (higher `t_par`); maximin's balanced blocks
  parallelize better (lower `t_par`) at a small quality cost on the harder
  families. VAT-found blocking is competitive-to-best on quality.

### Scale: n=5000 (blobs), where flat LKH is impractical

Flat LKH does not scale and `elkai` exposes no effort cap: one run is ~433 s on
uniform n=5000 and was **killed after >15 min on clustered blobs**. So above
n=2000 the flat-LKH reference is skipped and the reference becomes flat VAT+2-opt
(3.2 s). Blocking uses per-block LKH for blocks ≤ 700, else NN+2-opt.

| strategy (blobs n=5000) | % over flat VAT+2opt | t_par |
|---|---|---|
| flat VAT + 2-opt (reference) | 0.0 | 3.2 s |
| vat block naive | +5.6 | 0.15 s |
| vat block opt | +3.5 | 0.15 s |
| **vat block opt+polish** | **−1.2** | 1.9 s |
| maximin block naive | +8.7 | 15.7 s |
| maximin block opt | −0.7 | 15.7 s |
| **maximin block opt+polish** | **−3.6** | 17.7 s |

- **Both opt+polish variants beat flat VAT+2-opt** (vat −1.2%, maximin −3.6%),
  confirming the n=500 finding at 10× the size.
- **maximin (balanced) blocking gives the best quality** (−3.6%): each ~625-node
  block is small enough to be solved near-optimally by LKH. Its 17.7 s is the
  slowest single block's LKH solve — but that is **~25–50× faster than flat LKH**
  (7–15+ min) at better-than-flat-2-opt quality.
- **VAT single-linkage blocking is far faster** (1.9 s) at a smaller gain
  (−1.2%): its fat block exceeds the LKH cap and falls back to 2-opt. This is the
  quality/parallelism trade — block imbalance is the honest cost of single-linkage
  blocking, and choosing more/balanced blocks trades some quality for parallel
  speed.
- **Scale takeaway:** divide-and-conquer blocking produces a tour *better than
  flat 2-opt in seconds*, exactly where the strong flat solver (LKH) is
  minutes-to-impractical.

## Verdict

- **VAT is a poor closed-tour constructor but an excellent block finder.** The
  headline warm-start value of VAT for TSP is *not* seeding a single tour (the
  wrap edge kills it); it is (a) the open-path seriation start of the prior
  findings, and (b) supplying the *blocks* for a divide-and-conquer solve.
- **VAT-blocks + per-block LKH + optimized stitch + polish** is the real result:
  it beats flat 2-opt and reaches within a few % of LKH at a fraction of LKH's
  wall-clock on large n, because the expensive global solve is replaced by many
  cheap independent block solves plus a light global polish. This mirrors — and
  extends to TSP — the repo's stitched divide-and-conquer VAT story.
- **Honest limits:** LKH is still the quality winner; the blocked tour is an
  approximation whose gap is smallest on separable (blobs) data — where the
  optimal tour is itself block-contiguous — and larger on structureless/uniform
  data. Block imbalance from VAT cuts limits the parallel speedup.

**Status: research spike, not shipped.** Open: LKH with an explicit warm-started
`INITIAL_TOUR` (does the VAT/blocked tour help LKH itself?); multi-seed error
bars; a genuinely parallel (not proxy) block-solve timing; and recursive blocking
for very large n.

## References
- **[LKH]** K. Helsgaun, "An effective implementation of the Lin–Kernighan
  traveling salesman heuristic," *EJOR* 126(1):106–130, 2000.
  doi:10.1016/S0377-2217(99)00284-2. (`elkai` wraps LKH-2.)
- **[DoubleTree]** D. J. Rosenkrantz, R. E. Stearns, P. M. Lewis II, "An Analysis
  of Several Heuristics for the TSP," *SIAM J. Comput.* 6(3):563–581, 1977.
  doi:10.1137/0206041.
- **[ClusterTSP]** cluster-first-route-second / clustered TSP: e.g. G. Laporte
  and F. Semet, and the GTSP literature; the block-VAT connection is the repo's
  own divide-and-conquer VAT (`STITCHED_VAT_FINDINGS.md`).
- **[MST-SL]** J. C. Gower, G. J. S. Ross, "Minimum Spanning Trees and Single
  Linkage Cluster Analysis," *JRSS-C* 18(1):54–64, 1969. doi:10.2307/2346439.
- **[VAT]** J. C. Bezdek, R. J. Hathaway, "VAT," *Proc. IJCNN* 2002, 2225–2230.
  doi:10.1109/IJCNN.2002.1007487.

## Files
- `experiments/vat_tsp_benchmark.py` — integer-matrix instances, LKH (elkai)
  reference (cached), construction warm starts, and the cluster-blocking pipeline
  (VAT / maximin blocking, block sub-TSP, endpoint-TSP block ordering, orientation
  DP, global polish) + reports and figure.
- `experiments/figures/vat_tsp_benchmark.png`.
