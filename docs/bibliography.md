# Bibliography — Prior Art for VAT/iVAT, IVATMeans, and Fuzzy C-Means

Curated references for the novelty review of `tribble-clustering`. Every entry
below is a **real, verifiable publication**; DOIs / stable URLs are given so they
can be checked against the source of record. Open-access PDFs that could be
retrieved are committed under `docs/papers/`; any other retrieved PDFs are
cached in `docs/sources/` (git-ignored; do not commit).

The entries are grouped by the role they play in the novelty analysis. See
`docs/novelty-review.md` for how each maps onto the code.

---

## 1. The VAT / iVAT family (direct lineage of this code)

### [VAT] Bezdek & Hathaway (2002) — original VAT
J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment of (cluster)
tendency," *Proc. Int. Joint Conf. on Neural Networks (IJCNN)*, Honolulu, HI,
2002, vol. 3, pp. 2225–2230. doi:10.1109/IJCNN.2002.1007487.
- **Relevance:** Defines VAT: build the pairwise dissimilarity matrix, reorder it
  with a **modified Prim's MST** traversal (seed = the globally most-distant pair),
  and display the reordered dissimilarity image (RDI); clusters appear as dark
  diagonal blocks. This is exactly what `pvat.vat_prim_mst` / `compute_vat` and
  the C kernels in `pcvat.pyx` compute.

### [iVAT] Wang, Nguyen, Bezdek, Leckie & Ramamohanarao (2010) — original iVAT + aVAT
L. Wang, U. T. V. Nguyen, J. C. Bezdek, C. A. Leckie, and K. Ramamohanarao,
"iVAT and aVAT: Enhanced Visual Analysis for Cluster Tendency Assessment,"
*Advances in Knowledge Discovery and Data Mining (PAKDD 2010)*, LNCS vol. 6118,
pp. 16–27, Springer, 2010. doi:10.1007/978-3-642-13657-3_5.
- **Relevance:** Introduces the **path-based (minimax) distance transform** on the
  VAT ordering — the "improved" VAT — and **aVAT**, which *automatically determines
  the number of clusters* from the reordered image. This is the closest ancestor of
  `IVATMeans`' automatic-`k` idea.

### [iVAT-fast] Havens & Bezdek (2012) — O(n²) recursive iVAT
T. C. Havens and J. C. Bezdek, "An Efficient Formulation of the Improved Visual
Assessment of Cluster Tendency (iVAT) Algorithm," *IEEE Trans. Knowledge and Data
Engineering*, vol. 24, no. 5, pp. 813–822, May 2012. doi:10.1109/TKDE.2011.33.
- **Relevance:** Gives the exact recurrence implemented in `pvat.compute_ivat` and
  `pcvat._compute_ivat_kernel_*`:
  `D'[r,c] = max(D*[r,j], D'[j,c])` where `j = argmin_{k<r} D*[r,k]`.
  Reduces iVAT from O(n³) to O(n²). **Cite this for the iVAT recursion itself.**
  PDF: `docs/papers/Havens_Bezdek_2012_iVAT_efficient.pdf`.

### [SpecVAT/partition] Wang, Leckie, Bezdek & Ramamohanarao / Bezdek et al. (2010) — VAT-based data partitioning
L. Wang, C. Leckie, K. Ramamohanarao, and J. Bezdek, "Automatically Determining
the Number of Clusters in Unlabeled Data Sets" / "Enhanced Visual Analysis for
Cluster Tendency Assessment and Data Partitioning," *IEEE Trans. Knowledge and
Data Engineering*, vol. 22, no. 3, pp. 335–350, 2009/2010.
doi:10.1109/TKDE.2009.135.
- **Relevance:** Goes beyond *tendency* to actually **extract a partition** from the
  reordered matrix and estimate `k` automatically (SpecVAT / spectral embedding +
  VAT). Direct prior art for "VAT/iVAT as a clustering (not just visualization)
  tool." PDF: `docs/papers/Wang_2010_SpecVAT_DataPartitioning_TKDE.pdf`.

### [clusiVAT-conf] Kumar, Palaniswami, Rajasegarar, Leckie, Bezdek & Havens (2013)
D. Kumar, M. Palaniswami, S. Rajasegarar, C. Leckie, J. C. Bezdek, and T. C.
Havens, "clusiVAT: A mixed visual/numerical clustering algorithm for big data,"
*IEEE Int. Conf. on Big Data*, 2013, pp. 112–117. doi:10.1109/BigData.2013.6691561.

### [clusiVAT-journal] Kumar, Bezdek, Palaniswami, Rajasegarar, Leckie & Havens (2016) — **closest competitor**
D. Kumar, J. C. Bezdek, M. Palaniswami, S. Rajasegarar, C. Leckie, and T. C.
Havens, "A Hybrid Approach to Clustering in Big Data," *IEEE Trans. Cybernetics*,
vol. 46, no. 10, pp. 2372–2385, Oct. 2016. doi:10.1109/TCYB.2015.2477416.
- **Relevance:** **The single most important comparison.** clusiVAT = sample the
  data → iVAT-image to estimate `k` → cut to form **single-linkage** clusters →
  extend labels to all points by the **nearest-prototype rule**. `IVATMeans` is the
  same skeleton (iVAT ordering → cut → centroids → assign), differing mainly in
  (a) FCM refinement instead of one nearest-prototype pass, (b) the specific auto-`k`
  gap rule, and (c) exact (non-sampled) full-data iVAT. Any novelty claim must be
  argued *against this paper*.

### [VAT-survey] Kumar & Bezdek (2020) — authoritative survey
D. Kumar and J. C. Bezdek, "Visual Approaches for Exploratory Data Analysis: A
Survey of the Visual Assessment of Clustering Tendency (VAT) Family of Algorithms,"
*IEEE Systems, Man, and Cybernetics Magazine*, vol. 6, no. 2, pp. 10–48, Apr. 2020.
doi:10.1109/MSMC.2019.2961163.
- **Relevance:** The canonical map of the whole VAT family (VAT, iVAT, sVAT, bigVAT,
  clusiVAT, aVAT, SpecVAT, …). Use it to position this work and to make sure no
  variant already claims the exact contribution.

### [ConiVAT] Rathore, Bezdek, Palaniswami et al. (2020)
P. Rathore, J. C. Bezdek, et al., "ConiVAT: Cluster Tendency Assessment and
Clustering with Partial Background Knowledge," arXiv:2008.09570, 2020.
- **Relevance:** Constraint-guided iVAT that builds a *minimum-transitive*
  dissimilarity and addresses VAT/iVAT sensitivity to noise and "bridge" points — a
  known failure mode that also affects `IVATMeans`. PDF:
  `docs/papers/Rathore_2020_ConiVAT.pdf`.

### [bigVAT / sVAT] Hathaway, Bezdek & Huband (2006); Hathaway et al. (2006)
R. J. Hathaway, J. C. Bezdek, and J. M. Huband, "Scalable visual assessment of
cluster tendency for large data sets," *Pattern Recognition*, vol. 39, no. 7,
pp. 1315–1324, 2006. doi:10.1016/j.patcog.2006.02.011.
- **Relevance:** Scalable/sampled VAT — prior art for scaling VAT to large `n`,
  which the `tribble-clustering` performance work (priority-queue MST, C/OpenMP)
  targets by a different (exact-computation, systems) route.

---

## 2. MST / single-linkage foundations (the theory the cut step reduces to)

### [Prim] Prim (1957)
R. C. Prim, "Shortest connection networks and some generalizations," *Bell System
Technical Journal*, vol. 36, no. 6, pp. 1389–1401, 1957.
doi:10.1002/j.1538-7305.1957.tb01515.x.
- **Relevance:** The MST algorithm at the heart of VAT ordering and of the
  priority-queue / compact-active-set implementations in `pvat.py`, `pqvat.py`,
  `pcvat.pyx`.

### [MST↔SL] Gower & Ross (1969)
J. C. Gower and G. J. S. Ross, "Minimum Spanning Trees and Single Linkage Cluster
Analysis," *Journal of the Royal Statistical Society, Series C (Applied
Statistics)*, vol. 18, no. 1, pp. 54–64, 1969. doi:10.2307/2346439.
- **Relevance:** Proves that **all information for single-linkage clustering is
  contained in the MST**. Because VAT ordering is a Prim traversal, cutting the
  ordered off-diagonal (what `get_ivat_levels` does) is *single-linkage clustering*.
  This is the key theoretical citation for framing `IVATMeans`' cut step.

### [Zahn] Zahn (1971)
C. T. Zahn, "Graph-Theoretical Methods for Detecting and Describing Gestalt
Clusters," *IEEE Trans. Computers*, vol. C-20, no. 1, pp. 68–86, Jan. 1971.
doi:10.1109/T-C.1971.223083.
- **Relevance:** Cutting **inconsistent (long) MST edges** to form clusters — the
  classical form of the "abrupt change" / largest-gap cut used in
  `get_ivat_levels`. Reviewers will map the cut heuristic to Zahn directly.

### [gap-statistic] Tibshirani, Walther & Hastie (2001)
R. Tibshirani, G. Walther, and T. Hastie, "Estimating the number of clusters in a
data set via the gap statistic," *J. Royal Statistical Society, Series B*, vol. 63,
no. 2, pp. 411–423, 2001. doi:10.1111/1467-9868.00293.
- **Relevance:** Conceptual relative of the "largest difference in the sorted
  diagonal ⇒ number of clusters" heuristic. Different mechanism, same goal (auto-`k`);
  worth contrasting.

---

## 3. Fuzzy C-Means foundations (the `fcm.py` / `cfcm.pyx` side)

### [Dunn] Dunn (1973)
J. C. Dunn, "A Fuzzy Relative of the ISODATA Process and Its Use in Detecting
Compact Well-Separated Clusters," *Journal of Cybernetics*, vol. 3, no. 3,
pp. 32–57, 1973. doi:10.1080/01969727308546046.
- **Relevance:** Origin of fuzzy c-means (the `m = 2` case).

### [Bezdek-FCM] Bezdek (1981)
J. C. Bezdek, *Pattern Recognition with Fuzzy Objective Function Algorithms*,
Plenum Press, New York, 1981. doi:10.1007/978-1-4757-0450-1.
- **Relevance:** Generalizes FCM to arbitrary fuzzifier `m` and gives the
  alternating-optimization update equations implemented in `fcm.py`
  (`_get_weights`, `_get_v_ij`) and `cfcm.pyx`.

### [FCM++] Stetco, Zeng & Keane (2015)
A. Stetco, X.-J. Zeng, and J. Keane, "Fuzzy C-means++: Fuzzy C-means with effective
seeding initialization," *Expert Systems with Applications*, vol. 42, no. 21,
pp. 7541–7548, 2015. doi:10.1016/j.eswa.2015.05.014.
- **Relevance:** State-of-the-art **FCM seeding** (k-means++-style). This is the
  baseline `IVATMeans`-as-a-seeder must beat or match; the novelty of `IVATMeans`
  is precisely an *alternative, deterministic, structure-aware* seeding.

### [kmeans++] Arthur & Vassilvitskii (2007)
D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of Careful Seeding,"
*Proc. 18th ACM-SIAM Symp. on Discrete Algorithms (SODA)*, 2007, pp. 1027–1035.
- **Relevance:** The dominant seeding baseline for the k-means analogue of
  `IVATMeans` (which also does a nearest-centroid hard assignment in `predict`).

---

## 4. Adjacent auto-`k` / density methods worth a comparison

### [OPTICS] Ankerst, Breunig, Kriegel & Sander (1999)
M. Ankerst, M. M. Breunig, H.-P. Kriegel, and J. Sander, "OPTICS: Ordering Points
To Identify the Clustering Structure," *Proc. ACM SIGMOD*, 1999, pp. 49–60.
doi:10.1145/304182.304187.
- **Relevance:** OPTICS produces an **ordering + reachability plot**; extracting
  clusters from "valleys/peaks" of that 1-D profile is directly analogous to
  reading `IVATMeans`' off-diagonal profile. Strong conceptual prior art for the
  "1-D profile of an ordering ⇒ clusters" idea.

### [DBE] Sledge, Havens, Bezdek & Keller — Dark Block Extraction
I. J. Sledge, T. C. Havens, J. C. Bezdek, and J. M. Keller, and related
"(Enhanced) Dark Block Extraction" work on counting/projecting diagonal blocks of
the RDI to obtain `k` automatically (see the VAT survey [VAT-survey] for the
consolidated treatment and exact citations).
- **Relevance:** Image-processing route to the same auto-`k` that `get_ivat_levels`
  does numerically off the diagonal. Verify the precise DBE citation against the
  survey before using in the thesis.

---

## 5. Performance / systems (the integration story)
*See `docs/performance-novelty.md`. These are the fast-VAT/iVAT lines the integrated
engine unifies and must be benchmarked against.*

### [Fast-VAT] Avinash & Lachheb (2025) — **concurrent CPU-JIT competitor**
MSR Avinash and I. Lachheb, "Fast-VAT: Accelerating Cluster Tendency Visualization
using Cython and Numba," arXiv:2507.15904, 2025.
- Reimplements **VAT** (not iVAT) in Python with **Numba JIT + Cython**, up to **50×**
  over a baseline; validates against DBSCAN/k-means. No parallel/SIMD, no in-place
  memory reduction, no iVAT, no clustering front-end. **Cite and differentiate.**
  PDF: `docs/sources/Avinash_Lachheb_2025_FastVAT_Cython_Numba.pdf`.

### [eVAT] Meng & Yuan (2018) — parallel GPU VAT
T. Meng and B. Yuan, "Parallel edge-based visual assessment of cluster tendency on
GPU," *International Journal of Data Science and Analytics*, 2018.
doi:10.1007/s41060-018-0100-7.
- Edge-based VAT (eVAT) reproducing efiVAT output, parallelized on **NVIDIA CUDA**.
  The parallel-VAT prior art; requires a GPU and is not memory- or
  arbitrary-dissimilarity-focused.

### [scalableVAT-2024] *Information Sciences* (2024) — sub-quadratic-memory exact VAT
"Time and memory scalable algorithms for clustering tendency assessment of big data,"
*Information Sciences*, 2024, PII S0020025524002378.
- Proposes **kdT-VAT / TkdT-VAT / BB-VAT**: reduce the EMST-edge search space with a
  **k-d tree** to avoid building the full n×n matrix → sub-quadratic memory, **exact**.
  **Requires Euclidean coordinates** (no tree without them) and degrades in high
  dimensions — the boundary of your arbitrary-dissimilarity regime. **Authors/exact
  title to verify** (Elsevier page paywalled; likely Rathore/Kumar et al.).

### [InPlacePerm] Cate & Twigg (1977); Catanzaro et al. (2014) — in-situ permutation
E. G. Cate and D. W. Twigg, "Algorithm 513: Analysis of In-Situ Transposition,"
*ACM Trans. Mathematical Software*, vol. 3, no. 1, pp. 104–110, 1977.
doi:10.1145/355719.355729. — and —
B. Catanzaro, A. Keller, and M. Garland, "A Decomposition for In-place Matrix
Transposition," *Proc. PPoPP*, 2014, pp. 193–206. doi:10.1145/2555243.2555253.
- Classical basis for the **cycle-following in-place permutation** in
  `pvat.shuffle_ordered_column`. Cite so the technique is credited; the novelty is its
  *application to VAT reordering* for the ~2× memory reduction, not the technique.

### GPU minimum-spanning-tree / device-side Borůvka (for the on-device GPU VAT)
Basis for the exact device-side Borůvka MST behind `gpu_vat.vat_gpu` (see
`docs/novelty-review.md` §8 for the full novelty positioning). These target
sparse/edge-list or Euclidean graphs; the VAT use here is a **dense complete
graph over an arbitrary dissimilarity matrix kept GPU-resident**.
- **[GPU-MST]** V. Vineet, P. Harish, S. Patidar, P. J. Narayanan, "Fast Minimum
  Spanning Tree for Large Graphs on the GPU," *Proc. High Performance Graphics
  (HPG '09)*, 2009, pp. 167–171. doi:10.1145/1572769.1572796. Canonical recursive
  GPU-Borůvka (scan/segmented-scan/split); sparse graphs; reports 30–50×.
  https://dl.acm.org/doi/10.1145/1572769.1572796
- **[kNN-Borůvka-GPU]** M. M. A. Arefin, C. Riveros, et al., "kNN-Borůvka-GPU: A
  Fast and Scalable MST Construction from kNN Graphs on GPU," *ICA3PP 2012*, LNCS
  7439. doi:10.1007/978-3-642-31125-3_6. MST from a kNN graph → **approximate**
  for a complete graph (contrast: our dense graph is exact).
  https://link.springer.com/chapter/10.1007/978-3-642-31125-3_6
- **[cudaMST]** J. Pan, "cudaMST — CUDA-accelerated data-parallel Borůvka's
  algorithm," GitHub reference implementation.
  https://github.com/jiachengpan/cudaMST

---

## 6. VAT ↔ TSP, seriation, and clustered TSP
*Prior art for the `experiments/vat_tsp*.py` thread; see `docs/vat-tsp-prior-art.md`
for the full novelty/gaps/benchmarks analysis. PDFs are **not** committed — this
session's egress policy blocks scholarly hosts (arXiv/IEEE/Springer/Elsevier
403); entries verified via DOI + metadata. OA-PDF links are noted for retrieval
in an unrestricted environment.*

### Seriation-as-TSP lineage
- **[Lenstra1974]** J. K. Lenstra, "Clustering a Data Array and the Traveling-
  Salesman Problem," *Operations Research* 22(2):413–414, 1974.
  doi:10.1287/opre.22.2.413. — origin of seriation ≡ TSP.
- **[HubertBaker1978]** L. J. Hubert & F. B. Baker, "Applications of Combinatorial
  Programming to Data Analysis: The Traveling Salesman and Related Problems,"
  *Psychometrika* 43(1):81–91, 1978. doi:10.1007/BF02294091. (OA-PDF via Cambridge
  Core.)
- **[ClimerZhang2006]** S. Climer & W. Zhang, "Rearrangement Clustering: Pitfalls,
  Remedies, and Applications," *J. Machine Learning Research* 7:919–943, 2006.
  Open access: jmlr.org/papers/volume7/climer06a. — dummy-city open-path reduction.
- **[Hahsler2008]** M. Hahsler, K. Hornik & C. Buchta, "Getting Things in Order:
  An Introduction to the R Package seriation," *J. Statistical Software*
  25(3):1–34, 2008. doi:10.18637/jss.v025.i03. Open access.
- **[Liiv2010]** I. Liiv, "Seriation and Matrix Reordering Methods: An Historical
  Overview," *Statistical Analysis and Data Mining* 3(2):70–91, 2010.
  doi:10.1002/sam.10071.
- **[Behrisch2016]** M. Behrisch et al., "Matrix Reordering Methods for Table and
  Network Visualization," *Computer Graphics Forum* 35(3):693–716, 2016.
  doi:10.1111/cgf.12935. (OA-PDF on HAL: hal-01326759.)
- **[WilkinsonFriendly2009]** L. Wilkinson & M. Friendly, "The History of the
  Cluster Heat Map," *The American Statistician* 63(2):179–184, 2009.
  doi:10.1198/tas.2009.0033. (OA-PDF: datavis.ca.)

### TSP solvers, local search, constructions, benchmarks
- **[LK73]** S. Lin & B. W. Kernighan, "An Effective Heuristic Algorithm for the
  TSP," *Operations Research* 21(2):498–516, 1973. doi:10.1287/opre.21.2.498.
  (OA-PDF: cs.princeton.edu/~bwk.)
- **[Croes58]** G. A. Croes, "A Method for Solving Traveling-Salesman Problems,"
  *Operations Research* 6(6):791–812, 1958. doi:10.1287/opre.6.6.791. (2-opt.)
- **[Lin65]** S. Lin, "Computer Solutions of the TSP," *Bell System Tech. J.*
  44(10):2245–2269, 1965. doi:10.1002/j.1538-7305.1965.tb04146.x. (3-opt.)
- **[Or76]** I. Or, PhD thesis, Northwestern Univ., 1976. (Or-opt; thesis.)
- **[Hel00]** K. Helsgaun, "An Effective Implementation of the Lin-Kernighan TSP
  Heuristic," *EJOR* 126(1):106–130, 2000. doi:10.1016/S0377-2217(99)00284-2.
  (LKH; OA author copy: akira.ruc.dk/~keld/research/LKH/.)
- **[Hel09]** K. Helsgaun, "General k-opt Submoves for the Lin–Kernighan TSP
  Heuristic," *Math. Prog. Computation* 1(2–3):119–163, 2009.
  doi:10.1007/s12532-009-0004-6. (LKH-2.)
- **[RSL77]** D. J. Rosenkrantz, R. E. Stearns & P. M. Lewis II, "An Analysis of
  Several Heuristics for the TSP," *SIAM J. Computing* 6(3):563–581, 1977.
  doi:10.1137/0206041. (NN bound; MST double-tree 2-approx.)
- **[DeinekoTiskin2007]** V. Deineko & A. Tiskin, "Fast Minimum-Weight Double-Tree
  Shortcutting for Metric TSP," *WEA 2007*. doi:10.1007/978-3-540-72845-0_11.
  (OA-PDF: arXiv:0710.0318.)
- **[Chr76]** N. Christofides, "Worst-Case Analysis of a New Heuristic for the
  TSP," Tech. Report 388, CMU GSIA, 1976 (reprinted *Oper. Res. Forum* 3:20, 2022,
  doi:10.1007/s43069-021-00101-z). (3/2-approx.)
- **[Bentley92]** J. L. Bentley, "Fast Algorithms for Geometric TSP," *ORSA J.
  Computing* 4(4):387–411, 1992. doi:10.1287/ijoc.4.4.387. (NN/greedy/2-opt/Or-opt
  engineering + %-excess data.)
- **[PB89]** L. K. Platzman & J. J. Bartholdi III, "Spacefilling Curves and the
  Planar TSP," *JACM* 36(4):719–737, 1989. doi:10.1145/76359.76361; and
  **[BartholdiPlatzman1982]** *Oper. Res. Lett.* 1(4):121–125.
  doi:10.1016/0167-6377(82)90012-8.
- **[TH19]** É. D. Taillard & K. Helsgaun, "POPMUSIC for the TSP," *EJOR*
  272(2):420–429, 2019. doi:10.1016/j.ejor.2018.06.039. (Initial tour + candidates
  for LKH; the modern warm-start reference.)
- **[Taillard2022]** É. D. Taillard, "A linearithmic heuristic for the TSP," *EJOR*
  297(2):442–450, 2022. doi:10.1016/j.ejor.2021.05.034.
- **[CS03]** W. Cook & P. Seymour, "Tour Merging via Branch-Decomposition,"
  *INFORMS J. Computing* 15(3):233–248, 2003. doi:10.1287/ijoc.15.3.233.16078.
- **[HK70]** M. Held & R. M. Karp, "The Traveling-Salesman Problem and Minimum
  Spanning Trees," *Operations Research* 18(6):1138–1162, 1970.
  doi:10.1287/opre.18.6.1138. (Held–Karp / 1-tree bound.)
- **[Rein91]** G. Reinelt, "TSPLIB — A TSP Library," *ORSA J. Computing*
  3(4):376–384, 1991. doi:10.1287/ijoc.3.4.376.
- **[JM97]** D. S. Johnson & L. A. McGeoch, "The TSP: A Case Study in Local
  Optimization," in *Local Search in Combinatorial Optimization*, 1997, pp.
  215–310. — greedy>NN as a 2-opt start; construction ≠ warm-start; LKH is
  start-insensitive. **[JM02]** the DIMACS-8 companion analysis (Kluwer, 2002,
  doi:10.1007/0-306-48213-4_9).

### ACO and pheromone warm-starts
- **[AntSystem96]** M. Dorigo, V. Maniezzo, A. Colorni, "Ant System," *IEEE Trans.
  SMC-B* 26(1):29–41, 1996. doi:10.1109/3477.484436.
- **[ACS97]** M. Dorigo & L. M. Gambardella, "Ant Colony System," *IEEE Trans.
  Evol. Comput.* 1(1):53–66, 1997. doi:10.1109/4235.585892. — τ₀ scaled by the NN
  tour (heuristic-informed initial pheromone).
- **[MMAS00]** T. Stützle & H. H. Hoos, "MAX–MIN Ant System," *FGCS*
  16(8):889–914, 2000. doi:10.1016/S0167-739X(00)00043-1.
- **[DaiJi2009]** Q. Dai, J. Ji, C. Liu, "An effective initialization strategy of
  pheromone for ant colony optimization," *Proc. BIC-TA 2009* (IEEE doc 5338067).
  **[unverified DOI]** — **MST-seeded pheromone**; closest prior art to the VAT/MST
  hot start.
- **[Stodola22]** P. Stodola et al., "Adaptive ACO with node clustering for the
  TSP," *Swarm & Evol. Comput.* 70:101056, 2022. doi:10.1016/j.swevo.2022.101056.
- **[Anytime14]** M. López-Ibáñez & T. Stützle, "Automatically improving the
  anytime behaviour of optimisation algorithms," *EJOR* 235(3):569–582, 2014.
  doi:10.1016/j.ejor.2013.10.043. (SQT / anytime evaluation methodology.)

### Clustered / divide-and-conquer TSP
- **[Chisman1975]** J. A. Chisman, "The clustered traveling salesman problem,"
  *Computers & Oper. Res.* 2(2):115–119, 1975. doi:10.1016/0305-0548(75)90015-5.
  (Origin of CTSP; visit each cluster contiguously.)
- **[GuttmannBeck2000]** N. Guttmann-Beck, R. Hassin, S. Khuller, B. Raghavachari,
  "Approximation Algorithms with Bounded Performance Guarantees for the CTSP,"
  *Algorithmica* 28(4):422–437, 2000. doi:10.1007/s004530010045. — **closest prior
  art to the block-to-block stitch** (cluster order + entry/exit endpoints +
  fixed-endpoint Hamiltonian paths; ratio 2.75).
- **[AnilyBramelHertz1999]** S. Anily, J. Bramel, A. Hertz, "A 5/3-approximation
  for the clustered TSP tour and path problems," *Oper. Res. Lett.* 24(1–2):29–35,
  1999. doi:10.1016/S0167-6377(98)00046-7.
- **[Ding2007]** C. Ding, Y. Cheng, M. He, "Two-Level Genetic Algorithm for CTSP
  with Application in Large-Scale TSPs," *Tsinghua Sci. & Tech.* 12(4):459–465,
  2007. doi:10.1016/S1007-0214(07)70068-8. — CTSP as divide-and-conquer for large
  plain TSP; closest on intent.
- **[Karp1977]** R. M. Karp, "Probabilistic Analysis of Partitioning Algorithms
  for the TSP in the Plane," *Math. Oper. Res.* 2(3):209–224, 1977.
  doi:10.1287/moor.2.3.209.
- **[FisherJaikumar1981]** M. L. Fisher & R. Jaikumar, "A generalized assignment
  heuristic for vehicle routing," *Networks* 11(2):109–124, 1981.
  doi:10.1002/net.3230110205; **[GillettMiller1974]** "sweep," *Oper. Res.*
  22(2):340–349, doi:10.1287/opre.22.2.340. (Cluster-first-route-second origins.)
- **[CTSPviaTSP2022]** Y. Lu, J.-K. Hao, Q. Wu, "Solving the CTSP via TSP methods,"
  *PeerJ Computer Science* 8:e972, 2022. doi:10.7717/peerj-cs.972. (OA;
  arXiv:2007.05254.)
- Neural divide-and-conquer TSP (scale frontier): **Learning-to-Delegate**
  (NeurIPS 2021), **H-TSP** (AAAI 2023, doi:10.1609/aaai.v37i8.26120,
  arXiv:2304.09395), **GLOP** (AAAI 2024, doi:10.1609/aaai.v38i18.30009,
  arXiv:2312.08224).
- Background: single-linkage ≡ MST minus the k−1 heaviest edges (a VAT cut is an
  MST partition); cf. Gagolewski et al., "Clustering with MSTs," *J. Classification*
  2024, doi:10.1007/s00357-024-09483-1.

---

## Notes on verification
- All DOIs/venues above were cross-checked against dblp / publisher records during
  the review. **Before final thesis submission, re-verify page numbers and the
  Wang et al. TKDE title/year pairing** (the 2009 TKDE "Automatically Determining
  the Number of Clusters…" and the 2010 SpecVAT partitioning material are closely
  related and are sometimes conflated in secondary sources).
- PDFs retrieved: Havens & Bezdek 2012, Wang et al. 2010 (SpecVAT/partitioning),
  Rathore et al. 2020 (ConiVAT). Others are behind paywalls (IEEE/Springer/JSTOR)
  and were verified via metadata only.
- **§6 (VAT↔TSP) PDFs were not retrieved:** the review session's egress policy
  denied all scholarly hosts (arXiv/IEEE/Springer/Elsevier → 403). All §6 entries
  are verified by DOI + search/abstract metadata; **[unverified DOI]** marks the
  few (IEEE conference papers) whose DOI could not be confirmed behind the 403 —
  re-verify Dai-Ji 2009 and the neural-solver page/DOIs before thesis submission.
  Open-access items (Climer-Zhang, Hahsler, LK73, LKH author copy, Behrisch/HAL,
  CTSPviaTSP/PeerJ, H-TSP/GLOP arXiv) can be fetched into `docs/papers/` from an
  unrestricted environment.
