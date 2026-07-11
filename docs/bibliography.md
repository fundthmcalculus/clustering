# Bibliography — Prior Art for VAT/iVAT, IVATMeans, and Fuzzy C-Means

Curated references for the novelty review of `tribble-clustering`. Every entry
below is a **real, verifiable publication**; DOIs / stable URLs are given so they
can be checked against the source of record. Open-access PDFs that could be
retrieved are cached in `docs/sources/` (git-ignored; do not commit).

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
  PDF: `docs/sources/Havens_Bezdek_2012_iVAT_efficient.pdf`.

### [SpecVAT/partition] Wang, Leckie, Bezdek & Ramamohanarao / Bezdek et al. (2010) — VAT-based data partitioning
L. Wang, C. Leckie, K. Ramamohanarao, and J. Bezdek, "Automatically Determining
the Number of Clusters in Unlabeled Data Sets" / "Enhanced Visual Analysis for
Cluster Tendency Assessment and Data Partitioning," *IEEE Trans. Knowledge and
Data Engineering*, vol. 22, no. 3, pp. 335–350, 2009/2010.
doi:10.1109/TKDE.2009.135.
- **Relevance:** Goes beyond *tendency* to actually **extract a partition** from the
  reordered matrix and estimate `k` automatically (SpecVAT / spectral embedding +
  VAT). Direct prior art for "VAT/iVAT as a clustering (not just visualization)
  tool." PDF: `docs/sources/Wang_2010_SpecVAT_DataPartitioning_TKDE.pdf`.

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
  `docs/sources/Rathore_2020_ConiVAT.pdf`.

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

## Notes on verification
- All DOIs/venues above were cross-checked against dblp / publisher records during
  the review. **Before final thesis submission, re-verify page numbers and the
  Wang et al. TKDE title/year pairing** (the 2009 TKDE "Automatically Determining
  the Number of Clusters…" and the 2010 SpecVAT partitioning material are closely
  related and are sometimes conflated in secondary sources).
- PDFs retrieved: Havens & Bezdek 2012, Wang et al. 2010 (SpecVAT/partitioning),
  Rathore et al. 2020 (ConiVAT). Others are behind paywalls (IEEE/Springer/JSTOR)
  and were verified via metadata only.
