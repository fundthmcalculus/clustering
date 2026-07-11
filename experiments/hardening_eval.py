"""Hardening the divide-and-conquer VAT claim: the two experiments a reviewer
still demanded after the adversarial eval.

PART A — arbitrary / NON-METRIC dissimilarity.
  VAT's native input is a dissimilarity matrix, not coordinates. k-means, kd-tree
  EMST, and coordinate partitioners cannot consume a non-metric / non-Euclidean D.
  We build such matrices (fractional p=0.5 Minkowski — triangle-inequality-
  violating; cosine; kNN-geodesic on a manifold) and check that stitched (its
  coordinate-free MaxiMin-partition variant) still reproduces EXACT single-linkage
  — i.e. the stitch makes no metric assumption.

PART B — partition-adversarial robustness.
  On two-moons, does a deliberately bad partition break the stitch's
  reconstruction? Sweep partition quality (k-means -> MaxiMin -> coordinate ->
  random -> worst-case-by-label) x N x #representatives r, measuring stitched ARI.

Run:  python -m experiments.hardening_eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402
from scipy.spatial.distance import squareform, pdist  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c  # noqa: E402
from experiments.blockwise_vat import (  # noqa: E402
    ivat_image_from_order, adjusted_rand, labels_from_order,
)
from experiments.stitched_vat import (  # noqa: E402
    stitch_core, maximin_partition, kmeans_partition,
)
from experiments.adversarial_eval import two_moons, easy_blobs  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Dissimilarity builders
# ---------------------------------------------------------------------------
def d_euclidean(X):
    return squareform(pdist(X, "euclidean")).astype(np.float64)


def d_fractional(X, p=0.5):
    # Minkowski with p<1: violates the triangle inequality -> NON-METRIC
    return squareform(pdist(X, "minkowski", p=p)).astype(np.float64)


def d_cosine(X):
    return squareform(pdist(X, "cosine")).astype(np.float64)


def d_geodesic(X, k=10):
    """kNN-graph shortest-path (manifold) dissimilarity — non-Euclidean."""
    E = squareform(pdist(X, "euclidean"))
    n = len(X)
    knn = np.argsort(E, axis=1)[:, 1:k + 1]
    A = np.full((n, n), 0.0)
    for i in range(n):
        for j in knn[i]:
            A[i, j] = A[j, i] = E[i, j]
    G = shortest_path(A, method="D", directed=False)
    finite = G[np.isfinite(G)]
    G[~np.isfinite(G)] = finite.max() * 2.0 if finite.size else 1.0
    np.fill_diagonal(G, 0.0)
    return np.ascontiguousarray(G)


def triangle_violation_rate(D, trials=20000, seed=0):
    """Fraction of random triples violating d(i,k) <= d(i,j)+d(j,k)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    i, j, k = (rng.integers(0, n, trials) for _ in range(3))
    lhs = D[i, k]
    rhs = D[i, j] + D[j, k] + 1e-12
    return float(np.mean(lhs > rhs))


# ---------------------------------------------------------------------------
# clustering-from-D methods
# ---------------------------------------------------------------------------
def sl_labels(D, k):
    Z = linkage(squareform(D, checks=False), method="single")
    return fcluster(Z, k, criterion="maxclust")


def vat_labels(D, k):
    ivat, _, order = compute_ivat_c(D.copy(), inplace=False)
    return labels_from_order(order, ivat, k)


def stitched_D_labels(D, k, N=8, r=24, seed=0):
    order = stitch_core(D, maximin_partition(D, N, seed=seed), n_repr=r, seed=seed)
    return labels_from_order(order, ivat_image_from_order(D, order), k)


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------
def part_a():
    print("\n=== PART A — arbitrary / non-metric dissimilarity ===")
    print("(k-means / kd-tree need coordinates+metric; VAT-family consume D directly)\n")
    Xb, yb = easy_blobs(1200, seed=1)
    Xb = (Xb - Xb.mean(0)) + 20.0  # push off origin so cosine (angle) separates
    Xm, ym = two_moons(1200, noise=0.06, seed=1)
    cases = [
        ("blobs", Xb, yb, 3, "euclidean (metric)", d_euclidean),
        ("blobs", Xb, yb, 3, "fractional p=0.5 (NON-metric)", d_fractional),
        ("blobs", Xb, yb, 3, "cosine (non-Euclidean)", d_cosine),
        ("moons", Xm, ym, 2, "euclidean (metric)", d_euclidean),
        ("moons", Xm, ym, 2, "kNN-geodesic (non-Euclidean)", d_geodesic),
    ]
    print(f"{'data':6s} {'dissimilarity':30s} {'tri-viol':>9s} "
          f"{'SL':>6s} {'VAT':>6s} {'stitchD':>8s} {'agree':>6s}")
    for name, X, y, k, dname, dfn in cases:
        D = dfn(X)
        tv = triangle_violation_rate(D)
        sl = adjusted_rand(y, sl_labels(D, k))
        vat = adjusted_rand(y, vat_labels(D, k))
        st = adjusted_rand(y, stitched_D_labels(D, k))
        agree = adjusted_rand(vat_labels(D, k), stitched_D_labels(D, k))
        print(f"{name:6s} {dname:30s} {tv:9.2%} {sl:6.2f} {vat:6.2f} "
              f"{st:8.2f} {agree:6.2f}")


# ---------------------------------------------------------------------------
# PART B — partition-adversarial robustness
# ---------------------------------------------------------------------------
def adversarial_partition(y, N):
    """Worst case: spread each true cluster evenly across all N blocks (uses
    labels as an adversary would not be able to, i.e. an upper bound on damage)."""
    groups = [[] for _ in range(N)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        for t, i in enumerate(idx):
            groups[t % N].append(i)
    return [np.array(g) for g in groups if g]


def random_partition(n, N, seed=0):
    g = np.random.default_rng(seed).integers(0, N, n)
    return [np.where(g == j)[0] for j in range(N) if np.any(g == j)]


def coordinate_partition(X, N):
    proj = X @ (X.mean(0) - X[0])
    order = np.argsort(np.asarray(proj).ravel())
    return [np.array(a) for a in np.array_split(order, N)]


def part_b():
    print("\n=== PART B — partition-adversarial robustness (two-moons, k=2) ===")
    X, y = two_moons(1500, noise=0.07, seed=2)
    D = d_euclidean(X)
    k = 2

    def parts(N):
        return {
            "kmeans": kmeans_partition(X, N, seed=2),
            "maximin(D)": maximin_partition(D, N, seed=2),
            "coordinate": coordinate_partition(X, N),
            "random": random_partition(len(X), N, seed=2),
            "adversarial": adversarial_partition(y, N),
        }

    ptypes = ["kmeans", "maximin(D)", "coordinate", "random", "adversarial"]
    Ns = [2, 4, 8, 16, 32]
    rs = [2, 4, 8, 16, 32, 64, 128]

    # heatmap: partition type x N (r=24)
    heat = np.full((len(ptypes), len(Ns)), np.nan)
    for cj, N in enumerate(Ns):
        P = parts(N)
        for ri, pt in enumerate(ptypes):
            order = stitch_core(D, P[pt], n_repr=24, seed=2)
            heat[ri, cj] = adjusted_rand(
                y, labels_from_order(order, ivat_image_from_order(D, order), k))
    # r-sweep at N=8
    rcurves = {}
    P8 = parts(8)
    for pt in ptypes:
        vals = []
        for r in rs:
            order = stitch_core(D, P8[pt], n_repr=r, seed=2)
            vals.append(adjusted_rand(
                y, labels_from_order(order, ivat_image_from_order(D, order), k)))
        rcurves[pt] = vals

    print(f"{'partition':13s} " + " ".join(f"N={n:<4d}" for n in Ns))
    for ri, pt in enumerate(ptypes):
        print(f"{pt:13s} " + " ".join(f"{heat[ri,cj]:5.2f} " for cj in range(len(Ns))))
    print("\nr-sweep (N=8): " + " ".join(f"r={r}" for r in rs))
    for pt in ptypes:
        print(f"  {pt:13s} " + " ".join(f"{v:.2f}" for v in rcurves[pt]))

    _plot_robustness(heat, ptypes, Ns, rcurves, rs)


def _plot_robustness(heat, ptypes, Ns, rcurves, rs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    im = a1.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    a1.set_xticks(range(len(Ns))); a1.set_xticklabels(Ns)
    a1.set_yticks(range(len(ptypes))); a1.set_yticklabels(ptypes)
    a1.set_xlabel("N (partition size)"); a1.set_ylabel("partition strategy")
    a1.set_title("stitched ARI on two-moons (r=24)\nbenign -> adversarial partition")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            a1.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=a1, label="ARI")
    for pt, vals in rcurves.items():
        a2.plot(rs, vals, "o-", label=pt)
    a2.set_xscale("log", base=2)
    a2.set_xlabel("# representatives r (N=8)"); a2.set_ylabel("stitched ARI")
    a2.set_title("How many cross-block representatives\nrecover the moons?")
    a2.grid(True, alpha=0.3); a2.legend(fontsize=8); a2.set_ylim(-0.05, 1.05)
    fig.suptitle("Partition-adversarial robustness of the stitched divide-and-conquer VAT",
                 fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "hardening_partition_robustness.png"
    fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    print("Hardening evaluation — non-metric dissimilarity + partition robustness")
    print("======================================================================")
    part_a()
    part_b()
