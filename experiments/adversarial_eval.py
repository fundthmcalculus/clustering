"""Adversarial evaluation of the divide-and-conquer VAT family.

The earlier experiments used Gaussian blobs — the worst case for *demonstrating*
VAT, because k-means already solves them, so "stitched → ARI 1.0" may just be
measuring the k-means partition. This script back-tests every method on
datasets where centroid methods fail and single-linkage / VAT should win
(non-convex, anisotropic), plus a case where single-linkage itself fails
(a bridge), with the controls a reviewer demands.

Methods (all produce k labels; ARI vs ground truth):
  * kmeans            — centroid control (also ~ the partition stitched uses)
  * single-linkage    — the exact non-convex reference VAT should match
  * exact-VAT         — compute_ivat_c -> cut iVAT superdiagonal at k-1 gaps
                        (== Boruvka-VAT, since the VAT order is identical)
  * naive-block(N)    — coordinate partition, per-block VAT, concatenate
  * stitched(N)       — k-means partition + light cross-block stitch

Decisive questions:
  1. Does exact-VAT beat k-means on non-convex data? (does VAT add anything?)
  2. Does stitched PRESERVE that, or collapse to its k-means partition?

Run:  python -m experiments.adversarial_eval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402

from tribbleclustering.pcvat import (
    compute_ivat_c,
    pairwise_distances_c_64,
)  # noqa: E402
from experiments.blockwise_vat import (  # noqa: E402
    partition,
    blockwise_vat,
    ivat_image_from_order,
    adjusted_rand,
    labels_from_order,
)
from experiments.stitched_vat import stitched_vat  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Adversarial datasets (2-D, with ground-truth labels)
# ---------------------------------------------------------------------------
def two_moons(n=1500, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    m = n // 2
    t1 = np.pi * rng.random(m)
    t2 = np.pi * rng.random(n - m)
    x1 = np.c_[np.cos(t1), np.sin(t1)]
    x2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    X = np.vstack([x1, x2]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(m), np.ones(n - m)].astype(int)
    return X, y


def circles(n=1500, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    m = n // 2
    a1 = 2 * np.pi * rng.random(m)
    a2 = 2 * np.pi * rng.random(n - m)
    x1 = np.c_[np.cos(a1), np.sin(a1)]
    x2 = 0.4 * np.c_[np.cos(a2), np.sin(a2)]
    X = np.vstack([x1, x2]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(m), np.ones(n - m)].astype(int)
    return X, y


def aniso(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [3, 5]], float)
    lbl = rng.integers(0, 3, n)
    X = rng.normal(0, 1, (n, 2)) + centers[lbl]
    T = np.array([[0.6, -0.6], [-0.4, 0.8]])  # shear -> elongated clusters
    return X @ T, lbl


def varied_density(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 0], [4, 6]], float)
    stds = np.array([0.4, 1.4, 2.6])
    lbl = rng.integers(0, 3, n)
    X = rng.normal(0, 1, (n, 2)) * stds[lbl][:, None] + centers[lbl]
    return X, lbl


def bridged(n=1500, seed=0):
    """Two blobs joined by a thin bridge — single-linkage (hence VAT) chains
    them together. Included to show where VAT ITSELF fails (honesty)."""
    rng = np.random.default_rng(seed)
    m = (n - 120) // 2
    a = rng.normal([0, 0], 0.5, (m, 2))
    b = rng.normal([6, 0], 0.5, (n - 120 - m, 2))
    bridge = np.c_[np.linspace(0, 6, 120), rng.normal(0, 0.08, 120)]
    X = np.vstack([a, b, bridge])
    y = np.r_[np.zeros(m), np.ones(n - 120 - m), np.full(120, -1)].astype(int)
    return X, y  # bridge points labeled -1 (ambiguous)


def easy_blobs(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [5, 9]], float)
    lbl = rng.integers(0, 3, n)
    return rng.normal(0, 0.8, (n, 2)) + centers[lbl], lbl


DATASETS = [
    ("two_moons", two_moons, 2),
    ("circles", circles, 2),
    ("aniso", aniso, 3),
    ("varied_density", varied_density, 3),
    ("bridged", bridged, 2),
    ("easy_blobs", easy_blobs, 3),
]


# ---------------------------------------------------------------------------
# Methods -> labels
# ---------------------------------------------------------------------------
def m_kmeans(X, D, k, N):
    _, lab = kmeans2(X.astype(float), k, minit="++", seed=0, missing="raise")
    return lab


def m_single_linkage(X, D, k, N):
    Z = linkage(squareform(D, checks=False), method="single")
    return fcluster(Z, k, criterion="maxclust")


def m_exact_vat(X, D, k, N):
    ivat, _, order = compute_ivat_c(D.copy(), inplace=False)
    return labels_from_order(order, ivat, k)


def m_naive_block(X, D, k, N):
    groups = partition(len(X), N, X, "coordinate", seed=0)
    order, _, _ = blockwise_vat(D, N, groups, merge="concat")
    return labels_from_order(order, ivat_image_from_order(D, order), k)


def m_stitched(X, D, k, N):
    order = stitched_vat(D, X, N, n_repr=24, seed=0)
    return labels_from_order(order, ivat_image_from_order(D, order), k)


METHODS = [
    ("kmeans", m_kmeans),
    ("single-linkage", m_single_linkage),
    ("exact-VAT", m_exact_vat),
    ("naive-block N=8", m_naive_block),
    ("stitched N=8", m_stitched),
]
N_BLOCKS = 8


def run():
    results = {}  # (dataset, method) -> (ARI, labels)
    data_cache = {}
    print(f"{'dataset':16s} " + " ".join(f"{m:>16s}" for m, _ in METHODS))
    for dname, gen, k in DATASETS:
        X, y = gen()
        D = pairwise_distances_c_64(np.ascontiguousarray(X, dtype=np.float64))
        data_cache[dname] = (X, y, k)
        row = []
        for mname, fn in METHODS:
            try:
                lab = fn(X, D, k, N_BLOCKS)
                # for bridged, exclude the ambiguous bridge points from ARI
                mask = y >= 0
                ari = adjusted_rand(y[mask], np.asarray(lab)[mask])
            except Exception as e:
                lab, ari = None, float("nan")
                print(f"  [{dname}/{mname}] ERR {e}")
            results[(dname, mname)] = (ari, lab)
            row.append(ari)
        print(f"{dname:16s} " + " ".join(f"{a:16.3f}" for a in row))
    _figure(data_cache, results)
    _verdict(results)
    return results


def _figure(data_cache, results):
    nd, nm = len(DATASETS), len(METHODS)
    fig, axes = plt.subplots(nd, nm, figsize=(3.0 * nm, 2.8 * nd))
    for i, (dname, _, k) in enumerate(DATASETS):
        X, y, _ = data_cache[dname]
        for j, (mname, _) in enumerate(METHODS):
            ax = axes[i, j]
            ari, lab = results[(dname, mname)]
            if lab is not None:
                ax.scatter(
                    X[:, 0], X[:, 1], c=np.asarray(lab), s=3, cmap="tab10", linewidths=0
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(mname, fontsize=10)
            if j == 0:
                ax.set_ylabel(dname, fontsize=10)
            ax.text(
                0.5,
                0.02,
                f"ARI={ari:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color=("green" if ari > 0.9 else "red" if ari < 0.6 else "black"),
            )
    fig.suptitle(
        "Adversarial evaluation — who recovers non-convex structure? "
        "(k-means fails; does VAT win, and does the stitch keep it?)",
        fontsize=13,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "adversarial_eval.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"\nwrote {p}")


def _verdict(results):
    print("\n=== verdict per dataset ===")
    for dname, _, _ in DATASETS:
        km = results[(dname, "kmeans")][0]
        vat = results[(dname, "exact-VAT")][0]
        st = results[(dname, "stitched N=8")][0]
        sl = results[(dname, "single-linkage")][0]
        vat_wins = vat - km
        stitch_keeps = st - km
        print(
            f"  {dname:16s} kmeans={km:.2f} SL={sl:.2f} VAT={vat:.2f} "
            f"stitched={st:.2f} | VAT-vs-kmeans={vat_wins:+.2f} "
            f"stitched-vs-kmeans={stitch_keeps:+.2f}"
        )


if __name__ == "__main__":
    print("Adversarial evaluation of divide-and-conquer VAT")
    print("================================================")
    run()
