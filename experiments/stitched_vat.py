"""Spike: structure-aware partition + light cross-block stitch VAT.

The sweet spot between naive block-decomposition VAT (approximate, ~N^2 parallel,
seam artifacts) and exact Boruvka-VAT (all cross-block edges, no artifacts).

Pipeline:
  1. STRUCTURE-AWARE PARTITION: coarse k-means into N blocks, so blocks are
     spatially coherent and rarely split a true cluster (the naive method's
     random partition was catastrophic; even a coordinate sort left seams).
  2. PER-BLOCK MST: exact Prim on each block's O((n/N)^2) sub-matrix -> a forest
     of N sub-MSTs (embarrassingly parallel).
  3. LIGHT STITCH: pick r representatives per block, add the cheapest
     representative cross-edge for each block pair (O(N^2 r^2) work), then take
     the MST of {block-MST edges} U {cross candidate edges}. This connects the
     forest into an APPROXIMATE global MST that actually *interleaves* across
     block boundaries -- the step the naive concatenation skips, and the exact
     reason the seam artifact disappears. r trades accuracy for cost:
       r -> full block  == exact Boruvka-VAT (all cross edges)
       r small          == cheap, approximate
  4. VAT ORDER from that approximate MST (Prim traversal from the max seed).

This script compares exact / naive-blockwise / stitched on quality (ARI, runs)
and renders the three iVAT images.

Run:  python -m experiments.stitched_vat
"""
from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c, pairwise_distances_c_64  # noqa: E402
from experiments.blockwise_vat import (  # noqa: E402
    make_blobs, partition, blockwise_vat, ivat_image_from_order,
    n_label_runs, adjusted_rand, labels_from_order,
)

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Structure-aware partition: coarse k-means (few Lloyd iterations)
# ---------------------------------------------------------------------------
def kmeans_partition(X, N, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = X[rng.choice(n, N, replace=False)].astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        # update
        for j in range(N):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    groups = [np.where(labels == j)[0] for j in range(N)]
    return [g for g in groups if len(g) > 0]  # drop empty clusters


def maximin_partition(D, N, seed=0):
    """Coordinate-free partition from the dissimilarity matrix alone
    (farthest-first / MaxiMin seeds, assign each point to its nearest seed).
    Works on arbitrary/non-metric D where k-means cannot run."""
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(n))]
    mind = D[seeds[0]].copy()
    for _ in range(N - 1):
        nxt = int(np.argmax(mind))
        seeds.append(nxt)
        mind = np.minimum(mind, D[nxt])
    assign = np.argmin(D[:, np.array(seeds)], axis=1)
    return [np.where(assign == j)[0] for j in range(N) if np.any(assign == j)]


# ---------------------------------------------------------------------------
# Exact dense Prim on a sub-matrix -> parent array (local indices)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _prim_parent(D):
    n = D.shape[0]
    # seed at the globally most-distant pair (VAT convention)
    src = 0
    best = -1.0
    for i in range(n):
        for j in range(n):
            if D[i, j] > best:
                best = D[i, j]
                src = i
    key = np.full(n, np.inf)
    parent = np.full(n, -1, np.int64)
    used = np.zeros(n, np.bool_)
    key[src] = 0.0
    for _ in range(n):
        u = -1
        bk = np.inf
        for i in range(n):
            if not used[i] and key[i] < bk:
                bk = key[i]
                u = i
        if u == -1:
            break
        used[u] = True
        row = D[u]
        for v in range(n):
            if not used[v] and row[v] < key[v]:
                key[v] = row[v]
                parent[v] = u
    return parent


def _order_from_edges(D, edges, n, src):
    """VAT ordering = Prim traversal of the given tree (edge list) from src."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    visited[src] = True
    order[0] = src
    k = 1
    h = [(w, nb) for (w, nb) in adj[src]]
    heapq.heapify(h)
    while h and k < n:
        w, v = heapq.heappop(h)
        if visited[v]:
            continue
        visited[v] = True
        order[k] = v
        k += 1
        for (w2, nb) in adj[v]:
            if not visited[nb]:
                heapq.heappush(h, (w2, nb))
    # any leftover (disconnected safety) appended in index order
    if k < n:
        for v in range(n):
            if not visited[v]:
                order[k] = v
                k += 1
    return order


def _union_mst(edges, n):
    """Kruskal MST over an edge list [(u,v,w)] -> list of kept edges."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
            kept.append((int(u), int(v), w))
            if len(kept) == n - 1:
                break
    return kept


def _fps(subD, r, seed):
    """Farthest-point (MaxiMin) sample of r local indices from a block's sub-D
    — spreads representatives to the block's extremes/boundary."""
    m = subD.shape[0]
    if m <= r:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    s = [int(rng.integers(m))]
    mind = subD[s[0]].copy()
    for _ in range(r - 1):
        nx = int(np.argmax(mind))
        s.append(nx)
        mind = np.minimum(mind, subD[nx])
    return np.array(s)


def _topm_pairs(block, m):
    """Indices (i, j) of the m smallest entries of a 2-D block."""
    m = min(m, block.size)
    flat = np.argpartition(block.ravel(), m - 1)[:m]
    return flat // block.shape[1], flat % block.shape[1]


def stitch_core(D, groups, n_repr=24, seed=0, reps="random", m_edges=1,
                full=False):
    """Bounded cross-block stitch over a GIVEN partition -> VAT order.

    Per-block exact Prim MST (forest) + cross-block candidate edges -> MST of
    the union = approximate global MST -> VAT order. Uses only D + the
    partition, so it works on arbitrary/non-metric dissimilarity.

    Cross-edge strategy:
      reps   : 'random' (default) or 'fps' (farthest-point — boundary-aware).
      m_edges: keep the top-m cheapest cross-edges per block pair (redundancy so
               one wrong edge cannot dominate). Default 1 = the fragile light stitch.
      full   : if True, search ALL points (not representatives) for each block
               pair's min cross-edge -> the EXACT MST (oracle == Boruvka merge),
               O(n^2). Otherwise cost is O(N^2 r^2), bounded.
    """
    n = D.shape[0]
    Ng = len(groups)
    edges = []
    reps_idx = []
    rng = np.random.default_rng(seed + 1)
    for g in groups:
        sub = np.ascontiguousarray(D[np.ix_(g, g)])
        par = _prim_parent(sub)
        for i in range(len(g)):
            if par[i] >= 0:
                gi, gp = int(g[i]), int(g[par[i]])
                edges.append((gi, gp, float(D[gi, gp])))
        if not full:
            r = min(len(g), n_repr)
            if reps == "fps":
                reps_idx.append(g[_fps(sub, r, seed)])
            else:
                reps_idx.append(g[rng.choice(len(g), r, replace=False)]
                                if len(g) > r else g)

    for a in range(Ng):
        for b in range(a + 1, Ng):
            ra = groups[a] if full else reps_idx[a]
            rb = groups[b] if full else reps_idx[b]
            block = D[np.ix_(ra, rb)]
            ii, jj = _topm_pairs(block, 1 if full else m_edges)
            for ia, ib in zip(ii, jj):
                u, v = int(ra[ia]), int(rb[ib])
                edges.append((u, v, float(D[u, v])))

    mst = _union_mst(edges, n)
    src = int(np.argmax(D)) // n
    return _order_from_edges(D, mst, n, src)


def stitched_vat(D, X, N, n_repr=24, seed=0):
    """Structure-aware (k-means, coordinate) partition + light stitch."""
    return stitch_core(D, kmeans_partition(X, N, seed=seed), n_repr, seed)


def stitched_vat_from_D(D, N, n_repr=24, seed=0):
    """Coordinate-free variant: MaxiMin partition from D + light stitch.
    Applicable to arbitrary/non-metric dissimilarity matrices."""
    return stitch_core(D, maximin_partition(D, N, seed=seed), n_repr, seed)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def quality_report():
    print("\n=== quality: exact vs naive-blockwise vs stitched "
          "(n=4000, k=10; ideal runs=10, ARI=1.0) ===")
    n, d, k = 4000, 10, 10
    X, lbl = make_blobs(n, d, k, seed=2)
    D = pairwise_distances_c_64(X)
    ivat_ex, _, p_ex = compute_ivat_c(D.copy(), inplace=False)
    print(f"  exact serial     : runs={n_label_runs(p_ex, lbl):3d} "
          f"ARI={adjusted_rand(labels_from_order(p_ex, ivat_ex, k), lbl):.3f}")
    for N in (4, 8, 16):
        # naive blockwise with the best (coordinate) partition, for reference
        g_coord = partition(n, N, X, "coordinate", seed=2)
        o_naive, _, _ = blockwise_vat(D, N, g_coord, merge="concat")
        img_naive = ivat_image_from_order(D, o_naive)
        ari_naive = adjusted_rand(labels_from_order(o_naive, img_naive, k), lbl)
        # stitched
        o_st = stitched_vat(D, X, N, n_repr=24, seed=2)
        img_st = ivat_image_from_order(D, o_st)
        ari_st = adjusted_rand(labels_from_order(o_st, img_st, k), lbl)
        print(f"  N={N:2d}: naive(coord) runs={n_label_runs(o_naive, lbl):3d} "
              f"ARI={ari_naive:.3f}   |   stitched runs="
              f"{n_label_runs(o_st, lbl):3d} ARI={ari_st:.3f}")


def repr_sweep():
    print("\n=== stitched: accuracy vs #representatives r (n=4000, k=10, N=8) ===")
    n, d, k = 4000, 10, 10
    X, lbl = make_blobs(n, d, k, seed=5)
    D = pairwise_distances_c_64(X)
    for r in (4, 8, 16, 32, 64):
        o = stitched_vat(D, X, 8, n_repr=r, seed=5)
        img = ivat_image_from_order(D, o)
        print(f"  r={r:3d}: runs={n_label_runs(o, lbl):3d} "
              f"ARI={adjusted_rand(labels_from_order(o, img, k), lbl):.3f}")


def figure():
    n, d, k, N = 1600, 8, 6, 8
    X, lbl = make_blobs(n, d, k, seed=1)
    D = pairwise_distances_c_64(X)
    ivat_ex, _, _ = compute_ivat_c(D.copy(), inplace=False)
    g_coord = partition(n, N, X, "coordinate", seed=1)
    o_naive, bounds, _ = blockwise_vat(D, N, g_coord, merge="concat")
    o_st = stitched_vat(D, X, N, n_repr=24, seed=1)

    panels = [
        ("exact serial VAT", ivat_ex, None),
        (f"naive blockwise N={N}", ivat_image_from_order(D, o_naive), bounds),
        (f"stitched N={N} (r=24)", ivat_image_from_order(D, o_st), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    vmax = np.percentile(ivat_ex, 99)
    for ax, (title, img, bnd) in zip(axes, panels):
        ax.imshow(img, cmap="viridis", vmax=vmax, aspect="equal")
        if bnd is not None:
            for b in bnd:
                ax.axhline(b - 0.5, color="red", lw=0.7, alpha=0.6)
                ax.axvline(b - 0.5, color="red", lw=0.7, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Structure-aware partition + light cross-block stitch recovers "
                 "the exact VAT image (no seam pseudo-clusters)", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "stitched_vat_quality.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Structure-aware partition + light stitch VAT spike")
    print("==================================================")
    quality_report()
    repr_sweep()
    print(f"\nwrote {figure()}")
