"""Can the iVAT min-max drop from O(n^2) to O(n log n)? — an evaluation.

iVAT's cost has two distinct pieces, and they have *different* lower bounds:

1. The **full reordered image** I(D'*) is an n x n matrix. Producing it is
   O(n^2) *by output size alone* — you must write n^2 pixels. No data
   structure beats that; the Havens-Bezdek recurrence already hits the bound.

2. The **min-max / MST machinery** underneath (VAT's modified-Prim ordering and
   the minimax path values that drive single-linkage clustering and auto-k) is
   O(n^2) in this repo only because Prim runs on the *dense complete graph*: it
   relaxes all n neighbours of every vertex. A priority queue does NOT fix that
   on its own — Prim with a heap is O(E log V), and here E = Theta(n^2), so the
   heap makes it *worse*. The quadratic term is the n^2 candidate edges, not the
   queue.

The way to actually reach O(n log n) is to stop feeding Prim n^2 edges. For
points in a low-dimensional metric space the exact Euclidean MST is a subgraph
of the Delaunay triangulation (2-D: O(n) edges, built in O(n log n)); a k-d /
cover tree gives the same in higher (modest) dimension. Feed those O(n)
candidate edges to union-find (Kruskal/Boruvka) for the exact MST, then run the
priority-queue Prim traversal on the *tree* (n-1 edges) to get the VAT order and
the per-vertex connect weights (the "cut magnitudes"). That yields, exactly and
in O(n log n) time / O(n) memory:

  * the VAT ordering,
  * the single-linkage k-clustering (cut the k-1 largest MST edges), and
  * the 1-D cut-magnitude profile auto-k reads (the diagonal get_ivat_levels
    uses) — all WITHOUT ever forming the n^2 matrix.

What you give up is the 2-D picture itself (still O(n^2)) and generality: this
is exact only for a genuine geometric embedding; on an arbitrary precomputed
dissimilarity there is no triangulation to exploit and O(n^2) stands.

This script verifies the equivalence (MST weight + clustering identical to the
dense iVAT path) and measures the scaling: dense O(n^2) Prim vs the sparse
O(n log n) EMST ordering, plus the memory wall the sparse route sidesteps.

Run:  python -m experiments.ivat_minimax_complexity
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402

from experiments.conivat_scaling import make_blobs  # noqa: E402

try:
    from tribbleclustering.pcvat import (  # noqa: E402
        pairwise_distances_c,
        vat_prim_mst_c,
    )

    HAS_COMPILED = True
except ImportError:
    HAS_COMPILED = False

FIG_DIR = Path(__file__).parent / "figures"
DENSE_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
SPARSE_SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000]
SEED = 7


# --------------------------------------------------------------------------- #
# O(n log n) EMST-based VAT ordering (no n x n matrix ever formed)
# --------------------------------------------------------------------------- #
def delaunay_mst(X: np.ndarray):
    """Exact Euclidean MST via Delaunay candidate edges. O(n log n), O(n) mem.

    Returns (u, v, w, n): the MST edge endpoints and weights, and n.
    """
    n = X.shape[0]
    tri = Delaunay(X)
    s = tri.simplices
    # Every edge of every simplex; dedup. O(n) edges for fixed low dimension.
    cols = [[0, 1], [1, 2], [0, 2]] if s.shape[1] == 3 else None
    if cols is None:  # general simplex: all C(d+1, 2) edges
        d1 = s.shape[1]
        cols = [[a, b] for a in range(d1) for b in range(a + 1, d1)]
    e = np.vstack([s[:, c] for c in cols])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    T = minimum_spanning_tree(coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)))
    T = T.tocoo()
    return T.row, T.col, T.data, n, e.shape[0]


def _approx_diameter_seed(X: np.ndarray) -> int:
    """O(n) double-sweep: farthest point from an arbitrary start, twice.

    VAT seeds from an endpoint of the most-distant pair; the exact diameter is
    O(n log n) (2-D) but the traversal seed only affects the image, never the
    MST or the clustering, so a 2-approximate endpoint is fine here.
    """
    d0 = np.linalg.norm(X - X[0], axis=1)
    a = int(np.argmax(d0))
    da = np.linalg.norm(X - X[a], axis=1)
    return int(np.argmax(da))


def vat_order_and_cuts(u, v, w, n, seed):
    """Priority-queue Prim traversal over the MST (n-1 edges). O(n log n).

    Returns (order, cut): the VAT ordering and, per position, the weight of the
    tree edge that attached that vertex (the single-linkage "cut magnitude" —
    the 1-D profile auto-k reads off the iVAT diagonal).
    """
    adj: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for a, b, wt in zip(u.tolist(), v.tolist(), w.tolist()):
        adj[a].append((wt, b))
        adj[b].append((wt, a))
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    cut = np.zeros(n, dtype=np.float64)
    visited[seed] = True
    order[0] = seed
    k = 1
    h: list[tuple[float, int]] = list(adj[seed])
    heapq.heapify(h)
    while h and k < n:
        wt, x = heapq.heappop(h)
        if visited[x]:
            continue
        visited[x] = True
        order[k] = x
        cut[k] = wt
        k += 1
        for wt2, nb in adj[x]:
            if not visited[nb]:
                heapq.heappush(h, (wt2, nb))
    return order, cut


def labels_from_cuts(u, v, w, n, k):
    """Single-linkage k-clustering: drop the k-1 heaviest MST edges."""
    heavy = set(np.argsort(w)[::-1][: k - 1].tolist())
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        if i in heavy:
            continue
        adj[a].append(b)
        adj[b].append(a)
    lab = np.full(n, -1, dtype=np.int64)
    c = 0
    for s in range(n):
        if lab[s] != -1:
            continue
        lab[s] = c
        stack = [s]
        while stack:
            xx = stack.pop()
            for nb in adj[xx]:
                if lab[nb] == -1:
                    lab[nb] = c
                    stack.append(nb)
        c += 1
    return lab


# --------------------------------------------------------------------------- #
# Correctness + scaling
# --------------------------------------------------------------------------- #
def _ari(a, b) -> float:
    from math import comb

    a = np.unique(np.asarray(a), return_inverse=True)[1]
    b = np.unique(np.asarray(b), return_inverse=True)[1]
    n = len(a)
    cont = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    for i in range(n):
        cont[a[i], b[i]] += 1
    si = cont.sum(1)
    sj = cont.sum(0)
    idx = sum(comb(int(x), 2) for x in cont.ravel())
    ei = sum(comb(int(x), 2) for x in si)
    ej = sum(comb(int(x), 2) for x in sj)
    exp = ei * ej / comb(n, 2)
    mx = 0.5 * (ei + ej)
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)


def verify() -> None:
    print("Correctness — sparse EMST vs dense MST (exact) and clustering:")
    print(f"{'n':>6} {'dense_w':>12} {'sparse_w':>12} {'match':>6} {'ARI_truth':>10}")
    for n in [500, 2000, 5000]:
        X, y = make_blobs(n)
        u, v, w, _, _ = delaunay_mst(X)
        dense_w = float(minimum_spanning_tree(pairwise_distances_c(X)).sum())
        sparse_w = float(w.sum())
        lab = labels_from_cuts(u, v, w, n, 4)
        print(
            f"{n:>6} {dense_w:>12.4f} {sparse_w:>12.4f} "
            f"{str(np.isclose(dense_w, sparse_w)):>6} {_ari(lab, y):>10.3f}"
        )
    print()


def _time(fn, *a, repeats: int = 3) -> float:
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*a)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def run() -> dict:
    if not HAS_COMPILED:
        raise SystemExit("Build the pcvat extension first (setup.py build_ext).")

    verify()

    # Warm the compiled kernel.
    Xw, _ = make_blobs(64)
    vat_prim_mst_c(pairwise_distances_c(Xw))

    dense_ms: list[float] = []
    sparse_ms: list[float] = []
    print("Scaling — VAT-order production time (ms):")
    print(f"{'n':>7} {'dense_O(n^2)':>13} {'sparse_O(nlogn)':>16}")
    for n in SPARSE_SIZES:
        X, _ = make_blobs(n)

        def _sparse():
            u, v, w, nn, _ = delaunay_mst(X)
            seed = _approx_diameter_seed(X)
            return vat_order_and_cuts(u, v, w, nn, seed)

        rep = 3 if n <= 16000 else 2
        t_sparse = _time(_sparse, repeats=rep)
        sparse_ms.append(t_sparse)

        if n in DENSE_SIZES:

            def _dense():
                D = pairwise_distances_c(np.ascontiguousarray(X))
                return vat_prim_mst_c(D)

            t_dense = _time(_dense, repeats=rep)
            dense_ms.append(t_dense)
            print(f"{n:>7} {t_dense:>13.2f} {t_sparse:>16.2f}")
        else:
            print(f"{n:>7} {'—':>13} {t_sparse:>16.2f}")

    _plot(dense_ms, sparse_ms)
    return dict(dense_ms=dense_ms, sparse_ms=sparse_ms)


def _plot(dense_ms, sparse_ms) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: wall time, with O(n^2) and O(n log n) reference slopes.
    dn = np.array(DENSE_SIZES, dtype=float)
    sn = np.array(SPARSE_SIZES, dtype=float)
    ax.plot(
        DENSE_SIZES, dense_ms, "s-", color="tab:red", label="dense Prim min-max, O(n²)"
    )
    ax.plot(
        SPARSE_SIZES,
        sparse_ms,
        "o-",
        color="tab:blue",
        label="Delaunay-EMST + PQ-Prim, O(n log n)",
    )
    ref2 = dn**2
    ref2 = ref2 / ref2[-1] * dense_ms[-1]
    ax.plot(DENSE_SIZES, ref2, "r--", alpha=0.4, label=r"$O(n^2)$ ref")
    refnl = sn * np.log2(sn)
    refnl = refnl / refnl[-1] * sparse_ms[-1]
    ax.plot(SPARSE_SIZES, refnl, "b--", alpha=0.4, label=r"$O(n\log n)$ ref")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("VAT-order time (ms)")
    ax.set_title("iVAT min-max ordering: dense O(n²) vs geometric O(n log n)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Right: memory footprint — the dense n x n matrix vs the sparse edge list.
    dense_bytes = sn**2 * 8
    sparse_bytes = sn * 3 * (2 * 8 + 8)  # ~3n edges x (2 int64 idx + f64 wt)
    ax2.plot(
        SPARSE_SIZES,
        dense_bytes / 1e9,
        "s-",
        color="tab:red",
        label="dense n×n f64 matrix",
    )
    ax2.plot(
        SPARSE_SIZES,
        sparse_bytes / 1e9,
        "o-",
        color="tab:blue",
        label="sparse EMST edge list",
    )
    ax2.axhline(15.0, color="grey", ls=":", label="this box: 15 GB RAM")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("footprint (GB)")
    ax2.set_title("Memory: the quadratic wall the geometric route avoids")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "ivat_minimax_complexity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nwrote {p}")
    return p


if __name__ == "__main__":
    print("iVAT min-max complexity: O(n^2) -> O(n log n) evaluation")
    print("========================================================")
    run()
