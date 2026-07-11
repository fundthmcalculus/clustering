"""Spike: speculative Boruvka-based parallel MST for VAT.

VAT's ordering is Prim's vertex-insertion order, and Prim only ever traverses
edges of the MST (cut property). So if we build the MST by ANY method and then
run Prim *restricted to the MST tree* from the same seed (the global-maximum
dissimilarity vertex), we reproduce the exact VAT ordering — and therefore the
exact iVAT image. That turns "parallel VAT" into "parallel MST + an O(n log n)
tree traversal", and Boruvka is the classic parallel MST:

  * Each round, every current component finds its minimum outgoing edge (an
    O(n^2) scan of the dense dissimilarity matrix — embarrassingly parallel).
  * All those edges are added at once and their components merged (union-find).
  * O(log n) rounds.

Total work is O(n^2 log n) — a log factor MORE than the serial O(n^2)
compact-Prim — so Boruvka can only win by parallelism, and only if the extra
log-factor work is outrun by the core/throughput count. This spike measures
whether that happens on 32 CPU cores (Numba) and on the GPU (CuPy), and
verifies the output image is identical to the serial engine.

Outputs two figures under experiments/figures/:
  * boruvka_vat_quality.png  — serial vs Boruvka iVAT images + their difference
  * boruvka_vat_scaling.png  — MST build time vs n for the three backends

Run:  python -m experiments.boruvka_vat
"""
from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import (  # noqa: E402
    compute_ivat_c, vat_prim_mst_c, pairwise_distances_c_64,
)

try:
    import cupy as _cp
    _HAS_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _HAS_CUPY = False

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Boruvka MST — dense, Numba (parallel min-edge scan)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:  # path compression
        parent[x], x = root, parent[x]
    return root


@njit(cache=True, parallel=True)
def boruvka_mst_numba(D):
    n = D.shape[0]
    parent = np.arange(n)
    rank = np.zeros(n, np.int64)
    roots = np.empty(n, np.int64)
    best_w = np.empty(n, np.float64)
    best_v = np.empty(n, np.int64)
    mst_u = np.empty(n - 1, np.int64)
    mst_v = np.empty(n - 1, np.int64)
    ne = 0

    while ne < n - 1:
        for i in range(n):
            roots[i] = _find(parent, i)
        # Parallel: each vertex's minimum edge leaving its component.
        for u in prange(n):
            cu = roots[u]
            bw = np.inf
            bv = -1
            row = D[u]
            for v in range(n):
                if roots[v] != cu and row[v] < bw:
                    bw = row[v]
                    bv = v
            best_w[u] = bw
            best_v[u] = bv
        # Per-component best (serial reduction over vertices).
        comp_bw = np.full(n, np.inf)
        comp_bu = np.full(n, -1, np.int64)
        comp_bv = np.full(n, -1, np.int64)
        for u in range(n):
            r = roots[u]
            if best_v[u] != -1 and best_w[u] < comp_bw[r]:
                comp_bw[r] = best_w[u]
                comp_bu[r] = u
                comp_bv[r] = best_v[u]
        # Add each component's best edge, union (serial, O(#components)).
        added = 0
        for r in range(n):
            if comp_bu[r] != -1:
                ru = _find(parent, comp_bu[r])
                rv = _find(parent, comp_bv[r])
                if ru != rv:
                    if rank[ru] < rank[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    if rank[ru] == rank[rv]:
                        rank[ru] += 1
                    mst_u[ne] = comp_bu[r]
                    mst_v[ne] = comp_bv[r]
                    ne += 1
                    added += 1
        if added == 0:
            break
    return mst_u[:ne], mst_v[:ne]


# ---------------------------------------------------------------------------
# Boruvka MST — GPU (CuPy) min-edge scan per round, host union-find
# ---------------------------------------------------------------------------
def boruvka_mst_cupy(D):
    n = D.shape[0]
    Dg = _cp.asarray(D)
    parent = np.arange(n)
    rank = np.zeros(n, np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    mst_u, mst_v = [], []
    idx = _cp.arange(n)
    while len(mst_u) < n - 1:
        roots = np.array([find(i) for i in range(n)])
        rg = _cp.asarray(roots)
        same = rg[:, None] == rg[None, :]
        Dm = _cp.where(same, _cp.inf, Dg)
        best_v = _cp.argmin(Dm, axis=1)
        best_w = Dm[idx, best_v]
        best_v = _cp.asnumpy(best_v)
        best_w = _cp.asnumpy(best_w)
        comp_bw = {}
        for u in range(n):
            if np.isfinite(best_w[u]):
                r = roots[u]
                if r not in comp_bw or best_w[u] < comp_bw[r][0]:
                    comp_bw[r] = (best_w[u], u, int(best_v[u]))
        added = 0
        for _, u, v in comp_bw.values():
            ru, rv = find(u), find(v)
            if ru != rv:
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
                mst_u.append(u)
                mst_v.append(v)
                added += 1
        if added == 0:
            break
    return np.array(mst_u), np.array(mst_v)


# ---------------------------------------------------------------------------
# VAT ordering from an MST edge list (Prim traversal of the tree from the seed)
# ---------------------------------------------------------------------------
def vat_order_from_mst(D, mst_u, mst_v):
    n = D.shape[0]
    adj = [[] for _ in range(n)]
    for a, b in zip(mst_u.tolist(), mst_v.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    src = int(np.argmax(D)) // n  # VAT seed = an endpoint of the global-max edge
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    k = 0
    visited[src] = True
    order[k] = src
    k += 1
    h = [(D[src, nb], nb) for nb in adj[src]]
    heapq.heapify(h)
    while h:
        w, v = heapq.heappop(h)
        if visited[v]:
            continue
        visited[v] = True
        order[k] = v
        k += 1
        for nb in adj[v]:
            if not visited[nb]:
                heapq.heappush(h, (D[v, nb], nb))
    return order


@njit(cache=True)
def ivat_image_from_order(D, order):
    """Reorder D by `order` and apply the minimax iVAT recursion."""
    n = order.shape[0]
    V = np.empty((n, n), np.float64)
    for i in range(n):
        oi = order[i]
        for j in range(n):
            V[i, j] = D[oi, order[j]]
    for r in range(1, n):
        jj = 0
        mn = V[r, 0]
        for c in range(1, r):
            if V[r, c] < mn:
                mn = V[r, c]
                jj = c
        for c in range(r):
            if c == jj:
                V[r, c] = mn
            else:
                cur = V[jj, c] if jj > c else V[c, jj]
                V[r, c] = mn if mn > cur else cur
    for i in range(1, n):
        for j in range(i):
            V[j, i] = V[i, j]
    return V


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_blobs(n, d, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-40, 40, size=(k, d))
    lbl = rng.integers(0, k, n)
    X = rng.standard_normal((n, d)) * 2.0 + centers[lbl]
    return np.ascontiguousarray(X)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def quality_figure():
    cases = [(1500, 8, 6, 1), (1500, 8, 12, 4)]
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 7))
    diffs = []
    for row, (n, d, k, seed) in enumerate(cases):
        X = make_blobs(n, d, k, seed)
        D = pairwise_distances_c_64(X)
        ivat_serial, _, _ = compute_ivat_c(D.copy(), inplace=False)
        mu, mv = boruvka_mst_numba(D)
        order = vat_order_from_mst(D, mu, mv)
        ivat_bor = ivat_image_from_order(D, order)
        diff = np.abs(ivat_serial - ivat_bor)
        diffs.append(float(diff.max()))
        vmax = np.percentile(ivat_serial, 99)
        for col, (img, title) in enumerate((
            (ivat_serial, f"serial Prim iVAT\n(n={n}, k={k})"),
            (ivat_bor, "Boruvka iVAT"),
            (diff, f"|difference|\nmax={diff.max():.2e}"),
        )):
            ax = axes[row, col]
            im = ax.imshow(img, cmap="viridis" if col < 2 else "magma",
                           vmax=vmax if col < 2 else None, aspect="equal")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("VAT/iVAT image: serial Prim vs Boruvka-MST "
                 "(identical output — Boruvka is a parallel MST build only)",
                 fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "boruvka_vat_quality.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, diffs


def _time(fn, *a, rep=2, warm=1):
    for _ in range(warm):
        fn(*a)
    best = np.inf
    for _ in range(rep):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def _time_gpu(fn, arg):
    """Time a GPU call, synchronising around it (arg may be a device array)."""
    _cp.cuda.Stream.null.synchronize()
    fn(arg)
    _cp.cuda.Stream.null.synchronize()
    best = np.inf
    for _ in range(2):
        _cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        fn(arg)
        _cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def scaling_figure():
    from experiments.boruvka_gpu import boruvka_mst_gpu

    sizes = [1000, 2000, 4000, 8000, 16000, 32000]
    t_prim, t_bnumba, t_gpu_dev, t_gpu_xfer, order_match = [], [], [], [], []
    for n in sizes:
        X = make_blobs(n, 10, 25, seed=7)
        D = pairwise_distances_c_64(X)
        t_prim.append(_time(lambda M: vat_prim_mst_c(M), D))
        t_bnumba.append(_time(lambda M: boruvka_mst_numba(M), D))
        if _HAS_CUPY:
            # with host->device transfer of the n x n matrix
            t_gpu_xfer.append(_time_gpu(lambda M: boruvka_mst_gpu(M), D))
            # matrix already resident on the device (the on-device-pipeline case)
            Dg = _cp.asarray(D)
            t_gpu_dev.append(_time_gpu(lambda M: boruvka_mst_gpu(M), Dg))
            del Dg
            _cp.get_default_memory_pool().free_all_blocks()
        # correctness: does Boruvka-derived VAT order match serial Prim?
        _, _, p_serial = compute_ivat_c(D.copy(), inplace=False)
        mu, mv = boruvka_mst_numba(D)
        order = vat_order_from_mst(D, mu, mv)
        order_match.append(float(np.mean(order == p_serial)))
        gm = (f"  gpu(dev) {t_gpu_dev[-1]:8.1f}ms  gpu(+xfer) {t_gpu_xfer[-1]:8.1f}ms"
              if _HAS_CUPY else "")
        print(f"  n={n:6d}: prim {t_prim[-1]:8.1f}ms  boruvka(numba) "
              f"{t_bnumba[-1]:8.1f}ms" + gm +
              f"  order_match={order_match[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(sizes, t_prim, "o-", label="serial Prim (C/OpenMP, O(n^2))")
    ax.plot(sizes, t_bnumba, "s-", label="Boruvka (Numba, 32 cores)")
    if _HAS_CUPY:
        ax.plot(sizes, t_gpu_dev, "^-", color="tab:red",
                label="Boruvka (GPU, matrix resident)")
        ax.plot(sizes, t_gpu_xfer, "^--", color="tab:red", alpha=0.5,
                label="Boruvka (GPU, incl. host->device transfer)")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("MST build time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MST build: serial Prim vs parallel Boruvka\n"
                 "(VAT order/image identical; MST build time is the only axis "
                 "that differs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "boruvka_vat_scaling.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, dict(sizes=sizes, prim=t_prim, boruvka_numba=t_bnumba,
                      gpu_resident=t_gpu_dev, gpu_xfer=t_gpu_xfer,
                      order_match=order_match)


if __name__ == "__main__":
    print("Boruvka VAT spike\n=================")
    print(f"CuPy GPU available: {_HAS_CUPY}\n")
    print("Quality figure (serial vs Boruvka iVAT image)...")
    qpath, diffs = quality_figure()
    print(f"  max |serial - boruvka| per case: {diffs}")
    print(f"  wrote {qpath}\n")
    print("Scaling figure (MST build time vs n)...")
    spath, data = scaling_figure()
    print(f"  wrote {spath}")
