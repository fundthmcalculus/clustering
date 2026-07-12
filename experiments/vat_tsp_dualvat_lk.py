"""A real LK-style local search + a dual-VAT tour constructor (n=1000).

Two experiments requested on top of the recursive-IVAT TSP thread:

1. **LK step.** A proper Lin-Kernighan-family local search: neighbour-list 2-opt
   (the *full* neighbourhood — my earlier `neighbor_two_opt` skipped j<i moves,
   which is why it stalled at ~16-23% over LKH) plus Or-opt (relocate segments of
   length 1-3), with the sorted-neighbour gain criterion, run to convergence.

2. **Dual-VAT.** A two-source construction:
     2.1 pick a seed edge (default: the smallest **non-zero** dissimilarity pair;
         ``seed_mode='max'`` for the largest);
     2.2 seed cluster 1 at i (pq-1) and cluster 2 at j (pq-2);
     2.3 grow both single-linkage (Prim) trees at once, each city joining
         whichever front reaches it first — a dual-source MST partition into two
         clusters, each with its own MST;
     2.4 traverse each MST from its seed into a path, then find the optimal
         conjunction of the two paths (exhaustive over the endpoint pairings /
         orientations — the seed pair from 2.1 is one fixed junction) into a
         single closed tour.
   The dual-VAT tour is then offered as a TSP suggestion and polished with the LK
   step. We also plot the two-cluster assignment (the "clustering image").

Data: repeatable TSPLIB reference instances (nearest-size EUC_2D via
``vat_tsp_tsplib.nearest_euc_instance``), not random points.

Run:  python -m experiments.vat_tsp_dualvat_lk
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu  # noqa: E402
from experiments.vat_tsp_tsplib import (  # noqa: E402
    knn_device,
    nearest_euc_instance,
    optimal_length,
)
from experiments.vat_tsp_reslice import gpu_two_opt  # noqa: E402

try:
    import elkai  # type: ignore

    _HAS_LKH = True
except ImportError:  # pragma: no cover
    _HAS_LKH = False

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


@njit(cache=True)
def _d(coords, a, b):
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    return np.floor((dx * dx + dy * dy) ** 0.5 + 0.5)  # TSPLIB EUC_2D nint


@njit(cache=True)
def tour_len(tour, coords):
    n = tour.shape[0]
    s = 0.0
    for k in range(n):
        s += _d(coords, tour[k], tour[(k + 1) % n])
    return s


# ---------------------------------------------------------------------------
# 1. LK-style local search: full neighbour 2-opt + Or-opt(1,2,3)
# ---------------------------------------------------------------------------
@njit(cache=True)
def lk_search(tour, coords, knn, max_pass=80):
    """Neighbour-list 2-opt (both directions) + Or-opt(1,2,3), first improvement,
    to convergence. Distances are the TSPLIB nint euclidean (matches LKH)."""
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    K = knn.shape[1]
    for _ in range(max_pass):
        improved = False
        for i in range(n):
            a = tour[i]
            moved = False

            # --- full neighbour 2-opt (best improvement over a's candidates):
            # add edge (a,c); remove (a,succ a),(c,succ c); add (succ a, succ c).
            bg = 1e-7
            bp = -1
            bq = -1
            for t in range(K):
                c = knn[a, t]
                j = pos[c]
                p = i if i < j else j
                q = i if i > j else j
                if q <= p:
                    continue
                pn = (p + 1) % n
                qn = (q + 1) % n
                if pn == q and qn == p:
                    continue
                gain = (
                    _d(coords, tour[p], tour[pn])
                    + _d(coords, tour[q], tour[qn])
                    - _d(coords, tour[p], tour[q])
                    - _d(coords, tour[pn], tour[qn])
                )
                if gain > bg:
                    bg = gain
                    bp = p
                    bq = q
            if bp >= 0:
                lo, hi = bp + 1, bq
                while lo < hi:
                    tour[lo], tour[hi] = tour[hi], tour[lo]
                    pos[tour[lo]] = lo
                    pos[tour[hi]] = hi
                    lo += 1
                    hi -= 1
                if lo == hi:
                    pos[tour[lo]] = lo
                improved = True
                continue

            # --- Or-opt: relocate the segment tour[i..i+L-1] next to a neighbour
            for L in (1, 2, 3):
                i0 = i
                i1 = (i + L - 1) % n
                s0 = tour[i0]
                s1 = tour[i1]
                pr = tour[(i0 - 1) % n]
                nx = tour[(i1 + 1) % n]
                if pr == s1 or nx == s0:
                    break  # segment wraps the whole tour
                remove_gain = (
                    _d(coords, pr, s0) + _d(coords, s1, nx) - _d(coords, pr, nx)
                )
                if remove_gain <= 1e-9:
                    continue
                done = False
                for t in range(K):
                    c = knn[s0, t]
                    # insert between c and succ(c): ... c - s0..s1 - succ(c) ...
                    jc = pos[c]
                    within = False
                    for dd in range(L):
                        if (i0 + dd) % n == jc:
                            within = True
                            break
                    if within or c == pr:
                        continue
                    cn = tour[(jc + 1) % n]
                    if cn == s0:
                        continue
                    add_cost = (
                        _d(coords, c, s0) + _d(coords, s1, cn) - _d(coords, c, cn)
                    )
                    if remove_gain - add_cost > 1e-7:
                        seg = np.empty(L, np.int64)
                        for dd in range(L):
                            seg[dd] = tour[(i0 + dd) % n]
                        rest = np.empty(n - L, np.int64)
                        w = 0
                        k = (i1 + 1) % n
                        while k != i0:
                            rest[w] = tour[k]
                            w += 1
                            k = (k + 1) % n
                        # rebuild: rest with seg inserted after c
                        newt = np.empty(n, np.int64)
                        w = 0
                        for r in range(n - L):
                            newt[w] = rest[r]
                            w += 1
                            if rest[r] == c:
                                for dd in range(L):
                                    newt[w] = seg[dd]
                                    w += 1
                        for r in range(n):
                            tour[r] = newt[r]
                            pos[tour[r]] = r
                        improved = True
                        done = True
                        break
                if done:
                    moved = True
                    break
        if not improved:
            break
    return tour


# ---------------------------------------------------------------------------
# 2. Dual-VAT: dual-source Prim partition -> two MST paths -> optimal join
# ---------------------------------------------------------------------------
def _mst_edges(D):
    """Single-linkage (Prim) MST edges as (weight, u, v)."""
    n = D.shape[0]
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    d = D[0].astype(np.float64).copy()
    par = np.zeros(n, np.int64)
    edges = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, d)))
        edges.append((float(d[j]), int(par[j]), j))
        in_tree[j] = True
        upd = D[j] < d
        d[upd] = D[j][upd]
        par[upd] = j
    return edges


def choose_seeds(D, coords, mode, seed=0):
    """Return the two initialisation vertices (i0, j0) for a dual-VAT strategy:

    'min'      smallest non-zero dissimilarity pair (two coincident points);
    'max'      largest dissimilarity pair (classic VAT extreme);
    'mst_gap'  endpoints of the longest MST edge (the natural single-linkage
               2-way split — removing it separates the two components);
    'pca'      the extreme points along the 1st principal axis (balanced,
               geometric);
    'mean'     a pair whose distance is closest to the *mean* pairwise distance
               — far enough apart that the two fronts grow independently before
               meeting (not min: instant competition; not max: one sweeps all);
    'random'   a random pair (baseline).
    """
    n = D.shape[0]
    if mode == "min":
        Dm = D.astype(np.float64).copy()
        Dm[Dm <= 0] = np.inf
        flat = int(np.argmin(Dm))
        return flat // n, flat % n
    if mode == "max":
        flat = int(np.argmax(D))
        return flat // n, flat % n
    if mode == "mean":
        Dm = D.astype(np.float64)
        mean_d = Dm.sum() / (n * n - n)  # mean off-diagonal (diagonal is 0)
        A = np.abs(Dm - mean_d)
        np.fill_diagonal(A, np.inf)
        flat = int(np.argmin(A))
        return flat // n, flat % n
    if mode == "mst_gap":
        w, u, v = max(_mst_edges(D), key=lambda e: e[0])
        return u, v
    if mode == "pca":
        X = coords - coords.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ vt[0]
        return int(np.argmin(proj)), int(np.argmax(proj))
    if mode == "random":
        rng = np.random.default_rng(seed)
        a, b = rng.choice(n, size=2, replace=False)
        return int(a), int(b)
    raise ValueError(mode)


def dual_vat(D, coords, seed_mode="min"):
    """Dual-source Prim from two initialisation vertices (see `choose_seeds`).

    Returns (label, aorder1, aorder2, i0, j0): the 2-cluster assignment and, per
    cluster, the *assignment order* — the single-linkage (Prim) insertion order,
    i.e. the VAT order of that cluster (a good open path), seed first."""
    n = D.shape[0]
    i0, j0 = choose_seeds(D, coords, seed_mode)

    INF = np.inf
    label = np.full(n, -1, np.int64)  # 0 -> cluster 1 (seed i0), 1 -> cluster 2 (j0)
    best = np.empty((n, 2))
    who = np.zeros((n, 2), np.int64)
    best[:, 0] = D[i0]
    who[:, 0] = i0
    best[:, 1] = D[j0]
    who[:, 1] = j0
    label[i0] = 0
    label[j0] = 1
    best[i0] = INF
    best[j0] = INF
    aorder = [[i0], [j0]]  # per-cluster insertion order (VAT order), seed first

    for _ in range(n - 2):
        # the smaller of each city's two front-distances; pick the global min (2.3)
        cand = np.minimum(best[:, 0], best[:, 1])
        c = int(np.argmin(cand))
        side = 0 if best[c, 0] <= best[c, 1] else 1
        label[c] = side
        aorder[side].append(c)
        best[c] = INF
        # relax the chosen front with c's edges (single-linkage / Prim update),
        # but only for still-unassigned cities (else D[c,c]=0 re-activates c)
        dc = D[c]
        upd = (dc < best[:, side]) & (label < 0)
        best[upd, side] = dc[upd]
        who[upd, side] = c
    return (
        label,
        np.array(aorder[0], np.int64),
        np.array(aorder[1], np.int64),
        i0,
        j0,
    )


def dual_vat_tour(D, coords, seed_mode="min"):
    """Build the dual-VAT tour: the two clusters' VAT paths joined by the optimal
    conjunction (exhaustive over endpoint pairings + orientations)."""
    label, p1, p2, i0, j0 = dual_vat(D, coords, seed_mode=seed_mode)
    # optimal conjunction (2.4): join two open paths into a closed tour. Enumerate
    # the 4 orientations (each path forward/reversed); the two junction edges are
    # (p1 end -> p2 start) and (p2 end -> p1 start). Pick the cheapest.
    best_tour, best_cost = None, np.inf
    for r1 in (p1, p1[::-1]):
        for r2 in (p2, p2[::-1]):
            cost = D[r1[-1], r2[0]] + D[r2[-1], r1[0]]
            if cost < best_cost:
                best_cost = cost
                best_tour = np.concatenate([r1, r2])
    return np.ascontiguousarray(best_tour), label, i0, j0


# ---------------------------------------------------------------------------
# Run + figure
# ---------------------------------------------------------------------------
def run(n=1000):
    name, coords, dim = nearest_euc_instance(n)
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    print(f"Dual-VAT seed study on TSPLIB {name} (dim {dim}, nearest EUC_2D to n={n})")
    print("=" * 66)
    print(
        f"GPU: {gpu.is_available()}   published optimum = {ref:.0f} "
        "(reference — no LKH)\n"
    )
    Dg = gpu.pairwise_distances_device(coords, dtype="float64")
    D = cp.asnumpy(Dg)
    knn = knn_device(Dg, 16)

    def pct(t):
        return 100.0 * (tour_len(np.ascontiguousarray(t), coords) - ref) / ref

    # time-to-near-optimal: how fast dual-VAT + polish reaches the published opt
    modes = ("min", "max", "mean", "mst_gap", "pca", "random")
    labels = {
        "min": "min-nonzero",
        "max": "max-edge",
        "mean": "mean-dist",
        "mst_gap": "MST-gap",
        "pca": "PCA-axis",
        "random": "random",
    }
    print(
        f"  {'init':>11s} {'|C1|':>5s} {'|C2|':>5s} {'raw':>7s} "
        f"{'+neighLK':>9s} {'t_LK':>7s} {'+full2opt':>10s} {'t_2opt':>8s}"
    )
    out = {}
    for mode in modes:
        t0 = time.perf_counter()
        dv_tour, label, i0, j0 = dual_vat_tour(D, coords, seed_mode=mode)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        dv_lk = lk_search(dv_tour.copy(), coords, knn)
        t_lk = time.perf_counter() - t0
        t0 = time.perf_counter()
        dv_2opt, _ = gpu_two_opt(dv_tour.copy(), Dg)
        t_2opt = time.perf_counter() - t0
        out[mode] = dict(
            label=label,
            i0=i0,
            j0=j0,
            tour=dv_tour,
            tour2=dv_2opt,
            raw=pct(dv_tour),
            lk=pct(dv_lk),
            opt=pct(dv_2opt),
            c1=int((label == 0).sum()),
            c2=int((label == 1).sum()),
            t_lk=t_build + t_lk,
            t_2opt=t_build + t_2opt,
        )
        r = out[mode]
        print(
            f"  {labels[mode]:>11s} {r['c1']:5d} {r['c2']:5d} {r['raw']:6.0f}% "
            f"{r['lk']:8.1f}% {r['t_lk']:6.2f}s {r['opt']:9.1f}% {r['t_2opt']:7.2f}s"
        )
    out["coords"] = coords
    out["name"] = name
    out["ref"] = ref
    out["modes"] = modes
    out["labels"] = labels
    return out


def _plot_clustering(ax, coords, r, title):
    c1 = coords[r["label"] == 0]
    c2 = coords[r["label"] == 1]
    ax.plot(c1[:, 0], c1[:, 1], ".", color="tab:blue", ms=3, label="cluster 1")
    ax.plot(c2[:, 0], c2[:, 1], ".", color="tab:red", ms=3, label="cluster 2")
    ax.plot(*coords[r["i0"]], "*", color="navy", ms=15)
    ax.plot(*coords[r["j0"]], "*", color="darkred", ms=15)
    ax.plot(
        [coords[r["i0"], 0], coords[r["j0"], 0]],
        [coords[r["i0"], 1], coords[r["j0"], 1]],
        "k--",
        lw=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")


def figure(res):
    coords = res["coords"]
    modes, labels = res["modes"], res["labels"]
    ncol = 4
    npanel = len(modes) + 1  # clusterings + one bar chart
    nrow = (npanel + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 5.2 * nrow))
    flat = axes.ravel()
    for ax, mode in zip(flat, modes):
        r = res[mode]
        _plot_clustering(
            ax,
            coords,
            r,
            f"{labels[mode]} init  |C1|={r['c1']} |C2|={r['c2']}\n"
            f"raw {r['raw']:+.0f}% → +2opt {r['opt']:+.1f}% over opt",
        )
    # quality bar chart (raw / +neighLK / +full2opt per init)
    ax = flat[len(modes)]
    x = np.arange(len(modes))
    w = 0.27
    ax.bar(x - w, [res[m]["raw"] for m in modes], w, label="raw", color="0.6")
    ax.bar(
        x, [res[m]["lk"] for m in modes], w, label="+neighbour-LK", color="tab:orange"
    )
    ax.bar(
        x + w, [res[m]["opt"] for m in modes], w, label="+full 2-opt", color="tab:green"
    )
    ax.set_yscale("symlog")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in modes], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% over optimum (symlog)")
    ax.set_title("quality by initialisation")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for extra in flat[len(modes) + 1 :]:  # blank any unused grid slots
        extra.axis("off")

    fig.suptitle(
        f"Dual-VAT initialisation study on TSPLIB {res['name']} "
        f"(reference = published optimum {res['ref']:.0f})",
        fontsize=13,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_dualvat_seed.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    res = run(1000)
    print(f"\nwrote {figure(res)}")
