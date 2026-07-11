"""A real LK-style local search + a dual-VAT tour constructor (n=1000).

Two experiments requested on top of the recursive-IVAT TSP thread:

1. **LK step.** A proper Lin-Kernighan-family local search: neighbour-list 2-opt
   (the *full* neighbourhood — my earlier `neighbor_two_opt` skipped j<i moves,
   which is why it stalled at ~16-23% over LKH) plus Or-opt (relocate segments of
   length 1-3), with the sorted-neighbour gain criterion, run to convergence.

2. **Dual-VAT.** A two-source construction:
     2.1 pick the largest-dissimilarity pair (i, j);
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
from experiments.vat_tsp_tsplib import knn_device  # noqa: E402
from experiments.vat_tsp_recursive import (
    clustered_instance,
    lkh_reference,
)  # noqa: E402
from experiments.vat_tsp_reslice import gpu_two_opt  # noqa: E402
from experiments.vat_tsp_warmstart import nn_order  # noqa: E402

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
def dual_vat(D):
    """Dual-source Prim from the two ends of the global-max edge.

    Returns (label, aorder1, aorder2, i0, j0): the 2-cluster assignment and, per
    cluster, the *assignment order* — the single-linkage (Prim) insertion order,
    i.e. the VAT order of that cluster (a good open path), seed first."""
    n = D.shape[0]
    flat = int(np.argmax(D))
    i0, j0 = flat // n, flat % n  # the largest-dissimilarity pair (2.1)

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


def dual_vat_tour(D):
    """Build the dual-VAT tour: the two clusters' VAT paths joined by the optimal
    conjunction (exhaustive over endpoint pairings + orientations)."""
    label, p1, p2, i0, j0 = dual_vat(D)
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
def run(n=1000, seeds=(1, 2, 3)):
    print(f"LK step + dual-VAT tour (n={n}, {len(seeds)} seeds)")
    print("=" * 46)
    print(f"GPU: {gpu.is_available()}   LKH (elkai): {_HAS_LKH}\n")
    print(
        f"  {'seed':>4s} {'|C1|':>5s} {'dualVAT raw':>12s} {'+neighborLK':>12s} "
        f"{'+full2opt':>10s} {'NN+neighLK':>11s} {'NN+full2opt':>12s}"
    )
    agg = {k: [] for k in ("dv", "dv_lk", "dv_2opt", "nn_lk", "nn_2opt")}
    plot = None
    for si, seed in enumerate(seeds):
        coords = clustered_instance(n, k=12, seed=seed)
        Dg = gpu.pairwise_distances_device(coords, dtype="float64")
        D = cp.asnumpy(Dg)
        knn = knn_device(Dg, 16)
        ref = lkh_reference(coords) or 1.0

        def pct(t):
            return 100.0 * (tour_len(np.ascontiguousarray(t), coords) - ref) / ref

        dv_tour, label, i0, j0 = dual_vat_tour(D)
        dv_lk = lk_search(dv_tour.copy(), coords, knn)
        dv_2opt, _ = gpu_two_opt(dv_tour.copy(), Dg)
        nn = np.ascontiguousarray(nn_order(D, i0)).astype(np.int64)
        nn_lk = lk_search(nn.copy(), coords, knn)
        nn_2opt, _ = gpu_two_opt(nn.copy(), Dg)
        vals = dict(
            dv=pct(dv_tour),
            dv_lk=pct(dv_lk),
            dv_2opt=pct(dv_2opt),
            nn_lk=pct(nn_lk),
            nn_2opt=pct(nn_2opt),
        )
        for k in agg:
            agg[k].append(vals[k])
        print(
            f"  {seed:4d} {int((label==0).sum()):5d} {vals['dv']:11.0f}% "
            f"{vals['dv_lk']:11.1f}% {vals['dv_2opt']:9.1f}% {vals['nn_lk']:10.1f}% "
            f"{vals['nn_2opt']:11.1f}%"
        )
        if si == 0:
            plot = dict(
                coords=coords,
                label=label,
                i0=i0,
                j0=j0,
                dv_tour=dv_tour,
                dv_2opt=dv_2opt,
                p=vals,
            )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    print("\n  mean over seeds:")
    for k, name in [
        ("dv", "dual-VAT raw"),
        ("dv_lk", "dual-VAT + neighbour-LK"),
        ("dv_2opt", "dual-VAT + full 2-opt"),
        ("nn_lk", "NN + neighbour-LK"),
        ("nn_2opt", "NN + full 2-opt"),
    ]:
        print(f"    {name:26s} {np.mean(agg[k]):+6.1f}% over LKH")
    plot["mean"] = {k: float(np.mean(agg[k])) for k in agg}
    return plot


def figure(res):
    coords, label = res["coords"], res["label"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # A: the dual-VAT clustering image (two-source partition)
    ax = axes[0]
    c1 = coords[label == 0]
    c2 = coords[label == 1]
    ax.plot(c1[:, 0], c1[:, 1], ".", color="tab:blue", ms=3, label="cluster 1 (pq-1)")
    ax.plot(c2[:, 0], c2[:, 1], ".", color="tab:red", ms=3, label="cluster 2 (pq-2)")
    ax.plot(*coords[res["i0"]], "*", color="navy", ms=16, label="seed i0")
    ax.plot(*coords[res["j0"]], "*", color="darkred", ms=16, label="seed j0")
    ax.plot(
        [coords[res["i0"], 0], coords[res["j0"], 0]],
        [coords[res["i0"], 1], coords[res["j0"], 1]],
        "k--",
        lw=0.8,
        label="max-dissim edge",
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A. dual-VAT clustering (dual-source Prim)")
    ax.legend(fontsize=7, loc="upper right")

    # B: the dual-VAT raw tour (two MST paths + optimal join)
    ax = axes[1]
    t = np.append(res["dv_tour"], res["dv_tour"][0])
    ax.plot(coords[t, 0], coords[t, 1], "-", color="0.5", lw=0.6)
    ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"B. dual-VAT raw tour {res['p']['dv']:+.0f}% over LKH")

    # C: after the full 2-opt polish
    ax = axes[2]
    t = np.append(res["dv_2opt"], res["dv_2opt"][0])
    ax.plot(coords[t, 0], coords[t, 1], "-", color="tab:green", lw=0.6)
    ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"C. dual-VAT + full 2-opt {res['p']['dv_2opt']:+.1f}% over LKH")

    fig.suptitle(
        "Dual-VAT construction (two-source Prim partition -> two MST paths -> "
        "optimal join), n=1000; polished to near-LKH",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_dualvat_lk.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    res = run(1000)
    print(f"\nwrote {figure(res)}")
