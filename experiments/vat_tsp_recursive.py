"""Recursive IVAT-clustered TSP: in-cluster + cluster-to-cluster ordering (n=1000).

Follows up vat_tsp_reslice.py. Instead of cutting the VAT order into fixed
size-capped blocks, this uses IVATMeans' own cluster detection recursively:

  * At each node, run IVAT on the sub-block and let ``get_ivat_levels`` (the same
    abrupt-change-on-the-iVAT-superdiagonal detector IVATMeans uses) split it into
    natural sub-clusters — contiguous runs of the VAT order.
  * Recurse until a leaf sub-cluster has <= s points; solve the leaf's in-cluster
    TSP with LKH (cut to an open path).
  * Bottom-up, optimise the CLUSTER-TO-CLUSTER ordering at every internal level:
    order the child arcs (endpoint TSP) and orient each (DP), then open the result
    at its longest edge for the parent. The top level closes the tour.

So both the in-cluster ordering (leaf LKH) and the cluster-to-cluster ordering
(recursive stitch) are optimised, hierarchically. We sweep the target leaf size
s over [16, 256] (default 64) and report tour quality (% over LKH) and time,
including an ablation that drops the cluster-to-cluster optimisation, and a GPU
2-opt polish.

Run:  python -m experiments.vat_tsp_recursive
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu, gpu_vat  # noqa: E402
from tribbleclustering.pcvat import (
    pairwise_distances_c_64,
    compute_ivat_c,
)  # noqa: E402
from tribbleclustering.pvat import get_ivat_levels  # noqa: E402
from experiments.vat_tsp_reslice import (  # noqa: E402
    gpu_two_opt,
    _stitch_segments,
    uniform_instance,
    closed_len,
)

try:
    import elkai  # type: ignore

    _HAS_LKH = True
except ImportError:  # pragma: no cover
    _HAS_LKH = False

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _open_at_longest(tour, Dg):
    """Rotate a cyclic index array so its longest edge becomes the open ends."""
    t = cp.asarray(tour, dtype=cp.int64)
    e = cp.asnumpy(Dg[t, cp.roll(t, -1)].astype(cp.float64))
    wi = int(e.argmax())
    return np.concatenate([tour[wi + 1 :], tour[: wi + 1]])


def _solve_leaf(idx, Dg):
    """In-cluster TSP on a leaf sub-cluster -> open path over its global ids."""
    m = len(idx)
    if m <= 3:
        return np.ascontiguousarray(idx)
    g = cp.asarray(idx, dtype=cp.int64)
    sub_int = cp.asnumpy(cp.rint(Dg[cp.ix_(g, g)].astype(cp.float64))).astype(np.int64)
    if _HAS_LKH:
        st = np.asarray(
            elkai.DistanceMatrix(sub_int.tolist()).solve_tsp(runs=1)[:-1], np.int64
        )
    else:
        st = np.arange(m, dtype=np.int64)
    return _open_at_longest(np.ascontiguousarray(idx[st]), Dg)


def _ivat_split(idx, coords, s):
    """Leverage IVATMeans' detector to split a sub-block into ~m/s sub-clusters.

    Runs IVAT on the sub-block and asks ``get_ivat_levels`` for K = round(m/s)
    clusters — it places the K-1 boundaries at the most salient iVAT-superdiagonal
    jumps — so the children land near the target leaf size s (instead of the
    n_clusters=-1 mode, which over-fragments homogeneous blobs). Returns lists of
    global ids (contiguous VAT-order runs)."""
    m = len(idx)
    K = max(2, min(int(round(m / s)), m - 1))
    sub = np.ascontiguousarray(coords[idx])
    D = pairwise_distances_c_64(sub)
    img, _, order = compute_ivat_c(D.copy(), inplace=False)
    res = get_ivat_levels(sub, img, order, n_clusters=K)
    children = [idx[np.asarray(c, dtype=np.int64)] for c in res.cluster_city_ids]
    if len(children) <= 1:  # degenerate -> even chunks of the VAT order
        nchunk = max(2, int(round(m / s)))
        children = [idx[c] for c in np.array_split(order, nchunk)]
    return children


def recursive_route(idx, coords, Dg, s, stitch_clusters=True, top=True, leaves=None):
    """Recursively route a sub-block: split by IVAT until leaves <= s, solve leaf
    in-cluster TSP, and (bottom-up) order+orient the child arcs. ``top`` returns a
    closed tour; internal calls return an open path. ``stitch_clusters=False``
    ablates the cluster-to-cluster ordering (children kept in VAT order). Pass a
    list as ``leaves`` to collect leaf sizes."""
    idx = np.ascontiguousarray(idx)
    if len(idx) <= s:
        if leaves is not None:
            leaves.append(len(idx))
        return _solve_leaf(idx, Dg)  # open path
    children = _ivat_split(idx, coords, s)
    child_paths = [
        recursive_route(c, coords, Dg, s, stitch_clusters, top=False, leaves=leaves)
        for c in children
    ]
    if stitch_clusters:
        closed = _stitch_segments(child_paths, Dg)  # cyclic arrangement
    else:
        closed = np.concatenate(child_paths)  # keep IVAT/VAT order, no reordering
    return closed if top else _open_at_longest(closed, Dg)


def lkh_reference(coords):
    if not _HAS_LKH:
        return None
    n = len(coords)
    d = coords[:, None, :] - coords[None, :, :]
    D_int = np.rint(np.sqrt((d * d).sum(-1))).astype(np.int64)
    t = np.asarray(elkai.DistanceMatrix(D_int.tolist()).solve_tsp(runs=5)[:-1])
    return float(sum(D_int[t[i], t[(i + 1) % n]] for i in range(n)))


def clustered_instance(n=1000, k=12, seed=1, spread=8.0):
    """n points in k gaussian blobs — data with real cluster structure for IVAT
    to detect (blob size ~ n/k)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1000, size=(k, 2))
    lbl = rng.integers(0, k, n)
    X = centers[lbl] + rng.standard_normal((n, 2)) * spread
    return np.ascontiguousarray(X)


def _one(coords, Dg, ref, s):
    """One (instance, s) run -> dict of lengths/leaf-count/time."""
    leaves: list = []
    t0 = time.perf_counter()
    tour_no = recursive_route(
        np.arange(len(coords)), coords, Dg, s, stitch_clusters=False
    )
    L_no = closed_len(tour_no, Dg)
    tour_cc = recursive_route(
        np.arange(len(coords)), coords, Dg, s, stitch_clusters=True, leaves=leaves
    )
    L_cc = closed_len(tour_cc, Dg)
    dt = time.perf_counter() - t0
    tour_opt, _ = gpu_two_opt(tour_cc, Dg)
    L_opt = closed_len(tour_opt, Dg)
    p = lambda L: 100.0 * (L - ref) / ref  # noqa: E731
    return dict(
        no_cc=p(L_no),
        cc=p(L_cc),
        cc_opt=p(L_opt),
        time=dt,
        n_leaves=len(leaves),
        tour=tour_opt,
    )


def run(n=1000, sizes=(16, 32, 64, 128, 256), kind="clustered", seeds=(1, 2, 3, 4, 5)):
    print(f"Recursive IVAT-clustered TSP (n={n}, {kind} data, {len(seeds)} seeds)")
    print("=" * 58)
    print(f"GPU: {gpu.is_available()}   LKH (elkai): {_HAS_LKH}")
    print("  % over LKH, mean over seeds (lower = better)\n")
    print(
        f"  {'s':>4s} {'leaves':>7s} {'no cc-order':>12s} {'cc-order':>10s} "
        f"{'cc + 2opt':>11s} {'time s':>8s}"
    )
    # aggregate across seeds
    agg = {
        s: {k: [] for k in ("no_cc", "cc", "cc_opt", "time", "n_leaves")} for s in sizes
    }
    plot_coords, plot_tours = None, {}
    for si, seed in enumerate(seeds):
        coords = (
            clustered_instance(n, seed=seed)
            if kind == "clustered"
            else uniform_instance(n, seed=seed)
        )
        Dg = gpu.pairwise_distances_device(coords, dtype="float64")
        ref = lkh_reference(coords) or 1.0
        for s in sizes:
            r = _one(coords, Dg, ref, s)
            for k in ("no_cc", "cc", "cc_opt", "time", "n_leaves"):
                agg[s][k].append(r[k])
            if si == 0:
                plot_tours[s] = r["tour"]
        if si == 0:
            plot_coords = coords
        del Dg
        cp.get_default_memory_pool().free_all_blocks()

    out = {}
    for s in sizes:
        m = {k: float(np.mean(agg[s][k])) for k in agg[s]}
        out[s] = m
        print(
            f"  {s:4d} {m['n_leaves']:7.0f} {m['no_cc']:11.1f}% {m['cc']:9.1f}% "
            f"{m['cc_opt']:10.1f}% {m['time']:8.2f}"
        )
    return dict(coords=plot_coords, tours=plot_tours, out=out, kind=kind, seeds=seeds)


def figure(res):
    coords, out, tours = res["coords"], res["out"], res["tours"]
    ss = sorted(out)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(
        ss,
        [out[s]["no_cc"] for s in ss],
        "s--",
        color="0.6",
        label="in-cluster only (VAT order between clusters)",
    )
    ax.plot(
        ss,
        [out[s]["cc"] for s in ss],
        "o-",
        color="tab:orange",
        label="+ cluster-to-cluster ordering",
    )
    ax.plot(
        ss, [out[s]["cc_opt"] for s in ss], "^-", color="tab:green", label="+ GPU 2-opt"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("target leaf size s")
    ax.set_ylabel("% over LKH (mean over seeds)")
    ax.set_title("A. quality vs leaf size s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ss, [out[s]["time"] for s in ss], "o-", color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("target leaf size s")
    ax.set_ylabel("recursive build time (s)")
    ax.set_title("B. time vs leaf size s")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    best_s = min(ss, key=lambda s: out[s]["cc_opt"])
    tour = np.append(tours[best_s], tours[best_s][0])
    ax.plot(coords[tour, 0], coords[tour, 1], "-", color="tab:blue", lw=0.6)
    ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"C. recursive+2opt tour (s={best_s}, seed 1) "
        f"{out[best_s]['cc_opt']:+.1f}% over LKH"
    )

    fig.suptitle(
        "Recursive IVAT-clustered TSP (n=1000): in-cluster + cluster-to-cluster "
        "ordering, leaf size s in [16,256]",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_recursive.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    res = run(1000)
    print(f"\nwrote {figure(res)}")
