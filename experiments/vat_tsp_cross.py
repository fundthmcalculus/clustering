"""Intersection-driven uncrossing 2-opt for the VAT->TSP tours (cross-system).

For a 2-D euclidean tour, crossing edges are the tell-tale of sub-optimality (an
optimal euclidean tour has none). This strategy targets them directly:

  1. take the longest tour edge;
  2. find every other edge that geometrically intersects it (the crossing test is
     GPU-vectorised: one long edge vs all n edges via orientation cross-products);
  3. for each crossing edge, apply the 2-opt move that removes the crossing (it is
     always improving for a proper crossing) — "split the long edge and the
     intersecting edges and re-2-opt them";
  4. repeat over the top-k longest edges, and loop until none of the top-k cross.

Compared, from the dual-VAT raw tour, against the neighbour-list 2-opt+Or-opt, on
nearest-size TSPLIB instances (fp32, reference = published optimum).

Run:  python -m experiments.vat_tsp_cross
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu  # noqa: E402
from experiments.vat_tsp_tsplib import (  # noqa: E402
    knn_device,
    nearest_euc_instance,
    optimal_length,
)
from experiments.vat_tsp_dualvat_lk import (  # noqa: E402
    dual_vat_tour_device,
    lk_search,
    tour_len,
)

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _crossers_device(coords_g, tour_g, i):
    """Positions j of edges (tour[j] -> tour[j+1]) that *properly* cross the edge
    (tour[i] -> tour[i+1]). One long edge vs all n edges, vectorised on the GPU
    via the orientation (cross-product) segment-intersection test."""
    n = tour_g.shape[0]
    p1 = coords_g[tour_g[i]]
    p2 = coords_g[tour_g[(i + 1) % n]]
    p3 = coords_g[tour_g]  # (n,2) edge starts
    p4 = coords_g[cp.roll(tour_g, -1)]  # (n,2) edge ends

    def direction(ax, ay, bx, by, cx, cy):  # sign of (b-a) x (c-a)
        return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

    d1 = direction(p3[:, 0], p3[:, 1], p4[:, 0], p4[:, 1], p1[0], p1[1])
    d2 = direction(p3[:, 0], p3[:, 1], p4[:, 0], p4[:, 1], p2[0], p2[1])
    d3 = direction(p1[0], p1[1], p2[0], p2[1], p3[:, 0], p3[:, 1])
    d4 = direction(p1[0], p1[1], p2[0], p2[1], p4[:, 0], p4[:, 1])
    mask = (d1 * d2 < 0) & (d3 * d4 < 0)  # strict -> excludes shared-endpoint touches
    js = cp.where(mask)[0]
    # drop self and the two tour-adjacent edges (they share an endpoint)
    js = js[(js != i) & (js != (i - 1) % n) & (js != (i + 1) % n)]
    return cp.asnumpy(js)


def _two_opt_delta(tour, coords, i, j, ceil):
    from experiments.vat_tsp_dualvat_lk import _d

    n = len(tour)
    p, q = (i, j) if i < j else (j, i)
    return (
        _d(coords, tour[p], tour[q], ceil)
        + _d(coords, tour[(p + 1) % n], tour[(q + 1) % n], ceil)
        - _d(coords, tour[p], tour[(p + 1) % n], ceil)
        - _d(coords, tour[q], tour[(q + 1) % n], ceil)
    )


def _oropt1_best(tour, coords, i, js, ceil):
    """Best Or-opt(1) alternative for the long edge (tour[i]->tour[i+1]): relocate
    the city b=tour[i+1] into one of the crossing edges (between its endpoints).
    Removing b breaks the long edge too. Returns (delta, b_city, x_city)."""
    from experiments.vat_tsp_dualvat_lk import _d

    n = len(tour)
    a = tour[i]
    b = tour[(i + 1) % n]
    succ = tour[(i + 2) % n]
    if succ == a:
        return 0.0, -1, -1
    rem = _d(coords, a, succ, ceil) - _d(coords, a, b, ceil) - _d(coords, b, succ, ceil)
    best_d, best_x = 0.0, -1
    for j in js:
        x = tour[j]
        y = tour[(j + 1) % n]
        if x in (a, b, succ) or y == b:
            continue
        ins = _d(coords, x, b, ceil) + _d(coords, b, y, ceil) - _d(coords, x, y, ceil)
        d = rem + ins
        if d < best_d:
            best_d, best_x = d, int(x)
    return best_d, int(b), best_x


def _apply_oropt1(tour, b, x):
    """Remove city b and reinsert it directly after city x. Preserves permutation."""
    base = tour[tour != b]
    pos = int(np.where(base == x)[0][0])
    return np.concatenate([base[: pos + 1], [b], base[pos + 1 :]])


def crossing_repair(tour, coords, coords_g, topk=16, use_oropt=False, max_iter=100000):
    """Uncrossing local search driven by the top-k longest edges' geometric
    crossings. Each iteration scans the top-k longest edges (longest first),
    applies the single most-improving move that removes one of that edge's
    crossings, and recomputes; stops when no top-k longest edge has an improving
    move. With use_oropt, an Or-opt(1) relocation of the long edge competes with
    the 2-opt reversal. Returns (tour, n_moves)."""
    tour = np.ascontiguousarray(tour, dtype=np.int64)
    ceil = False  # euclidean-only geometry (EUC_2D)
    moves = 0
    for _ in range(max_iter):
        tour_g = cp.asarray(tour)
        el = cp.asnumpy(
            cp.linalg.norm(coords_g[tour_g] - coords_g[cp.roll(tour_g, -1)], axis=1)
        )
        topk_pos = np.argsort(el)[::-1][:topk]
        applied = False
        for i in topk_pos:
            i = int(i)
            js = _crossers_device(coords_g, tour_g, i)
            if len(js) == 0:
                continue
            # best uncrossing 2-opt among this edge's crossers
            best_d, best_j, kind = 0.0, -1, None
            for j in js:
                d = _two_opt_delta(tour, coords, i, int(j), ceil)
                if d < best_d:
                    best_d, best_j, kind = d, int(j), "2opt"
            if use_oropt:  # let an Or-opt(1) relocation compete
                od, ob, ox = _oropt1_best(tour, coords, i, [int(j) for j in js], ceil)
                if od < best_d and ox >= 0:
                    best_d, kind = od, "oropt"
                    or_b, or_x = ob, ox
            if kind == "2opt":
                p, q = (i, best_j) if i < best_j else (best_j, i)
                tour[p + 1 : q + 1] = tour[p + 1 : q + 1][::-1]
                moves += 1
                applied = True
                break  # positions changed -> recompute lengths/crossers
            elif kind == "oropt":
                tour = _apply_oropt1(tour, or_b, or_x)
                moves += 1
                applied = True
                break
        if not applied:
            break
    return tour, moves


def crossing_2opt(tour, coords, coords_g, topk=16, max_iter=100000):
    """Back-compat wrapper: 2-opt-only uncrossing over the top-k longest edges."""
    return crossing_repair(
        tour, coords, coords_g, topk=topk, use_oropt=False, max_iter=max_iter
    )


def run(targets=(200, 500, 1000, 2000, 5000)):
    print("Intersection-driven uncrossing 2-opt vs neighbour 2-opt+Or-opt")
    print("=" * 70)
    print(f"GPU: {gpu.is_available()}   (reference = published optimum)\n")
    print(
        f"  {'instance':>9s} {'n':>6s} {'raw %':>7s} {'cross2opt':>10s} "
        f"{'moves':>6s} {'t_cross':>8s} {'2opt+Or':>8s} {'cross→2opt':>11s}"
    )
    rows = []
    for tgt in targets:
        name, coords, dim = nearest_euc_instance(tgt)  # EUC_2D only (geometry)
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        coords_g = cp.asarray(coords)
        knn = knn_device(Dg, 10)
        raw, _, _, _ = dual_vat_tour_device(Dg, "min")

        def q(t):
            return (
                100.0 * (tour_len(np.ascontiguousarray(t), coords, False) - ref) / ref
            )

        t0 = time.perf_counter()
        cx, moves = crossing_2opt(raw.copy(), coords, coords_g)
        t_cross = time.perf_counter() - t0
        o2 = lk_search(raw.copy(), coords, knn, False)
        cx_then = lk_search(cx.copy(), coords, knn, False)  # uncross, then full 2-opt
        rows.append(
            dict(
                name=name,
                n=dim,
                raw=q(raw),
                cross=q(cx),
                moves=moves,
                tcross=t_cross,
                o2=q(o2),
                cross_o2=q(cx_then),
            )
        )
        r = rows[-1]
        print(
            f"  {name:>9s} {dim:6d} {r['raw']:6.0f}% {r['cross']:9.1f}% "
            f"{r['moves']:6d} {r['tcross']:7.2f}s {r['o2']:7.1f}% {r['cross_o2']:10.1f}%"
        )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def figure(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ns, [r["raw"] for r in rows], "d--", color="0.6", label="dual-VAT raw")
    ax.plot(
        ns,
        [r["cross"] for r in rows],
        "s-",
        color="tab:orange",
        label="uncrossing 2-opt (top-k longest)",
    )
    ax.plot(
        ns,
        [r["o2"] for r in rows],
        "o-",
        color="tab:blue",
        label="neighbour 2-opt+Or-opt",
    )
    ax.plot(
        ns,
        [r["cross_o2"] for r in rows],
        "^-",
        color="tab:green",
        label="uncross → 2-opt+Or-opt",
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("n (cities)")
    ax.set_ylabel("% over optimum")
    ax.set_title("Intersection-driven uncrossing 2-opt (from dual-VAT tour)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_cross.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def tour_figure(target=2103):
    """Before/after tour plot for one instance: dual-VAT raw -> uncrossed -> +2-opt."""
    name, coords, dim = nearest_euc_instance(target)
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    Dg = gpu.pairwise_distances_device(coords, dtype="float32")
    coords_g = cp.asarray(coords)
    knn = knn_device(Dg, 10)
    raw, _, _, _ = dual_vat_tour_device(Dg, "min")
    cx, moves = crossing_2opt(raw.copy(), coords, coords_g)
    cxo = lk_search(cx.copy(), coords, knn, False)

    def q(t):
        return 100.0 * (tour_len(np.ascontiguousarray(t), coords, False) - ref) / ref

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, t, ttl in [
        (axes[0], raw, f"dual-VAT raw  (+{q(raw):.0f}%)"),
        (axes[1], cx, f"uncrossed ({moves} moves)  (+{q(cx):.0f}%)"),
        (axes[2], cxo, f"uncross -> 2-opt+Or  (+{q(cxo):.1f}%)"),
    ]:
        loop = np.append(t, t[0])
        ax.plot(coords[loop, 0], coords[loop, 1], "-", lw=0.5, color="tab:blue")
        ax.plot(coords[:, 0], coords[:, 1], ".", ms=1.5, color="k")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"{name} (n={dim}): intersection-driven uncrossing", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_cross_tour.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    del Dg
    cp.get_default_memory_pool().free_all_blocks()
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    rows = run()
    print(f"\nwrote {figure(rows)}")
    print(f"wrote {tour_figure()}")
