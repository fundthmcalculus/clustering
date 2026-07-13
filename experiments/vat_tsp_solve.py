"""Canonical VAT->TSP solver — the validated default pipeline.

This is the recommended entry point, distilled from the whole VAT<->TSP thread:

    multi-start nearest-neighbour construction
      -> neighbour-list 2-opt   (to convergence)
      -> neighbour-list 3-opt   (to convergence)
      -> take the shortest tour over the starts.

It reaches ~+2-4% over the published optimum from n=2k-18k in seconds on the GPU
pipeline (see VAT_TSP_SCALE_FINDINGS.md), and it sidesteps every dead end the
thread found: the VAT insertion-order seams (VAT_TSP_KOPT / MPRIM), the k-NN
quality cap (which only bit the seam-heavy VAT tour), the one-move/pass GPU 2-opt,
and tuning the run-cap m (VAT_TSP_MPRIM_SWEEP: NN ~ best m, multi-start beats it).

Uses the GPU distance matrix / kNN when available, else a NumPy fallback.

Library:
    from experiments.vat_tsp_solve import solve_tsp
    tour, length, info = solve_tsp(coords, n_starts=8)

CLI:
    python -m experiments.vat_tsp_solve 1000            # nearest EUC_2D TSPLIB
    python -m experiments.vat_tsp_solve 5000 --starts 16 --plot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.vat_tsp_dualvat_lk import tour_len
from experiments.vat_tsp_kopt import two_opt_converge, three_opt_converge
from experiments.vat_tsp_mprim import mprim_order

try:
    from tribbleclustering import gpu

    _HAS_GPU = gpu.is_available()
except Exception:  # pragma: no cover - gpu module always importable in-repo
    _HAS_GPU = False

__all__ = ["solve_tsp"]

FIG_DIR = Path(__file__).parent / "figures"


def _distances_and_knn(coords, k, dtype):
    """(D_host, knn) from coords; GPU-built when a CUDA device is present."""
    n = len(coords)
    k = min(k, n - 1)
    if _HAS_GPU:
        import cupy as cp
        from experiments.vat_tsp_tsplib import knn_device

        Dg = gpu.pairwise_distances_device(coords, dtype=dtype)
        D = cp.asnumpy(Dg).astype(np.float64)
        knn = knn_device(Dg, k)
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
        return D, knn
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    knn = np.argsort(D, axis=1)[:, 1 : k + 1].astype(np.int32)
    return D, knn


def solve_tsp(
    coords,
    n_starts=8,
    k=10,
    ceil=False,
    seed=None,
    dtype="float64",
    return_starts=False,
):
    """Solve a Euclidean TSP with the default pipeline (see module docstring).

    Parameters
    ----------
    coords : (n, 2) array of point coordinates.
    n_starts : number of nearest-neighbour starts to try (take-best).
    k : neighbour-list size for 2-opt / 3-opt.
    ceil : TSPLIB rounding — False = EUC_2D nint (default), True = CEIL_2D.
    seed : if given, starts are drawn at random (reproducibly); else evenly spread.
    dtype : device distance-matrix dtype ("float64" recommended).
    return_starts : also return the per-start final lengths.

    Returns
    -------
    tour : (n,) int64 permutation (the shortest tour found).
    length : float, its tour length (official rounding).
    info : dict with 'time', 'starts', 'raw_best', 'final_lengths', 'n'.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 4:
        order = np.arange(n, dtype=np.int64)
        return order, tour_len(order, coords, ceil), {"time": 0.0, "starts": 0, "n": n}
    D, knn = _distances_and_knn(coords, k, dtype)

    ns = min(n_starts, n)
    if seed is None:
        starts = np.linspace(0, n - 1, ns, dtype=int)
    else:
        starts = np.random.default_rng(seed).choice(n, size=ns, replace=False)

    best_len = np.inf
    best_tour = None
    raw_best = np.inf
    finals = []
    t0 = time.perf_counter()
    for s in starts:
        order, _, _ = mprim_order(D, int(s), n)  # m >= n  =>  nearest-neighbour
        raw_best = min(raw_best, tour_len(order, coords, ceil))
        t = np.ascontiguousarray(order)
        two_opt_converge(t, coords, knn, ceil)
        three_opt_converge(t, coords, knn, ceil)
        L = tour_len(t, coords, ceil)
        finals.append(L)
        if L < best_len:
            best_len = L
            best_tour = t.copy()
    dt = time.perf_counter() - t0
    info = {
        "time": dt,
        "starts": int(ns),
        "raw_best": float(raw_best),
        "final_lengths": np.array(finals),
        "n": n,
        "gpu": _HAS_GPU,
    }
    if return_starts:
        return best_tour, float(best_len), info
    return best_tour, float(best_len), info


def _plot(coords, tour, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = np.append(tour, tour[0])
    fig, ax = plt.subplots(figsize=(7, 7))
    lw = 0.5 if len(tour) <= 5000 else 0.3
    ms = 2.0 if len(tour) <= 5000 else 0.8
    ax.plot(coords[loop, 0], coords[loop, 1], "-", lw=lw, color="tab:blue")
    ax.plot(coords[:, 0], coords[:, 1], ".", ms=ms, color="k")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description="Default VAT->TSP solver "
        "(multi-start NN + 2-opt/3-opt, take-best)."
    )
    ap.add_argument(
        "target",
        type=int,
        help="target n; solves the nearest-size EUC_2D TSPLIB instance",
    )
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    from experiments.vat_tsp_tsplib import nearest_euc_instance, optimal_length

    name, coords, dim = nearest_euc_instance(args.target)
    opt = optimal_length(name)
    tour, length, info = solve_tsp(
        coords, n_starts=args.starts, k=args.k, seed=args.seed
    )
    print(f"{name}  n={dim}  ({'GPU' if info['gpu'] else 'CPU'})")
    if opt:
        pct = 100.0 * (length - opt) / opt
        raw_pct = 100.0 * (info["raw_best"] - opt) / opt
        print(f"  tour length {length:.0f}  (opt {opt}, +{pct:.2f}%)")
        print(
            f"  {info['starts']} starts, take-best, {info['time']:.2f}s  "
            f"(raw-NN best +{raw_pct:.1f}%)"
        )
    else:
        pct = float("nan")
        print(f"  tour length {length:.0f}")
        print(f"  {info['starts']} starts, take-best, {info['time']:.2f}s")
    if args.plot:
        ttl = f"{name} n={dim}: NN + 2-opt/3-opt take-best  (+{pct:.2f}%)"
        p = FIG_DIR / f"vat_tsp_solve_{name}.png"
        _plot(coords, tour, ttl, p)
        print(f"  wrote {p}")


if __name__ == "__main__":
    _main()
