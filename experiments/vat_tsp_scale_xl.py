"""XL scale run: multi-start NN + 2-opt/3-opt take-best at 33k / 86k cities.

Same validated pipeline as vat_tsp_solve, but memory-tuned for the O(n^2) wall:
the device/host distance matrix is kept in **float32** (only the NN construction
reads it; 2-opt/3-opt use coords + kNN), so pla85900 fits in ~59 GB peak on the
128 GB unified memory instead of 118 GB at f64. CEIL_2D rounding (pla instances).

Run:  python -m experiments.vat_tsp_scale_xl 33810 85900
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu  # noqa: E402
from experiments.vat_tsp_tsplib import (  # noqa: E402
    knn_device,
    nearest_coord_instance,
    optimal_length,
)
from experiments.vat_tsp_dualvat_lk import tour_len  # noqa: E402
from experiments.vat_tsp_kopt import two_opt_converge, three_opt_converge  # noqa: E402
from experiments.vat_tsp_mprim import mprim_order  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _warmup():
    wc = (np.random.default_rng(0).random((48, 2)) * 100.0).astype(np.float64)
    wD = np.sqrt(((wc[:, None] - wc[None]) ** 2).sum(-1)).astype(np.float32)
    wknn = np.argsort(wD, axis=1)[:, 1:11].astype(np.int32)
    mprim_order(wD, 0, 48)
    two_opt_converge(np.arange(48), wc, wknn, True)
    three_opt_converge(np.arange(48), wc, wknn, True)


def run_size(target, n_starts):
    name, coords, dim, ewt = nearest_coord_instance(target)
    ceil = ewt == "CEIL_2D"
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    print(f"\n{name}  n={dim}  ewt={ewt}  opt={opt}  ({n_starts} starts)", flush=True)

    t_build = time.perf_counter()
    Dg = gpu.pairwise_distances_device(coords, dtype="float32")
    knn = knn_device(Dg, 10)
    D = cp.asnumpy(Dg)  # float32 host — only the NN construction reads it
    del Dg
    cp.get_default_memory_pool().free_all_blocks()
    print(
        f"  matrix+kNN built ({time.perf_counter()-t_build:.1f}s, "
        f"D {D.nbytes/1e9:.1f} GB f32)",
        flush=True,
    )
    _warmup()

    starts = np.linspace(0, dim - 1, n_starts, dtype=int)

    def pct(L):
        return 100.0 * (L - ref) / ref

    best_L = np.inf
    best_tour = None
    best_raw = None
    raw_best = np.inf
    finals = []
    t0 = time.perf_counter()
    for i, s in enumerate(starts):
        ts = time.perf_counter()
        order, _, _ = mprim_order(D, int(s), dim)  # NN
        rawL = tour_len(order, coords, ceil)
        raw_best = min(raw_best, rawL)
        t = np.ascontiguousarray(order)
        two_opt_converge(t, coords, knn, ceil)
        three_opt_converge(t, coords, knn, ceil)
        L = tour_len(t, coords, ceil)
        finals.append(L)
        if L < best_L:
            best_L = L
            best_tour = t.copy()
            best_raw = order.copy()
        print(
            f"  start {i+1}/{n_starts}: raw +{pct(rawL):.1f}%  -> "
            f"+{pct(L):.2f}%  ({time.perf_counter()-ts:.1f}s)",
            flush=True,
        )
    dt = time.perf_counter() - t0
    finals = np.array(finals)
    print(
        f"  TAKE-BEST +{pct(best_L):.2f}%  (mean +{pct(finals.mean()):.2f}%, "
        f"raw-NN best +{pct(raw_best):.1f}%)  total {dt:.1f}s",
        flush=True,
    )

    # tour image
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, tr, ttl in [
        (axes[0], best_raw, f"raw NN tour  (+{pct(raw_best):.0f}%)"),
        (axes[1], best_tour, f"NN + 2-opt* + 3-opt* take-best  (+{pct(best_L):.2f}%)"),
    ]:
        loop = np.append(tr, tr[0])
        ax.plot(coords[loop, 0], coords[loop, 1], "-", lw=0.15, color="tab:blue")
        ax.plot(coords[:, 0], coords[:, 1], ".", ms=0.3, color="k")
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{name} (n={dim}, opt {opt}): multi-start NN + 2-opt/3-opt  —  "
        f"best +{pct(best_L):.2f}% in {dt:.0f}s ({n_starts} starts, GPU)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"vat_tsp_scale_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    del D
    return name


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    targets = [int(x) for x in sys.argv[1:]] or [33810, 85900]
    for tgt in targets:
        n_starts = 4 if tgt < 50000 else 2
        run_size(tgt, n_starts)
