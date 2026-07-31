"""Scale run: multi-start nearest-neighbour + take-best, polished to convergence.

The m-sweep concluded: use NN-style construction (not VAT insertion order), and
multi-start + take-best beats tuning m. So the scalable pipeline is:

    for S spread starts:  NN construct -> neighbour-list 2-opt* -> 3-opt*  (converge)
    keep the shortest tour.

Run at n = 2000, 5000, ~18k (nearest-size EUC_2D TSPLIB), GPU-built distance
matrix / kNN. Reports best / mean quality (% over published optimum) and wall
time, and writes a raw-NN -> best-polished tour image per size.

Run:  python -m experiments.vat_tsp_scale
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
from experiments.vat_tsp_dualvat_lk import tour_len  # noqa: E402
from experiments.vat_tsp_kopt import two_opt_converge, three_opt_converge  # noqa: E402
from experiments.vat_tsp_mprim import mprim_order  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"
TARGETS = (2000, 5000, 18000)
N_STARTS = 8


def _warmup(coords, knn):
    wc = (np.random.default_rng(0).random((48, 2)) * 100.0).astype(np.float64)
    wD = np.sqrt(((wc[:, None] - wc[None]) ** 2).sum(-1))
    wknn = np.argsort(wD, axis=1)[:, 1:11].astype(np.int32)
    mprim_order(wD, 0, 48)
    two_opt_converge(np.arange(48), wc, wknn, False)
    three_opt_converge(np.arange(48), wc, wknn, False)


def run_size(target, n_starts=N_STARTS, warm=True):
    name, coords, dim = nearest_euc_instance(target)
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    Dg = gpu.pairwise_distances_device(coords, dtype="float64")
    D = cp.asnumpy(Dg)
    knn = knn_device(Dg, 10)
    if warm:
        _warmup(coords, knn)
    starts = np.linspace(0, dim - 1, n_starts, dtype=int)

    def pct(L):
        return 100.0 * (L - ref) / ref

    best_L = np.inf
    best_tour = None
    best_raw = None
    raw_pcts, fin_pcts = [], []
    t0 = time.perf_counter()
    for s in starts:
        order, _, _ = mprim_order(D, int(s), dim)  # m>=n => nearest-neighbour
        raw_pcts.append(pct(tour_len(order, coords, False)))
        t = order.copy()
        two_opt_converge(t, coords, knn, False)
        three_opt_converge(t, coords, knn, False)
        L = tour_len(t, coords, False)
        fin_pcts.append(pct(L))
        if L < best_L:
            best_L = L
            best_tour = t.copy()
            best_raw = order.copy()
    dt = time.perf_counter() - t0
    raw_pcts = np.array(raw_pcts)
    fin_pcts = np.array(fin_pcts)
    row = dict(
        name=name,
        n=dim,
        opt=opt,
        best=pct(best_L),
        mean=fin_pcts.mean(),
        worst=fin_pcts.max(),
        raw_best=raw_pcts.min(),
        time=dt,
        starts=n_starts,
        coords=coords,
        best_tour=best_tour,
        best_raw=best_raw,
    )
    print(
        f"{name:>9s} n={dim:5d}  best {row['best']:5.2f}%  "
        f"mean {row['mean']:5.2f}%  (worst {row['worst']:5.2f}%)  "
        f"raw-NN best {row['raw_best']:5.1f}%   {dt:6.2f}s / {n_starts} starts"
    )
    del Dg, D
    cp.get_default_memory_pool().free_all_blocks()
    return row


def tour_figure(row):
    coords = row["coords"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, tr, ttl in [
        (axes[0], row["best_raw"], f"raw NN tour  (+{row['raw_best']:.0f}%)"),
        (
            axes[1],
            row["best_tour"],
            f"multi-start best -> 2-opt* + 3-opt*  (+{row['best']:.2f}%)",
        ),
    ]:
        loop = np.append(tr, tr[0])
        lw = 0.5 if row["n"] <= 5000 else 0.3
        ax.plot(coords[loop, 0], coords[loop, 1], "-", lw=lw, color="tab:blue")
        ms = 2.0 if row["n"] <= 5000 else 0.8
        ax.plot(coords[:, 0], coords[:, 1], ".", ms=ms, color="k")
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{row['name']} (n={row['n']}, opt {row['opt']}): multi-start NN + "
        f"2-opt/3-opt  —  best {row['best']:.2f}% in {row['time']:.1f}s "
        f"({row['starts']} starts)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"vat_tsp_scale_{row['name']}.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def summary_figure(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(
        ns,
        [r["raw_best"] for r in rows],
        "d--",
        color="0.6",
        label="raw NN (best start)",
    )
    ax[0].plot(
        ns,
        [r["mean"] for r in rows],
        "s-",
        color="tab:orange",
        label="polished (mean/start)",
    )
    ax[0].plot(
        ns,
        [r["best"] for r in rows],
        "o-",
        color="tab:green",
        label="polished (take-best)",
    )
    ax[0].set_title("quality vs n")
    ax[0].set_ylabel("% over optimum")
    ax[0].set_yscale("log")
    ax[1].plot(ns, [r["time"] for r in rows], "o-", color="tab:blue")
    ax[1].set_title(f"wall time ({rows[0]['starts']} starts, construct+2opt+3opt)")
    ax[1].set_ylabel("seconds")
    for a in ax:
        a.set_xlabel("n (cities)")
        a.set_xscale("log")
        a.grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Multi-start NN + 2-opt/3-opt at scale", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "vat_tsp_scale_summary.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    print("Multi-start NN + 2-opt* + 3-opt* (take-best) at scale")
    print("=" * 74)
    rows = []
    for i, tgt in enumerate(TARGETS):
        rows.append(run_size(tgt, warm=(i == 0)))
    print()
    for r in rows:
        print(f"wrote {tour_figure(r)}")
    print(f"wrote {summary_figure(rows)}")
