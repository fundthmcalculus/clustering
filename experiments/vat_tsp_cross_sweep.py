"""Top-k sweep for the intersection-driven uncrossing pre-pass.

Extends vat_tsp_cross: sweeps the number of longest edges attacked
(top-k in {8,16,32,64}) and whether Or-opt(1) relocations compete with the 2-opt
uncrossing reversal, across a range of TSPLIB sizes. Reports, for the pipeline
`dual-VAT raw -> crossing-repair(top-k) -> 2-opt+Or-opt`:

  * final tour quality (% over published optimum),
  * pre-pass wall-time and move count,

against the baseline `dual-VAT raw -> 2-opt+Or-opt` (no uncrossing). fp32,
EUC_2D instances (geometry). Source figure: experiments/figures/vat_tsp_cross_sweep.png

Run:  python -m experiments.vat_tsp_cross_sweep
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
from experiments.vat_tsp_cross import crossing_repair  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"
SIZES = (200, 500, 1000, 2000, 5000, 10000)
TOPKS = (8, 16, 32, 64)


def run(sizes=SIZES, topks=TOPKS):
    print("Top-k sweep: intersection-driven uncrossing pre-pass -> 2-opt+Or-opt")
    print("=" * 74)
    print(f"GPU: {gpu.is_available()}   (reference = published optimum, fp32)\n")
    grid = {}  # (n, topk, oropt) -> dict
    base = {}  # n -> (quality, time)
    inst = {}  # n -> name
    for tgt in sizes:
        name, coords, dim = nearest_euc_instance(tgt)
        inst[tgt] = (name, dim)
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
        b0 = lk_search(raw.copy(), coords, knn, False)
        base[tgt] = (q(b0), time.perf_counter() - t0)
        print(
            f"{name} (n={dim}):  raw +{q(raw):.0f}%  baseline 2opt+Or +{base[tgt][0]:.1f}%"
        )
        for topk in topks:
            for oropt in (False, True):
                t0 = time.perf_counter()
                cx, mv = crossing_repair(
                    raw.copy(), coords, coords_g, topk=topk, use_oropt=oropt
                )
                t_pre = time.perf_counter() - t0
                fin = lk_search(cx.copy(), coords, knn, False)
                grid[(tgt, topk, oropt)] = dict(uq=q(cx), fq=q(fin), t=t_pre, mv=mv)
                tag = "2opt+Or" if oropt else "2opt   "
                r = grid[(tgt, topk, oropt)]
                print(
                    f"    top-{topk:<3d} [{tag}]  uncross +{r['uq']:5.1f}%  "
                    f"final +{r['fq']:5.1f}%  ({r['mv']:4d} mv, {r['t']:5.2f}s)"
                )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    return grid, base, inst


def figure(grid, base, inst, sizes=SIZES, topks=TOPKS):
    ns = [inst[s][1] for s in sizes]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(topks)))
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (a) final quality vs n, one line per top-k (Or-opt on), + baseline
    for c, k in zip(colors, topks):
        ax[0, 0].plot(
            ns,
            [grid[(s, k, True)]["fq"] for s in sizes],
            "o-",
            color=c,
            label=f"top-{k}",
        )
    ax[0, 0].plot(ns, [base[s][0] for s in sizes], "k--", label="no uncross")
    ax[0, 0].set_title("Final quality (uncross+Or-opt -> 2-opt+Or-opt)")
    ax[0, 0].set_ylabel("% over optimum")

    # (b) pre-pass time vs n, one line per top-k (Or-opt on)
    for c, k in zip(colors, topks):
        ax[0, 1].plot(
            ns,
            [grid[(s, k, True)]["t"] for s in sizes],
            "o-",
            color=c,
            label=f"top-{k}",
        )
    ax[0, 1].set_title("Uncrossing pre-pass wall-time")
    ax[0, 1].set_ylabel("seconds")
    ax[0, 1].set_yscale("log")

    # (c) final quality vs top-k, one line per size (Or-opt on)
    for s in sizes:
        ax[1, 0].plot(
            topks,
            [grid[(s, k, True)]["fq"] for k in topks],
            "o-",
            label=f"n={inst[s][1]}",
        )
    ax[1, 0].set_title("Final quality vs top-k")
    ax[1, 0].set_xlabel("top-k longest edges attacked")
    ax[1, 0].set_ylabel("% over optimum")
    ax[1, 0].set_xticks(topks)

    # (d) moveset: 2-opt only vs 2-opt+Or-opt, final quality vs n (top-32)
    ax[1, 1].plot(
        ns,
        [grid[(s, 32, False)]["fq"] for s in sizes],
        "s-",
        color="tab:blue",
        label="top-32, 2-opt only",
    )
    ax[1, 1].plot(
        ns,
        [grid[(s, 32, True)]["fq"] for s in sizes],
        "^-",
        color="tab:red",
        label="top-32, 2-opt + Or-opt",
    )
    ax[1, 1].plot(ns, [base[s][0] for s in sizes], "k--", label="no uncross")
    ax[1, 1].set_title("Move set at top-32: 2-opt vs 2-opt+Or-opt")
    ax[1, 1].set_ylabel("% over optimum")

    for a in ax.flat:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a.set_xscale("log")
        a.set_xlabel("n (cities)")
    fig.suptitle(
        "Intersection-driven uncrossing: top-k sweep (quality & time)", fontsize=13
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_cross_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    grid, base, inst = run()
    print(f"\nwrote {figure(grid, base, inst)}")
