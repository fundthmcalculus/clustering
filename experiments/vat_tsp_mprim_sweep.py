"""Pin the run-cap m for bounded-run modified Prim: sweep m x instances x starts.

Follows vat_tsp_mprim (single start, noisy). Here we average the FINAL quality
(construction -> neighbour-list 2-opt -> 3-opt, all to convergence) over several
starts per instance, across a range of sizes, to find a robust sweet-spot m and
whether it scales with n. % over published optimum; nearest-size EUC_2D TSPLIB.

Run:  python -m experiments.vat_tsp_mprim_sweep
"""

from __future__ import annotations

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
TARGETS = (200, 500, 1000, 2000, 5000)
MS = (0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
N_STARTS = 8


def run(targets=TARGETS, ms=MS, n_starts=N_STARTS):
    # warm up numba
    wc = (np.random.default_rng(0).random((48, 2)) * 100.0).astype(np.float64)
    wD = np.sqrt(((wc[:, None] - wc[None]) ** 2).sum(-1))
    wknn = np.argsort(wD, axis=1)[:, 1:11].astype(np.int32)
    mprim_order(wD, 0, 4)
    two_opt_converge(np.arange(48), wc, wknn, False)
    three_opt_converge(np.arange(48), wc, wknn, False)

    print(
        f"run-cap m sweep — {n_starts} starts/instance, final = "
        f"construct -> 2-opt* -> 3-opt*  (% over optimum)"
    )
    print("=" * 78)
    results = (
        {}
    )  # name -> {"n":, "m":[...], "mean":[...], "std":[...], "nn":(mean,std)}
    for tgt in targets:
        name, coords, dim = nearest_euc_instance(tgt)
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        Dg = gpu.pairwise_distances_device(coords, dtype="float64")
        D = cp.asnumpy(Dg)
        knn = knn_device(Dg, 10)
        starts = np.linspace(0, dim - 1, n_starts, dtype=int)

        def final_pct(m, s):
            order, _, _ = mprim_order(D, int(s), m)
            t = order.copy()
            two_opt_converge(t, coords, knn, False)
            three_opt_converge(t, coords, knn, False)
            L = tour_len(np.ascontiguousarray(t), coords, False)
            return 100.0 * (L - ref) / ref

        all_m = list(ms) + [dim]  # include NN
        means, stds = [], []
        for m in all_m:
            qs = np.array([final_pct(m, s) for s in starts])
            means.append(qs.mean())
            stds.append(qs.std())
        means = np.array(means)
        stds = np.array(stds)
        # best over the finite-m grid (exclude the NN endpoint from "best m")
        grid_means = means[: len(ms)]
        bi = int(np.argmin(grid_means))
        results[name] = dict(
            n=dim, m=all_m, mean=means, std=stds, best_m=ms[bi], best_q=grid_means[bi]
        )
        print(
            f"\n{name} n={dim} (opt {opt}):  VAT(m=0) {means[0]:.2f}%  "
            f"NN {means[-1]:.2f}%   best m={ms[bi]} -> {grid_means[bi]:.2f}% "
            f"(±{stds[bi]:.2f})"
        )
        line = "   " + "  ".join(f"m{m}:{mu:.1f}" for m, mu in zip(ms, grid_means))
        print(line)
        del Dg, D
        cp.get_default_memory_pool().free_all_blocks()
    return results


def figure(results, ms=MS):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    for c, (name, r) in zip(cmap, results.items()):
        mg = np.array(ms, dtype=float)
        mean_g = r["mean"][: len(ms)]
        std_g = r["std"][: len(ms)]
        lab = f"{name} (n={r['n']})"
        ax[0].plot(mg, mean_g, "o-", color=c, label=lab)
        ax[0].fill_between(mg, mean_g - std_g, mean_g + std_g, color=c, alpha=0.12)
        # collapse test: quality vs m/sqrt(n)
        ax[2].plot(mg / np.sqrt(r["n"]), mean_g, "o-", color=c, label=lab)
    ax[0].set_title("mean final quality vs m (band = ±1 std over starts)")
    ax[0].set_xlabel("run-cap m")
    ax[0].set_ylabel("% over optimum")
    ax[0].set_xscale("symlog")
    ax[0].set_yscale("log")

    ns = [r["n"] for r in results.values()]
    bm = [r["best_m"] for r in results.values()]
    ax[1].plot(ns, bm, "o-", color="tab:red", label="best m (mean)")
    ax[1].plot(ns, [np.sqrt(n) for n in ns], "k--", lw=1, label="sqrt(n)")
    ax[1].set_title("best m vs n")
    ax[1].set_xlabel("n (cities)")
    ax[1].set_ylabel("best run-cap m")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[2].set_title("collapse test: quality vs m / sqrt(n)")
    ax[2].set_xlabel("m / sqrt(n)")
    ax[2].set_ylabel("% over optimum")
    ax[2].set_yscale("log")

    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(
        "Pinning the bounded-run modified Prim m (mean over starts)", fontsize=13
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_mprim_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    results = run()
    print(f"\nwrote {figure(results)}")
