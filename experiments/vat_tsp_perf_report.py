"""Performance report (quality + time) for the dual-VAT + LK TSP pipeline.

Sweeps target sizes 50 -> 50000, using for each the nearest-size EUC_2D TSPLIB
instance (repeatable reference data; `nearest_euc_instance`). Note: the largest
EUC_2D instance is d18512, so the 20000 and 50000 targets both resolve to it —
euclidean TSPLIB has nothing near 50k, so the sweep tops out at ~18.5k.

Pipeline (all fp32 on the GB10, matrix resident):
  1. distances on the device (fp32);
  2. dual-VAT construction (min-non-zero seed) -> raw closed tour;
  3. neighbour-list LK polish (2-opt + Or-opt, candidates from the resident-matrix
     kNN) -> final tour.
Reference = the published optimum from the TSPLIB `solutions` file (no LKH). We
report % over optimum (raw and polished) and wall-clock (build / polish).

Run:  python -m experiments.vat_tsp_perf_report
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
    nearest_coord_instance,
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


def run(targets=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)):
    print("Dual-VAT (GPU build) + LK performance report (fp32, ref = optimum)")
    print("=" * 74)
    print(f"GPU: {gpu.is_available()}\n")
    # resolve each target to its nearest coordinate instance, de-duplicated
    seen = {}
    for tgt in targets:
        name, coords, dim, ewt = nearest_coord_instance(tgt)
        seen.setdefault(name, (coords, dim, ewt))
    instances = sorted(seen.items(), key=lambda kv: kv[1][1])  # by dimension

    print(
        f"  {'instance':>10s} {'n':>6s} {'ewt':>7s} {'opt':>11s} {'raw %':>7s} "
        f"{'final %':>8s} {'build s':>8s} {'polish s':>9s} {'total s':>8s}"
    )
    rows = []
    for name, (coords, dim, ewt) in instances:
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        ceil = ewt == "CEIL_2D"
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        knn = knn_device(Dg, 10)

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        raw, _label, _i, _j = dual_vat_tour_device(Dg, seed_mode="min")
        cp.cuda.Stream.null.synchronize()
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        final = lk_search(raw.copy(), coords, knn, ceil)
        t_polish = time.perf_counter() - t0

        raw_pct = 100.0 * (tour_len(raw, coords, ceil) - ref) / ref
        fin_pct = 100.0 * (tour_len(final, coords, ceil) - ref) / ref
        rows.append(
            dict(
                name=name,
                n=dim,
                ewt=ewt,
                raw=raw_pct,
                final=fin_pct,
                t_build=t_build,
                t_polish=t_polish,
                t_total=t_build + t_polish,
            )
        )
        print(
            f"  {name:>10s} {dim:6d} {ewt:>7s} {ref:11.0f} {raw_pct:6.0f}% "
            f"{fin_pct:7.1f}% {t_build:8.2f} {t_polish:9.2f} "
            f"{t_build + t_polish:8.2f}"
        )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def figure(rows):
    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        ns, [r["raw"] for r in rows], "s--", color="0.6", label="dual-VAT raw tour"
    )
    ax1.plot(
        ns,
        [r["final"] for r in rows],
        "o-",
        color="tab:green",
        label="dual-VAT + LK polish",
    )
    ax1.axhline(0, color="k", lw=0.8, ls=":", label="optimum")
    ax1.set_xscale("log")
    ax1.set_yscale("symlog")
    ax1.set_xlabel("n (cities)")
    ax1.set_ylabel("% over published optimum")
    ax1.set_title("A. tour quality vs n")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    for r in rows:
        ax1.annotate(
            r["name"],
            (r["n"], r["final"]),
            fontsize=6,
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    ax2.plot(
        ns, [r["t_build"] for r in rows], "^-", color="tab:blue", label="dual-VAT build"
    )
    ax2.plot(
        ns, [r["t_polish"] for r in rows], "D-", color="tab:orange", label="LK polish"
    )
    ax2.plot(ns, [r["t_total"] for r in rows], "o-", color="k", label="total")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("n (cities)")
    ax2.set_ylabel("wall-clock (s)")
    ax2.set_title("B. time vs n (fp32, GB10)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Dual-VAT (GPU build) + LK TSP performance on TSPLIB (fp32): quality & "
        f"time, n = {min(ns)} → {max(ns)}",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_perf_report.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    rows = run()
    print(f"\nwrote {figure(rows)}")
