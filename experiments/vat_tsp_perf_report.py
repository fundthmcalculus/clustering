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
    nearest_euc_instance,
    optimal_length,
)
from experiments.vat_tsp_dualvat_lk import dual_vat_tour, lk_search, tour_len  # noqa

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def run(targets=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)):
    print("Dual-VAT + LK performance report (fp32, reference = published optimum)")
    print("=" * 72)
    print(f"GPU: {gpu.is_available()}\n")
    # resolve each target to its nearest EUC_2D instance, de-duplicated
    seen = {}
    for tgt in targets:
        name, coords, dim = nearest_euc_instance(tgt)
        seen.setdefault(name, (coords, dim))
    instances = sorted(seen.items(), key=lambda kv: kv[1][1])  # by dimension

    print(
        f"  {'instance':>10s} {'n':>6s} {'opt':>10s} {'raw %':>8s} "
        f"{'final %':>8s} {'build s':>8s} {'polish s':>9s} {'total s':>8s}"
    )
    rows = []
    for name, (coords, dim) in instances:
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        knn = knn_device(Dg, 10)
        D = cp.asnumpy(Dg)  # fp32 host matrix for the dual-VAT growth

        t0 = time.perf_counter()
        raw, _label, _i, _j = dual_vat_tour(D, coords, seed_mode="min")
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        final = lk_search(raw.copy(), coords, knn)
        t_polish = time.perf_counter() - t0

        raw_pct = 100.0 * (tour_len(raw, coords) - ref) / ref
        fin_pct = 100.0 * (tour_len(final, coords) - ref) / ref
        rows.append(
            dict(
                name=name,
                n=dim,
                raw=raw_pct,
                final=fin_pct,
                t_build=t_build,
                t_polish=t_polish,
                t_total=t_build + t_polish,
            )
        )
        print(
            f"  {name:>10s} {dim:6d} {ref:10.0f} {raw_pct:7.0f}% {fin_pct:7.1f}% "
            f"{t_build:8.2f} {t_polish:9.2f} {t_build + t_polish:8.2f}"
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
        "Dual-VAT + LK TSP performance on TSPLIB (EUC_2D, fp32): quality & time, "
        "n = 51 → 18512",
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
