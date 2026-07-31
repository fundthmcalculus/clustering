"""Local-search comparison: 2-opt+Or-opt vs a variable-depth Lin-Kernighan.

`lk_search`     — neighbour-list 2-opt + Or-opt(1,2,3), best improvement.
`lk_search_vd`  — a variable-depth sequential LK: from each anchored city, a chain
                  of reverse-suffix 2-opt steps under the cumulative positive-gain
                  criterion, keeping the best-improving prefix (arbitrary depth).

Both start from the dual-VAT raw tour; compared on nearest-size TSPLIB instances
(fp32, reference = published optimum). We report the LK-vd on its own, and as a
refinement stage applied after 2-opt+Or-opt (does the deep chain find moves the
2-opt neighbourhood misses?).

Run:  python -m experiments.vat_tsp_lk
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
    lk_search_vd,
    tour_len,
)

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def run(targets=(200, 500, 1000, 2000, 5000)):
    print("Local search: 2-opt+Or-opt vs variable-depth LK (from dual-VAT raw)")
    print("=" * 72)
    print(f"GPU: {gpu.is_available()}   (reference = published optimum)\n")
    print(
        f"  {'instance':>9s} {'n':>6s} {'raw %':>7s} {'2opt+Or':>8s} {'LK-vd':>7s} "
        f"{'2opt→LKvd':>10s} {'t 2opt':>7s} {'t LKvd':>7s}"
    )
    rows = []
    for tgt in targets:
        name, coords, dim, ewt = nearest_coord_instance(tgt)
        ceil = ewt == "CEIL_2D"
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        knn = knn_device(Dg, 10)
        raw, _, _, _ = dual_vat_tour_device(Dg, "min")

        def q(t):
            return 100.0 * (tour_len(np.ascontiguousarray(t), coords, ceil) - ref) / ref

        t0 = time.perf_counter()
        a = lk_search(raw.copy(), coords, knn, ceil)
        t_2opt = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = lk_search_vd(raw.copy(), coords, knn, ceil)
        t_lkvd = time.perf_counter() - t0
        ab = lk_search_vd(a.copy(), coords, knn, ceil)  # LK-vd refining the 2-opt tour
        rows.append(
            dict(
                name=name,
                n=dim,
                raw=q(raw),
                o2=q(a),
                lkvd=q(b),
                o2_lkvd=q(ab),
                t2=t_2opt,
                tv=t_lkvd,
            )
        )
        r = rows[-1]
        print(
            f"  {name:>9s} {dim:6d} {r['raw']:6.0f}% {r['o2']:7.1f}% "
            f"{r['lkvd']:6.1f}% {r['o2_lkvd']:9.1f}% {r['t2']:6.2f}s {r['tv']:6.2f}s"
        )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def figure(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ns, [r["o2"] for r in rows], "o-", color="tab:blue", label="2-opt + Or-opt")
    ax.plot(
        ns,
        [r["lkvd"] for r in rows],
        "s-",
        color="tab:red",
        label="variable-depth LK (standalone)",
    )
    ax.plot(
        ns,
        [r["o2_lkvd"] for r in rows],
        "^-",
        color="tab:green",
        label="2-opt+Or-opt → LK-vd refine",
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("n (cities)")
    ax.set_ylabel("% over optimum")
    ax.set_title(
        "Local search from the dual-VAT tour: 2-opt+Or-opt vs " "variable-depth LK"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_lk.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    rows = run()
    print(f"\nwrote {figure(rows)}")
