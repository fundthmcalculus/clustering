"""Dual-VAT MST join mechanisms: endpoint vs a GPU N x M cycle-merge.

The dual-VAT construction gives two cluster VAT paths (P1, P2); joining them into
one closed tour ("closing the loop") is a separate choice. Two mechanisms:

  * endpoint  — connect the two paths at their 2x2 endpoints, best of 4
    orientations (only the path ends are candidates). O(1).
  * nxm       — GPU N x M cycle-merge: close each path into a sub-cycle, then take
    the best 2-opt-across move over ALL N x M cross edge pairs (remove one edge
    from each cycle, reconnect crosswise). The full N x M delta is evaluated on
    the device; O(N*M).

We compare on a **balanced** partition (two-density-peak seed, so N ~ M and the
N x M grid is meaningful), on nearest-size TSPLIB instances, reference = the
published optimum. Report tour cost (raw + LK-polished) and the join wall-clock.

Run:  python -m experiments.vat_tsp_join
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
    dual_vat_device,
    choose_seeds,
    join_endpoint,
    join_nxm_device,
    lk_search,
    tour_len,
)

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _sync():
    cp.cuda.Stream.null.synchronize()


def run(targets=(200, 500, 1000, 2000, 5000, 10000)):
    print("Dual-VAT join mechanisms: endpoint vs GPU N x M cycle-merge")
    print("=" * 70)
    print(f"GPU: {gpu.is_available()}   (balanced two-density-peak seed)\n")
    print(
        f"  {'instance':>9s} {'n':>6s} {'|C1|':>5s} {'|C2|':>5s} "
        f"{'endpt raw':>10s} {'endpt+LK':>9s} {'nxm raw':>8s} {'nxm+LK':>8s} "
        f"{'endpt s':>8s} {'nxm s':>8s}"
    )
    rows = []
    for tgt in targets:
        name, coords, dim, ewt = nearest_coord_instance(tgt)
        ceil = ewt == "CEIL_2D"
        opt = optimal_length(name)
        ref = float(opt) if opt else 1.0
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        knn = knn_device(Dg, 10)
        i0, j0 = choose_seeds(cp.asnumpy(Dg), coords, "two_dpeaks")
        _label, p1, p2 = dual_vat_device(Dg, i0, j0)

        def q(tour):
            return (
                100.0 * (tour_len(np.ascontiguousarray(tour), coords, ceil) - ref) / ref
            )

        out = {}
        for jn, fn in (("endpoint", join_endpoint), ("nxm", join_nxm_device)):
            _sync()
            t0 = time.perf_counter()
            tour = fn(Dg, p1, p2)
            _sync()
            t_join = time.perf_counter() - t0
            raw = q(tour)
            pol = q(lk_search(tour.copy(), coords, knn, ceil))
            out[jn] = dict(raw=raw, pol=pol, t=t_join)
        rows.append(dict(name=name, n=dim, c1=len(p1), c2=len(p2), **out))
        e, x = out["endpoint"], out["nxm"]
        print(
            f"  {name:>9s} {dim:6d} {len(p1):5d} {len(p2):5d} "
            f"{e['raw']:9.0f}% {e['pol']:8.1f}% {x['raw']:7.0f}% {x['pol']:7.1f}% "
            f"{e['t']:8.4f} {x['t']:8.4f}"
        )
        del Dg
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def figure(rows):
    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        ns,
        [r["endpoint"]["pol"] for r in rows],
        "o-",
        color="tab:blue",
        label="endpoint join + LK",
    )
    ax1.plot(
        ns,
        [r["nxm"]["pol"] for r in rows],
        "s-",
        color="tab:green",
        label="N×M cycle-merge + LK",
    )
    ax1.plot(
        ns,
        [r["endpoint"]["raw"] for r in rows],
        "o--",
        color="tab:blue",
        alpha=0.4,
        label="endpoint raw",
    )
    ax1.plot(
        ns,
        [r["nxm"]["raw"] for r in rows],
        "s--",
        color="tab:green",
        alpha=0.4,
        label="N×M raw",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("symlog")
    ax1.set_xlabel("n (cities)")
    ax1.set_ylabel("% over optimum")
    ax1.set_title("A. tour quality by join mechanism")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(
        ns,
        [r["endpoint"]["t"] for r in rows],
        "o-",
        color="tab:blue",
        label="endpoint join (O(1))",
    )
    ax2.plot(
        ns,
        [r["nxm"]["t"] for r in rows],
        "s-",
        color="tab:green",
        label="N×M cycle-merge (O(N·M), GPU)",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("n (cities)")
    ax2.set_ylabel("join wall-clock (s)")
    ax2.set_title("B. join time")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Dual-VAT join: endpoint vs GPU N×M cycle-merge (balanced partition, "
        "TSPLIB, fp32)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_join.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    rows = run()
    print(f"\nwrote {figure(rows)}")
