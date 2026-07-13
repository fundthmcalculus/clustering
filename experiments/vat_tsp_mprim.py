"""Bounded-run modified Prim tour construction (chain m steps, then switch branch).

The raw VAT tour is poor (+94% at n=1000) because Prim's *insertion order* takes
the globally-nearest frontier vertex every step, so consecutive **tour** edges
jump between branches and are long. This variant biases toward continuing the
current chain:

  * for up to `m` consecutive steps, extend greedily from the *current* vertex
    (nearest unvisited to it) — a nearest-neighbour chain of short tour edges;
  * on step m+1, take one standard Prim step (globally-nearest unvisited to the
    whole tree) — a minimal-cost jump to "another branch"; reset the counter.

m = 0  => pure Prim insertion order (= the VAT tour).
m -> n => pure nearest-neighbour.
Bounded m is an A*/beam-like compromise: commit to a branch for m moves, then
re-evaluate globally. We sweep m and measure raw quality, then neighbour-list
2-opt and 3-opt to convergence (does a better construction let the *scalable*
local search reach good quality, i.e. the 18k path?).

nearest-size EUC_2D TSPLIB, % over published optimum.
Run:  python -m experiments.vat_tsp_mprim [target_n]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numba import njit  # noqa: E402

from tribbleclustering import gpu  # noqa: E402
from experiments.vat_tsp_tsplib import (  # noqa: E402
    knn_device,
    nearest_euc_instance,
    optimal_length,
)
from experiments.vat_tsp_dualvat_lk import tour_len  # noqa: E402
from experiments.vat_tsp_kopt import two_opt_converge, three_opt_converge  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


@njit(cache=True)
def mprim_order(D, start, m):
    """Bounded-run modified Prim visiting order. `key[v]` = distance from v to the
    current tree; chain steps use distance to the *current* vertex, switch steps
    use `key`. Returns (order, n_switch_edges, switch_edge_total_len)."""
    n = D.shape[0]
    visited = np.zeros(n, np.bool_)
    order = np.empty(n, np.int64)
    key = D[start].astype(np.float64).copy()
    key[start] = np.inf
    visited[start] = True
    order[0] = start
    cur = start
    run = 0
    n_sw = 0
    sw_len = 0.0
    for step in range(1, n):
        if run < m:  # extend the chain: nearest unvisited to the current vertex
            bd = np.inf
            bv = -1
            for v in range(n):
                if (not visited[v]) and D[cur, v] < bd:
                    bd = D[cur, v]
                    bv = v
            nxt = bv
            run += 1
        else:  # forced switch: globally-nearest unvisited to the whole tree
            bd = np.inf
            bv = -1
            for v in range(n):
                if (not visited[v]) and key[v] < bd:
                    bd = key[v]
                    bv = v
            nxt = bv
            run = 0
            n_sw += 1
            sw_len += D[cur, nxt]
        visited[nxt] = True
        order[step] = nxt
        cur = nxt
        for v in range(n):
            if (not visited[v]) and D[nxt, v] < key[v]:
                key[v] = D[nxt, v]
    return order, n_sw, sw_len


def run(target=1000, ms=(0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)):
    name, coords, dim = nearest_euc_instance(target)
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    Dg = gpu.pairwise_distances_device(coords, dtype="float64")
    D = cp.asnumpy(Dg)
    knn = knn_device(Dg, 10)
    start = int(cp.asnumpy(cp.argmax(Dg))) // dim

    def pct(t):
        return 100.0 * (tour_len(np.ascontiguousarray(t), coords, False) - ref) / ref

    # warm up numba operators
    mprim_order(D[:32, :32].copy(), 0, 4)
    two_opt_converge(np.arange(32), coords[:32], knn[:32] % 32, False)
    three_opt_converge(np.arange(32), coords[:32], knn[:32] % 32, False)

    ms = tuple(m for m in ms if m < dim) + (dim,)  # include pure NN (m>=n)
    print(f"bounded-run modified Prim — {name} n={dim} (opt {opt})")
    print(f"  m=0 => VAT insertion order · m>=n => nearest-neighbour\n")
    print(
        f"  {'m':>5s} {'raw%':>7s} {'+2opt%':>7s} {'+3opt%':>7s} "
        f"{'switches':>8s} {'t_build':>8s}"
    )
    rows = []
    for m in ms:
        t0 = time.perf_counter()
        order, n_sw, _ = mprim_order(D, start, m)
        t_build = time.perf_counter() - t0
        raw = pct(order)
        t = order.copy()
        two_opt_converge(t, coords, knn, False)
        q2 = pct(t)
        three_opt_converge(t, coords, knn, False)
        q3 = pct(t)
        rows.append(dict(m=m, raw=raw, q2=q2, q3=q3, sw=n_sw, tb=t_build))
        tag = " (VAT)" if m == 0 else (" (NN)" if m >= dim else "")
        print(
            f"  {m:5d} {raw:6.1f}% {q2:6.2f}% {q3:6.2f}% {n_sw:8d} "
            f"{t_build:7.3f}s{tag}"
        )
    del Dg, D
    cp.get_default_memory_pool().free_all_blocks()
    return name, dim, rows


def figure(name, dim, rows):
    ms = [r["m"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].plot(ms, [r["raw"] for r in rows], "o-", color="0.5", label="raw tour")
    ax[0].plot(
        ms, [r["q2"] for r in rows], "s-", color="tab:blue", label="+ 2-opt* (nbr-list)"
    )
    ax[0].plot(ms, [r["q3"] for r in rows], "^-", color="tab:green", label="+ 3-opt*")
    ax[0].set_title(f"quality vs run-cap m — {name} n={dim}")
    ax[0].set_xlabel("m  (0 = VAT insertion order,  n = nearest-neighbour)")
    ax[0].set_ylabel("% over optimum")
    ax[0].set_xscale("symlog")
    ax[0].set_yscale("log")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(ms, [r["sw"] for r in rows], "o-", color="tab:red")
    ax[1].set_title("forced branch switches vs m")
    ax[1].set_xlabel("m")
    ax[1].set_ylabel("# switch (jump) edges")
    ax[1].set_xscale("symlog")
    ax[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Bounded-run modified Prim construction (chain m, then switch)", fontsize=12
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_mprim.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    name, dim, rows = run(tgt)
    print(f"\nwrote {figure(name, dim, rows)}")
