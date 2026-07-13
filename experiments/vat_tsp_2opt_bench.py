"""Benchmark: consensus-restricted 2-opt vs full 2-opt, n = 50 .. 20 000.

From the VAT sequence-variation study (`VAT_TSP_SEQVAR_FINDINGS.md`): most of the
VAT ordering is stable across starts; only the low-consensus *swap points* (the
long between-cluster seams) move. So restrict 2-opt to initiate moves only from
the **seam cities**, freezing the consistent subsequences — and see whether that
matches full 2-opt's quality for a fraction of the time.

Three tours per instance, all from the same VAT reference start:
  * raw VAT tour (the shared start),
  * **full** neighbour-list 2-opt (every city may initiate),
  * **consensus** neighbour-list 2-opt (only seam cities initiate).

Consensus is built by sampling S VAT orders from different starts and counting how
often each reference-tour link recurs (a dict over consecutive pairs — no n x n
matrix, so it scales to 20k). Time AND quality (% over published optimum) reported;
nearest-size EUC_2D TSPLIB instances, official nint rounding.

Run:  python -m experiments.vat_tsp_2opt_bench
"""

from __future__ import annotations

import time
from collections import Counter
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
    _d,
)
from experiments.vat_tsp_dualvat_lk import tour_len  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"
SIZES = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)


@njit(cache=True)
def vat_order_nb(D, start):
    """VAT ordering (Prim insertion order over the MST) grown from `start`."""
    n = D.shape[0]
    in_tree = np.zeros(n, np.bool_)
    key = np.empty(n, np.float64)
    for v in range(n):
        key[v] = D[start, v]
    key[start] = np.inf
    order = np.empty(n, np.int64)
    in_tree[start] = True
    order[0] = start
    for i in range(1, n):
        best = -1
        bestv = np.inf
        for v in range(n):
            if (not in_tree[v]) and key[v] < bestv:
                bestv = key[v]
                best = v
        u = best
        in_tree[u] = True
        order[i] = u
        for v in range(n):
            if not in_tree[v]:
                du = D[u, v]
                if du < key[v]:
                    key[v] = du
    return order


@njit(cache=True)
def two_opt_only(tour, coords, knn, ceil, allow, max_pass=80):
    """Strong pure 2-opt: best-improvement neighbour-list 2-opt (the same operator
    as lk_search minus Or-opt), handling the closed-tour wrap edge (q+1 wraps).
    Only cities with allow[city] may *initiate* a move; segment reversals still
    apply to any span, so a frozen consistent subsequence is preserved (a reversal
    keeps its internal adjacency). allow all-True = full 2-opt; allow = consensus
    seam cities = consensus-restricted 2-opt."""
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    K = knn.shape[1]
    for _ in range(max_pass):
        improved = False
        for i in range(n):
            a = tour[i]
            if not allow[a]:
                continue
            bg = 1e-7
            bp = -1
            bq = -1
            for t in range(K):
                c = knn[a, t]
                j = pos[c]
                p = i if i < j else j
                q = i if i > j else j
                if q <= p:
                    continue
                pn = (p + 1) % n
                qn = (q + 1) % n
                if pn == q and qn == p:
                    continue
                gain = (
                    _d(coords, tour[p], tour[pn], ceil)
                    + _d(coords, tour[q], tour[qn], ceil)
                    - _d(coords, tour[p], tour[q], ceil)
                    - _d(coords, tour[pn], tour[qn], ceil)
                )
                if gain > bg:
                    bg = gain
                    bp = p
                    bq = q
            if bp >= 0:
                lo, hi = bp + 1, bq
                while lo < hi:
                    tour[lo], tour[hi] = tour[hi], tour[lo]
                    pos[tour[lo]] = lo
                    pos[tour[hi]] = hi
                    lo += 1
                    hi -= 1
                if lo == hi:
                    pos[tour[lo]] = lo
                improved = True
        if not improved:
            break
    return tour


def consensus_seams(D, ref, starts, tau, N):
    """Seam cities = endpoints of reference-tour links whose recurrence across the
    sampled VAT starts is below tau. Dict of consecutive pairs, no n x n matrix."""
    cnt = Counter()
    for s in starts:
        o = vat_order_nb(D, int(s))
        a, b = o[:-1], o[1:]
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        cnt.update((lo * N + hi).tolist())
    ra, rb = ref[:-1], ref[1:]
    rk = np.minimum(ra, rb).astype(np.int64) * N + np.maximum(ra, rb).astype(np.int64)
    freq = np.array([cnt.get(int(k), 0) for k in rk], dtype=np.float64) / len(starts)
    swaps = np.where(freq < tau)[0]
    seam = np.zeros(N, dtype=np.bool_)
    for i in swaps:
        seam[ref[i]] = True
        seam[ref[i + 1]] = True
    return swaps, seam


def run(sizes=SIZES, n_starts=24, tau=0.5):
    # numba warm-up at a realistic size (keep JIT/cache load out of the timings)
    wc = np.random.default_rng(0).random((256, 2)) * 100.0
    Dw = np.sqrt(((wc[:, None] - wc[None]) ** 2).sum(-1))
    wknn = np.argsort(Dw, axis=1)[:, 1:11].astype(np.int32)
    wo = vat_order_nb(Dw, 0)
    two_opt_only(wo.copy(), wc, wknn, False, np.ones(256, np.bool_), 5)
    two_opt_only(wo.copy(), wc, wknn, False, np.zeros(256, np.bool_), 5)
    consensus_seams(Dw, wo, np.arange(4), 0.5, 256)

    print(f"consensus-restricted vs full 2-opt  (S={n_starts} starts, tau={tau})")
    print("=" * 78)
    print(
        f"  {'inst':>9s} {'n':>6s} {'raw%':>6s} {'full%':>6s} {'cons%':>6s} "
        f"{'seam%':>6s} {'t_full':>8s} {'t_cons':>8s} {'t_build':>8s} {'speedup':>7s}"
    )
    rows = []
    for tgt in sizes:
        name, coords, dim = nearest_euc_instance(tgt)
        opt = optimal_length(name)
        ref_len = float(opt) if opt else None
        Dg = gpu.pairwise_distances_device(coords, dtype="float32")
        D = cp.asnumpy(Dg)
        knn = knn_device(Dg, 10)
        start = int(cp.asnumpy(cp.argmax(Dg))) // dim
        ref = vat_order_nb(D, start)

        def q(t):
            L = tour_len(np.ascontiguousarray(t), coords, False)
            return 100.0 * (L - ref_len) / ref_len if ref_len else float("nan")

        starts = np.linspace(0, dim - 1, min(n_starts, dim), dtype=int)
        t0 = time.perf_counter()
        swaps, seam = consensus_seams(D, ref, starts, tau, dim)
        t_build = time.perf_counter() - t0

        allow_all = np.ones(dim, dtype=np.bool_)
        t0 = time.perf_counter()
        full = two_opt_only(ref.copy(), coords, knn, False, allow_all)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        cons = two_opt_only(ref.copy(), coords, knn, False, seam)
        t_cons = time.perf_counter() - t0

        rows.append(
            dict(
                name=name,
                n=dim,
                raw=q(ref),
                full=q(full),
                cons=q(cons),
                seam=100.0 * seam.mean(),
                t_full=t_full,
                t_cons=t_cons,
                t_build=t_build,
                speedup=t_full / max(t_cons, 1e-9),
            )
        )
        r = rows[-1]
        print(
            f"  {name:>9s} {dim:6d} {r['raw']:5.0f}% {r['full']:5.1f}% "
            f"{r['cons']:5.1f}% {r['seam']:5.0f}% {r['t_full']:7.3f}s "
            f"{r['t_cons']:7.3f}s {r['t_build']:7.3f}s {r['speedup']:6.1f}x"
        )
        del Dg, D
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def figure(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    ax[0].plot(ns, [r["raw"] for r in rows], "d--", color="0.6", label="raw VAT tour")
    ax[0].plot(
        ns, [r["full"] for r in rows], "o-", color="tab:blue", label="full 2-opt"
    )
    ax[0].plot(
        ns,
        [r["cons"] for r in rows],
        "^-",
        color="tab:red",
        label="consensus 2-opt (seams only)",
    )
    ax[0].set_title("quality: % over optimum")
    ax[0].set_ylabel("% over optimum")
    ax[0].set_yscale("symlog")

    ax[1].plot(
        ns, [r["t_full"] for r in rows], "o-", color="tab:blue", label="full 2-opt"
    )
    ax[1].plot(
        ns, [r["t_cons"] for r in rows], "^-", color="tab:red", label="consensus 2-opt"
    )
    ax[1].plot(
        ns,
        [r["t_build"] for r in rows],
        "s:",
        color="tab:green",
        label="consensus build (S starts)",
    )
    ax[1].set_title("time (2-opt run; build shown separately)")
    ax[1].set_ylabel("seconds")
    ax[1].set_yscale("log")

    # end-to-end wall time: full 2-opt needs no build; consensus pays build + run
    ax[2].plot(
        ns,
        [r["t_full"] for r in rows],
        "o-",
        color="tab:blue",
        label="full 2-opt (end-to-end)",
    )
    ax[2].plot(
        ns,
        [r["t_build"] + r["t_cons"] for r in rows],
        "^-",
        color="tab:red",
        label="consensus end-to-end (build+run)",
    )
    ax[2].set_title("end-to-end wall time (build included)")
    ax[2].set_ylabel("seconds")
    ax[2].set_yscale("log")
    ax2b = ax[2].twinx()
    ax2b.plot(
        ns,
        [r["seam"] for r in rows],
        "s:",
        color="tab:purple",
        label="seam cities (active %)",
    )
    ax2b.set_ylabel("% seam cities (active)", color="tab:purple")
    ax2b.legend(fontsize=8, loc="center right")

    for a in ax:
        a.set_xlabel("n (cities)")
        a.set_xscale("log")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(
        "Consensus-restricted vs full 2-opt from the VAT tour (n=50..20k)", fontsize=13
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_2opt_bench.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    rows = run()
    print(f"\nwrote {figure(rows)}")
