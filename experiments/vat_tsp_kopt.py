"""Full 2-opt and 3-opt to convergence on the GPU-accelerated pipeline.

"Full round ... until no new moves are made": run each operator with NO pass cap,
looping until a complete sweep makes zero improving moves (a true local optimum).
Reports time, sweeps, moves, and quality (% over published optimum), starting from
the VAT tour, on the GPU-built distance matrix / kNN. Escalation: raw VAT ->
2-opt* -> 3-opt* (3-opt started from the 2-opt optimum, since 3-opt ⊇ 2-opt).

Operators:
  * two_opt_converge  — neighbour-list bidirectional 2-opt (scales; the 18k path).
  * gpu_two_opt       — exact full-neighbourhood (all O(n^2) pairs) 2-opt on the
                        GPU, for the true full-2-opt number at n=1000.
  * three_opt_converge — neighbour-list 3-opt: for each first cut, second/third
                        cuts drawn from kNN candidates; all 7 reconnections
                        evaluated by a recipe whose delta and application are the
                        same construction (self-tested vs brute force).

Run:  python -m experiments.vat_tsp_kopt [target_n]   (default 1000)
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

from tribbleclustering import gpu
from experiments.vat_tsp_tsplib import (
    knn_device,
    nearest_euc_instance,
    optimal_length,
    _d,
)
from experiments.vat_tsp_dualvat_lk import tour_len
from experiments.vat_tsp_2opt_bench import vat_order_nb
from experiments.vat_tsp_reslice import gpu_two_opt

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


# --------------------------------------------------------------------------- #
# 2-opt to convergence (neighbour-list, uncapped)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def two_opt_converge(tour, coords, knn, ceil):
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    K = knn.shape[1]
    sweeps = 0
    moves = 0
    while True:
        improved = False
        sweeps += 1
        for i in range(n):
            a = tour[i]
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
                moves += 1
        if not improved:
            break
    return sweeps, moves


# --------------------------------------------------------------------------- #
# 3-opt to convergence (neighbour-list, uncapped, recipe-based)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _dnew_code(coords, a, b, c, d, e, f, ceil, code):
    """New-edge distance sum for 3-opt reconnection `code` (1..7). Segments
    B=[i+1..j] (ends b,c), C=[j+1..k] (ends d,e); a=tour[i], f=tour[k+1]."""
    if code == 1:  # BC, reverse B
        return _d(coords, a, c, ceil) + _d(coords, b, d, ceil) + _d(coords, e, f, ceil)
    if code == 2:  # BC, reverse C
        return _d(coords, a, b, ceil) + _d(coords, c, e, ceil) + _d(coords, d, f, ceil)
    if code == 3:  # BC, reverse B and C
        return _d(coords, a, c, ceil) + _d(coords, b, e, ceil) + _d(coords, d, f, ceil)
    if code == 4:  # CB
        return _d(coords, a, d, ceil) + _d(coords, e, b, ceil) + _d(coords, c, f, ceil)
    if code == 5:  # CB, reverse B
        return _d(coords, a, d, ceil) + _d(coords, e, c, ceil) + _d(coords, b, f, ceil)
    if code == 6:  # CB, reverse C
        return _d(coords, a, e, ceil) + _d(coords, d, b, ceil) + _d(coords, c, f, ceil)
    # code == 7: CB, reverse B and C
    return _d(coords, a, e, ceil) + _d(coords, d, c, ceil) + _d(coords, b, f, ceil)


@njit(cache=True)
def _apply3(tour, pos, i, j, k, code, buf):
    order_cb = code >= 4
    rev_b = code == 1 or code == 3 or code == 5 or code == 7
    rev_c = code == 2 or code == 3 or code == 6 or code == 7
    w = 0
    if not order_cb:
        if rev_b:
            for x in range(j, i, -1):
                buf[w] = tour[x]
                w += 1
        else:
            for x in range(i + 1, j + 1):
                buf[w] = tour[x]
                w += 1
        if rev_c:
            for x in range(k, j, -1):
                buf[w] = tour[x]
                w += 1
        else:
            for x in range(j + 1, k + 1):
                buf[w] = tour[x]
                w += 1
    else:
        if rev_c:
            for x in range(k, j, -1):
                buf[w] = tour[x]
                w += 1
        else:
            for x in range(j + 1, k + 1):
                buf[w] = tour[x]
                w += 1
        if rev_b:
            for x in range(j, i, -1):
                buf[w] = tour[x]
                w += 1
        else:
            for x in range(i + 1, j + 1):
                buf[w] = tour[x]
                w += 1
    for t in range(w):
        tour[i + 1 + t] = buf[t]
        pos[buf[t]] = i + 1 + t


@njit(cache=True)
def three_opt_converge(tour, coords, knn, ceil):
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    K = knn.shape[1]
    buf = np.empty(n, np.int64)
    sweeps = 0
    moves = 0
    while True:
        improved = False
        sweeps += 1
        for i in range(n - 2):
            a = tour[i]
            b = tour[i + 1]
            d0_ab = _d(coords, a, b, ceil)
            best_gain = 1e-7
            best_code = -1
            best_j = -1
            best_k = -1
            for tj in range(K):
                j = pos[knn[a, tj]]
                if j <= i or j >= n - 1:
                    continue
                c = tour[j]
                d = tour[j + 1]
                d0_cd = _d(coords, c, d, ceil)
                for tk in range(K):
                    kk = pos[knn[b, tk]]
                    if kk <= j or kk >= n:
                        continue
                    e = tour[kk]
                    f = tour[(kk + 1) % n]
                    d0 = d0_ab + d0_cd + _d(coords, e, f, ceil)
                    for code in range(1, 8):
                        g = d0 - _dnew_code(coords, a, b, c, d, e, f, ceil, code)
                        if g > best_gain:
                            best_gain = g
                            best_code = code
                            best_j = j
                            best_k = kk
            if best_code > 0:
                _apply3(tour, pos, i, best_j, best_k, best_code, buf)
                improved = True
                moves += 1
                b = tour[i + 1]
        if not improved:
            break
    return sweeps, moves


# --------------------------------------------------------------------------- #
# self-test: recipe delta must match applied length change, tours stay valid
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(1)
    n = 11
    coords = (rng.random((n, 2)) * 100.0).astype(np.float64)
    buf = np.empty(n, np.int64)
    for _ in range(300):
        tour = rng.permutation(n).astype(np.int64)
        pos = np.empty(n, np.int64)
        for x in range(n):
            pos[tour[x]] = x
        i = rng.integers(0, n - 3)
        j = rng.integers(i + 1, n - 2)
        k = rng.integers(j + 1, n)
        a, b = tour[i], tour[i + 1]
        c, d = tour[j], tour[j + 1]
        e, f = tour[k], tour[(k + 1) % n]
        L0 = tour_len(tour, coords, False)
        d0 = _d(coords, a, b, False) + _d(coords, c, d, False) + _d(coords, e, f, False)
        for code in range(1, 8):
            t2 = tour.copy()
            p2 = pos.copy()
            dnew = _dnew_code(coords, a, b, c, d, e, f, False, code)
            _apply3(t2, p2, i, j, k, code, buf)
            assert len(np.unique(t2)) == n, "not a permutation"
            L1 = tour_len(t2, coords, False)
            if abs((L0 - d0 + dnew) - L1) > 0.5:
                raise AssertionError(
                    f"recipe {code} delta mismatch: " f"{L0-d0+dnew} vs {L1}"
                )
    return True


def run(target=1000):
    name, coords, dim = nearest_euc_instance(target)
    opt = optimal_length(name)
    ref = float(opt) if opt else None
    # float64 device matrix: the GPU all-pairs 2-opt kernel expects double
    Dg = gpu.pairwise_distances_device(coords, dtype="float64")
    D = cp.asnumpy(Dg)
    knn = knn_device(Dg, 10)
    start = int(cp.asnumpy(cp.argmax(Dg))) // dim
    raw = vat_order_nb(D, start)

    def pct(t):
        L = tour_len(np.ascontiguousarray(t), coords, False)
        return 100.0 * (L - ref) / ref if ref else float("nan"), L

    # warm up the numba operators on a small independent instance (JIT out of timings)
    wc = (np.random.default_rng(0).random((64, 2)) * 100.0).astype(np.float64)
    wD = np.sqrt(((wc[:, None] - wc[None]) ** 2).sum(-1))
    wknn = np.argsort(wD, axis=1)[:, 1:11].astype(np.int32)
    two_opt_converge(np.arange(64), wc, wknn, False)
    three_opt_converge(np.arange(64), wc, wknn, False)

    q_raw, _ = pct(raw)
    print(f"instance {name}  n={dim}  (published optimum = {opt})")
    print(f"  raw VAT tour:                 +{q_raw:6.2f}%")

    # --- full 2-opt to convergence (neighbour-list) ---
    t = raw.copy()
    t0 = time.perf_counter()
    sw, mv = two_opt_converge(t, coords, knn, False)
    dt2 = time.perf_counter() - t0
    q2, _ = pct(t)
    print(
        f"  2-opt* (neighbour-list):      +{q2:6.2f}%   {dt2:7.3f}s   "
        f"{sw} sweeps, {mv} moves  (converged: last sweep 0 moves)"
    )

    # --- exact full-neighbourhood 2-opt on the GPU (all pairs) ---
    # one move/pass (O(n^2) kernel per move) — only tractable at small n
    tg = None
    if dim <= 4000:
        t0 = time.perf_counter()
        tg, pg = gpu_two_opt(raw.copy(), Dg)
        dtg = time.perf_counter() - t0
        qg, _ = pct(tg)
        print(
            f"  2-opt (GPU all-pairs):        +{qg:6.2f}%   {dtg:7.3f}s   "
            f"{pg} passes (1 move/pass)"
        )
    else:
        print(
            "  2-opt (GPU all-pairs):        skipped (1 move/pass; "
            "neighbour-list is the scalable path)"
        )

    # --- full 3-opt to convergence, from the neighbour-list 2-opt optimum ---
    t3 = t.copy()
    t0 = time.perf_counter()
    sw3, mv3 = three_opt_converge(t3, coords, knn, False)
    dt3 = time.perf_counter() - t0
    q3, _ = pct(t3)
    print(
        f"  3-opt* (from nbr-2-opt):      +{q3:6.2f}%   {dt3:7.3f}s   "
        f"{sw3} sweeps, {mv3} moves  (converged: last sweep 0 moves)"
    )

    # --- full 3-opt to convergence, from the exact full 2-opt optimum ---
    t3b = None
    if tg is not None:
        t3b = tg.copy()
        t0 = time.perf_counter()
        sw3b, mv3b = three_opt_converge(t3b, coords, knn, False)
        dt3b = time.perf_counter() - t0
        q3b, _ = pct(t3b)
        print(
            f"  3-opt* (from full 2-opt):     +{q3b:6.2f}%   {dt3b:7.3f}s   "
            f"{sw3b} sweeps, {mv3b} moves"
        )

    # --- tour image (reporting template): raw -> full 2-opt -> +3-opt ---
    mid = tg if tg is not None else t
    mid_lab = "full 2-opt (all-pairs)" if tg is not None else "2-opt* (nbr-list)"
    fin = t3b if t3b is not None else t3
    panels = [
        (raw, f"raw VAT tour  (+{pct(raw)[0]:.0f}%)"),
        (mid, f"{mid_lab}  (+{pct(mid)[0]:.1f}%)"),
        (fin, f"then 3-opt*  (+{pct(fin)[0]:.1f}%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (tr, ttl) in zip(axes, panels):
        loop = np.append(tr, tr[0])
        ax.plot(coords[loop, 0], coords[loop, 1], "-", lw=0.5, color="tab:blue")
        ax.plot(coords[:, 0], coords[:, 1], ".", ms=1.5, color="k")
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{name} (n={dim}): full 2-opt and 3-opt to convergence (GPU pipeline)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"vat_tsp_kopt_{name}.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print(f"  wrote {path}")

    del Dg, D
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    assert _selftest(), "3-opt recipe self-test failed"
    print("3-opt recipe self-test: OK\n")
    tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run(tgt)
