"""Spike (follow-up to vat_tsp_warmstart.py): benchmark VAT/MST warm starts and a
VAT-cluster-blocking strategy against a real Lin-Kernighan (LKH) solver.

Two questions the earlier spikes deferred:

  1. Against a STRONG solver (LKH, via `elkai`), how good are the cheap VAT/MST
     warm start and 2-opt refinement -- measured as % over the LKH tour on
     synthetic instances at n ~ 50, 500, 5000? (No TSPLIB; self-contained
     generators so the cluster-blocking strategy has real structure to exploit.)

  2. CLUSTER-BLOCKING: find good blocks (VAT single-linkage cut / k-means), solve
     each block's sub-TSP, then optimize the block-to-block connections into one
     global tour -- "cluster-first, route-second". Does using VAT to FIND the
     blocks, plus an optimized stitch, approach flat-LKH quality at lower
     (parallel) cost, and where does it break?

LKH baseline: `elkai` (Lin-Kernighan-Helsgaun) if importable; the script degrades
to "no-LKH" (LKH columns blank, best-of-our-methods as the reference) otherwise --
there is no LK implementation in this repo, so the real solver is the honest
baseline. All costs use an integer distance matrix (what LKH consumes), so every
method is compared on exactly the same objective.

Run:  python -m experiments.vat_tsp_benchmark
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c  # noqa: E402
from experiments.vat_tsp import two_opt as two_opt_tour  # noqa: E402
from experiments.vat_tsp_warmstart import (  # noqa: E402
    nn_order,
    greedy_edge_order,
    mst_dfs_order,
    _seed_of,
)
from experiments.adversarial_eval import two_moons, circles, easy_blobs  # noqa: E402
from experiments.hardening_eval import d_geodesic  # noqa: E402
from experiments.stitched_vat import maximin_partition  # noqa: E402

try:
    import elkai  # type: ignore

    _HAS_LKH = True
except ImportError:  # pragma: no cover - optional experiment dep
    _HAS_LKH = False

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Integer distance matrices (what LKH consumes; keeps every method on one metric)
# ---------------------------------------------------------------------------
def _int_euclid(X, scale=1.0):
    d = X[:, None, :] - X[None, :, :]
    D = np.sqrt((d * d).sum(-1)) * scale
    return np.rint(D).astype(np.int64)


def instance(name, n, seed=1):
    """Return an integer symmetric distance matrix for a named synthetic family."""
    if name == "blobs":
        X, _ = easy_blobs(n, seed=seed)
        return _int_euclid(X, scale=40.0)
    if name == "uniform":
        X = np.random.default_rng(seed).random((n, 2)) * 1000.0
        return _int_euclid(X)
    if name == "moons":
        X, _ = two_moons(n, noise=0.08, seed=seed)
        return _int_euclid(X, scale=300.0)
    if name == "circles":
        X, _ = circles(n, noise=0.06, seed=seed)
        return _int_euclid(X, scale=300.0)
    if name == "geodesic":
        X, _ = two_moons(n, noise=0.08, seed=seed)
        G = d_geodesic(X)
        return np.rint(G / G.max() * 5000.0).astype(np.int64)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Closed-tour helpers over an integer matrix (used as float64 for the njit kernels)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _closed_cost(tour, D):
    n = tour.shape[0]
    s = 0.0
    for k in range(n):
        s += D[tour[k], tour[(k + 1) % n]]
    return s


def vat_tour(Df):
    _, _, p = compute_ivat_c(Df.copy(), inplace=False)
    return np.ascontiguousarray(p, dtype=np.int64)


def lkh_tour(D_int, runs=1):
    """LKH tour (closed) via elkai -> order array (length n). None if no elkai."""
    if not _HAS_LKH:
        return None
    sol = elkai.DistanceMatrix(D_int.tolist()).solve_tsp(runs=runs)
    return np.asarray(sol[:-1], dtype=np.int64)  # drop the repeated return-to-start


def constructions(Df):
    """{name: closed tour order} for the cheap construction heuristics."""
    D_int = Df.astype(np.int64)
    s = _seed_of(Df)
    return {
        "nearest-neighbour": nn_order(Df, s),
        "greedy-edge": greedy_edge_order(Df),
        "MST double-tree": mst_dfs_order(D_int, s),
        "VAT (free)": vat_tour(Df),
    }


# LKH is expensive (a single run at n=5000 ~ 7 min), so compute each instance's
# reference tour exactly once and share it across Part 1 and Part 2.
_LKH_CACHE = {}


def lkh_reference(fam, n, D_int, Df):
    key = (fam, n)
    if key not in _LKH_CACHE:
        t = time.perf_counter()
        tour = lkh_tour(D_int, runs=1)
        dt = time.perf_counter() - t
        ref = _closed_cost(tour, Df) if tour is not None else None
        _LKH_CACHE[key] = (ref, dt)
    return _LKH_CACHE[key]


# ---------------------------------------------------------------------------
# Part 1 — warm starts + 2-opt vs LKH, across families and n
# ---------------------------------------------------------------------------
def benchmark_report(plan):
    print("\n=== 1. warm start + 2-opt vs LKH (Lin-Kernighan-Helsgaun) ===")
    print("    numbers are % over the LKH tour (lower = better; LKH ~= optimal)")
    if not _HAS_LKH:
        print("    [elkai/LKH not installed -> reference = best of our methods]")
    results = {}
    for fam, n in plan:
        D_int = instance(fam, n)
        Df = np.ascontiguousarray(D_int.astype(np.float64))
        ref, _ = lkh_reference(fam, n, D_int, Df)

        row = {}
        starts = constructions(Df)
        r_ls = []
        for sd in range(3):
            o = np.random.default_rng(50 + sd).permutation(n)
            two_opt_tour(o, Df)
            r_ls.append(_closed_cost(o, Df))
        row["random (x3)"] = (np.nan, float(np.mean(r_ls)))
        for m, o in starts.items():
            init_c = _closed_cost(o, Df)
            o2 = o.copy()
            two_opt_tour(o2, Df)
            row[m] = (init_c, _closed_cost(o2, Df))
        if ref is None:
            ref = min(v[1] for v in row.values())
        results[(fam, n)] = (row, ref)

        print(f"\n  {fam} n={n}  (LKH ref = {ref:.0f})")
        print(f"    {'method':18s} {'init %':>8s} {'+2opt %':>8s}")
        for m in (
            "random (x3)",
            "nearest-neighbour",
            "greedy-edge",
            "MST double-tree",
            "VAT (free)",
        ):
            init_c, ls_c = row[m]
            ip = "   -   " if np.isnan(init_c) else f"{100*(init_c-ref)/ref:7.1f}"
            print(f"    {m:18s} {ip:>8s} {100*(ls_c-ref)/ref:8.1f}")
    return results


# ---------------------------------------------------------------------------
# Part 2 — cluster-blocking + optimized block-to-block connections
# ---------------------------------------------------------------------------
def vat_blocks(Df, B):
    """Find B blocks as VAT/single-linkage segments: cut the iVAT superdiagonal
    at its top B-1 gaps -> contiguous runs of the VAT order."""
    img, _, p = compute_ivat_c(Df.copy(), inplace=False)
    p = np.ascontiguousarray(p, dtype=np.int64)
    diag = np.diag(img, 1)
    cuts = np.sort(np.argsort(diag)[-(B - 1) :]) if B > 1 else np.array([], int)
    blocks, prev = [], 0
    for c in list(cuts) + [len(p) - 1]:
        blocks.append(p[prev : c + 1])
        prev = c + 1
    return [b for b in blocks if len(b) > 0]


def _solve_block(sub_int, lkh_cap=700):
    """Best cheap block sub-tour: LKH for small blocks (m <= lkh_cap), else
    NN + 2-opt (keeps a fat VAT block from costing minutes of LKH)."""
    m = sub_int.shape[0]
    if m <= 3:
        return np.arange(m, dtype=np.int64)
    if _HAS_LKH and m <= lkh_cap:
        t = lkh_tour(sub_int, runs=1)
        if t is not None:
            return t
    Ds = np.ascontiguousarray(sub_int.astype(np.float64))
    o = nn_order(Ds, 0)
    two_opt_tour(o, Ds)
    return o


@njit(cache=True)
def _open_from_subtour(subtour, sub):
    """Cut a closed block sub-tour at its longest edge -> open local path."""
    m = subtour.shape[0]
    worst, wi = -1.0, 0
    for k in range(m):
        d = sub[subtour[k], subtour[(k + 1) % m]]
        if d > worst:
            worst, wi = d, k
    out = np.empty(m, np.int64)
    q = 0
    for k in range(wi + 1, m):
        out[q] = subtour[k]
        q += 1
    for k in range(wi + 1):
        out[q] = subtour[k]
        q += 1
    return out


def _orient_cycle(cyc, ep, D_int):
    """Given a block cycle and each block's 2 endpoints, choose per-block
    orientation minimizing the sum of junction edges (cyclic DP, 2 states)."""
    B = len(cyc)
    if B == 1:
        return [0]

    def exit_id(bi, o):
        return ep[cyc[bi], 1 - o]

    def enter_id(bi, o):
        return ep[cyc[bi], o]

    best_total, best_choice = None, None
    for o0 in (0, 1):
        # position 0 fixed to o0; dp[o] = best cost with current block at orient o
        dp = {o0: 0.0}
        paths = {o0: [o0]}
        for pos in range(1, B):
            ndp, npaths = {}, {}
            for o in (0, 1):
                bestc, besto = None, None
                for op in dp:
                    c = dp[op] + D_int[exit_id(pos - 1, op), enter_id(pos, o)]
                    if bestc is None or c < bestc:
                        bestc, besto = c, op
                ndp[o] = bestc
                npaths[o] = paths[besto] + [o]
            dp, paths = ndp, npaths
        # close the cycle: exit of last -> enter of first (o0)
        for o in dp:
            total = dp[o] + D_int[exit_id(B - 1, o), enter_id(0, o0)]
            if best_total is None or total < best_total:
                best_total, best_choice = total, paths[o]
    return best_choice


def solve_blocks(Df, B, blocking):
    """Partition into blocks and solve each block's sub-TSP ONCE -> (paths, ep,
    block_solve_times). Reused for the naive / opt / opt+polish stitches."""
    D_int = Df.astype(np.int64)
    blocks = maximin_partition(Df, B) if blocking == "maximin" else vat_blocks(Df, B)
    paths, btimes = [], []
    for g in blocks:
        if len(g) == 1:
            paths.append(g.copy())
            btimes.append(0.0)
            continue
        sub = np.ascontiguousarray(D_int[np.ix_(g, g)])
        t0 = time.perf_counter()
        st = _solve_block(sub)
        btimes.append(time.perf_counter() - t0)
        loc = _open_from_subtour(np.ascontiguousarray(st), sub.astype(np.float64))
        paths.append(g[loc])
    ep = np.array([[pp[0], pp[-1]] for pp in paths], dtype=np.int64)
    return paths, ep, btimes


def stitch(Df, paths, ep, optimized=True, polish=False):
    """Stitch solved block paths into one closed tour. optimized reorders blocks
    (TSP over endpoints) + picks per-block orientation; polish runs a global
    2-opt. Returns (tour, t_stitch, t_polish)."""
    D_int = Df.astype(np.int64)
    Bn = len(paths)
    t_stitch = time.perf_counter()
    if not optimized or Bn == 1:
        tour = np.concatenate(paths)
    else:
        Bd = np.zeros((Bn, Bn), dtype=np.int64)
        for i in range(Bn):
            for j in range(Bn):
                if i != j:
                    Bd[i, j] = min(
                        D_int[ep[i, a], ep[j, b]] for a in (0, 1) for b in (0, 1)
                    )
        if Bn <= 3:
            cyc = list(range(Bn))
        else:
            c = lkh_tour(Bd, runs=2)
            cyc = list(c) if c is not None else list(range(Bn))
        orient = _orient_cycle(cyc, ep, D_int)
        seq = [
            paths[bi] if orient[pos] == 0 else paths[bi][::-1]
            for pos, bi in enumerate(cyc)
        ]
        tour = np.concatenate(seq)
    dt_stitch = time.perf_counter() - t_stitch

    t_polish = 0.0
    if polish:
        tp = time.perf_counter()
        tour = np.ascontiguousarray(tour, dtype=np.int64)
        two_opt_tour(tour, Df)
        t_polish = time.perf_counter() - tp
    return tour, dt_stitch, t_polish


def blocking_report(plan, block_of=None):
    print("\n=== 2. cluster-blocking + optimized block-to-block connections ===")
    print("    tour % over LKH. naive = block-order concat; opt = block-TSP order +")
    print("    best orientation; +polish = one global 2-opt. t_par = max-block solve")
    print("    + stitch (+ polish) -- the parallel-proxy wall-clock; t_lkh = flat LKH.")
    block_of = block_of or {"blobs": 8, "uniform": 16, "moons": 8, "circles": 8}
    out = {}
    for fam, n in plan:
        D_int = instance(fam, n)
        Df = np.ascontiguousarray(D_int.astype(np.float64))
        base, t_lkh = lkh_reference(fam, n, D_int, Df)

        fo = vat_tour(Df)
        two_opt_tour(fo, Df)
        flat = _closed_cost(fo, Df)
        B = block_of[fam]
        if base is None:
            base = flat

        rows = {"flat": (100 * (flat - base) / base, None)}
        print(f"\n  {fam} n={n}  B={B}   LKH ref={base:.0f} (flat LKH {t_lkh:.1f}s)")
        print(f"    {'strategy':26s} {'% over LKH':>11s} {'t_par s':>9s}")
        print(f"    {'flat VAT + 2opt':26s} {rows['flat'][0]:11.1f} {'-':>9s}")
        for blk in ("vat", "maximin"):
            paths, ep, btimes = solve_blocks(Df, B, blk)
            for tag, opt, pol in (
                ("naive", False, False),
                ("opt", True, False),
                ("opt+polish", True, True),
            ):
                tour, ds, tp = stitch(Df, paths, ep, optimized=opt, polish=pol)
                g = 100 * (_closed_cost(tour, Df) - base) / base
                t_par = max(btimes) + ds + tp
                rows[(blk, tag)] = (g, t_par)
                print(f"    {'  ' + blk + ' block ' + tag:26s} {g:11.1f} {t_par:9.2f}")
        out[(fam, n)] = (base, t_lkh, rows)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure(bench, blk):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))

    # A: Part 1 -- % over LKH after 2-opt, by method, at n=500
    ax = axes[0]
    methods = [
        "random (x3)",
        "nearest-neighbour",
        "greedy-edge",
        "MST double-tree",
        "VAT (free)",
    ]
    fams = [k for k in bench if k[1] == 500]
    fam_names = [f for f, _ in fams]
    colors = ["#999", "#e8a", "#6b9", "#c85", "#268"]
    x = np.arange(len(fams))
    w = 0.16
    for mi, m in enumerate(methods):
        vals = [bench[k][0][m][1] for k in fams]
        ref = [bench[k][1] for k in fams]
        pct = [100 * (v - r) / r for v, r in zip(vals, ref)]
        ax.bar(x + (mi - 2) * w, pct, w, label=m, color=colors[mi])
    ax.set_xticks(x)
    ax.set_xticklabels(fam_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% over LKH (after 2-opt)")
    ax.set_title("A. warm start + 2-opt vs LKH (n=500)", fontsize=11)
    ax.legend(fontsize=7)

    # B: Part 2 -- blocking strategies % over LKH at n=500
    ax = axes[1]
    b500 = [k for k in blk if k[1] == 500]
    strat = [
        "flat",
        ("vat", "naive"),
        ("vat", "opt"),
        ("vat", "opt+polish"),
        ("maximin", "opt+polish"),
    ]
    labels = [
        "flat VAT+2opt",
        "vat naive",
        "vat opt",
        "vat opt+polish",
        "maximin opt+polish",
    ]
    scol = ["#345", "#c44", "#e93", "#268", "#6b9"]
    x = np.arange(len(b500))
    w = 0.17
    for si, s in enumerate(strat):
        vals = [blk[k][2][s][0] for k in b500]
        ax.bar(x + (si - 2) * w, vals, w, label=labels[si], color=scol[si])
    ax.set_xticks(x)
    ax.set_xticklabels([f for f, _ in b500], fontsize=8)
    ax.set_ylabel("% over LKH")
    ax.set_title("B. cluster-blocking + stitch (n=500)", fontsize=11)
    ax.legend(fontsize=7)

    # C: scale -- flat LKH time vs blocked opt+polish, at n=5000
    ax = axes[2]
    big = [k for k in blk if k[1] == 5000]
    if big:
        names, tl, tp, gp = [], [], [], []
        for k in big:
            base, t_lkh, rows = blk[k]
            names.append(k[0])
            tl.append(t_lkh)
            tp.append(rows[("vat", "opt+polish")][1])
            gp.append(rows[("vat", "opt+polish")][0])
        xx = np.arange(len(names))
        ax.bar(xx - 0.2, tl, 0.4, label="flat LKH", color="#c44")
        ax.bar(xx + 0.2, tp, 0.4, label="blocked opt+polish", color="#268")
        ax.set_yscale("log")
        ax.set_xticks(xx)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("wall-clock s (log)")
        for i, g in enumerate(gp):
            ax.text(i + 0.2, tp[i] * 1.3, f"+{g:.0f}%", ha="center", fontsize=8)
        ax.set_title(
            "C. scale n=5000: blocked is far faster\n(+% = gap over LKH)", fontsize=11
        )
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    fig.suptitle(
        "VAT warm starts and VAT-cluster-blocking vs Lin-Kernighan (LKH): "
        "blocked LKH + optimized stitch + polish approaches LKH quality far faster at scale",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_benchmark.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("VAT warm starts + cluster-blocking vs LKH")
    print("=========================================")
    print(f"LKH (elkai) available: {_HAS_LKH}")
    t0 = time.perf_counter()
    small = ("blobs", "uniform", "moons", "circles")
    # n=5000 uses one clustered instance (blobs) shared across both parts: a
    # single flat-LKH reference there costs ~7 min, so we spend it once.
    bench_plan = (
        [(f, 50) for f in small] + [(f, 500) for f in small] + [("blobs", 5000)]
    )
    block_plan = [(f, 500) for f in ("blobs", "uniform", "moons")] + [("blobs", 5000)]
    bench = benchmark_report(bench_plan)
    blk = blocking_report(block_plan)
    print(f"\nwrote {figure(bench, blk)}")
    print(f"(total {time.perf_counter() - t0:.1f}s)")
