"""VAT->TSP cluster-blocking on real TSPLIB instances (DGX Spark GB10).

Complements experiments/vat_tsp_dgx_scale.py (synthetic blobs) with real
benchmark instances from the TSPLIB collection (git submodule
experiments/tsplib, https://github.com/mastqe/tsplib; the number in each
filename is the city count). Samples span ~50 to ~34000 cities.

The pipeline is the same VAT-cluster-blocking solver run on the DGX Spark's
unified-memory GPU front-end: build the n x n euclidean distance matrix and the
exact VAT ordering on the device (`gpu_vat.vat_gpu`), cut the VAT order into
size-capped blocks at its single-linkage gaps, solve each block with LKH
(`elkai`), and optimize the block-to-block stitch. Tour lengths use TSPLIB's
official rounding (EUC_2D = nint, CEIL_2D = ceil) so the % gap over the
published optimum is exactly comparable; flat LKH is run as a reference where it
is still affordable (n <= 2000).

Run:  python -m experiments.vat_tsp_tsplib
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu, gpu_vat  # noqa: E402
from experiments.vat_tsp import two_opt  # noqa: E402
from experiments.vat_tsp_warmstart import nn_order  # noqa: E402
from experiments.vat_tsp_dgx_scale import (  # noqa: E402
    _vat_gap_blocks,
    _orient_cycle_dist,
)
from experiments.vat_tsp_benchmark import _open_from_subtour  # noqa: E402

try:
    import elkai  # type: ignore

    _HAS_LKH = True
except ImportError:  # pragma: no cover
    _HAS_LKH = False

if gpu.is_available():
    import cupy as cp

TSPLIB = Path(__file__).parent / "tsplib"
FIG_DIR = Path(__file__).parent / "figures"

# Published TSPLIB optimal tour lengths (for the % gap).
OPTIMA = {
    "berlin52": 7542,
    "a280": 2579,
    "pr1002": 259045,
    "pcb3038": 137694,
    "fnl4461": 182566,
    "rl5915": 565530,
    "d15112": 1573084,
    "d18512": 645238,
}

# Instances to run (name, city count) — a spread from ~50 to ~18500. The benign
# coordinate instances top out here; the only larger TSPLIB instances are the
# pla* VLSI cases (CEIL_2D, huge integer distances) where per-block LKH is
# impractically slow — the synthetic scale study (experiments/vat_tsp_dgx_scale)
# carries the pipeline to n=80000 instead.
PLAN = [
    "berlin52",
    "a280",
    "pr1002",
    "pcb3038",
    "fnl4461",
    "rl5915",
    "d15112",
    "d18512",
]
LKH_FLAT_MAX = 2000  # flat LKH reference only where still affordable


def _sync():
    cp.cuda.Stream.null.synchronize()


# ---------------------------------------------------------------------------
# TSPLIB parsing + official distance rounding
# ---------------------------------------------------------------------------
def parse_tsplib(name):
    """Return (coords (n,2) float64, edge_weight_type) for a coordinate instance."""
    path = TSPLIB / f"{name}.tsp"
    ewt = "EUC_2D"
    coords = []
    with open(path, errors="ignore") as fh:
        in_nodes = False
        for line in fh:
            u = line.strip().upper()
            if u.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.split(":")[-1].strip()
            elif u.startswith("NODE_COORD_SECTION"):
                in_nodes = True
            elif u.startswith("EOF"):
                break
            elif in_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return np.ascontiguousarray(np.array(coords, dtype=np.float64)), ewt


def _round_fn(ewt):
    """TSPLIB integer distance rounding for a given edge-weight type."""
    if ewt == "CEIL_2D":
        return np.ceil
    # EUC_2D / default: nearest integer (TSPLIB nint = floor(x + 0.5))
    return lambda x: np.floor(x + 0.5)


def tour_length(tour, Dg, rnd):
    """Official closed-tour length: sum of rounded consecutive edge distances."""
    tg = cp.asarray(tour, dtype=cp.int64)
    edges = cp.asnumpy(Dg[tg, cp.roll(tg, -1)].astype(cp.float64))
    return float(rnd(edges).sum())


@njit(cache=True)
def _d(coords, a, b, ceil):
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    r = (dx * dx + dy * dy) ** 0.5
    if ceil:
        return np.ceil(r)
    return np.floor(r + 0.5)  # TSPLIB EUC_2D nint


def knn_device(Dg, k, tile=4096):
    """k nearest neighbours per row from the resident GPU distance matrix,
    sorted ascending (self excluded). Computed in row-tiles so the transient
    argpartition index buffer is only (tile x n), never a second full n x n —
    keeping this O(n^2) step within the unified pool at large n. Returns a host
    (n, k) int32 array of neighbour ids."""
    n = Dg.shape[0]
    k = min(k, n - 1)
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, tile):
        e = min(s + tile, n)
        block = Dg[s:e]  # (R, n) view
        part = cp.argpartition(block, k, axis=1)[:, : k + 1]  # (R, k+1) incl self
        rows = cp.arange(e - s)[:, None]
        ordr = cp.argsort(block[rows, part], axis=1)
        knn = cp.take_along_axis(part, ordr, axis=1)[:, 1 : k + 1]  # drop self
        out[s:e] = cp.asnumpy(knn).astype(np.int32)
    return out


@njit(cache=True)
def neighbor_two_opt(tour, coords, knn, ceil, max_pass=30):
    """2-opt over each city's k-nearest-neighbour candidates (spatial, not
    tour-local), the standard way large TSP local search scales: O(n*k) per pass
    with the sorted-neighbour early-out. Fixes the long seam edges a tour-window
    misses (spatially-close cities sit far apart in tour position after
    stitching). Distances from coordinates, official rounding."""
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    K = knn.shape[1]
    for _ in range(max_pass):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            dab = _d(coords, a, b, ceil)
            for t in range(K):
                c = knn[a, t]
                dac = _d(coords, a, c, ceil)
                if dac >= dab:
                    break  # neighbours sorted: no 2-opt gain past this
                j = pos[c]
                if j <= i + 1 or j == n - 1 and i == 0:
                    continue
                dd = tour[(j + 1) % n]
                delta = dac + _d(coords, b, dd, ceil) - dab - _d(coords, c, dd, ceil)
                if delta < -1e-9:
                    lo, hi = i + 1, j
                    while lo < hi:
                        tour[lo], tour[hi] = tour[hi], tour[lo]
                        pos[tour[lo]] = lo
                        pos[tour[hi]] = hi
                        lo += 1
                        hi -= 1
                    if lo == hi:
                        pos[tour[lo]] = lo
                    improved = True
                    b = tour[i + 1]
                    dab = _d(coords, a, b, ceil)
        if not improved:
            break
    return tour


@njit(cache=True)
def windowed_two_opt(tour, coords, W, ceil, max_pass=6):
    """2-opt restricted to tour-position windows |i-j| <= W, distances computed
    from coordinates on the fly (matches the official rounded objective). O(n*W)
    per pass — near-linear, so it scales where a full O(n^2) 2-opt cannot. After
    a proximity-ordered stitch the costly edges sit at block seams (tour-local),
    which is exactly what a windowed pass repairs."""
    n = tour.shape[0]
    for _ in range(max_pass):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            jmax = i + W
            if jmax > n - 1:
                jmax = n - 1
            for j in range(i + 2, jmax + 1):
                c = tour[j]
                dd = tour[(j + 1) % n]
                if (j + 1) % n == i:
                    continue
                delta = (
                    _d(coords, a, c, ceil)
                    + _d(coords, b, dd, ceil)
                    - _d(coords, a, b, ceil)
                    - _d(coords, c, dd, ceil)
                )
                if delta < -1e-9:
                    lo, hi = i + 1, j
                    while lo < hi:
                        tour[lo], tour[hi] = tour[hi], tour[lo]
                        lo += 1
                        hi -= 1
                    improved = True
                    b = tour[i + 1]
        if not improved:
            break
    return tour


# ---------------------------------------------------------------------------
# Blocked TSP with official-rounded integer sub-problems
# ---------------------------------------------------------------------------
def _solve_block(gids, Dg, rnd, lkh_cap=800):
    m = len(gids)
    if m <= 3:
        return gids.copy()
    g = cp.asarray(gids, dtype=cp.int64)
    sub = cp.asnumpy(Dg[cp.ix_(g, g)].astype(cp.float64))
    sub_int = rnd(sub).astype(np.int64)
    if _HAS_LKH and m <= lkh_cap:
        st = np.asarray(
            elkai.DistanceMatrix(sub_int.tolist()).solve_tsp(runs=1)[:-1], np.int64
        )
    else:
        Ds = np.ascontiguousarray(sub_int.astype(np.float64))
        st = nn_order(Ds, 0)
        two_opt(st, Ds)
    loc = _open_from_subtour(
        np.ascontiguousarray(st), np.ascontiguousarray(sub_int.astype(np.float64))
    )
    return gids[loc]


def _stitch(paths, Dg, rnd):
    Bn = len(paths)
    if Bn == 1:
        return np.ascontiguousarray(paths[0])
    ep = np.array([[p[0], p[-1]] for p in paths], dtype=np.int64)
    ep_ids = ep.reshape(-1)
    e = cp.asarray(ep_ids, dtype=cp.int64)
    Dep = rnd(cp.asnumpy(Dg[cp.ix_(e, e)].astype(cp.float64)))
    pos_of = {int(pid): i for i, pid in enumerate(ep_ids.tolist())}

    def dist(a, b):
        return Dep[pos_of[int(a)], pos_of[int(b)]]

    Bd = np.zeros((Bn, Bn), dtype=np.int64)
    for i in range(Bn):
        for j in range(Bn):
            if i != j:
                Bd[i, j] = int(
                    min(dist(ep[i, a], ep[j, b]) for a in (0, 1) for b in (0, 1))
                )
    if Bn <= 3:
        cyc = list(range(Bn))
    elif _HAS_LKH:
        cyc = list(np.asarray(elkai.DistanceMatrix(Bd.tolist()).solve_tsp(runs=2)[:-1]))
    else:
        cyc = list(range(Bn))
    orient = _orient_cycle_dist(cyc, ep, dist)
    seq = [paths[bi] if orient[p] == 0 else paths[bi][::-1] for p, bi in enumerate(cyc)]
    return np.ascontiguousarray(np.concatenate(seq))


def blocked_tsp(coords, ewt, cap=1000, dtype="float64"):
    """Full VAT-cluster-blocking TSP on a coordinate instance, GPU front-end.

    Stages: GPU front-end (distances + VAT order) -> size-capped VAT gap blocks
    -> per-block LKH -> optimized stitch -> windowed 2-opt polish (O(n*W),
    scalable). Returns tours + a per-stage time breakdown.
    """
    rnd = _round_fn(ewt)
    ceil = ewt == "CEIL_2D"
    t = {}
    _sync()
    t0 = time.perf_counter()
    order, _, Dg = gpu_vat.vat_gpu(coords, dtype=dtype, return_distances=True)
    _sync()
    t["frontend"] = time.perf_counter() - t0

    raw_len = tour_length(order, Dg, rnd)
    blocks = _vat_gap_blocks(order, Dg, target=cap)

    btimes, paths = [], []
    t0 = time.perf_counter()
    for gg in blocks:
        tb = time.perf_counter()
        paths.append(_solve_block(gg, Dg, rnd))
        btimes.append(time.perf_counter() - tb)
    t["block_solve"] = time.perf_counter() - t0
    t["max_block_solve"] = max(btimes) if btimes else 0.0

    t0 = time.perf_counter()
    tour = _stitch(paths, Dg, rnd)
    t["stitch"] = time.perf_counter() - t0
    stitched_len = tour_length(tour, Dg, rnd)

    # k-nearest-neighbour candidate lists from the resident GPU matrix (unified
    # memory: the whole n x n matrix is on the device, so this O(n^2) step scales
    # with the front-end) -> neighbour-list 2-opt polish (spatial, O(n*k)).
    t0 = time.perf_counter()
    knn = knn_device(Dg, k=10)
    t["knn"] = time.perf_counter() - t0
    del Dg
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.perf_counter()
    tour = neighbor_two_opt(np.ascontiguousarray(tour), coords, knn, ceil)
    t["polish"] = time.perf_counter() - t0

    # cost after polish (coords, official rounding)
    blocked_len = _tour_len_coords(tour, coords, ceil)
    t["total"] = t["frontend"] + t["block_solve"] + t["stitch"] + t["knn"] + t["polish"]
    t["t_par"] = (
        t["frontend"] + t["max_block_solve"] + t["stitch"] + t["knn"] + t["polish"]
    )
    return {
        "order": order,
        "tour": tour,
        "raw_len": raw_len,
        "stitched_len": stitched_len,
        "blocked_len": blocked_len,
        "n_blocks": len(blocks),
        "times": t,
    }


@njit(cache=True)
def _tour_len_coords(tour, coords, ceil):
    n = tour.shape[0]
    s = 0.0
    for k in range(n):
        s += _d(coords, tour[k], tour[(k + 1) % n], ceil)
    return s


def flat_lkh(coords, ewt):
    """Flat LKH over the whole instance (reference), if affordable. Returns
    (length, seconds) or (None, None)."""
    n = len(coords)
    if not _HAS_LKH or n > LKH_FLAT_MAX:
        return None, None
    rnd = _round_fn(ewt)
    d = coords[:, None, :] - coords[None, :, :]
    D_int = rnd(np.sqrt((d * d).sum(-1))).astype(np.int64)
    t = time.perf_counter()
    tour = np.asarray(elkai.DistanceMatrix(D_int.tolist()).solve_tsp(runs=1)[:-1])
    dt = time.perf_counter() - t
    length = float(sum(D_int[tour[k], tour[(k + 1) % n]] for k in range(n)))
    return length, dt


# ---------------------------------------------------------------------------
# Run + plot
# ---------------------------------------------------------------------------
def run(plan=PLAN):
    print("VAT-cluster-blocking on real TSPLIB instances (DGX Spark GB10)")
    print("=" * 62)
    print(f"GPU: {gpu.is_available()}   LKH (elkai): {_HAS_LKH}")
    print(
        f"\n  {'instance':10s} {'n':>6s} {'ewt':8s} {'opt':>10s} "
        f"{'blocked':>10s} {'gap%':>7s} {'rawVATgap%':>10s} "
        f"{'flatLKH%':>9s} {'front s':>8s} {'t_par s':>8s} {'total s':>8s}"
    )
    results = {}
    for name in plan:
        coords, ewt = parse_tsplib(name)
        n = len(coords)
        r = blocked_tsp(coords, ewt)
        opt = OPTIMA.get(name)
        flkh, flkh_t = flat_lkh(coords, ewt)
        gap = 100 * (r["blocked_len"] - opt) / opt if opt else np.nan
        rawgap = 100 * (r["raw_len"] - opt) / opt if opt else np.nan
        flatgap = 100 * (flkh - opt) / opt if (opt and flkh) else np.nan
        tt = r["times"]
        results[name] = dict(
            n=n,
            ewt=ewt,
            opt=opt,
            blocked=r["blocked_len"],
            gap=gap,
            rawgap=rawgap,
            flat=flkh,
            flatgap=flatgap,
            flat_t=flkh_t,
            coords=coords,
            tour=r["tour"],
            times=tt,
            n_blocks=r["n_blocks"],
        )
        print(
            f"  {name:10s} {n:6d} {ewt:8s} {opt if opt else 0:10d} "
            f"{r['blocked_len']:10.0f} {gap:7.1f} {rawgap:10.1f} "
            f"{flatgap:9.2f} {tt['frontend']:8.2f} {tt['t_par']:8.2f} "
            f"{tt['total']:8.2f}"
        )
    return results


def figure(results):
    names = list(results)
    ns = [results[k]["n"] for k in names]
    fig = plt.figure(figsize=(16, 5))

    # A: % over optimum vs n
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(
        ns,
        [results[k]["gap"] for k in names],
        "o-",
        color="tab:blue",
        label="VAT-blocked (this)",
    )
    ax.plot(
        ns,
        [results[k]["rawgap"] for k in names],
        "s--",
        color="0.6",
        label="raw VAT tour",
    )
    fl = [
        (results[k]["n"], results[k]["flatgap"])
        for k in names
        if not np.isnan(results[k]["flatgap"])
    ]
    if fl:
        ax.plot(
            [a for a, _ in fl],
            [b for _, b in fl],
            "^:",
            color="tab:green",
            label="flat LKH (n<=2000)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("cities (n)")
    ax.set_ylabel("% over TSPLIB optimum")
    ax.set_title("A. tour quality vs known optimum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # B: time vs n (front-end, total, t_par)
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(
        ns,
        [results[k]["times"]["frontend"] for k in names],
        "^-",
        color="tab:blue",
        label="GPU front-end (dist+VAT)",
    )
    ax.plot(
        ns,
        [results[k]["times"]["t_par"] for k in names],
        "D-",
        color="tab:orange",
        label="t_par (parallel proxy)",
    )
    ax.plot(
        ns,
        [results[k]["times"]["total"] for k in names],
        "o--",
        color="0.5",
        label="total (serial blocks)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cities (n)")
    ax.set_ylabel("seconds")
    ax.set_title("B. solve time vs n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # C: the actual tour on a representative mid-size instance
    ax = fig.add_subplot(1, 3, 3)
    pick = min(
        (k for k in names if 800 <= results[k]["n"] <= 5000),
        key=lambda k: abs(results[k]["n"] - 1500),
        default=names[len(names) // 2],
    )
    r = results[pick]
    C = r["coords"]
    tour = np.append(r["tour"], r["tour"][0])
    ax.plot(C[tour, 0], C[tour, 1], "-", color="tab:blue", lw=0.6)
    ax.plot(C[:, 0], C[:, 1], ".", color="tab:red", ms=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"C. VAT-blocked tour: {pick} (n={r['n']})\n"
        f"{r['gap']:.1f}% over optimum, {r['times']['total']:.1f}s"
    )

    fig.suptitle(
        "VAT-cluster-blocking TSP on real TSPLIB instances (GB10 unified-memory "
        "GPU front-end): near-optimal tours, scaling to tens of thousands of cities",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_tsplib.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    res = run()
    print(f"\nwrote {figure(res)}")
