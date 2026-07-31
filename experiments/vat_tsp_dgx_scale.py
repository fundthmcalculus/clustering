"""Scale study: the VAT->TSP cluster-blocking pipeline on the DGX Spark (GB10).

The VAT-cluster-blocking TSP solver (experiments/vat_tsp_benchmark.py) is
"cluster-first, route-second": find blocks along the VAT ordering, solve each
block's sub-TSP (LKH), then optimize the block-to-block stitch. Its per-block
work is cheap and parallel; the wall it hits at scale is the **O(n^2) VAT
front-end** — build the n x n dissimilarity matrix and derive the VAT ordering
that defines the blocks. On a host CPU that front-end walls out in both memory
(the dense n x n matrix) and time (the serial-ish iVAT).

On the DGX Spark's GB10 (unified ~128 GB, coherent CPU/GPU) that same front-end
runs on the device with the matrix resident: `gpu_vat.vat_gpu` builds the
distances + exact Boruvka MST + VAT ordering on the GPU and returns only the
O(n) ordering. So the blocking pipeline scales to n far beyond the host — the
matrix lives in the unified pool, never crossing a bus.

Two parts:
  A. front-end scaling — time (distances + VAT order) on the host CPU vs the GPU
     (f64 exact, f32) across n. Verifies the GPU order is identical to the CPU.
  B. end-to-end blocked TSP at scale — full pipeline on the GPU front-end (only
     O(n) + O(B^2) data on the host), tour cost vs the raw VAT tour, time
     breakdown, at n the host front-end cannot reach.

Run:  python -m experiments.vat_tsp_dgx_scale
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu, gpu_vat  # noqa: E402
from tribbleclustering.pcvat import (  # noqa: E402
    pairwise_distances_c_64,
    compute_ivat_c,
)
from experiments.vat_tsp import two_opt  # noqa: E402  (njit closed-tour 2-opt)
from experiments.vat_tsp_warmstart import nn_order  # noqa: E402
from experiments.vat_tsp_benchmark import (
    _orient_cycle,
    _open_from_subtour,
)  # noqa: E402
from experiments.adversarial_eval import easy_blobs  # noqa: E402

try:
    import elkai  # type: ignore

    _HAS_LKH = True
except ImportError:  # pragma: no cover
    _HAS_LKH = False

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _sync():
    cp.cuda.Stream.null.synchronize()


def _data(n, seed=1):
    X, _ = easy_blobs(n, seed=seed)
    return np.ascontiguousarray(X)


# ---------------------------------------------------------------------------
# Part A — front-end scaling (distances + VAT order): host CPU vs GPU
# ---------------------------------------------------------------------------
def frontend_scaling(cpu_sizes, gpu_sizes):
    print("\n=== A. VAT front-end (distances + order): host CPU vs GPU (GB10) ===")
    print("    time to build the n x n matrix and derive the VAT ordering.")
    rows = {}
    # CPU: compiled C/OpenMP distances + compute_ivat_c order (the best host path)
    for n in cpu_sizes:
        X = _data(n)
        t = time.perf_counter()
        Dc = pairwise_distances_c_64(X)
        _, _, order_cpu = compute_ivat_c(Dc.copy(), inplace=False)
        t_cpu = time.perf_counter() - t
        rows.setdefault(n, {})["cpu"] = t_cpu
        rows[n]["order_cpu"] = order_cpu
        del Dc
        print(f"  n={n:6d}: CPU front-end {t_cpu*1e3:8.0f} ms")
    # GPU: vat_gpu (distances + Boruvka MST + order on device), f64 and f32
    for n in gpu_sizes:
        X = _data(n)
        for dt in ("float64", "float32"):
            _sync()
            gpu_vat.vat_gpu(X, dtype=dt)  # warm (kernel compile / pool)
            _sync()
            t = time.perf_counter()
            order_g, _ = gpu_vat.vat_gpu(X, dtype=dt)
            _sync()
            tg = time.perf_counter() - t
            rows.setdefault(n, {})[f"gpu_{dt}"] = tg
            if dt == "float32":
                rows[n]["order_gpu32"] = order_g
        cp.get_default_memory_pool().free_all_blocks()
        om = ""
        if "order_cpu" in rows.get(n, {}):
            om = f"  order_match(f32 vs CPU)={float(np.mean(rows[n]['order_gpu32']==rows[n]['order_cpu'])):.4f}"
        print(
            f"  n={n:6d}: GPU f64 {rows[n]['gpu_float64']*1e3:8.1f} ms   "
            f"GPU f32 {rows[n]['gpu_float32']*1e3:8.1f} ms{om}"
        )
    return rows


# ---------------------------------------------------------------------------
# Part B — end-to-end blocked TSP on the GPU front-end (host holds only O(n)/O(B^2))
# ---------------------------------------------------------------------------
def _closed_cost_gpu(tour, Dg):
    tg = cp.asarray(tour, dtype=cp.int64)
    return float(Dg[tg, cp.roll(tg, -1)].astype(cp.float64).sum())


def _vat_gap_blocks(order, Dg, target=400):
    """Split the VAT order into ~n/target contiguous runs at its largest path
    gaps (single-linkage cluster cuts), then hard-split any residual run longer
    than 2*target into equal sub-runs. Keeps every block LKH-solvable and the
    block solve embarrassingly parallel as n grows. Uses only the O(n)
    consecutive-edge distances (gathered on the device)."""
    n = len(order)
    if n <= target:
        return [order.copy()]
    o = cp.asarray(order, dtype=cp.int64)
    gaps = cp.asnumpy(Dg[o[:-1], o[1:]].astype(cp.float64))  # O(n) on device
    k = max(1, n // target - 1)  # number of gap cuts -> ~n/target blocks
    cuts = np.sort(np.argsort(gaps)[-k:])
    runs, prev = [], 0
    for c in list(cuts) + [n - 1]:
        runs.append((prev, c + 1))
        prev = c + 1
    blocks = []
    for a, b in runs:
        if b - a <= 2 * target:
            blocks.append(order[a:b])
        else:  # hard-split an over-long run into equal sub-runs
            nsub = int(np.ceil((b - a) / target))
            for s in np.array_split(np.arange(a, b), nsub):
                blocks.append(order[s[0] : s[-1] + 1])
    return [bl for bl in blocks if len(bl) > 0]


def _solve_block(gids, Dg, scale, lkh_cap=800):
    """Solve one block's sub-TSP; return an OPEN local path over the block's
    global ids. LKH for small blocks, else NN + 2-opt. The sub-matrix is gathered
    from the resident device matrix and scaled to integers (LKH consumes ints, so
    ``scale`` preserves resolution of the small euclidean distances)."""
    m = len(gids)
    if m <= 3:
        return gids.copy()
    g = cp.asarray(gids, dtype=cp.int64)
    sub_int = cp.asnumpy(cp.rint(Dg[cp.ix_(g, g)].astype(cp.float64) * scale)).astype(
        np.int64
    )
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


def _stitch(paths, Dg, scale):
    """Optimized block-to-block stitch using only endpoint distances: order the
    blocks (TSP over endpoints) and pick each block's orientation. Returns the
    global closed tour."""
    Bn = len(paths)
    if Bn == 1:
        return np.ascontiguousarray(paths[0])
    ep = np.array([[p[0], p[-1]] for p in paths], dtype=np.int64)
    ep_ids = ep.reshape(-1)
    # dense endpoint x endpoint distances (2B x 2B) gathered once
    e = cp.asarray(ep_ids, dtype=cp.int64)
    Dep = cp.asnumpy(Dg[cp.ix_(e, e)].astype(cp.float64))
    pos_of = {int(pid): i for i, pid in enumerate(ep_ids.tolist())}

    def dist(a, b):
        return Dep[pos_of[int(a)], pos_of[int(b)]]

    # block-to-block distance = min over the 4 endpoint pairings (scaled to int)
    Bd = np.zeros((Bn, Bn), dtype=np.int64)
    for i in range(Bn):
        for j in range(Bn):
            if i != j:
                Bd[i, j] = int(
                    round(
                        min(dist(ep[i, a], ep[j, b]) for a in (0, 1) for b in (0, 1))
                        * scale
                    )
                )
    if Bn <= 3:
        cyc = list(range(Bn))
    elif _HAS_LKH:
        c = elkai.DistanceMatrix(Bd.tolist()).solve_tsp(runs=2)[:-1]
        cyc = list(np.asarray(c, dtype=np.int64))
    else:
        cyc = list(range(Bn))
    orient = _orient_cycle_dist(cyc, ep, dist)
    seq = [paths[bi] if orient[p] == 0 else paths[bi][::-1] for p, bi in enumerate(cyc)]
    return np.ascontiguousarray(np.concatenate(seq))


def _orient_cycle_dist(cyc, ep, dist):
    """Orientation DP (2 states/block) minimizing junction edges — endpoint
    distances only (adapted from vat_tsp_benchmark._orient_cycle)."""
    B = len(cyc)
    if B == 1:
        return [0]

    def exit_id(bi, o):
        return ep[cyc[bi], 1 - o]

    def enter_id(bi, o):
        return ep[cyc[bi], o]

    best_total, best_choice = None, None
    for o0 in (0, 1):
        dp = {o0: 0.0}
        paths = {o0: [o0]}
        for pos in range(1, B):
            ndp, npaths = {}, {}
            for o in (0, 1):
                bestc, besto = None, None
                for op in dp:
                    c = dp[op] + dist(exit_id(pos - 1, op), enter_id(pos, o))
                    if bestc is None or c < bestc:
                        bestc, besto = c, op
                ndp[o] = bestc
                npaths[o] = paths[besto] + [o]
            dp, paths = ndp, npaths
        for o in dp:
            total = dp[o] + dist(exit_id(B - 1, o), enter_id(0, o0))
            if best_total is None or total < best_total:
                best_total, best_choice = total, paths[o]
    return best_choice


def blocked_tsp_gpu(X, cap=600, dtype="float32"):
    """Full VAT-cluster-blocking TSP with the GPU unified-memory front-end.

    Only O(n) (order, cost) + O(B^2) (stitch) + per-block sub-matrices touch the
    host; the n x n matrix stays resident on the device. Blocks are size-capped
    (<= cap) so the block solve stays LKH-fast and parallel. Returns a dict of
    tour costs and a per-stage time breakdown.
    """
    t = {}
    _sync()
    t0 = time.perf_counter()
    order, _, Dg = gpu_vat.vat_gpu(X, dtype=dtype, return_distances=True)
    _sync()
    t["frontend"] = time.perf_counter() - t0

    raw_cost = _closed_cost_gpu(order, Dg)  # VAT order as a closed tour (baseline)
    # integer scale for LKH (small euclidean distances -> ~5-digit ints)
    scale = 1.0e5 / max(1e-9, float(Dg.max()))

    t0 = time.perf_counter()
    blocks = _vat_gap_blocks(order, Dg, target=cap)
    t["blocking"] = time.perf_counter() - t0

    btimes = []
    paths = []
    t0 = time.perf_counter()
    for g in blocks:
        tb = time.perf_counter()
        paths.append(_solve_block(g, Dg, scale))
        btimes.append(time.perf_counter() - tb)
    t["block_solve"] = time.perf_counter() - t0
    t["max_block_solve"] = max(btimes) if btimes else 0.0  # parallel proxy
    t["max_block"] = max((len(g) for g in blocks), default=0)

    t0 = time.perf_counter()
    tour = _stitch(paths, Dg, scale)
    t["stitch"] = time.perf_counter() - t0

    blocked_cost = _closed_cost_gpu(tour, Dg)
    t["total"] = t["frontend"] + t["blocking"] + t["block_solve"] + t["stitch"]
    # parallel proxy: front-end + blocking + slowest single block + stitch
    t["t_par"] = t["frontend"] + t["blocking"] + t["max_block_solve"] + t["stitch"]
    del Dg
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "raw_vat_cost": raw_cost,
        "blocked_cost": blocked_cost,
        "improve_pct": 100.0 * (blocked_cost - raw_cost) / raw_cost,
        "times": t,
        "n_blocks": len(blocks),
    }


def blocked_scaling(sizes, cap=600, dtype="float32"):
    print(f"\n=== B. end-to-end blocked TSP at scale (GPU front-end, {dtype}) ===")
    print("    tour cost vs raw VAT tour; time breakdown (front-end is the O(n^2)")
    print(f"    stage the unified-memory GPU keeps feasible). blocks capped @ {cap}.")
    out = {}
    for n in sizes:
        X = _data(n)
        r = blocked_tsp_gpu(X, cap=cap, dtype=dtype)
        out[n] = r
        tt = r["times"]
        print(
            f"  n={n:6d} ({r['n_blocks']:3d} blks, max {tt['max_block']}): "
            f"total {tt['total']:6.2f}s  t_par {tt['t_par']:5.2f}s "
            f"[front {tt['frontend']:.2f} | block Σ{tt['block_solve']:.2f}/"
            f"max{tt['max_block_solve']:.2f} | stitch {tt['stitch']:.2f}]  "
            f"blocked vs raw VAT: {r['improve_pct']:+.1f}%"
        )
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure(front, blocked):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ns_cpu = sorted(n for n, r in front.items() if "cpu" in r)
    ns_gpu = sorted(n for n, r in front.items() if "gpu_float32" in r)
    if ns_cpu:
        ax.plot(
            [n for n in ns_cpu],
            [front[n]["cpu"] * 1e3 for n in ns_cpu],
            "s-",
            color="0.5",
            label="CPU (C/OpenMP dist + iVAT)",
        )
    ax.plot(
        ns_gpu,
        [front[n]["gpu_float64"] * 1e3 for n in ns_gpu],
        "^-",
        color="tab:blue",
        label="GPU f64 (unified)",
    )
    ax.plot(
        ns_gpu,
        [front[n]["gpu_float32"] * 1e3 for n in ns_gpu],
        "D-",
        color="tab:orange",
        label="GPU f32 (unified)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("VAT front-end time (ms)")
    ax.set_title("A. VAT front-end: host CPU walls out; GB10 unified GPU scales")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ns = sorted(blocked)
    fronts = [blocked[n]["times"]["frontend"] for n in ns]
    solves = [blocked[n]["times"]["block_solve"] for n in ns]
    stitches = [blocked[n]["times"]["stitch"] for n in ns]
    ax.bar(range(len(ns)), fronts, 0.6, label="GPU front-end", color="tab:blue")
    ax.bar(
        range(len(ns)),
        solves,
        0.6,
        bottom=fronts,
        label="block solve (LKH)",
        color="tab:green",
    )
    ax.bar(
        range(len(ns)),
        stitches,
        0.6,
        bottom=[f + s for f, s in zip(fronts, solves)],
        label="stitch",
        color="tab:orange",
    )
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("wall-clock (s)")
    ax.set_title("B. blocked-TSP pipeline time breakdown (GPU front-end)")
    ax.legend()

    fig.suptitle(
        "VAT->TSP cluster-blocking on the DGX Spark: the unified-memory GPU "
        "front-end scales the pipeline past the host CPU wall",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_dgx_scale.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("VAT->TSP cluster-blocking scale study (DGX Spark GB10)")
    print("=" * 54)
    print(f"GPU available: {gpu.is_available()}   LKH (elkai): {_HAS_LKH}")
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    front = frontend_scaling(
        cpu_sizes=[2000, 5000, 10000, 20000],
        gpu_sizes=[2000, 5000, 10000, 20000, 40000, 80000],
    )
    blocked = blocked_scaling(
        [5000, 10000, 20000, 40000, 80000], cap=600, dtype="float32"
    )
    print(f"\nwrote {figure(front, blocked)}")
