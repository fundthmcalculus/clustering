"""VAT hot-start seed study + unified-GPU 2-opt + largest-k reslicer (n=1000).

Three experiments on one n=1000 instance, all on the DGX Spark GB10 with the
dissimilarity matrix resident in unified memory:

  1. VAT hot-start seed.  VAT seeds its ordering at the global-MAX dissimilarity
     vertex (a peripheral/outlier point). Does seeding at the SMALLEST non-zero
     dissimilarity (a dense-core point) give a better TSP hot start? We build one
     MST and derive the VAT order from each seed, then compare tour length before
     and after local optimisation.

  2. Unified GPU 2-opt.  A best-improvement 2-opt implemented as a CuPy RawKernel
     that reads the resident n x n matrix directly (the LKH-style local-opt core;
     the tour never leaves the device). One move per pass, O(n^2) delta scan on
     the GPU each pass.

  3. Largest-k reslicer.  Break the k longest edges of the tour (the long
     "intersection lines"), splitting it into k arcs, then reconnect the arcs
     optimally (endpoint TSP + per-arc orientation DP) — a targeted k-opt that
     spends its effort only on the worst edges.

Run:  python -m experiments.vat_tsp_reslice
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import gpu, gpu_vat  # noqa: E402
from experiments.vat_tsp_dgx_scale import _orient_cycle_dist  # noqa: E402

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


# ---------------------------------------------------------------------------
# Unified GPU 2-opt (resident matrix, best improvement, one move per pass)
# ---------------------------------------------------------------------------
_TWO_OPT_SRC = r"""
extern "C" __global__
void two_opt_row(const double* __restrict__ D, const int* __restrict__ tour, int n,
                 double* __restrict__ row_delta, int* __restrict__ row_j) {
    int i = blockIdx.x;                       // one block per tour position i
    if (i >= n - 1) return;
    int ti = tour[i], ti1 = tour[i + 1];
    double rem_i = D[(size_t)ti * n + ti1];    // edge (t_i, t_{i+1}) removed
    double bd = -1e-9; int bj = -1;            // best improving delta for this row
    for (int j = i + 2 + threadIdx.x; j < n; j += blockDim.x) {
        int jn = (j + 1 == n) ? 0 : j + 1;
        if (jn == i) continue;                 // adjacent wrap-around: skip
        int tj = tour[j], tjn = tour[jn];
        double delta = D[(size_t)ti * n + tj] + D[(size_t)ti1 * n + tjn]
                     - rem_i - D[(size_t)tj * n + tjn];
        if (delta < bd) { bd = delta; bj = j; }
    }
    __shared__ double sd[256];
    __shared__ int sj[256];
    sd[threadIdx.x] = bd; sj[threadIdx.x] = bj;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && sd[threadIdx.x + s] < sd[threadIdx.x]) {
            sd[threadIdx.x] = sd[threadIdx.x + s];
            sj[threadIdx.x] = sj[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { row_delta[i] = sd[0]; row_j[i] = sj[0]; }
}
"""
_TWO_OPT = None


def _two_opt_kernel():
    global _TWO_OPT
    if _TWO_OPT is None:
        _TWO_OPT = cp.RawKernel(_TWO_OPT_SRC, "two_opt_row")
    return _TWO_OPT


def gpu_two_opt(tour, Dg, max_pass=100000):
    """Best-improvement 2-opt on the GPU using the resident matrix ``Dg``.

    ``tour`` (host int array) is a closed-tour permutation. Returns (tour, passes)
    with the tour improved to a 2-opt local optimum (or until max_pass). The delta
    scan and the reversal both run on the device; only one scalar move descriptor
    crosses to the host per pass.
    """
    n = len(tour)
    t = cp.asarray(tour, dtype=cp.int32)
    row_delta = cp.empty(n, dtype=cp.float64)
    row_j = cp.empty(n, dtype=cp.int32)
    kern = _two_opt_kernel()
    tpb = 256
    passes = 0
    for _ in range(max_pass):
        kern((n - 1,), (tpb,), (Dg, t, np.int32(n), row_delta, row_j))
        bi = int(cp.argmin(row_delta).item())
        bd = float(row_delta[bi].item())
        if bd >= -1e-9:
            break
        bj = int(row_j[bi].item())
        # reverse the segment t[bi+1 .. bj] on the device
        t[bi + 1 : bj + 1] = t[bi + 1 : bj + 1][::-1]
        passes += 1
    return cp.asnumpy(t).astype(np.int64), passes


# ---------------------------------------------------------------------------
# Largest-k reslicer: break the k longest edges, reconnect the arcs optimally
# ---------------------------------------------------------------------------
def _stitch_segments(segs, Dg):
    """Reconnect open arcs into one closed tour: order them (endpoint TSP) and
    orient each (DP), using only the arc-endpoint distances."""
    Bn = len(segs)
    if Bn == 1:
        return np.ascontiguousarray(segs[0])
    ep = np.array([[s[0], s[-1]] for s in segs], dtype=np.int64)
    ep_ids = ep.reshape(-1)
    e = cp.asarray(ep_ids, dtype=cp.int64)
    Dep = cp.asnumpy(Dg[cp.ix_(e, e)].astype(cp.float64))
    pos_of = {int(pid): i for i, pid in enumerate(ep_ids.tolist())}

    def dist(a, b):
        return Dep[pos_of[int(a)], pos_of[int(b)]]

    Bd = np.zeros((Bn, Bn), dtype=np.int64)
    for i in range(Bn):
        for j in range(Bn):
            if i != j:
                Bd[i, j] = int(
                    round(min(dist(ep[i, a], ep[j, b]) for a in (0, 1) for b in (0, 1)))
                )
    if Bn <= 3:
        cyc = list(range(Bn))
    elif _HAS_LKH:
        cyc = list(np.asarray(elkai.DistanceMatrix(Bd.tolist()).solve_tsp(runs=2)[:-1]))
    else:
        cyc = list(range(Bn))
    orient = _orient_cycle_dist(cyc, ep, dist)
    seq = [segs[bi] if orient[p] == 0 else segs[bi][::-1] for p, bi in enumerate(cyc)]
    return np.ascontiguousarray(np.concatenate(seq))


def largest_k_reslice(tour, Dg, k):
    """Cut the k longest closed-tour edges into k arcs and reconnect optimally.

    Targets exactly the worst edges (the long crossing/seam lines): a k-opt move
    whose cost is set by k (arc-endpoint TSP over 2k endpoints), not n."""
    n = len(tour)
    tour = np.ascontiguousarray(tour, dtype=np.int64)
    t = cp.asarray(tour, dtype=cp.int64)
    edges = cp.asnumpy(Dg[t, cp.roll(t, -1)].astype(cp.float64))  # edge p: t[p]->t[p+1]
    cutp = np.sort(np.argsort(edges)[-k:])  # positions of the k longest edges
    m = len(cutp)
    segs = []
    for a in range(m):
        start = (cutp[a] + 1) % n
        end = cutp[(a + 1) % m]  # inclusive end of this arc
        if start <= end:
            segs.append(tour[start : end + 1])
        else:  # arc wraps past index 0
            segs.append(np.concatenate([tour[start:], tour[: end + 1]]))
    return _stitch_segments(segs, Dg)


# ---------------------------------------------------------------------------
# Instance + VAT hot starts from two seeds
# ---------------------------------------------------------------------------
def uniform_instance(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((n, 2)) * 1000.0)


def closed_len(tour, Dg):
    t = cp.asarray(tour, dtype=cp.int64)
    return float(Dg[t, cp.roll(t, -1)].astype(cp.float64).sum())


def vat_orders(coords, dtype="float64"):
    """Build the MST once; return (Dg, order_max_seed, order_min_seed).

    order_max_seed: VAT order seeded at the global-max dissimilarity vertex
    (classic VAT). order_min_seed: seeded at the smallest non-zero dissimilarity.
    """
    Dg = gpu.pairwise_distances_device(coords, dtype=dtype)
    n = Dg.shape[0]
    mu, mv = gpu_vat.boruvka_mst_device(Dg)
    w = cp.asnumpy(Dg[mu, mv].astype(cp.float64))
    mu_h, mv_h = cp.asnumpy(mu), cp.asnumpy(mv)

    src_max = int(cp.argmax(Dg).item()) // n
    # smallest non-zero distance: mask the zero diagonal
    D_masked = Dg + cp.eye(n, dtype=Dg.dtype) * cp.asarray(Dg.max() + 1)
    src_min = int(cp.argmin(D_masked).item()) // n
    del D_masked

    order_max, _ = gpu_vat._order_from_mst(mu_h, mv_h, w, n, src_max)
    order_min, _ = gpu_vat._order_from_mst(mu_h, mv_h, w, n, src_min)
    return Dg, order_max.astype(np.int64), order_min.astype(np.int64), src_max, src_min


def lkh_reference(coords):
    if not _HAS_LKH:
        return None
    n = len(coords)
    d = coords[:, None, :] - coords[None, :, :]
    D_int = np.rint(np.sqrt((d * d).sum(-1))).astype(np.int64)
    tour = np.asarray(elkai.DistanceMatrix(D_int.tolist()).solve_tsp(runs=5)[:-1])
    return float(sum(D_int[tour[i], tour[(i + 1) % n]] for i in range(n)))


# ---------------------------------------------------------------------------
# Run + figure
# ---------------------------------------------------------------------------
def run(n=1000):
    print(f"VAT hot-start seed study + GPU 2-opt + largest-k reslicer (n={n})")
    print("=" * 64)
    print(f"GPU: {gpu.is_available()}   LKH (elkai): {_HAS_LKH}")
    coords = uniform_instance(n)
    Dg, o_max, o_min, s_max, s_min = vat_orders(coords)
    ref = lkh_reference(coords)
    ref = ref if ref else 1.0

    def pct(L):
        return 100.0 * (L - ref) / ref

    print(f"\nLKH reference tour = {ref:.0f}")
    print(f"seeds: max-dissimilarity vertex = {s_max}, min-nonzero = {s_min}")
    rows = {}
    for tag, order in (("max-seed", o_max), ("min-seed", o_min)):
        L_raw = closed_len(order, Dg)
        t0 = time.perf_counter()
        t_2opt, passes = gpu_two_opt(order, Dg)
        dt2 = time.perf_counter() - t0
        L_2opt = closed_len(t_2opt, Dg)
        rows[tag] = dict(
            raw=L_raw,
            opt=L_2opt,
            tour_raw=order,
            tour_opt=t_2opt,
            passes=passes,
            t2opt=dt2,
        )
        print(
            f"\n  {tag}: raw {L_raw:.0f} ({pct(L_raw):+.1f}%)  ->  "
            f"GPU 2-opt {L_2opt:.0f} ({pct(L_2opt):+.1f}%)  "
            f"[{passes} passes, {dt2:.2f}s]"
        )

    # largest-k reslicer, applied to the raw max-seed VAT tour, for several k
    print("\n  largest-k reslicer on the raw VAT tour (max-seed):")
    print(f"    {'k':>4s} {'reslice %':>10s} {'+2opt %':>9s} {'t_reslice':>10s}")
    reslice = {}
    base = o_max
    for k in (5, 10, 20, 50, 100):
        t0 = time.perf_counter()
        tr = largest_k_reslice(base, Dg, k)
        dtr = time.perf_counter() - t0
        Lr = closed_len(tr, Dg)
        tro, _ = gpu_two_opt(tr, Dg)
        Lro = closed_len(tro, Dg)
        reslice[k] = dict(resliced=Lr, then_opt=Lro, tour=tr, t=dtr)
        print(f"    {k:4d} {pct(Lr):10.1f} {pct(Lro):9.1f} {dtr:9.3f}s")

    return dict(
        coords=coords, ref=ref, rows=rows, reslice=reslice, s_max=s_max, s_min=s_min
    )


def figure(res):
    coords = res["coords"]
    rows = res["rows"]

    def _pct(L):
        return 100.0 * (L - res["ref"]) / res["ref"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # A: raw VAT tour (max-seed) with the longest edges highlighted
    ax = axes[0]
    tour = np.append(rows["max-seed"]["tour_raw"], rows["max-seed"]["tour_raw"][0])
    ax.plot(coords[tour, 0], coords[tour, 1], "-", color="0.7", lw=0.5)
    seg = coords[tour]
    elen = np.sqrt(((seg[1:] - seg[:-1]) ** 2).sum(1))
    longest = np.argsort(elen)[-20:]
    for p in longest:
        ax.plot(
            [seg[p, 0], seg[p + 1, 0]],
            [seg[p, 1], seg[p + 1, 1]],
            "-",
            color="tab:red",
            lw=1.3,
        )
    ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1.5)
    ax.set_title(
        f"A. raw VAT tour (max-seed) {_pct(rows['max-seed']['raw']):+.0f}%\n"
        "20 longest edges (red) = the 'intersection lines'",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # B: after GPU 2-opt
    ax = axes[1]
    tour = np.append(rows["max-seed"]["tour_opt"], rows["max-seed"]["tour_opt"][0])
    ax.plot(coords[tour, 0], coords[tour, 1], "-", color="tab:blue", lw=0.5)
    ax.plot(coords[:, 0], coords[:, 1], ".", color="k", ms=1.5)
    ax.set_title(
        f"B. after unified GPU 2-opt {_pct(rows['max-seed']['opt']):+.0f}%\n"
        f"{rows['max-seed']['passes']} passes, {rows['max-seed']['t2opt']:.2f}s",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # C: seed comparison + reslicer-vs-k
    ax = axes[2]
    ks = sorted(res["reslice"])
    ax.plot(
        ks,
        [_pct(res["reslice"][k]["resliced"]) for k in ks],
        "o-",
        color="tab:orange",
        label="largest-k reslice (raw)",
    )
    ax.plot(
        ks,
        [_pct(res["reslice"][k]["then_opt"]) for k in ks],
        "s-",
        color="tab:green",
        label="reslice + GPU 2-opt",
    )
    ax.axhline(
        _pct(rows["max-seed"]["raw"]),
        color="0.6",
        ls="--",
        lw=1,
        label=f"raw VAT max-seed ({_pct(rows['max-seed']['raw']):+.0f}%)",
    )
    ax.axhline(
        _pct(rows["max-seed"]["opt"]),
        color="tab:blue",
        ls=":",
        lw=1.2,
        label=f"GPU 2-opt max-seed ({_pct(rows['max-seed']['opt']):+.0f}%)",
    )
    ax.axhline(
        _pct(rows["min-seed"]["opt"]),
        color="tab:red",
        ls=":",
        lw=1.2,
        label=f"GPU 2-opt min-seed ({_pct(rows['min-seed']['opt']):+.0f}%)",
    )
    ax.set_xlabel("k (longest edges broken)")
    ax.set_ylabel("% over LKH")
    ax.set_title("C. largest-k reslicer + seed comparison", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "VAT hot start + unified-GPU 2-opt + largest-k reslicer (n=1000, "
        "matrix resident on GB10)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_reslice.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    res = run(1000)
    print(f"\nwrote {figure(res)}")
