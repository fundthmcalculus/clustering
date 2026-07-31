"""Closing the VAT loop on the GPU: order + iVAT image, no n x n host round-trip.

The MST is already built on the device (``boruvka_gpu.boruvka_mst_gpu``). The two
remaining VAT stages used to run on the host, which forced the (up to 80 GB)
dissimilarity matrix back across the bus:

  * ``vat_order_from_mst``   — a heap traversal of the MST tree from the global
    -max seed. It only ever reads MST *edge weights*, so it needs O(n) values,
    not the n x n matrix. Here the seed (device argmax) and the edge weights
    (device gather at the MST edges) are computed on the GPU and only O(n) scalars
    cross to the host, where the light O(n log n) traversal runs.
  * ``ivat_image_from_order`` — the O(n^2) reorder + minimax recurrence. This is
    the stage that needed the whole matrix on the host. It is moved to the device
    here.

The iVAT recurrence looks strictly serial (row r reads rows < r), but the pivot
of each row — ``h_r = min_{c<r} V[r,c]`` and ``jj_r = argmin`` — reads only the
*original* reordered distances (earlier rows never overwrite row r's entries
before row r is processed). So all (h_r, jj_r) are computed in one parallel pass
(``gather_rowmin``); only the max-propagation ``V[r,c] = max(h_r, V[parent,c])``
keeps the serial-in-r dependency (``ivat_row``, one launch per row). A final
parallel ``symmetrize`` mirrors the lower triangle. Output is bit-identical to
the numba ``ivat_image_from_order`` at f64.

Run:  python -m experiments.gpu_vat
"""

from __future__ import annotations

import heapq

import numpy as np

from experiments.boruvka_gpu import (
    _HAS_CUPY,
    _SUBST,
    boruvka_mst_gpu,
    pairwise_distances_gpu,
    alloc_unified,
)

if _HAS_CUPY:
    import cupy as cp

# V (image) type tokens: f64 for bit-exact parity with the numba reference, f32
# to halve the image footprint. D is read through the same TREAL/TLOAD widening
# as the MST kernels (so f16/f32/f64 storage all work).
_VOUT = {"float64": ("double", "unsigned"), "float32": ("float", "unsigned")}

_IVAT_TEMPLATE = r"""
#include <math_constants.h>
TEXTRAHDR
extern "C" {

// One block per row r: gather V[r, 0..r] = D[order[r], order[0..r]] (widening
// the stored TREAL to TVOUT) and block-reduce the row pivot (min + lowest-index
// argmin over c < r). h_r/jj_r depend only on these original values.
__global__ void gather_rowmin(const TREAL* __restrict__ D, const int* __restrict__ order,
                              int n, TVOUT* __restrict__ V,
                              TVOUT* __restrict__ hrow, int* __restrict__ jrow) {
    int r = blockIdx.x;
    if (r >= n) return;
    const TREAL* Drow = D + (size_t)order[r] * n;
    TVOUT lb = (TVOUT)CUDART_INF; int lj = 0;
    for (int c = threadIdx.x; c <= r; c += blockDim.x) {
        TVOUT v = (TVOUT)TLOAD(Drow[order[c]]);
        V[(size_t)r * n + c] = v;
        if (c < r && v < lb) { lb = v; lj = c; }
    }
    __shared__ TVOUT sw[256];
    __shared__ int sj[256];
    sw[threadIdx.x] = lb; sj[threadIdx.x] = lj;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sw[threadIdx.x + s] < sw[threadIdx.x]) {
                sw[threadIdx.x] = sw[threadIdx.x + s];
                sj[threadIdx.x] = sj[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { hrow[r] = sw[0]; jrow[r] = sj[0]; }
}

// Serial-in-r: V[r,c] = h_r if c==jj_r else max(h_r, V[max(jj,c), min(jj,c)]).
// Reads only rows < r (lower triangle), so one launch per row is race-free.
__global__ void ivat_row(TVOUT* __restrict__ V, const TVOUT* __restrict__ hrow,
                         const int* __restrict__ jrow, int n, int r) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= r) return;
    TVOUT hr = hrow[r];
    int jr = jrow[r];
    if (c == jr) { V[(size_t)r * n + c] = hr; return; }
    int a = jr > c ? jr : c;
    int b = jr > c ? c : jr;
    TVOUT cur = V[(size_t)a * n + b];
    V[(size_t)r * n + c] = hr > cur ? hr : cur;
}

// Mirror the lower triangle into the upper: V[r,c] = V[c,r] for c > r.
__global__ void symmetrize(TVOUT* __restrict__ V, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < n && c < n && c > r) V[(size_t)r * n + c] = V[(size_t)c * n + r];
}

}  // extern C
"""

_IVAT_MODS: dict = {}


def _ivat_module(d_dtype, v_dtype):
    key = (d_dtype, v_dtype)
    if key not in _IVAT_MODS:
        tvout, _ = _VOUT[v_dtype]
        src = _IVAT_TEMPLATE
        src = src.replace("TREAL", _SUBST[d_dtype]["TREAL"])
        src = src.replace("TLOAD", _SUBST[d_dtype]["TLOAD"])
        src = src.replace("TEXTRAHDR", _SUBST[d_dtype]["TEXTRAHDR"])
        src = src.replace("TVOUT", tvout)
        _IVAT_MODS[key] = cp.RawModule(code=src, options=("--std=c++14",))
    return _IVAT_MODS[key]


def vat_order_from_mst_gpu(Dg, mu, mv):
    """VAT order from a device matrix + MST edges, without a host n x n copy.

    Only the seed (device argmax) and the MST edge weights (device gather, O(n))
    touch the GPU-resident matrix; the O(n log n) tree traversal runs on the host
    over those O(n) scalars. Returns an int64 host order array of length n.
    """
    n = Dg.shape[0]
    mu = cp.asnumpy(mu) if _HAS_CUPY and isinstance(mu, cp.ndarray) else np.asarray(mu)
    mv = cp.asnumpy(mv) if _HAS_CUPY and isinstance(mv, cp.ndarray) else np.asarray(mv)
    # edge weights, read on the device (O(n)); seed = an endpoint of the global max
    w = cp.asnumpy(Dg[cp.asarray(mu), cp.asarray(mv)].astype(cp.float64))
    src = int(cp.asnumpy(cp.argmax(Dg))) // n
    adj = [[] for _ in range(n)]
    for a, b, ww in zip(mu.tolist(), mv.tolist(), w.tolist()):
        adj[a].append((b, ww))
        adj[b].append((a, ww))
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    order[0] = src
    visited[src] = True
    k = 1
    h = [(ww, nb) for nb, ww in adj[src]]
    heapq.heapify(h)
    while h:
        ww, v = heapq.heappop(h)
        if visited[v]:
            continue
        visited[v] = True
        order[k] = v
        k += 1
        for nb, w2 in adj[v]:
            if not visited[nb]:
                heapq.heappush(h, (w2, nb))
    return order


def ivat_image_gpu(Dg, order, v_dtype="float64", out=None):
    """On-device iVAT image from a device matrix and a VAT order.

    ``Dg`` is a device (or host) array in float16/float32/float64; ``order`` is a
    length-n permutation. The reorder-gather, the minimax recurrence and the
    symmetrisation all run on the GPU; nothing but the (optional) final image
    leaves the device. ``v_dtype`` is the image precision (f64 = bit-identical to
    the numba reference; f32 halves the image footprint). Returns a device array.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    d_dtype = cp.dtype(getattr(Dg, "dtype", cp.float64)).name
    if d_dtype not in _SUBST:
        d_dtype = "float64"
    Dg = cp.ascontiguousarray(cp.asarray(Dg, dtype=d_dtype))
    n = Dg.shape[0]
    order_g = cp.asarray(order, dtype=cp.int32)
    mod = _ivat_module(d_dtype, v_dtype)
    k_gather = mod.get_function("gather_rowmin")
    k_row = mod.get_function("ivat_row")
    k_sym = mod.get_function("symmetrize")

    V = alloc_unified((n, n), dtype=v_dtype) if out is None else out
    hrow = cp.empty(n, dtype=v_dtype)
    jrow = cp.empty(n, dtype=cp.int32)

    tpb = 256
    k_gather((n,), (tpb,), (Dg, order_g, np.int32(n), V, hrow, jrow))
    for r in range(1, n):  # serial-in-r max propagation
        grid = (r + tpb - 1) // tpb
        k_row((grid,), (tpb,), (V, hrow, jrow, np.int32(n), np.int32(r)))
    tb = 16
    k_sym(
        ((n + tb - 1) // tb, (n + tb - 1) // tb),
        (tb, tb),
        (V, np.int32(n)),
    )
    return V


def vat_ivat_gpu(X, d_dtype=None, v_dtype="float64"):
    """End-to-end device-resident VAT/iVAT: X -> distances -> MST -> order -> image.

    The n x n dissimilarity matrix is built on the device and never returns to the
    host; only the O(n) MST edges/order do. Returns (V_image_device, order).
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    d_dtype = cp.float64 if d_dtype is None else cp.dtype(d_dtype)
    Dg = pairwise_distances_gpu(X, dtype=d_dtype)
    mu, mv = boruvka_mst_gpu(Dg)
    order = vat_order_from_mst_gpu(Dg, mu, mv)
    V = ivat_image_gpu(Dg, order, v_dtype=v_dtype)
    return V, order


def _validate():
    from tribbleclustering.pcvat import pairwise_distances_c_64
    from experiments.boruvka_vat import (
        make_blobs,
        boruvka_mst_numba,
        vat_order_from_mst,
        ivat_image_from_order,
    )

    print("Correctness (GPU device image vs numba host reference):")
    for n in [1500, 3000, 6000]:
        X = make_blobs(n, 10, 25, seed=7)
        Dc = pairwise_distances_c_64(X)
        mu, mv = boruvka_mst_numba(Dc)
        order_cpu = vat_order_from_mst(Dc, mu, mv)
        iv_cpu = ivat_image_from_order(Dc, order_cpu)
        Dg = cp.asarray(Dc)
        mug, mvg = boruvka_mst_gpu(Dg)
        order_gpu = vat_order_from_mst_gpu(Dg, mug, mvg)
        V = ivat_image_gpu(Dg, order_gpu, v_dtype="float64")
        om = float(np.mean(order_gpu == order_cpu))
        diff = float(np.abs(iv_cpu - cp.asnumpy(V)).max())
        print(f"  n={n:5d}: order_match={om:.4f}  iVAT max|GPU-CPU|={diff:.2e}")
        del Dg, V
        cp.get_default_memory_pool().free_all_blocks()


def pipeline_figure():
    """Two panels: the iVAT stage (GPU vs CPU) and the end-to-end pipeline.

    End-to-end is the honest 'X on host -> iVAT image on host' comparison:
      * device-resident : X -> GPU pairwise -> MST -> order -> GPU iVAT, image
        copied to host once at the end (the n x n matrix never leaves the device).
      * host iVAT        : X -> GPU pairwise -> MST -> order, then the n x n matrix
        is copied back to the host for the numba iVAT (the round-trip we remove).
    """
    import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from tribbleclustering.pcvat import pairwise_distances_c_64, compute_ivat_c
    from experiments.boruvka_vat import make_blobs, ivat_image_from_order

    def sync():
        cp.cuda.Stream.null.synchronize()

    sizes = [4000, 8000, 16000, 32000]
    ivat_cpu, ivat_g64, ivat_g32 = [], [], []
    e2e_dev, e2e_host = [], []
    # warm numba
    ivat_image_from_order(np.zeros((4, 4)), np.arange(4))
    for n in sizes:
        X = make_blobs(n, 10, 25, seed=7)
        Dc = pairwise_distances_c_64(X)
        _, _, order = compute_ivat_c(Dc.copy(), inplace=False)

        t = time.perf_counter()
        compute_ivat_c(Dc.copy(), inplace=False)
        ivat_cpu.append((time.perf_counter() - t) * 1e3)

        Dg = cp.asarray(Dc)
        ivat_image_gpu(Dg, order, v_dtype="float64")
        sync()
        t = time.perf_counter()
        ivat_image_gpu(Dg, order, v_dtype="float64")
        sync()
        ivat_g64.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        ivat_image_gpu(Dg, order, v_dtype="float32")
        sync()
        ivat_g32.append((time.perf_counter() - t) * 1e3)

        # end-to-end, f32 distances born on device
        sync()
        t = time.perf_counter()
        V, _ = vat_ivat_gpu(X, d_dtype=cp.float32, v_dtype="float32")
        _ = cp.asnumpy(V)  # image to host once
        sync()
        e2e_dev.append((time.perf_counter() - t) * 1e3)
        del V

        # host-iVAT alternative: GPU pairwise+MST+order, copy D back, numba iVAT
        sync()
        t = time.perf_counter()
        Dg2 = pairwise_distances_gpu(X, dtype=cp.float32)
        mu, mv = boruvka_mst_gpu(Dg2)
        order_h = vat_order_from_mst_gpu(Dg2, mu, mv)
        Dhost = cp.asnumpy(Dg2).astype(np.float64)  # the n x n round-trip
        ivat_image_from_order(Dhost, order_h)
        e2e_host.append((time.perf_counter() - t) * 1e3)
        del Dg, Dg2
        cp.get_default_memory_pool().free_all_blocks()
        print(
            f"  n={n:6d}: iVAT cpu {ivat_cpu[-1]:7.0f}  gpu-f64 {ivat_g64[-1]:7.0f}  "
            f"gpu-f32 {ivat_g32[-1]:7.0f}  |  e2e device {e2e_dev[-1]:7.0f}  "
            f"e2e host-iVAT {e2e_host[-1]:7.0f}  (ms)"
        )

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(sizes, ivat_cpu, "s-", color="0.5", label="CPU compute_ivat_c")
    ax.plot(sizes, ivat_g64, "^-", color="tab:blue", label="GPU iVAT f64 (device)")
    ax.plot(sizes, ivat_g32, "D-", color="tab:orange", label="GPU iVAT f32 (device)")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("iVAT stage time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("iVAT image stage: on-device vs host")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax2.plot(
        sizes,
        e2e_host,
        "o--",
        color="tab:red",
        alpha=0.7,
        label="GPU MST, host iVAT (n×n copied back)",
    )
    ax2.plot(
        sizes, e2e_dev, "^-", color="tab:blue", label="fully device-resident (f32)"
    )
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("end-to-end X→iVAT image (ms)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("End-to-end: closing the loop removes the n×n round-trip")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig.suptitle(
        "Closing the VAT loop on GB10 — the whole pipeline stays on the device",
        fontsize=12,
    )
    fig.tight_layout()
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    path = fig_dir / "gpu_vat_pipeline.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("On-device VAT/iVAT (closing the loop)\n" + "=" * 37)
    print(f"CuPy GPU available: {_HAS_CUPY}")
    if not _HAS_CUPY:
        raise SystemExit("no CUDA device — nothing to measure")
    _validate()
    print("\nPipeline benchmark (iVAT stage + end-to-end)...")
    p = pipeline_figure()
    print(f"  wrote {p}")
