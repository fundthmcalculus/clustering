"""Real device-side GPU Boruvka MST (CuPy RawKernels), for the VAT spike.

The naive GPU attempt (experiments/boruvka_vat.py::boruvka_mst_cupy) lost badly:
it materialised an n x n mask every round and ran union-find on the host in a
Python loop, so it was allocation- and host-sync-bound. This version keeps the
entire round on the device:

  1. min_out_edge   — one block per row, threads scan the row coalesced and
     block-reduce to each vertex's minimum edge leaving its own component.
  2. reduce_minw    — per component, atomicMin the (monotonic bit-cast of the)
     minimum outgoing weight. Non-negative IEEE doubles are order-preserving as
     u64, so atomicMin on the bits gives the true min.
  3. pick_vertex    — per component, atomicMin the vertex index among those
     achieving that min weight (deterministic tie-break -> smallest index).
  4. hook           — each component root hooks to its chosen neighbour's
     component; mutual (2-cycle) picks are resolved by letting the larger id
     hook so the shared edge is emitted exactly once. Edges are appended via an
     atomic counter.
  5. relabel + jump — parallel pointer-jumping flattens the component forest to
     roots (no host union-find).

Only a single scalar (edge count) is copied to the host per round to test for
termination. The O(n^2) matrix read per round is coalesced and bandwidth-bound,
which is where a GPU's memory bandwidth can beat the CPU.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _HAS_CUPY = False

# Kernel source is templated over three type tokens so the identical algorithm
# runs at f64/f32/f16 storage. TREAL is the on-device element type of the n x n
# matrix (the memory that dominates); TCOMP is the compare/reduce type (float for
# f16/f32, double for f64 — f16 is widened on load so we never rely on 16-bit
# atomics or comparisons); TKEY is the monotonic bit-cast of a *non-negative*
# TCOMP weight, on which atomicMin gives the true min (u32 for float, u64 for
# double). TLOAD widens a stored TREAL to TCOMP (identity for f32/f64,
# __half2float for f16); TASKEY is the TCOMP->TKEY bit-cast intrinsic.
_TEMPLATE = r"""
#include <math_constants.h>
TEXTRAHDR
extern "C" {

__global__ void min_out_edge(const TREAL* __restrict__ D, const int* __restrict__ comp,
                             int n, TCOMP* __restrict__ best_w, int* __restrict__ best_j) {
    int i = blockIdx.x;                 // one block per row
    if (i >= n) return;
    int ci = comp[i];
    const TREAL* row = D + (size_t)i * n;
    TCOMP lb = TINF; int lj = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (comp[j] != ci) {
            TCOMP d = TLOAD(row[j]);
            if (d < lb) { lb = d; lj = j; }
        }
    }
    __shared__ TCOMP sw[256];
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
    if (threadIdx.x == 0) { best_w[i] = sw[0]; best_j[i] = sj[0]; }
}

__global__ void reduce_minw(const TCOMP* __restrict__ best_w, const int* __restrict__ comp,
                            int n, TKEY* __restrict__ comp_min_key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // non-negative TCOMP weights are monotonic as TKEY bit patterns
    TKEY k = (TKEY)TASKEY(best_w[i]);
    atomicMin(&comp_min_key[comp[i]], k);
}

__global__ void pick_vertex(const TCOMP* __restrict__ best_w, const int* __restrict__ comp,
                            int n, const TKEY* __restrict__ comp_min_key,
                            int* __restrict__ comp_win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    TKEY k = (TKEY)TASKEY(best_w[i]);
    if (k == comp_min_key[comp[i]]) atomicMin(&comp_win[comp[i]], i);
}

__global__ void hook(const int* __restrict__ comp, const int* __restrict__ best_j,
                     const int* __restrict__ comp_win, int n,
                     int* __restrict__ root_parent, int* __restrict__ mst_u,
                     int* __restrict__ mst_v, int* __restrict__ ne) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    root_parent[c] = c;                 // default: remain a root
    if (comp[c] != c) return;           // only component roots act
    int wv = comp_win[c];
    if (wv == 0x7fffffff) return;       // no outgoing edge -> isolated/done
    int tv = best_j[wv];
    int tc = comp[tv];
    // mutual (2-cycle) detection
    bool mutual = false;
    int wv2 = comp_win[tc];
    if (wv2 != 0x7fffffff) {
        int tv2 = best_j[wv2];
        if (comp[tv2] == c) mutual = true;
    }
    if (mutual && c < tc) return;       // larger id emits the shared edge
    root_parent[c] = tc;
    int slot = atomicAdd(ne, 1);
    mst_u[slot] = wv;
    mst_v[slot] = tv;
}

__global__ void relabel(int* __restrict__ comp, const int* __restrict__ root_parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    comp[i] = root_parent[comp[i]];
}

__global__ void jump(int* __restrict__ comp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    comp[i] = comp[comp[i]];
}

}  // extern C
"""

# Per-dtype token substitutions. f16 and f32 share the float compute/key path;
# only the stored TREAL and the widening TLOAD differ.
_SUBST = {
    "float64": dict(
        TREAL="double",
        TCOMP="double",
        TKEY="unsigned long long",
        TLOAD="",
        TASKEY="__double_as_longlong",
        TINF="CUDART_INF",
        TEXTRAHDR="",
    ),
    "float32": dict(
        TREAL="float",
        TCOMP="float",
        TKEY="unsigned int",
        TLOAD="",
        TASKEY="__float_as_uint",
        TINF="CUDART_INF_F",
        TEXTRAHDR="",
    ),
    "float16": dict(
        TREAL="__half",
        TCOMP="float",
        TKEY="unsigned int",
        TLOAD="__half2float",
        TASKEY="__float_as_uint",
        TINF="CUDART_INF_F",
        TEXTRAHDR="#include <cuda_fp16.h>",
    ),
}
# COMP/KEY host dtypes and the "no edge" key sentinel per storage dtype.
_COMP_DT = {"float64": "float64", "float32": "float32", "float16": "float32"}
_KEY_DT = {"float64": "uint64", "float32": "uint32", "float16": "uint32"}
_KEY_MAX = {
    "float64": 0xFFFFFFFFFFFFFFFF,
    "float32": 0xFFFFFFFF,
    "float16": 0xFFFFFFFF,
}
_SUPPORTED = tuple(_SUBST)

_MODS: dict = {}


def _module(dtype_name):
    if dtype_name not in _MODS:
        src = _TEMPLATE
        for tok, val in _SUBST[dtype_name].items():
            src = src.replace(tok, val)
        _MODS[dtype_name] = cp.RawModule(code=src, options=("--std=c++14",))
    return _MODS[dtype_name]


def alloc_unified(shape, dtype=None):
    """Allocate a matrix in CUDA *managed* (unified) memory.

    On a unified-memory device (e.g. DGX Spark / GB10 Grace-Blackwell) the
    returned array is backed by pages the CPU and GPU share coherently, so the
    CPU can fill it (e.g. write a pairwise-distance matrix) and ``boruvka_mst_gpu``
    can consume it with **no explicit host->device copy** and only one ``n x n``
    allocation instead of two. Returns a cupy ndarray viewing the managed buffer.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    dtype = cp.float64 if dtype is None else dtype
    n_bytes = int(np.prod(shape)) * cp.dtype(dtype).itemsize
    mem = cp.cuda.malloc_managed(n_bytes)
    return cp.ndarray(shape, dtype=dtype, memptr=mem)


def as_unified(D, dtype=None):
    """Copy a host ndarray into a managed (unified-memory) cupy array, once.

    Convenience for the common case where ``D`` already lives in host numpy: the
    single copy here replaces the per-call ``cp.asarray`` inside the MST loop, so
    repeated MST builds (or a downstream device pipeline) pay it zero more times.
    Pass ``dtype`` (``float16``/``float32``/``float64``) to also downcast on the
    way in, halving/quartering the resident ``n x n`` footprint.
    """
    dtype = D.dtype if dtype is None else dtype
    Dg = alloc_unified(D.shape, dtype=dtype)
    Dg[...] = cp.asarray(D)
    cp.cuda.Stream.null.synchronize()
    return Dg


def pairwise_distances_gpu(X, out=None, tile=4096, dtype=None):
    """Dense Euclidean pairwise-distance matrix, built entirely on the GPU.

    ``X`` is a host or device ``(n, d)`` array; the ``(n, n)`` result is written
    into ``out`` (allocate it with :func:`alloc_unified` to keep the whole VAT
    pipeline copy-free on a unified-memory device). Rows are computed in tiles so
    the transient ``||xi||^2 + ||xj||^2 - 2 xi.xj`` expansion never materialises a
    second full ``n x n`` buffer. Tiles are computed in f64 and cast to ``dtype``
    (``float16``/``float32``/``float64``, default f64) on store, so the stored
    matrix is at the target precision while the expansion stays accurate. Matches
    the CPU pairwise output to fp tolerance (at f64).
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    Xg = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float64))
    n = Xg.shape[0]
    sq = cp.einsum("ij,ij->i", Xg, Xg)  # per-row squared norm, length n
    D = alloc_unified((n, n), dtype=dtype) if out is None else out
    for s in range(0, n, tile):
        e = min(s + tile, n)
        # (tile, n) block: ||xi||^2 + ||xj||^2 - 2 xi.xj, clamped and sqrt'd
        g = Xg[s:e] @ Xg.T
        block = sq[s:e, None] + sq[None, :] - 2.0 * g
        cp.maximum(block, 0.0, out=block)
        cp.sqrt(block, out=block)
        D[s:e] = block.astype(dtype, copy=False)  # downcast on store
    cp.cuda.Stream.null.synchronize()
    return D


def boruvka_mst_gpu(D):
    """Device-side Boruvka MST of a dense symmetric dissimilarity matrix.

    Accepts a host ndarray or a cupy array in float16/float32/float64. The stored
    dtype is honoured (it is the ``n x n`` memory footprint); f16 is widened to
    f32 for all comparison/reduction so the result stays a valid MST of the stored
    matrix. Any non-supported dtype is promoted to float64. Returns (mst_u, mst_v)
    as host int32 arrays of length n-1.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    dt = cp.dtype(getattr(D, "dtype", cp.float64)).name
    if dt not in _SUPPORTED:
        dt = "float64"
    mod = _module(dt)
    k_scan = mod.get_function("min_out_edge")
    k_redw = mod.get_function("reduce_minw")
    k_pick = mod.get_function("pick_vertex")
    k_hook = mod.get_function("hook")
    k_relabel = mod.get_function("relabel")
    k_jump = mod.get_function("jump")

    Dg = cp.ascontiguousarray(cp.asarray(D, dtype=dt))
    n = Dg.shape[0]

    comp = cp.arange(n, dtype=cp.int32)
    best_w = cp.empty(n, dtype=_COMP_DT[dt])
    best_j = cp.empty(n, dtype=cp.int32)
    comp_min_key = cp.empty(n, dtype=_KEY_DT[dt])
    comp_win = cp.empty(n, dtype=cp.int32)
    root_parent = cp.empty(n, dtype=cp.int32)
    mst_u = cp.empty(n - 1, dtype=cp.int32)
    mst_v = cp.empty(n - 1, dtype=cp.int32)
    ne = cp.zeros(1, dtype=cp.int32)

    tpb = 256
    grid1d = (n + tpb - 1) // tpb
    U64_MAX = cp.dtype(_KEY_DT[dt]).type(_KEY_MAX[dt])
    I32_MAX = np.int32(0x7FFFFFFF)
    # A single round's hook forest has depth < n, and each pointer-jump halves
    # it, so ceil(log2 n)+2 sync-free jumps always flatten to roots.
    n_jumps = int(np.ceil(np.log2(max(2, n)))) + 2

    max_rounds = 2 * int(np.ceil(np.log2(max(2, n)))) + 3
    for _ in range(max_rounds):
        comp_min_key.fill(U64_MAX)
        comp_win.fill(I32_MAX)
        k_scan((n,), (tpb,), (Dg, comp, np.int32(n), best_w, best_j))
        k_redw((grid1d,), (tpb,), (best_w, comp, np.int32(n), comp_min_key))
        k_pick((grid1d,), (tpb,), (best_w, comp, np.int32(n), comp_min_key, comp_win))
        k_hook(
            (grid1d,),
            (tpb,),
            (comp, best_j, comp_win, np.int32(n), root_parent, mst_u, mst_v, ne),
        )
        k_relabel((grid1d,), (tpb,), (comp, root_parent, np.int32(n)))
        for _ in range(n_jumps):  # sync-free pointer-jumping
            k_jump((grid1d,), (tpb,), (comp, np.int32(n)))
        if int(ne.get()[0]) >= n - 1:  # one scalar host sync per round
            break

    cnt = int(ne.get()[0])
    return cp.asnumpy(mst_u[:cnt]), cp.asnumpy(mst_v[:cnt])
