"""Fully on-device VAT front-end (CuPy): distances -> Borůvka MST -> ordering,
with the dense dissimilarity matrix kept resident on the GPU.

Motivation (see experiments/BORUVKA_VAT_FINDINGS.md): a device-side Borůvka MST
builds the exact MST ~5x faster than the serial CPU Prim and the win does not
erode with n — but ONLY if the n x n matrix is already on the GPU, since a
host->device copy of that matrix erases the gain. This module satisfies that
condition end-to-end: it computes the distance matrix on the device
(gpu.pairwise_distances_device), builds the MST on the device, and returns just
the length-n VAT ordering to the host (plus the parent map). The output is the
*exact* VAT ordering — Prim's insertion order — because Prim only ever traverses
MST edges, so re-deriving the order by traversing the MST from the same
max-dissimilarity seed reproduces it bit-for-bit.

Everything is gated on CuPy/CUDA availability; callers should check
gpu.is_available() and fall back to the CPU VAT (pcvat.compute_vat_c) otherwise.
"""

from __future__ import annotations

import heapq
import warnings

import numpy as np
from numba import njit

from . import gpu as _gpu

_cp = _gpu._cp

# The Borůvka kernels are templated over three type tokens so the identical
# algorithm runs at f64/f32/f16 storage (the n x n matrix is the memory
# footprint, and the per-round scan is bandwidth-bound, so narrower storage is
# both smaller and ~2x faster per step). TREAL is the stored element type; TCOMP
# is the compare/reduce type (float for f16/f32, double for f64 — f16 is widened
# on load, so no 16-bit atomics); TKEY is the monotonic bit-cast of a
# non-negative TCOMP weight (u32 for float, u64 for double), on which atomicMin
# gives the true min. TLOAD widens a stored TREAL to TCOMP; TASKEY is the
# TCOMP->TKEY bit-cast intrinsic.
_BORUVKA_TEMPLATE = r"""
#include <math_constants.h>
TEXTRAHDR
extern "C" {

__global__ void bv_min_out_edge(const TREAL* __restrict__ D, const int* __restrict__ comp,
                                int n, TCOMP* __restrict__ best_w, int* __restrict__ best_j) {
    int i = blockIdx.x;                 // one block per row (coalesced scan)
    if (i >= n) return;
    int ci = comp[i];
    const TREAL* row = D + (size_t)i * n;
    TCOMP lb = TINF; int lj = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (comp[j] != ci) { TCOMP d = TLOAD(row[j]); if (d < lb) { lb = d; lj = j; } }
    }
    __shared__ TCOMP sw[256];
    __shared__ int sj[256];
    sw[threadIdx.x] = lb; sj[threadIdx.x] = lj;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && sw[threadIdx.x + s] < sw[threadIdx.x]) {
            sw[threadIdx.x] = sw[threadIdx.x + s];
            sj[threadIdx.x] = sj[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { best_w[i] = sw[0]; best_j[i] = sj[0]; }
}

__global__ void bv_reduce_minw(const TCOMP* __restrict__ best_w, const int* __restrict__ comp,
                               int n, TKEY* __restrict__ comp_min_key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // non-negative TCOMP weights are monotonic as TKEY bit patterns
    TKEY k = (TKEY)TASKEY(best_w[i]);
    atomicMin(&comp_min_key[comp[i]], k);
}

__global__ void bv_pick_vertex(const TCOMP* __restrict__ best_w, const int* __restrict__ comp,
                               int n, const TKEY* __restrict__ comp_min_key,
                               int* __restrict__ comp_win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    TKEY k = (TKEY)TASKEY(best_w[i]);
    if (k == comp_min_key[comp[i]]) atomicMin(&comp_win[comp[i]], i);
}

__global__ void bv_hook(const int* __restrict__ comp, const int* __restrict__ best_j,
                        const int* __restrict__ comp_win, int n,
                        int* __restrict__ root_parent, int* __restrict__ mst_u,
                        int* __restrict__ mst_v, int* __restrict__ ne) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    root_parent[c] = c;
    if (comp[c] != c) return;           // only roots act
    int wv = comp_win[c];
    if (wv == 0x7fffffff) return;       // no outgoing edge
    int tv = best_j[wv];
    int tc = comp[tv];
    bool mutual = false;
    int wv2 = comp_win[tc];
    if (wv2 != 0x7fffffff) { if (comp[best_j[wv2]] == c) mutual = true; }
    if (mutual && c < tc) return;       // larger id emits the shared edge
    root_parent[c] = tc;
    int slot = atomicAdd(ne, 1);
    mst_u[slot] = wv; mst_v[slot] = tv;
}

__global__ void bv_relabel(int* __restrict__ comp, const int* __restrict__ root_parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) comp[i] = root_parent[comp[i]];
}

__global__ void bv_jump(int* __restrict__ comp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) comp[i] = comp[comp[i]];
}

}  // extern C
"""

# storage-dtype -> (TREAL, TCOMP, TKEY, TLOAD, TASKEY, TINF, TEXTRAHDR)
_MST_SUBST = {
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
_COMP_DT = {"float64": "float64", "float32": "float32", "float16": "float32"}
_KEY_DT = {"float64": "uint64", "float32": "uint32", "float16": "uint32"}
_KEY_MAX = {"float64": 0xFFFFFFFFFFFFFFFF, "float32": 0xFFFFFFFF, "float16": 0xFFFFFFFF}

_MST_MODS: dict = {}


def _mst_module(dtype_name):
    if dtype_name not in _MST_MODS:
        src = _BORUVKA_TEMPLATE
        for tok, val in _MST_SUBST[dtype_name].items():
            src = src.replace(tok, val)
        _MST_MODS[dtype_name] = _cp.RawModule(code=src, options=("--std=c++14",))
    return _MST_MODS[dtype_name]


def boruvka_mst_device(Dg):
    """Exact MST of a device-resident dense symmetric dissimilarity matrix.

    ``Dg`` is a CuPy array in float16/float32/float64 (its stored dtype is
    honoured — it is the n x n memory footprint; f16 is widened to f32 for
    comparison). Returns device int32 arrays ``(mst_u, mst_v)``.
    """
    dt = _cp.dtype(getattr(Dg, "dtype", _cp.float64)).name
    if dt not in _MST_SUBST:
        dt = "float64"
    mod = _mst_module(dt)
    scan = mod.get_function("bv_min_out_edge")
    redw = mod.get_function("bv_reduce_minw")
    pick = mod.get_function("bv_pick_vertex")
    hook = mod.get_function("bv_hook")
    relabel = mod.get_function("bv_relabel")
    jump = mod.get_function("bv_jump")

    Dg = _cp.ascontiguousarray(Dg, dtype=dt)
    n = Dg.shape[0]
    comp = _cp.arange(n, dtype=_cp.int32)
    best_w = _cp.empty(n, dtype=_COMP_DT[dt])
    best_j = _cp.empty(n, dtype=_cp.int32)
    comp_min_key = _cp.empty(n, dtype=_KEY_DT[dt])
    comp_win = _cp.empty(n, dtype=_cp.int32)
    root_parent = _cp.empty(n, dtype=_cp.int32)
    mst_u = _cp.empty(max(1, n - 1), dtype=_cp.int32)
    mst_v = _cp.empty(max(1, n - 1), dtype=_cp.int32)
    ne = _cp.zeros(1, dtype=_cp.int32)

    tpb = 256
    grid = (n + tpb - 1) // tpb
    key_max = _cp.dtype(_KEY_DT[dt]).type(_KEY_MAX[dt])
    I32_MAX = np.int32(0x7FFFFFFF)
    n_jumps = int(np.ceil(np.log2(max(2, n)))) + 2
    max_rounds = 2 * int(np.ceil(np.log2(max(2, n)))) + 3

    for _ in range(max_rounds):
        comp_min_key.fill(key_max)
        comp_win.fill(I32_MAX)
        scan((n,), (tpb,), (Dg, comp, np.int32(n), best_w, best_j))
        redw((grid,), (tpb,), (best_w, comp, np.int32(n), comp_min_key))
        pick((grid,), (tpb,), (best_w, comp, np.int32(n), comp_min_key, comp_win))
        hook(
            (grid,),
            (tpb,),
            (comp, best_j, comp_win, np.int32(n), root_parent, mst_u, mst_v, ne),
        )
        relabel((grid,), (tpb,), (comp, root_parent, np.int32(n)))
        for _ in range(n_jumps):
            jump((grid,), (tpb,), (comp, np.int32(n)))
        if int(ne.get()[0]) >= n - 1:
            break
    cnt = int(ne.get()[0])
    return mst_u[:cnt], mst_v[:cnt]


def _order_from_mst(mst_u, mst_v, weights, n, src):
    """Prim traversal of the MST tree from `src` -> the exact VAT ordering."""
    adj = [[] for _ in range(n)]
    for a, b, w in zip(mst_u.tolist(), mst_v.tolist(), weights.tolist()):
        adj[a].append((w, b))
        adj[b].append((w, a))
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    visited[src] = True
    order[0] = src
    k = 1
    h = [(w, nb, src) for (w, nb) in adj[src]]
    heapq.heapify(h)
    while h:
        w, v, par = heapq.heappop(h)
        if visited[v]:
            continue
        visited[v] = True
        order[k] = v
        parent[v] = par
        k += 1
        for w2, nb in adj[v]:
            if not visited[nb]:
                heapq.heappush(h, (w2, nb, v))
    return order, parent


def vat_gpu(X, high_precision: bool = True, return_distances: bool = False, dtype=None):
    """Compute the exact VAT ordering fully on the GPU.

    Distances and the MST are built on the device (the n x n matrix stays
    resident); only the length-n ordering and parent map return to host.

    Parameters
    ----------
    X : (n, d) float32/float64 array of coordinates.
    return_distances : if True, also return the resident CuPy distance matrix.
    dtype : matrix storage precision (``None`` keeps the input dtype;
        ``float16``/``float32``/``float64`` force it). Narrower storage is
        smaller and faster; the VAT ordering is exact at f32/f64 and near-exact
        at f16.

    Returns
    -------
    order : (n,) int32 VAT permutation (identical to serial VAT).
    parent : (n,) int32 MST parent of each vertex in the traversal (-1 for the
        seed).
    [distances] : (n, n) CuPy array, only if return_distances=True.
    """
    if not _gpu.is_available():
        raise RuntimeError("CuPy/CUDA device not available")
    Dg = _gpu.pairwise_distances_device(X, high_precision=high_precision, dtype=dtype)
    n = Dg.shape[0]
    mu, mv = boruvka_mst_device(Dg)
    # gather the n-1 edge weights on-device, then bring only O(n) data to host
    w = _cp.asnumpy(Dg[mu, mv])
    mu_h = _cp.asnumpy(mu)
    mv_h = _cp.asnumpy(mv)
    src = int(_cp.argmax(Dg).get()) // n  # VAT seed: global-max-dissimilarity vertex
    order, parent = _order_from_mst(mu_h, mv_h, w, n, src)
    if return_distances:
        return order, parent, Dg
    del Dg
    _cp.get_default_memory_pool().free_all_blocks()
    return order, parent


@njit(cache=True)
def _ivat_from_vat_ordered(V):
    """In-place minimax iVAT recurrence on an already-VAT-ordered matrix V
    (host/numba fallback). Mirrors pcvat's _compute_ivat_kernel (lower triangle
    then back-copy)."""
    n = V.shape[0]
    for r in range(1, n):
        jj = 0
        mn = V[r, 0]
        for c in range(1, r):
            if V[r, c] < mn:
                mn = V[r, c]
                jj = c
        for c in range(r):
            if c == jj:
                V[r, c] = mn
            else:
                cur = V[jj, c] if jj > c else V[c, jj]
                V[r, c] = mn if mn > cur else cur
    for i in range(1, n):
        for j in range(i):
            V[j, i] = V[i, j]
    return V


# On-device iVAT: gather + parallel row-pivot, serial-in-r max propagation,
# symmetrise. The recurrence looks strictly serial, but each row's pivot
# (h_r = min_{c<r} V[r,c], jj_r = argmin) reads only the *original* reordered
# distances (earlier rows never overwrite row r before it is processed), so all
# (h_r, jj_r) compute in one parallel pass; only the max-propagation keeps the
# serial-in-r dependency. TVOUT is the image type; D is read through TREAL/TLOAD.
_IVAT_TEMPLATE = r"""
#include <math_constants.h>
TEXTRAHDR
extern "C" {

__global__ void iv_gather_rowmin(const TREAL* __restrict__ D, const int* __restrict__ order,
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
        if (threadIdx.x < s && sw[threadIdx.x + s] < sw[threadIdx.x]) {
            sw[threadIdx.x] = sw[threadIdx.x + s];
            sj[threadIdx.x] = sj[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { hrow[r] = sw[0]; jrow[r] = sj[0]; }
}

__global__ void iv_row(TVOUT* __restrict__ V, const TVOUT* __restrict__ hrow,
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

__global__ void iv_symmetrize(TVOUT* __restrict__ V, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < n && c < n && c > r) V[(size_t)r * n + c] = V[(size_t)c * n + r];
}

}  // extern C
"""

_VOUT_C = {"float64": "double", "float32": "float"}
_IVAT_MODS: dict = {}


def _ivat_module(d_dtype, v_dtype):
    key = (d_dtype, v_dtype)
    if key not in _IVAT_MODS:
        src = _IVAT_TEMPLATE
        src = src.replace("TREAL", _MST_SUBST[d_dtype]["TREAL"])
        src = src.replace("TLOAD", _MST_SUBST[d_dtype]["TLOAD"])
        src = src.replace("TEXTRAHDR", _MST_SUBST[d_dtype]["TEXTRAHDR"])
        src = src.replace("TVOUT", _VOUT_C[v_dtype])
        _IVAT_MODS[key] = _cp.RawModule(code=src, options=("--std=c++14",))
    return _IVAT_MODS[key]


def ivat_image_device(Dg, order, v_dtype="float64"):
    """On-device iVAT image from a device matrix and a VAT order.

    The reorder-gather, minimax recurrence and symmetrisation all run on the GPU
    (no host copy of the n x n matrix). ``v_dtype='float64'`` is bit-identical to
    the CPU engine. Returns a device (CuPy) array.
    """
    d_dtype = _cp.dtype(getattr(Dg, "dtype", _cp.float64)).name
    if d_dtype not in _MST_SUBST:
        d_dtype = "float64"
    Dg = _cp.ascontiguousarray(_cp.asarray(Dg, dtype=d_dtype))
    n = Dg.shape[0]
    order_g = _cp.asarray(order, dtype=_cp.int32)
    mod = _ivat_module(d_dtype, v_dtype)
    k_gather = mod.get_function("iv_gather_rowmin")
    k_row = mod.get_function("iv_row")
    k_sym = mod.get_function("iv_symmetrize")

    V = _cp.empty((n, n), dtype=v_dtype)
    hrow = _cp.empty(n, dtype=v_dtype)
    jrow = _cp.empty(n, dtype=_cp.int32)
    tpb = 256
    k_gather((n,), (tpb,), (Dg, order_g, np.int32(n), V, hrow, jrow))
    for r in range(1, n):
        grid = (r + tpb - 1) // tpb
        k_row((grid,), (tpb,), (V, hrow, jrow, np.int32(n), np.int32(r)))
    tb = 16
    k_sym(
        ((n + tb - 1) // tb, (n + tb - 1) // tb),
        (tb, tb),
        (V, np.int32(n)),
    )
    return V


def _resolve_vat_dtype(dtype) -> str:
    """GPU-VAT precision policy -> storage dtype name.

    float32 (default) and float16 (opt-in) are used as requested; float64 is
    downgraded to float32 with a warning so the caller knows the precision was
    changed on the fly (the CPU backend is the exact-float64 path).
    """
    name = np.dtype(dtype).name
    if name == "float64":
        warnings.warn(
            "GPU VAT backend computes in float32; the requested float64 was "
            "converted to float32 on the fly. Use backend/on_device off (the CPU "
            "engine) for exact float64.",
            stacklevel=3,
        )
        return "float32"
    if name in ("float16", "float32"):
        return name
    raise TypeError(f"dtype must be float16/float32/float64, got {name}")


def ivat_gpu(X, high_precision: bool = True, dtype=None, device_recurrence=True):
    """Compute the IVAT matrix and VAT ordering fully on the GPU.

    Distances, the exact MST, the ordering, and (by default) the iVAT minimax
    recurrence all run on the device with the n x n matrix resident — closing the
    loop end-to-end, so nothing but the O(n) order and the final image return to
    the host.

    ``dtype`` selects the matrix storage precision: ``None`` (default) keeps the
    input dtype (float32/float64) — **bit-identical to ``compute_ivat_c``**;
    ``float16``/``float32`` shrink the resident matrix (f16 near-exact). This is
    a faithful low-level primitive: it does not apply the f32-default policy —
    that lives in :class:`IVATMeans`. ``device_recurrence=False`` falls back to
    the host numba recurrence (copies the matrix back once) for parity/debugging.

    Returns (ivat_matrix, order) — a host image and the exact VAT permutation.
    """
    if not _gpu.is_available():
        raise RuntimeError("CuPy/CUDA device not available")
    order, parent, Dg = vat_gpu(
        X, high_precision=high_precision, return_distances=True, dtype=dtype
    )
    store = _cp.dtype(Dg.dtype).name
    if device_recurrence:
        # f64 storage -> f64 image (bit-exact); f32/f16 storage -> f32 image
        v_dtype = "float64" if store == "float64" else "float32"
        V = ivat_image_device(Dg, order, v_dtype=v_dtype)
        ivat = _cp.asnumpy(V)
        del Dg, V
        _cp.get_default_memory_pool().free_all_blocks()
        return ivat, order
    D_host = _cp.asnumpy(Dg)  # host copy for the serial CPU recurrence
    del Dg
    _cp.get_default_memory_pool().free_all_blocks()
    V = np.ascontiguousarray(D_host[np.ix_(order, order)].astype(np.float64))
    ivat = _ivat_from_vat_ordered(V)
    return ivat, order
