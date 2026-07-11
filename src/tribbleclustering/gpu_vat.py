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

import numpy as np
from numba import njit

from . import gpu as _gpu

_cp = _gpu._cp

_BORUVKA_SRC = r"""
#include <math_constants.h>
extern "C" {

__global__ void bv_min_out_edge(const double* __restrict__ D, const int* __restrict__ comp,
                                int n, double* __restrict__ best_w, int* __restrict__ best_j) {
    int i = blockIdx.x;                 // one block per row (coalesced scan)
    if (i >= n) return;
    int ci = comp[i];
    const double* row = D + (size_t)i * n;
    double lb = CUDART_INF; int lj = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (comp[j] != ci) { double d = row[j]; if (d < lb) { lb = d; lj = j; } }
    }
    __shared__ double sw[256];
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

__global__ void bv_reduce_minw(const double* __restrict__ best_w, const int* __restrict__ comp,
                               int n, unsigned long long* __restrict__ comp_min_key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // non-negative doubles are monotonic as u64 bit patterns
    unsigned long long k = (unsigned long long)__double_as_longlong(best_w[i]);
    atomicMin(&comp_min_key[comp[i]], k);
}

__global__ void bv_pick_vertex(const double* __restrict__ best_w, const int* __restrict__ comp,
                               int n, const unsigned long long* __restrict__ comp_min_key,
                               int* __restrict__ comp_win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long k = (unsigned long long)__double_as_longlong(best_w[i]);
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

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = _cp.RawModule(code=_BORUVKA_SRC, options=("--std=c++14",))
    return _MOD


def boruvka_mst_device(Dg):
    """Exact MST of a device-resident dense symmetric dissimilarity matrix
    (float64 CuPy array). Returns device int32 arrays (mst_u, mst_v)."""
    mod = _module()
    scan = mod.get_function("bv_min_out_edge")
    redw = mod.get_function("bv_reduce_minw")
    pick = mod.get_function("bv_pick_vertex")
    hook = mod.get_function("bv_hook")
    relabel = mod.get_function("bv_relabel")
    jump = mod.get_function("bv_jump")

    Dg = _cp.ascontiguousarray(Dg, dtype=_cp.float64)
    n = Dg.shape[0]
    comp = _cp.arange(n, dtype=_cp.int32)
    best_w = _cp.empty(n, dtype=_cp.float64)
    best_j = _cp.empty(n, dtype=_cp.int32)
    comp_min_key = _cp.empty(n, dtype=_cp.uint64)
    comp_win = _cp.empty(n, dtype=_cp.int32)
    root_parent = _cp.empty(n, dtype=_cp.int32)
    mst_u = _cp.empty(max(1, n - 1), dtype=_cp.int32)
    mst_v = _cp.empty(max(1, n - 1), dtype=_cp.int32)
    ne = _cp.zeros(1, dtype=_cp.int32)

    tpb = 256
    grid = (n + tpb - 1) // tpb
    U64_MAX = _cp.uint64(0xFFFFFFFFFFFFFFFF)
    I32_MAX = np.int32(0x7FFFFFFF)
    n_jumps = int(np.ceil(np.log2(max(2, n)))) + 2
    max_rounds = 2 * int(np.ceil(np.log2(max(2, n)))) + 3

    for _ in range(max_rounds):
        comp_min_key.fill(U64_MAX)
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


def vat_gpu(X, high_precision: bool = True, return_distances: bool = False):
    """Compute the exact VAT ordering fully on the GPU.

    Distances and the MST are built on the device (the n x n matrix stays
    resident); only the length-n ordering and parent map return to host.

    Parameters
    ----------
    X : (n, d) float32/float64 array of coordinates.
    return_distances : if True, also return the resident CuPy distance matrix.

    Returns
    -------
    order : (n,) int32 VAT permutation (identical to serial VAT).
    parent : (n,) int32 MST parent of each vertex in the traversal (-1 for the
        seed).
    [distances] : (n, n) CuPy array, only if return_distances=True.
    """
    if not _gpu.is_available():
        raise RuntimeError("CuPy/CUDA device not available")
    Dg = _gpu.pairwise_distances_device(X, high_precision=high_precision)
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
    """In-place minimax iVAT recurrence on an already-VAT-ordered matrix V.
    Mirrors pcvat's _compute_ivat_kernel (lower triangle then back-copy)."""
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


def ivat_gpu(X, high_precision: bool = True):
    """Compute the IVAT matrix and VAT ordering using the on-device front-end.

    Distances and the exact MST are built on the GPU (matrix resident); the
    ordering is derived on-device. The iVAT minimax recurrence itself is still
    serial and runs on the host (moving it on-device is future work), so the
    resident matrix is copied to the host once, reordered, and transformed.

    Returns (ivat_matrix, order) — ivat_matrix is bit-identical to
    ``compute_ivat_c`` and ``order`` is the exact VAT permutation.
    """
    if not _gpu.is_available():
        raise RuntimeError("CuPy/CUDA device not available")
    order, parent, Dg = vat_gpu(X, high_precision=high_precision, return_distances=True)
    n = Dg.shape[0]
    D_host = _cp.asnumpy(Dg)  # host copy needed for the serial CPU recurrence
    del Dg
    _cp.get_default_memory_pool().free_all_blocks()
    V = np.ascontiguousarray(D_host[np.ix_(order, order)])
    ivat = _ivat_from_vat_ordered(V)
    return ivat, order
