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

_SRC = r"""
#include <math_constants.h>
extern "C" {

__global__ void min_out_edge(const double* __restrict__ D, const int* __restrict__ comp,
                             int n, double* __restrict__ best_w, int* __restrict__ best_j) {
    int i = blockIdx.x;                 // one block per row
    if (i >= n) return;
    int ci = comp[i];
    const double* row = D + (size_t)i * n;
    double lb = CUDART_INF; int lj = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (comp[j] != ci) {
            double d = row[j];
            if (d < lb) { lb = d; lj = j; }
        }
    }
    __shared__ double sw[256];
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

__global__ void reduce_minw(const double* __restrict__ best_w, const int* __restrict__ comp,
                            int n, unsigned long long* __restrict__ comp_min_key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // non-negative doubles are monotonic as u64 bit patterns
    unsigned long long k = (unsigned long long)__double_as_longlong(best_w[i]);
    atomicMin(&comp_min_key[comp[i]], k);
}

__global__ void pick_vertex(const double* __restrict__ best_w, const int* __restrict__ comp,
                            int n, const unsigned long long* __restrict__ comp_min_key,
                            int* __restrict__ comp_win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long k = (unsigned long long)__double_as_longlong(best_w[i]);
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

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=("--std=c++14",))
    return _MOD


def boruvka_mst_gpu(D):
    """Device-side Boruvka MST of a dense symmetric dissimilarity matrix.

    Accepts a host ndarray or a cupy array (float64). Returns (mst_u, mst_v) as
    host int32 arrays of length n-1.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy/CUDA device not available")
    mod = _module()
    k_scan = mod.get_function("min_out_edge")
    k_redw = mod.get_function("reduce_minw")
    k_pick = mod.get_function("pick_vertex")
    k_hook = mod.get_function("hook")
    k_relabel = mod.get_function("relabel")
    k_jump = mod.get_function("jump")

    Dg = cp.ascontiguousarray(cp.asarray(D, dtype=cp.float64))
    n = Dg.shape[0]

    comp = cp.arange(n, dtype=cp.int32)
    best_w = cp.empty(n, dtype=cp.float64)
    best_j = cp.empty(n, dtype=cp.int32)
    comp_min_key = cp.empty(n, dtype=cp.uint64)
    comp_win = cp.empty(n, dtype=cp.int32)
    root_parent = cp.empty(n, dtype=cp.int32)
    mst_u = cp.empty(n - 1, dtype=cp.int32)
    mst_v = cp.empty(n - 1, dtype=cp.int32)
    ne = cp.zeros(1, dtype=cp.int32)

    tpb = 256
    grid1d = (n + tpb - 1) // tpb
    U64_MAX = cp.uint64(0xFFFFFFFFFFFFFFFF)
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
