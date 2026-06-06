# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange


# ---------------------------------------------------------------------------
# Core MST kernel — no allocation, no GIL, no Python calls.
# All working arrays are passed by the caller.
# ---------------------------------------------------------------------------
cdef void _prim_mst_kernel(
    const double* adj, int n,
    double* key, double* h_keys, int* h_verts, int* h_pos, int* h_par,
    char* in_mst,
    int* out_seq, int* out_par_seq
) noexcept nogil:
    cdef int i, j, u, v, v_at_idx
    cdef int src_i = 0, src_j = 0
    cdef int heap_size = 0
    cdef int seq_idx = 0
    cdef int idx, pidx, left, right, smallest
    cdef double max_val = -INFINITY
    cdef double adj_uv, saved_key, tmp_key
    cdef int tmp_vert

    for i in range(n):
        key[i]    = INFINITY
        h_pos[i]  = -1
        in_mst[i] = 0

    # Find global max to seed source
    for i in range(n):
        for j in range(n):
            if adj[i * n + j] > max_val:
                max_val = adj[i * n + j]
                src_i = i
                src_j = j

    # Push source vertex into heap
    key[src_i]   = max_val
    h_keys[0]    = max_val
    h_verts[0]   = src_i
    h_pos[src_i] = 0
    h_par[src_i] = src_j
    heap_size = 1

    while heap_size > 0:
        # Pop minimum-key vertex
        u = h_verts[0]
        out_seq[seq_idx]     = u
        out_par_seq[seq_idx] = h_par[u]
        seq_idx += 1

        in_mst[u] = 1
        h_pos[u]  = -1
        heap_size -= 1

        if heap_size > 0:
            # Move last → root, bubble down
            v_at_idx        = h_verts[heap_size]
            h_keys[0]       = h_keys[heap_size]
            h_verts[0]      = v_at_idx
            h_pos[v_at_idx] = 0

            idx = 0
            while True:
                smallest = idx
                left  = 2 * idx + 1
                right = left + 1
                if left  < heap_size and h_keys[left]  < h_keys[smallest]:
                    smallest = left
                if right < heap_size and h_keys[right] < h_keys[smallest]:
                    smallest = right
                if smallest == idx:
                    break
                tmp_key               = h_keys[idx]
                tmp_vert              = h_verts[idx]
                h_keys[idx]           = h_keys[smallest]
                h_verts[idx]          = h_verts[smallest]
                h_pos[h_verts[idx]]   = idx
                h_keys[smallest]      = tmp_key
                h_verts[smallest]     = tmp_vert
                h_pos[tmp_vert]       = smallest
                idx = smallest

        # Scan neighbors — decrease-key or insert
        for v in range(n):
            if in_mst[v]:
                continue
            adj_uv = adj[u * n + v]
            if adj_uv >= key[v]:
                continue
            key[v]   = adj_uv
            h_par[v] = seq_idx

            if h_pos[v] == -1:
                idx          = heap_size
                h_keys[idx]  = adj_uv
                h_verts[idx] = v
                h_pos[v]     = idx
                heap_size   += 1
            else:
                idx          = h_pos[v]
                h_keys[idx]  = adj_uv

            # Bubble up
            saved_key = adj_uv
            while idx > 0:
                pidx = (idx - 1) >> 1
                if h_keys[pidx] <= saved_key:
                    break
                h_keys[idx]           = h_keys[pidx]
                h_verts[idx]          = h_verts[pidx]
                h_pos[h_verts[idx]]   = idx
                idx = pidx
            h_keys[idx]  = saved_key
            h_verts[idx] = v
            h_pos[v]     = idx


# ---------------------------------------------------------------------------
# Shared helper: allocate working buffers, run kernel, free buffers.
# Raises MemoryError on failure; runs kernel with nogil.
# ---------------------------------------------------------------------------
cdef _run_mst(const double* adj_ptr, int n, int[:] heap_seq, int[:] parent_seq):
    cdef double* key    = <double*>malloc(n * sizeof(double))
    cdef double* h_keys = <double*>malloc(n * sizeof(double))
    cdef int*   h_verts = <int*>  malloc(n * sizeof(int))
    cdef int*   h_pos   = <int*>  malloc(n * sizeof(int))
    cdef int*   h_par   = <int*>  malloc(n * sizeof(int))
    cdef char*  in_mst  = <char*> malloc(n * sizeof(char))

    if not (key and h_keys and h_verts and h_pos and h_par and in_mst):
        free(key); free(h_keys); free(h_verts)
        free(h_pos); free(h_par); free(in_mst)
        raise MemoryError("MST workspace allocation failed")

    with nogil:
        _prim_mst_kernel(
            adj_ptr, n,
            key, h_keys, h_verts, h_pos, h_par, in_mst,
            &heap_seq[0], &parent_seq[0]
        )

    free(key); free(h_keys); free(h_verts)
    free(h_pos); free(h_par); free(in_mst)


# ---------------------------------------------------------------------------
# Public: Prim's MST only
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def vat_prim_mst_c(double[:, ::1] adj):
    """
    Optimized Prim's MST via decrease-key binary heap (O(n) heap size, nogil).
    Returns (heap_seq, parent_seq) as int32 numpy arrays.
    """
    cdef int n = adj.shape[0]
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    cdef int[:] heap_seq   = heap_seq_np
    cdef int[:] parent_seq = parent_seq_np

    _run_mst(&adj[0, 0], n, heap_seq, parent_seq)
    return heap_seq_np, parent_seq_np


# ---------------------------------------------------------------------------
# Public: full VAT pipeline  (MST + permutation gather, OpenMP parallel)
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_vat_c(double[:, ::1] adj):
    """
    C/OpenMP implementation of compute_ordered_dis_njit_merge.

    Steps:
      1. Decrease-key Prim's MST to get permutation p (all nogil, O(n) heap).
      2. Parallel gather: ordered[i,j] = adj[p[i], p[j]]
         Each OpenMP thread owns a stripe of output rows — sequential writes,
         within-row gather of adj[p[i],:] which fits in L1/L2 cache.

    Returns (ordered_matrix, p_seq, q_seq).
    """
    cdef int n = adj.shape[0]
    cdef int i, j, pi

    out_np        = np.empty((n, n), dtype=np.float64)
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)

    cdef double[:, ::1] out = out_np
    cdef int[:]  p           = heap_seq_np
    cdef int[:]  q           = parent_seq_np

    # Step 1: MST (nogil)
    _run_mst(&adj[0, 0], n, p, q)

    # Step 2: Parallel permutation gather
    with nogil:
        for i in prange(n, schedule='static'):
            pi = p[i]
            for j in range(n):
                out[i, j] = adj[pi, p[j]]

    return out_np, heap_seq_np, parent_seq_np
