# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY, INFINITY as INFINITY_F
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, threadid
cimport openmp


# Below this many vertices, OpenMP fork/join overhead dominates the O(n^2)
# work, so the serial path wins. Used to gate the two parallel regions (the
# global-max seed scan and the permutation gather). Tuned empirically; the
# overhead is far worse under MSVC's vcomp runtime than under libgomp.
cdef int _PAR_THRESHOLD = 512


# ---------------------------------------------------------------------------
# Core MST kernel — float64 variant
# ---------------------------------------------------------------------------
cdef void _prim_mst_kernel_64(
    const double* adj, int n,
    int* act_vert, double* act_key, int* act_par,
    int* out_seq, int* out_par_seq,
    double* sd, int* si, int nthreads
) noexcept nogil:
    # Compact-active-set dense Prim. `act_*` are parallel arrays over the
    # currently-unvisited vertices, packed into slots [0, m). Removing a vertex
    # is an O(1) swap-with-last, so round r scans only m = n-r slots — total
    # work is n(n-1)/2 instead of n^2, and the inner loop is branch-free over
    # contiguous memory.
    #
    # Threading note: Prim adds exactly one vertex per round and each round
    # depends on the previous, so the round loop is inherently sequential and
    # runs serial. The ONE part worth threading is the initial global-max scan
    # — a single O(n^2) pass with one fork/join — which is gated on nthreads.
    cdef int i, j, w, u, bk, rnd, m, tid
    cdef int src_i = 0, src_j = 0
    cdef double max_val = -INFINITY
    cdef double d, best
    cdef const double* row

    for i in range(n):
        act_vert[i] = i
        act_key[i]  = INFINITY
        act_par[i]  = -1

    # Find global maximum to seed the source vertex.
    if nthreads > 1:
        # Parallel global-max over row stripes; per-thread (value, flat-index)
        # then a serial combine. Static schedule gives thread t the lowest
        # contiguous block of rows, and every comparison is strict `>`, so ties
        # resolve to the lowest flat index — identical to the serial scan.
        for tid in range(nthreads):
            sd[tid] = -INFINITY
            si[tid] = 0
        for i in prange(n, schedule='static', num_threads=nthreads):
            tid = threadid()
            for j in range(n):
                if adj[i * n + j] > sd[tid]:
                    sd[tid] = adj[i * n + j]
                    si[tid] = i * n + j
        for tid in range(nthreads):
            if sd[tid] > max_val:
                max_val = sd[tid]
                src_i = si[tid] // n
                src_j = si[tid] % n
    else:
        for i in range(n):
            row = adj + i * n
            for j in range(n):
                if row[j] > max_val:
                    max_val = row[j]
                    src_i = i
                    src_j = j

    # Source occupies slot src_i (act_vert[i] == i initially)
    act_key[src_i] = max_val
    act_par[src_i] = src_j
    bk = src_i
    m  = n

    for rnd in range(n):
        u = act_vert[bk]
        out_seq[rnd]     = u
        out_par_seq[rnd] = act_par[bk]

        # Remove slot bk by swapping in the last active slot
        m -= 1
        act_vert[bk] = act_vert[m]
        act_key[bk]  = act_key[m]
        act_par[bk]  = act_par[m]

        # Fused relax of the remaining active set + next-min selection (serial).
        row  = adj + u * n
        best = INFINITY
        bk   = -1
        for i in range(m):
            w = act_vert[i]
            d = row[w]
            if d < act_key[i]:
                act_key[i] = d
                act_par[i] = rnd + 1
            if act_key[i] < best:
                best = act_key[i]
                bk   = i


# ---------------------------------------------------------------------------
# Shared helper: allocate working buffers, run kernel, free buffers — float64
# ---------------------------------------------------------------------------
cdef _run_mst_64(const double* adj_ptr, int n, int[:] heap_seq, int[:] parent_seq):
    cdef int*    act_vert = <int*>   malloc(n * sizeof(int))
    cdef double* act_key  = <double*>malloc(n * sizeof(double))
    cdef int*    act_par  = <int*>   malloc(n * sizeof(int))

    # Thread the one-shot global-max scan only when it pays off.
    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1
    cdef double* sd = NULL
    cdef int* si    = NULL

    if not (act_vert and act_key and act_par):
        free(act_vert); free(act_key); free(act_par)
        raise MemoryError("MST workspace allocation failed")

    if nthreads > 1:
        sd = <double*>malloc(nthreads * sizeof(double))
        si = <int*>   malloc(nthreads * sizeof(int))
        if not (sd and si):
            free(act_vert); free(act_key); free(act_par); free(sd); free(si)
            raise MemoryError("MST workspace allocation failed")

    with nogil:
        _prim_mst_kernel_64(
            adj_ptr, n,
            act_vert, act_key, act_par,
            &heap_seq[0], &parent_seq[0],
            sd, si, nthreads
        )

    free(sd); free(si)
    free(act_vert); free(act_key); free(act_par)


# ---------------------------------------------------------------------------
# Public: Prim's MST only — float64
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def vat_prim_mst_c_64(double[:, ::1] adj):
    """
    Compact dense Prim's MST (O(n) workspace, nogil round loop).
    Returns (heap_seq, parent_seq) as int32 numpy arrays.
    """
    cdef int n = adj.shape[0]
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    cdef int[:] heap_seq   = heap_seq_np
    cdef int[:] parent_seq = parent_seq_np

    _run_mst_64(&adj[0, 0], n, heap_seq, parent_seq)
    return heap_seq_np, parent_seq_np


# ---------------------------------------------------------------------------
# Public: full VAT pipeline — float64 (MST + permutation gather, OpenMP parallel)
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_vat_c_64(double[:, ::1] adj):
    """
    C/OpenMP implementation of compute_ordered_dis_njit_merge (float64).

    Steps:
      1. Compact dense Prim's MST to get permutation p (nogil).
      2. Parallel gather: ordered[i,j] = adj[p[i], p[j]]
         Each OpenMP thread owns a stripe of output rows — sequential writes,
         within-row gather of adj[p[i],:] which fits in L1/L2 cache. A single
         fork/join over O(n^2) work, so it is gated on _PAR_THRESHOLD.

    Returns (ordered_matrix, p_seq, q_seq).
    """
    cdef int n = adj.shape[0]
    cdef int i, c, pi
    cdef Py_ssize_t base, orow
    cdef const double* src
    cdef double* dst

    out_np        = np.empty((n, n), dtype=np.float64)
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    invp_np       = np.empty(n, dtype=np.int32)

    cdef double[:, ::1] out = out_np
    cdef int[:]  p           = heap_seq_np
    cdef int[:]  q           = parent_seq_np
    cdef int[:]  invp        = invp_np

    # Step 1: MST (nogil)
    _run_mst_64(&adj[0, 0], n, p, q)

    # Step 2: Permutation re-order, out[i, j] = adj[p[i], p[j]].
    cdef const double* A = &adj[0, 0]
    cdef double* O       = &out[0, 0]
    cdef const int* P    = &p[0]
    cdef const int* IP   = &invp[0]

    for i in range(n):
        invp[P[i]] = i

    cdef int gthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or gthreads < 2:
        gthreads = 1

    with nogil:
        for i in prange(n, schedule='static', num_threads=gthreads):
            pi   = P[i]
            base = <Py_ssize_t>pi * n
            orow = <Py_ssize_t>i * n
            src  = A + base
            dst  = O + orow
            for c in range(n):
                dst[IP[c]] = src[c]

    return out_np, heap_seq_np, parent_seq_np


# =========================================================================
# FLOAT32 IMPLEMENTATIONS
# =========================================================================

# ---------------------------------------------------------------------------
# Core MST kernel — float32 variant
# ---------------------------------------------------------------------------
cdef void _prim_mst_kernel_32(
    const float* adj, int n,
    int* act_vert, float* act_key, int* act_par,
    int* out_seq, int* out_par_seq,
    float* sd, int* si, int nthreads
) noexcept nogil:
    cdef int i, j, w, u, bk, rnd, m, tid
    cdef int src_i = 0, src_j = 0
    cdef float max_val = -INFINITY
    cdef float d, best
    cdef const float* row

    for i in range(n):
        act_vert[i] = i
        act_key[i]  = INFINITY
        act_par[i]  = -1

    if nthreads > 1:
        for tid in range(nthreads):
            sd[tid] = -INFINITY
            si[tid] = 0
        for i in prange(n, schedule='static', num_threads=nthreads):
            tid = threadid()
            for j in range(n):
                if adj[i * n + j] > sd[tid]:
                    sd[tid] = adj[i * n + j]
                    si[tid] = i * n + j
        for tid in range(nthreads):
            if sd[tid] > max_val:
                max_val = sd[tid]
                src_i = si[tid] // n
                src_j = si[tid] % n
    else:
        for i in range(n):
            row = adj + i * n
            for j in range(n):
                if row[j] > max_val:
                    max_val = row[j]
                    src_i = i
                    src_j = j

    act_key[src_i] = max_val
    act_par[src_i] = src_j
    bk = src_i
    m  = n

    for rnd in range(n):
        u = act_vert[bk]
        out_seq[rnd]     = u
        out_par_seq[rnd] = act_par[bk]

        m -= 1
        act_vert[bk] = act_vert[m]
        act_key[bk]  = act_key[m]
        act_par[bk]  = act_par[m]

        row  = adj + u * n
        best = INFINITY
        bk   = -1
        for i in range(m):
            w = act_vert[i]
            d = row[w]
            if d < act_key[i]:
                act_key[i] = d
                act_par[i] = rnd + 1
            if act_key[i] < best:
                best = act_key[i]
                bk   = i


# ---------------------------------------------------------------------------
# Shared helper: allocate working buffers, run kernel, free buffers — float32
# ---------------------------------------------------------------------------
cdef _run_mst_32(const float* adj_ptr, int n, int[:] heap_seq, int[:] parent_seq):
    cdef int*   act_vert = <int*>  malloc(n * sizeof(int))
    cdef float* act_key  = <float*>malloc(n * sizeof(float))
    cdef int*   act_par  = <int*>  malloc(n * sizeof(int))

    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1
    cdef float* sd = NULL
    cdef int* si   = NULL

    if not (act_vert and act_key and act_par):
        free(act_vert); free(act_key); free(act_par)
        raise MemoryError("MST workspace allocation failed")

    if nthreads > 1:
        sd = <float*>malloc(nthreads * sizeof(float))
        si = <int*>  malloc(nthreads * sizeof(int))
        if not (sd and si):
            free(act_vert); free(act_key); free(act_par); free(sd); free(si)
            raise MemoryError("MST workspace allocation failed")

    with nogil:
        _prim_mst_kernel_32(
            adj_ptr, n,
            act_vert, act_key, act_par,
            &heap_seq[0], &parent_seq[0],
            sd, si, nthreads
        )

    free(sd); free(si)
    free(act_vert); free(act_key); free(act_par)


# ---------------------------------------------------------------------------
# Public: Prim's MST only — float32
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def vat_prim_mst_c_32(float[:, ::1] adj):
    """
    Compact dense Prim's MST (O(n) workspace, nogil round loop) — float32.
    Returns (heap_seq, parent_seq) as int32 numpy arrays.
    """
    cdef int n = adj.shape[0]
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    cdef int[:] heap_seq   = heap_seq_np
    cdef int[:] parent_seq = parent_seq_np

    _run_mst_32(&adj[0, 0], n, heap_seq, parent_seq)
    return heap_seq_np, parent_seq_np


# ---------------------------------------------------------------------------
# Public: full VAT pipeline — float32 (MST + permutation gather, OpenMP parallel)
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_vat_c_32(float[:, ::1] adj):
    """
    C/OpenMP implementation of compute_ordered_dis_njit_merge (float32).

    Steps:
      1. Compact dense Prim's MST to get permutation p (nogil).
      2. Parallel gather: ordered[i,j] = adj[p[i], p[j]]
         Each OpenMP thread owns a stripe of output rows — sequential writes,
         within-row gather of adj[p[i],:] which fits in L1/L2 cache. A single
         fork/join over O(n^2) work, so it is gated on _PAR_THRESHOLD.

    Returns (ordered_matrix, p_seq, q_seq).
    """
    cdef int n = adj.shape[0]
    cdef int i, c, pi
    cdef Py_ssize_t base, orow
    cdef const float* src
    cdef float* dst

    out_np        = np.empty((n, n), dtype=np.float32)
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    invp_np       = np.empty(n, dtype=np.int32)

    cdef float[:, ::1] out = out_np
    cdef int[:]  p          = heap_seq_np
    cdef int[:]  q          = parent_seq_np
    cdef int[:]  invp       = invp_np

    # Step 1: MST (nogil)
    _run_mst_32(&adj[0, 0], n, p, q)

    # Step 2: Permutation re-order, out[i, j] = adj[p[i], p[j]].
    cdef const float* A = &adj[0, 0]
    cdef float* O       = &out[0, 0]
    cdef const int* P   = &p[0]
    cdef const int* IP  = &invp[0]

    for i in range(n):
        invp[P[i]] = i

    cdef int gthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or gthreads < 2:
        gthreads = 1

    with nogil:
        for i in prange(n, schedule='static', num_threads=gthreads):
            pi   = P[i]
            base = <Py_ssize_t>pi * n
            orow = <Py_ssize_t>i * n
            src  = A + base
            dst  = O + orow
            for c in range(n):
                dst[IP[c]] = src[c]

    return out_np, heap_seq_np, parent_seq_np


# ---------------------------------------------------------------------------
# Public dispatch: automatic dtype selection
# ---------------------------------------------------------------------------
def vat_prim_mst_c(adj):
    """
    Compact dense Prim's MST.
    Automatically dispatches to float32 or float64 based on input dtype.
    Returns (heap_seq, parent_seq) as int32 numpy arrays.
    """
    if adj.dtype == np.float32:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        return vat_prim_mst_c_32(adj_c)
    elif adj.dtype == np.float64:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        return vat_prim_mst_c_64(adj_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {adj.dtype}")


def compute_vat_c(adj):
    """
    C/OpenMP implementation of compute_ordered_dis_njit_merge.
    Automatically dispatches to float32 or float64 based on input dtype.
    Returns (ordered_matrix, p_seq, q_seq).
    """
    if adj.dtype == np.float32:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        return compute_vat_c_32(adj_c)
    elif adj.dtype == np.float64:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        return compute_vat_c_64(adj_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {adj.dtype}")
