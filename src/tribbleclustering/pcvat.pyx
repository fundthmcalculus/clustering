# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, threadid
cimport openmp


# Below this many vertices, threading overhead dominates the O(n^2) work,
# so the serial kernel wins. Tuned empirically.
cdef int _PAR_THRESHOLD = 512


# ---------------------------------------------------------------------------
# Core MST kernel — dense array-based Prim (no heap).
#
# For a complete graph (a distance matrix is complete), a binary-heap Prim is
# O(n^2 log n) — every one of the O(n^2) edges can trigger an O(log n) heap
# operation. The dense array variant is O(n^2): each of the n rounds does a
# single fused linear pass that BOTH relaxes neighbor keys AND tracks the
# minimum-key unvisited vertex for the next round. One sequential pass over
# contiguous memory per round — SIMD-friendly, no pointer chasing, no log factor.
#
# No allocation, no GIL, no Python calls. Caller supplies all buffers.
# ---------------------------------------------------------------------------
cdef void _prim_mst_kernel(
    const double* adj, int n,
    int* act_vert, double* act_key, int* act_par,
    int* out_seq, int* out_par_seq
) noexcept nogil:
    # Compact-active-set dense Prim. `act_*` are parallel arrays over the
    # currently-unvisited vertices, packed into slots [0, m). Removing a vertex
    # is an O(1) swap-with-last, so round r scans only m = n-r slots — total
    # work is n(n-1)/2 instead of n^2, and the inner loop is branch-free over
    # contiguous memory. The per-vertex edge weight row[w] is gathered from a
    # single row of adj, which fits in L1 for the sizes of interest.
    cdef int i, j, w, u, bk, rnd, m
    cdef int src_i = 0, src_j = 0
    cdef double max_val = -INFINITY
    cdef double d, best
    cdef const double* row

    for i in range(n):
        act_vert[i] = i
        act_key[i]  = INFINITY
        act_par[i]  = -1

    # Find global maximum to seed the source vertex
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

        # Fused relax of the remaining active set + next-min selection
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
# Parallel compact dense Prim. Identical algorithm and active-set compaction
# as _prim_mst_kernel, but the two O(n^2) phases are threaded:
#   * global-max search: each thread reduces a static row stripe;
#   * each round's fused relax + next-min: each thread owns a static slot
#     stripe of the active set, writing only its own (best, idx) scratch slot.
# The O(1) swap-removal and the tiny per-thread reduction stay serial between
# rounds. Once the active set shrinks below _INNER_SERIAL, the round is cheaper
# to run serially than to fork — so we fall through to a plain loop.
# ---------------------------------------------------------------------------
cdef int _INNER_SERIAL = 1024

cdef void _prim_mst_kernel_par(
    const double* adj, int n,
    int* act_vert, double* act_key, int* act_par,
    int* out_seq, int* out_par_seq,
    double* sd, int* si, int nthreads
) noexcept nogil:
    cdef int i, j, w, u, bk, rnd, m, tid
    cdef int src_i = 0, src_j = 0
    cdef double max_val = -INFINITY
    cdef double d, best
    cdef const double* row

    for i in range(n):
        act_vert[i] = i
        act_key[i]  = INFINITY
        act_par[i]  = -1

    # Parallel global-max search over row stripes
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

        if m >= _INNER_SERIAL:
            # Threaded fused relax + per-thread argmin
            for tid in range(nthreads):
                sd[tid] = INFINITY
                si[tid] = -1
            for i in prange(m, schedule='static', num_threads=nthreads):
                tid = threadid()
                w = act_vert[i]
                d = row[w]
                if d < act_key[i]:
                    act_key[i] = d
                    act_par[i] = rnd + 1
                if act_key[i] < sd[tid]:
                    sd[tid] = act_key[i]
                    si[tid] = i
            for tid in range(nthreads):
                if sd[tid] < best:
                    best = sd[tid]
                    bk   = si[tid]
        else:
            # Cheap round — serial pass avoids fork/join overhead
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
# Shared helper: allocate working buffers, run kernel, free buffers.
# Raises MemoryError on failure; runs kernel with nogil.
# Dispatches to the parallel kernel above _PAR_THRESHOLD, else serial.
# ---------------------------------------------------------------------------
cdef _run_mst(const double* adj_ptr, int n, int[:] heap_seq, int[:] parent_seq):
    cdef int*    act_vert = <int*>   malloc(n * sizeof(int))
    cdef double* act_key  = <double*>malloc(n * sizeof(double))
    cdef int*    act_par  = <int*>   malloc(n * sizeof(int))

    cdef int nthreads = openmp.omp_get_max_threads()
    cdef double* sd = NULL
    cdef int* si    = NULL

    if not (act_vert and act_key and act_par):
        free(act_vert); free(act_key); free(act_par)
        raise MemoryError("MST workspace allocation failed")

    if n >= _PAR_THRESHOLD and nthreads > 1:
        sd = <double*>malloc(nthreads * sizeof(double))
        si = <int*>   malloc(nthreads * sizeof(int))
        if not (sd and si):
            free(act_vert); free(act_key); free(act_par); free(sd); free(si)
            raise MemoryError("MST workspace allocation failed")
        with nogil:
            _prim_mst_kernel_par(
                adj_ptr, n,
                act_vert, act_key, act_par,
                &heap_seq[0], &parent_seq[0],
                sd, si, nthreads
            )
        free(sd); free(si)
    else:
        with nogil:
            _prim_mst_kernel(
                adj_ptr, n,
                act_vert, act_key, act_par,
                &heap_seq[0], &parent_seq[0]
            )

    free(act_vert); free(act_key); free(act_par)


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
    _run_mst(&adj[0, 0], n, p, q)

    # Step 2: Permutation re-order, out[i, j] = adj[p[i], p[j]].
    #
    # A permutation reads each source element exactly once — there is no
    # temporal reuse, so throughput is set by cache-line utilisation. Reading
    # the source row in permuted column order (adj[p[i], p[j]]) touches a row's
    # cache lines in random order: poor prefetch, partial line use.
    #
    # Instead read the source row sequentially and SCATTER through the inverse
    # permutation: out[i, invp[c]] = adj[p[i], c]. The source read is now fully
    # sequential (hardware prefetch, every line fully used); the random write
    # lands inside the current output row, which fits in L1 for the sizes of
    # interest. Each thread owns a contiguous block of output rows.
    cdef const double* A = &adj[0, 0]
    cdef double* O       = &out[0, 0]
    cdef const int* P    = &p[0]
    cdef const int* IP   = &invp[0]

    for i in range(n):
        invp[P[i]] = i

    with nogil:
        for i in prange(n, schedule='static'):
            pi   = P[i]
            base = <Py_ssize_t>pi * n
            orow = <Py_ssize_t>i * n
            src  = A + base
            dst  = O + orow
            for c in range(n):
                dst[IP[c]] = src[c]

    return out_np, heap_seq_np, parent_seq_np
