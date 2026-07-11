# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY, INFINITY as INFINITY_F, sqrt
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
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
    double* sd, int* sbi, int* sbj, int nthreads
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
        # Parallel global-max over row stripes; each thread keeps its best
        # (value, row, col), then a serial combine. Static schedule gives
        # thread t the lowest contiguous block of rows, and every comparison is
        # strict `>`, so ties resolve to the lowest (i, j) — identical to the
        # serial scan. Row/col are stored separately (each < n) rather than a
        # packed i*n+j, which overflows int32 once n >~ 46340. The matrix offset
        # itself is computed in Py_ssize_t for the same reason.
        for tid in range(nthreads):
            sd[tid]  = -INFINITY
            sbi[tid] = 0
            sbj[tid] = 0
        for i in prange(n, schedule='static', num_threads=nthreads):
            tid = threadid()
            for j in range(n):
                if adj[<Py_ssize_t>i * n + j] > sd[tid]:
                    sd[tid]  = adj[<Py_ssize_t>i * n + j]
                    sbi[tid] = i
                    sbj[tid] = j
        for tid in range(nthreads):
            if sd[tid] > max_val:
                max_val = sd[tid]
                src_i = sbi[tid]
                src_j = sbj[tid]
    else:
        for i in range(n):
            row = adj + <Py_ssize_t>i * n
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
        row  = adj + <Py_ssize_t>u * n
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
    cdef double* sd  = NULL
    cdef int*    sbi = NULL
    cdef int*    sbj = NULL

    if not (act_vert and act_key and act_par):
        free(act_vert); free(act_key); free(act_par)
        raise MemoryError("MST workspace allocation failed")

    if nthreads > 1:
        sd  = <double*>malloc(nthreads * sizeof(double))
        sbi = <int*>   malloc(nthreads * sizeof(int))
        sbj = <int*>   malloc(nthreads * sizeof(int))
        if not (sd and sbi and sbj):
            free(act_vert); free(act_key); free(act_par)
            free(sd); free(sbi); free(sbj)
            raise MemoryError("MST workspace allocation failed")

    with nogil:
        _prim_mst_kernel_64(
            adj_ptr, n,
            act_vert, act_key, act_par,
            &heap_seq[0], &parent_seq[0],
            sd, sbi, sbj, nthreads
        )

    free(sd); free(sbi); free(sbj)
    free(act_vert); free(act_key); free(act_par)


# ---------------------------------------------------------------------------
# Shared back-copy helpers — mirror lower triangle to upper triangle.
#
# Extracted into their own cdef functions so that the prange inside has no
# shared state with the fill-prange in the caller. MSVC's vcomp OpenMP runtime
# can conflate private variables across two consecutive prange regions in the
# same function; a separate function guarantees a clean variable scope.
# ---------------------------------------------------------------------------
cdef void _backcopy_lower_to_upper_64(double* M, int n, int nthreads) noexcept nogil:
    cdef int i, j
    for i in prange(1, n, schedule='static', num_threads=nthreads):
        for j in range(i):
            M[<Py_ssize_t>j * n + i] = M[<Py_ssize_t>i * n + j]

cdef void _backcopy_lower_to_upper_32(float* M, int n, int nthreads) noexcept nogil:
    cdef int i, j
    for i in prange(1, n, schedule='static', num_threads=nthreads):
        for j in range(i):
            M[<Py_ssize_t>j * n + i] = M[<Py_ssize_t>i * n + j]


# ---------------------------------------------------------------------------
# In-place symmetric permutation: M[i, j] <- M[p[i], p[j]] for all i, j.
#
# This is the memory-frugal alternative to the out-of-place gather. It applies
# the permutation as P·M·Pᵀ in two independent 1-D passes:
#
#   1. Permute rows   so row i becomes old row p[i]  (M1[i,j] = M[p[i], j]).
#   2. Permute columns so col j becomes old col p[j] (M2[i,j] = M1[i, p[j]]
#      = M[p[i], p[j]]).
#
# Each pass is a standard cycle-following in-place permutation of a 1-D index
# (rows, then columns-within-each-row), which is provably correct: the walk
# reads only the *next* (not-yet-visited) element of a cycle before overwriting
# the current one. Workspace is O(n) — one length-n scratch row plus a length-n
# visited flag — far less than the n^2 output buffer it replaces.
#
# NB: the earlier cycle-following-on-cell-pairs scheme with symmetric
# mirror-writes (pvat.shuffle_ordered_column) is NOT correct — a mirror-written
# cell can be read as another cycle's "next" after it has already been given
# its final value. This row-then-column formulation avoids that entirely by
# never coupling two cells in a single write.
#
# Serial by nature (cycles have data-dependent length); the trade is speed for
# memory, opted into via inplace=True when the input is destroyable.
# ---------------------------------------------------------------------------
cdef void _permute_sym_inplace_64(
    double* M, int n, const int* p, double* tmp, unsigned char* seen
) noexcept nogil:
    cdef int start, i, j, nxt
    cdef double* row

    # Phase 1: permute rows so new row i = old row p[i].
    for i in range(n):
        seen[i] = 0
    for start in range(n):
        if seen[start] or p[start] == start:
            seen[start] = 1
            continue
        memcpy(tmp, M + <Py_ssize_t>start * n, <size_t>n * sizeof(double))
        i = start
        while True:
            seen[i] = 1
            nxt = p[i]
            if nxt == start:
                memcpy(M + <Py_ssize_t>i * n, tmp, <size_t>n * sizeof(double))
                break
            memcpy(M + <Py_ssize_t>i * n, M + <Py_ssize_t>nxt * n,
                   <size_t>n * sizeof(double))
            i = nxt

    # Phase 2: permute columns within each row: new[i,j] = cur[i, p[j]].
    for i in range(n):
        row = M + <Py_ssize_t>i * n
        for j in range(n):
            tmp[j] = row[j]
        for j in range(n):
            row[j] = tmp[p[j]]


cdef void _permute_sym_inplace_32(
    float* M, int n, const int* p, float* tmp, unsigned char* seen
) noexcept nogil:
    cdef int start, i, j, nxt
    cdef float* row

    for i in range(n):
        seen[i] = 0
    for start in range(n):
        if seen[start] or p[start] == start:
            seen[start] = 1
            continue
        memcpy(tmp, M + <Py_ssize_t>start * n, <size_t>n * sizeof(float))
        i = start
        while True:
            seen[i] = 1
            nxt = p[i]
            if nxt == start:
                memcpy(M + <Py_ssize_t>i * n, tmp, <size_t>n * sizeof(float))
                break
            memcpy(M + <Py_ssize_t>i * n, M + <Py_ssize_t>nxt * n,
                   <size_t>n * sizeof(float))
            i = nxt

    for i in range(n):
        row = M + <Py_ssize_t>i * n
        for j in range(n):
            tmp[j] = row[j]
        for j in range(n):
            row[j] = tmp[p[j]]


# Run MST then permute the matrix in place. Returns (p_seq, q_seq). The caller
# owns `M` (already C-contiguous, correct dtype) and it IS the VAT result.
cdef _vat_inplace_64(double[:, ::1] M):
    cdef int n = M.shape[0]
    heap_seq_np = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    cdef int[:] p = heap_seq_np
    cdef int[:] q = parent_seq_np
    _run_mst_64(&M[0, 0], n, p, q)

    cdef double* tmp = <double*>malloc(<size_t>n * sizeof(double))
    cdef unsigned char* seen = <unsigned char*>malloc(<size_t>n)
    if not (tmp and seen):
        free(tmp); free(seen)
        raise MemoryError("permutation workspace allocation failed")
    with nogil:
        _permute_sym_inplace_64(&M[0, 0], n, &p[0], tmp, seen)
    free(tmp); free(seen)
    return heap_seq_np, parent_seq_np


cdef _vat_inplace_32(float[:, ::1] M):
    cdef int n = M.shape[0]
    heap_seq_np = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)
    cdef int[:] p = heap_seq_np
    cdef int[:] q = parent_seq_np
    _run_mst_32(&M[0, 0], n, p, q)

    cdef float* tmp = <float*>malloc(<size_t>n * sizeof(float))
    cdef unsigned char* seen = <unsigned char*>malloc(<size_t>n)
    if not (tmp and seen):
        free(tmp); free(seen)
        raise MemoryError("permutation workspace allocation failed")
    with nogil:
        _permute_sym_inplace_32(&M[0, 0], n, &p[0], tmp, seen)
    free(tmp); free(seen)
    return heap_seq_np, parent_seq_np


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
      2. Parallel lower-triangle gather: out[i,j] = adj[p[i], p[j]] for j<=i.
         No inverse-permutation array needed. A second parallel pass mirrors
         lower → upper. Two fork/joins, each O(n^2/2), gated on _PAR_THRESHOLD.

    Returns (ordered_matrix, p_seq, q_seq).
    """
    cdef int n = adj.shape[0]
    cdef int i, j
    cdef Py_ssize_t pi_row, i_row

    out_np        = np.empty((n, n), dtype=np.float64)
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)

    cdef double[:, ::1] out = out_np
    cdef int[:]  p           = heap_seq_np
    cdef int[:]  q           = parent_seq_np

    # Step 1: MST (nogil)
    _run_mst_64(&adj[0, 0], n, p, q)

    # Step 2: Lower-triangle gather + back-copy. No invp needed.
    cdef const double* A = &adj[0, 0]
    cdef double* O       = &out[0, 0]
    cdef const int* P    = &p[0]

    cdef int gthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or gthreads < 2:
        gthreads = 1

    with nogil:
        # Fill lower triangle (including diagonal): out[i,j] = adj[P[i],P[j]], j<=i.
        # Thread i reads row P[i] of adj (scattered by P[j]) and writes row i of
        # out sequentially — each thread owns an exclusive output row.
        for i in prange(n, schedule='static', num_threads=gthreads):
            pi_row = <Py_ssize_t>P[i] * n
            i_row  = <Py_ssize_t>i * n
            for j in range(i + 1):
                O[i_row + j] = A[pi_row + P[j]]
        # Back-copy in a dedicated helper (clean variable scope, avoids MSVC
        # vcomp variable aliasing across two consecutive prange regions).
        _backcopy_lower_to_upper_64(O, n, gthreads)

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
    float* sd, int* sbi, int* sbj, int nthreads
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
        # Per-thread best (value, row, col); row/col stored separately to avoid
        # the int32 overflow of a packed i*n+j, and the matrix offset is taken
        # in Py_ssize_t. See the float64 kernel for the full rationale.
        for tid in range(nthreads):
            sd[tid]  = -INFINITY
            sbi[tid] = 0
            sbj[tid] = 0
        for i in prange(n, schedule='static', num_threads=nthreads):
            tid = threadid()
            for j in range(n):
                if adj[<Py_ssize_t>i * n + j] > sd[tid]:
                    sd[tid]  = adj[<Py_ssize_t>i * n + j]
                    sbi[tid] = i
                    sbj[tid] = j
        for tid in range(nthreads):
            if sd[tid] > max_val:
                max_val = sd[tid]
                src_i = sbi[tid]
                src_j = sbj[tid]
    else:
        for i in range(n):
            row = adj + <Py_ssize_t>i * n
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

        row  = adj + <Py_ssize_t>u * n
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
    cdef float* sd  = NULL
    cdef int*   sbi = NULL
    cdef int*   sbj = NULL

    if not (act_vert and act_key and act_par):
        free(act_vert); free(act_key); free(act_par)
        raise MemoryError("MST workspace allocation failed")

    if nthreads > 1:
        sd  = <float*>malloc(nthreads * sizeof(float))
        sbi = <int*>  malloc(nthreads * sizeof(int))
        sbj = <int*>  malloc(nthreads * sizeof(int))
        if not (sd and sbi and sbj):
            free(act_vert); free(act_key); free(act_par)
            free(sd); free(sbi); free(sbj)
            raise MemoryError("MST workspace allocation failed")

    with nogil:
        _prim_mst_kernel_32(
            adj_ptr, n,
            act_vert, act_key, act_par,
            &heap_seq[0], &parent_seq[0],
            sd, sbi, sbj, nthreads
        )

    free(sd); free(sbi); free(sbj)
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
    See compute_vat_c_64 for the full rationale.

    Returns (ordered_matrix, p_seq, q_seq).
    """
    cdef int n = adj.shape[0]
    cdef int i, j
    cdef Py_ssize_t pi_row, i_row

    out_np        = np.empty((n, n), dtype=np.float32)
    heap_seq_np   = np.empty(n, dtype=np.int32)
    parent_seq_np = np.empty(n, dtype=np.int32)

    cdef float[:, ::1] out = out_np
    cdef int[:]  p          = heap_seq_np
    cdef int[:]  q          = parent_seq_np

    # Step 1: MST (nogil)
    _run_mst_32(&adj[0, 0], n, p, q)

    # Step 2: Lower-triangle gather + back-copy. No invp needed.
    cdef const float* A = &adj[0, 0]
    cdef float* O       = &out[0, 0]
    cdef const int* P   = &p[0]

    cdef int gthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or gthreads < 2:
        gthreads = 1

    with nogil:
        for i in prange(n, schedule='static', num_threads=gthreads):
            pi_row = <Py_ssize_t>P[i] * n
            i_row  = <Py_ssize_t>i * n
            for j in range(i + 1):
                O[i_row + j] = A[pi_row + P[j]]
        _backcopy_lower_to_upper_32(O, n, gthreads)

    return out_np, heap_seq_np, parent_seq_np


# =========================================================================
# PAIRWISE DISTANCES (OpenMP)
# =========================================================================
#
# Full dense Euclidean distance matrix: out[i, j] = ||data[i] - data[j]||_2,
# the C/OpenMP equivalent of util.pairwise_distances. The work is the upper
# triangle (i < j), mirrored into the lower triangle; the diagonal stays 0.
#
# Threading: prange over the outer row index i. Row i does (n - i - 1) inner
# pairs, so the per-row cost shrinks as i grows — a 'guided' schedule keeps
# threads balanced over that triangular profile. Write safety: the thread
# owning row i is the ONLY writer of both out[i, j] and its mirror out[j, i]
# for every j > i, so no two threads ever touch the same cell. Gated on
# _PAR_THRESHOLD, like the other parallel regions.


# ---------------------------------------------------------------------------
# Core pairwise-distance kernel — float64 variant
# ---------------------------------------------------------------------------
cdef void _pairwise_distances_kernel_64(
    const double* data, int n, int d, double* out, int nthreads
) noexcept nogil:
    cdef int i, j, k
    cdef double diff, acc
    cdef const double* ri
    cdef const double* rj
    for i in prange(n, schedule='guided', num_threads=nthreads):
        ri = data + <Py_ssize_t>i * d
        for j in range(i + 1, n):
            rj = data + <Py_ssize_t>j * d
            acc = 0.0
            for k in range(d):
                diff = ri[k] - rj[k]
                acc = acc + diff * diff
            acc = sqrt(acc)
            out[<Py_ssize_t>i * n + j] = acc
            out[<Py_ssize_t>j * n + i] = acc


# ---------------------------------------------------------------------------
# Public: pairwise distances — float64
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_distances_c_64(double[:, ::1] data):
    """
    Dense Euclidean pairwise-distance matrix (float64), OpenMP parallel.

    `data` is (n_samples, n_features), C-contiguous. Returns an
    (n_samples, n_samples) float64 matrix with a zero diagonal.
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    out_np = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return out_np
    cdef double[:, ::1] out = out_np

    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _pairwise_distances_kernel_64(&data[0, 0], n, d, &out[0, 0], nthreads)
    return out_np


# ---------------------------------------------------------------------------
# Core pairwise-distance kernel — float32 variant
# ---------------------------------------------------------------------------
cdef void _pairwise_distances_kernel_32(
    const float* data, int n, int d, float* out, int nthreads
) noexcept nogil:
    # Accumulate in double to keep the squared-sum well-conditioned, then store
    # the float32 result — matches numpy's higher-precision reduction.
    cdef int i, j, k
    cdef double diff, acc
    cdef const float* ri
    cdef const float* rj
    for i in prange(n, schedule='guided', num_threads=nthreads):
        ri = data + <Py_ssize_t>i * d
        for j in range(i + 1, n):
            rj = data + <Py_ssize_t>j * d
            acc = 0.0
            for k in range(d):
                diff = <double>ri[k] - <double>rj[k]
                acc = acc + diff * diff
            out[<Py_ssize_t>i * n + j] = <float>sqrt(acc)
            out[<Py_ssize_t>j * n + i] = <float>sqrt(acc)


# ---------------------------------------------------------------------------
# Public: pairwise distances — float32
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_distances_c_32(float[:, ::1] data):
    """
    Dense Euclidean pairwise-distance matrix (float32), OpenMP parallel.

    `data` is (n_samples, n_features), C-contiguous. Returns an
    (n_samples, n_samples) float32 matrix with a zero diagonal.
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    out_np = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return out_np
    cdef float[:, ::1] out = out_np

    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _pairwise_distances_kernel_32(&data[0, 0], n, d, &out[0, 0], nthreads)
    return out_np


def pairwise_distances_c(data):
    """
    Dense Euclidean pairwise-distance matrix.
    Automatically dispatches to float32 or float64 based on input dtype.
    Returns an (n, n) distance matrix of the same dtype.
    """
    if data.dtype == np.float32:
        data_c = np.require(data, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        return pairwise_distances_c_32(data_c)
    elif data.dtype == np.float64:
        data_c = np.require(data, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        return pairwise_distances_c_64(data_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {data.dtype}")


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


def compute_vat_c(adj, inplace=False):
    """
    C/OpenMP implementation of compute_ordered_dis_njit_merge.
    Automatically dispatches to float32 or float64 based on input dtype.
    Returns (ordered_matrix, p_seq, q_seq).

    If ``inplace=True`` the input matrix is reordered in place (via a
    serial, memory-frugal cycle-following permutation with an ~n^2/8-byte
    bitmask) instead of into a fresh n x n buffer, and the returned VAT matrix
    IS the input array. This roughly halves peak memory at the cost of a
    serial permutation pass. The input must be destroyable; if it is not
    already C-contiguous and of the dispatched dtype, a conforming copy is made
    and only that copy is modified.
    """
    if adj.dtype == np.float32:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        if inplace:
            p_np, q_np = _vat_inplace_32(adj_c)
            return adj_c, p_np, q_np
        return compute_vat_c_32(adj_c)
    elif adj.dtype == np.float64:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        if inplace:
            p_np, q_np = _vat_inplace_64(adj_c)
            return adj_c, p_np, q_np
        return compute_vat_c_64(adj_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {adj.dtype}")


# =========================================================================
# IVAT COMPUTATION (builds on VAT)
# =========================================================================
#
# compute_ivat_c: Takes a distance matrix, computes VAT, then builds IVAT.
# The IVAT algorithm refines the VAT matrix by computing the "reachability"
# distance — the maximum of the minimum distances along a path in the MST.


cdef void _compute_ivat_kernel_64(
    double* M, int n, int* argmin_seq, int nthreads
) noexcept nogil:
    """
    Build the IVAT matrix from the VAT matrix **in place** (float64).

    `M` enters as the VAT matrix and leaves as the IVAT matrix; no separate
    output buffer is allocated. This is safe because of the data dependencies
    of the construction:

      * Row r is finalised only after rows 1..r-1, in increasing order.
      * The per-row minimum scan reads the *entire* VAT row r (columns [0, r))
        BEFORE any write to row r, so those VAT values are fully consumed into
        (min_val, best_jj) before being overwritten.
      * The minimax fill for row r then reads only ivat[best_jj, c] and
        ivat[c, best_jj] — both in rows < r, already finalised — and writes
        only into row r. No read in row r's fill touches row r, so there is no
        read-after-write hazard against the VAT values still needed elsewhere.

    Only the lower triangle (row r, columns 0..r-1) is written during the
    O(n^2) construction; a final parallel back-copy mirrors lower → upper,
    overwriting the stale VAT values left in the upper triangle. The diagonal
    is left untouched (VAT's zero diagonal is already correct for IVAT).
    """
    cdef int r, c, best_jj
    cdef double min_val, cur_val, max_val
    cdef double* row

    for r in range(1, n):
        row = M + <Py_ssize_t>r * n

        # Find minimum distance in columns [0, r). Consumes the whole VAT row r
        # before the write loop below overwrites it.
        min_val = row[0]
        best_jj = 0
        for c in range(1, r):
            if row[c] < min_val:
                min_val = row[c]
                best_jj = c

        argmin_seq[r - 1] = best_jj

        # Overwrite row r's lower triangle with IVAT values. For c == best_jj
        # the value is the edge weight min_val; otherwise it is the minimax
        # ivat[r, c] = max(min_val, ivat[best_jj, c]), reading the (finalised)
        # lower-triangle canonical cell of rows < r.
        for c in range(r):
            if c == best_jj:
                row[c] = min_val
            else:
                if best_jj > c:
                    cur_val = M[<Py_ssize_t>best_jj * n + c]
                else:
                    cur_val = M[<Py_ssize_t>c * n + best_jj]
                max_val = min_val if min_val > cur_val else cur_val
                row[c] = max_val

    # Back-copy via dedicated helper (clean variable scope, avoids MSVC vcomp
    # variable aliasing across two consecutive prange regions).
    _backcopy_lower_to_upper_64(M, n, nthreads)


cdef void _compute_ivat_kernel_32(
    float* M, int n, int* argmin_seq, int nthreads
) noexcept nogil:
    """Build IVAT from VAT **in place** (float32). See _compute_ivat_kernel_64 for the full rationale."""
    cdef int r, c, best_jj
    cdef float min_val, cur_val, max_val
    cdef float* row

    for r in range(1, n):
        row = M + <Py_ssize_t>r * n

        # Consume the whole VAT row r before overwriting it.
        min_val = row[0]
        best_jj = 0
        for c in range(1, r):
            if row[c] < min_val:
                min_val = row[c]
                best_jj = c

        argmin_seq[r - 1] = best_jj

        # Overwrite row r's lower triangle with IVAT values (reads rows < r).
        for c in range(r):
            if c == best_jj:
                row[c] = min_val
            else:
                if best_jj > c:
                    cur_val = M[<Py_ssize_t>best_jj * n + c]
                else:
                    cur_val = M[<Py_ssize_t>c * n + best_jj]
                max_val = min_val if min_val > cur_val else cur_val
                row[c] = max_val

    _backcopy_lower_to_upper_32(M, n, nthreads)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ivat_c_64(double[:, ::1] adj):
    """
    Compute IVAT (improved VAT) for float64 distance matrix.

    The VAT matrix is computed first (modified in-place from MST), then IVAT is
    built from VAT using a sequential O(n^2) kernel followed by a parallelized
    back-copy to mirror the lower triangle to the upper triangle.

    Returns (ivat_matrix, argmin_seq, p_seq) where:
      - ivat_matrix: improved VAT matrix (n x n)
      - argmin_seq: sequence of minimum indices from IVAT construction
      - p_seq: permutation sequence from VAT
    """
    cdef int n = adj.shape[0]

    # Compute VAT, then transform it into IVAT in place (see
    # _compute_ivat_kernel_64). The VAT buffer IS the returned IVAT buffer, so
    # the IVAT path holds two n x n matrices (caller's `adj` + this one) rather
    # than three — the single biggest lever on the maximum feasible n.
    vat_np, p_seq_np, q_seq_np = compute_vat_c_64(adj)
    argmin_seq_np = np.zeros(n - 1, dtype=np.int32)

    cdef double[:, ::1] vat = vat_np
    cdef int[:] argmin_seq = argmin_seq_np
    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _compute_ivat_kernel_64(&vat[0, 0], n, &argmin_seq[0], nthreads)

    return vat_np, argmin_seq_np, p_seq_np


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ivat_c_32(float[:, ::1] adj):
    """
    Compute IVAT (improved VAT) for float32 distance matrix.

    The VAT matrix is computed first (modified in-place from MST), then IVAT is
    built from VAT using a sequential O(n^2) kernel followed by a parallelized
    back-copy to mirror the lower triangle to the upper triangle.

    Returns (ivat_matrix, argmin_seq, p_seq) where:
      - ivat_matrix: improved VAT matrix (n x n)
      - argmin_seq: sequence of minimum indices from IVAT construction
      - p_seq: permutation sequence from VAT
    """
    cdef int n = adj.shape[0]

    # See compute_ivat_c_64: VAT is transformed into IVAT in place, so the
    # returned IVAT buffer is the VAT buffer (two n x n matrices, not three).
    vat_np, p_seq_np, q_seq_np = compute_vat_c_32(adj)
    argmin_seq_np = np.zeros(n - 1, dtype=np.int32)

    cdef float[:, ::1] vat = vat_np
    cdef int[:] argmin_seq = argmin_seq_np
    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _compute_ivat_kernel_32(&vat[0, 0], n, &argmin_seq[0], nthreads)

    return vat_np, argmin_seq_np, p_seq_np


cdef _ivat_inplace_impl_64(adj_c):
    cdef int n = adj_c.shape[0]
    cdef double[:, ::1] M = adj_c
    # Permute in place -> M is now VAT; only an ~n^2/8-byte bitmask is allocated.
    p_np, q_np = _vat_inplace_64(M)
    argmin_np = np.zeros(n - 1, dtype=np.int32)
    cdef int[:] argmin = argmin_np
    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1
    # Transform VAT -> IVAT in place (PR: in-place IVAT kernel).
    with nogil:
        _compute_ivat_kernel_64(&M[0, 0], n, &argmin[0], nthreads)
    return adj_c, argmin_np, p_np


cdef _ivat_inplace_impl_32(adj_c):
    cdef int n = adj_c.shape[0]
    cdef float[:, ::1] M = adj_c
    p_np, q_np = _vat_inplace_32(M)
    argmin_np = np.zeros(n - 1, dtype=np.int32)
    cdef int[:] argmin = argmin_np
    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1
    with nogil:
        _compute_ivat_kernel_32(&M[0, 0], n, &argmin[0], nthreads)
    return adj_c, argmin_np, p_np


def compute_ivat_c(adj, inplace=False):
    """
    Compute IVAT (improved VAT) for a distance matrix.
    Automatically dispatches to float32 or float64 based on input dtype.

    The IVAT construction always runs in place over its VAT work buffer (the
    VAT matrix is transformed into IVAT without a third allocation).

    ``inplace`` controls whether the *input* matrix is consumed:
      - inplace=False (default): the input is not modified. The VAT/IVAT work
        buffer is a fresh n x n array, so peak memory is ~2 matrices (input +
        buffer).
      - inplace=True: the input is reordered in place and transformed into the
        returned IVAT matrix. Peak memory is ~1 matrix plus an ~n^2/8-byte
        permutation bitmask, at the cost of a serial in-place permutation pass.
        The input must be destroyable (e.g. a throwaway distance matrix); if it
        is not already C-contiguous and of the dispatched dtype, a conforming
        copy is made and only that copy is consumed.

    Returns (ivat_matrix, argmin_seq, p_seq) where:
      - ivat_matrix: improved VAT matrix (n x n, symmetric)
      - argmin_seq: sequence of minimum indices from IVAT construction (n-1,)
      - p_seq: permutation sequence from VAT (n,)
    """
    if adj.dtype == np.float32:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        if inplace:
            return _ivat_inplace_impl_32(adj_c)
        return compute_ivat_c_32(adj_c)
    elif adj.dtype == np.float64:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        if inplace:
            return _ivat_inplace_impl_64(adj_c)
        return compute_ivat_c_64(adj_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {adj.dtype}")
