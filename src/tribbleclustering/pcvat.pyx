# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY, INFINITY as INFINITY_F, sqrt
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


# =========================================================================
# IVAT COMPUTATION (builds on VAT)
# =========================================================================
#
# compute_ivat_c: Takes a distance matrix, computes VAT, then builds IVAT.
# The IVAT algorithm refines the VAT matrix by computing the "reachability"
# distance — the maximum of the minimum distances along a path in the MST.


cdef void _compute_ivat_kernel_64(
    const double* vat_matrix, int n, double* ivat_matrix,
    int* argmin_seq, int nthreads
) noexcept nogil:
    """
    Build IVAT matrix from VAT matrix (float64).

    Exploits symmetry: the main construction loop only writes to the lower
    triangle (row r, columns 0..r-1). Reads of previously-computed ivat values
    are canonicalised to the lower triangle (swap indices when best_jj < c).
    After the sequential loop finishes, a single parallelisable back-copy pass
    mirrors lower → upper. This halves the scattered write traffic during the
    O(n^2) construction phase.
    """
    cdef int r, c, best_jj
    cdef double min_val, cur_val, max_val
    cdef const double* vat_row

    for r in range(1, n):
        vat_row = vat_matrix + <Py_ssize_t>r * n

        # Find minimum distance in columns [0, r) — lower-triangle VAT reads.
        min_val = vat_row[0]
        best_jj = 0
        for c in range(1, r):
            if vat_row[c] < min_val:
                min_val = vat_row[c]
                best_jj = c

        argmin_seq[r - 1] = best_jj

        # Write edge weight — lower triangle only (r > best_jj always holds).
        ivat_matrix[<Py_ssize_t>r * n + best_jj] = min_val

        # For every other already-visited vertex c, the minimax distance is
        # ivat[r, c] = max(ivat[r, best_jj], ivat[best_jj, c]).
        # Reads are canonicalised to the lower triangle: swap when best_jj < c.
        for c in range(r):
            if c != best_jj:
                if best_jj > c:
                    cur_val = ivat_matrix[<Py_ssize_t>best_jj * n + c]
                else:
                    cur_val = ivat_matrix[<Py_ssize_t>c * n + best_jj]
                max_val = min_val if min_val > cur_val else cur_val
                ivat_matrix[<Py_ssize_t>r * n + c] = max_val

    # Back-copy: mirror lower triangle to upper. All lower-triangle values are
    # final, so threads own exclusive (r, c) pairs — no races.
    for r in prange(1, n, schedule='static', num_threads=nthreads):
        for c in range(r):
            ivat_matrix[<Py_ssize_t>c * n + r] = ivat_matrix[<Py_ssize_t>r * n + c]


cdef void _compute_ivat_kernel_32(
    const float* vat_matrix, int n, float* ivat_matrix,
    int* argmin_seq, int nthreads
) noexcept nogil:
    """Build IVAT matrix from VAT matrix (float32). See _compute_ivat_kernel_64 for rationale."""
    cdef int r, c, best_jj
    cdef float min_val, cur_val, max_val
    cdef const float* vat_row

    for r in range(1, n):
        vat_row = vat_matrix + <Py_ssize_t>r * n

        # Find minimum distance in columns [0, r) — lower-triangle VAT reads.
        min_val = vat_row[0]
        best_jj = 0
        for c in range(1, r):
            if vat_row[c] < min_val:
                min_val = vat_row[c]
                best_jj = c

        argmin_seq[r - 1] = best_jj

        # Write edge weight — lower triangle only (r > best_jj always holds).
        ivat_matrix[<Py_ssize_t>r * n + best_jj] = min_val

        # For every other already-visited vertex c, the minimax distance is
        # ivat[r, c] = max(ivat[r, best_jj], ivat[best_jj, c]).
        # Reads are canonicalised to the lower triangle: swap when best_jj < c.
        for c in range(r):
            if c != best_jj:
                if best_jj > c:
                    cur_val = ivat_matrix[<Py_ssize_t>best_jj * n + c]
                else:
                    cur_val = ivat_matrix[<Py_ssize_t>c * n + best_jj]
                max_val = min_val if min_val > cur_val else cur_val
                ivat_matrix[<Py_ssize_t>r * n + c] = max_val

    # Back-copy: mirror lower triangle to upper. All lower-triangle values are
    # final, so threads own exclusive (r, c) pairs — no races.
    for r in prange(1, n, schedule='static', num_threads=nthreads):
        for c in range(r):
            ivat_matrix[<Py_ssize_t>c * n + r] = ivat_matrix[<Py_ssize_t>r * n + c]


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ivat_c_64(double[:, ::1] adj):
    """
    Compute IVAT (improved VAT) for float64 distance matrix.

    Returns (ivat_matrix, vat_matrix, argmin_seq, p_seq) where:
      - ivat_matrix: improved VAT matrix (n x n)
      - vat_matrix: VAT matrix (n x n)
      - argmin_seq: sequence of minimum indices
      - p_seq: permutation sequence from VAT
    """
    cdef int n = adj.shape[0]

    # Compute VAT using the C implementation
    vat_np, p_seq_np, q_seq_np = compute_vat_c_64(adj)
    cdef double[:, ::1] vat = vat_np

    argmin_seq_np = np.zeros(n - 1, dtype=np.int32)
    cdef int[:] argmin_seq = argmin_seq_np

    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _compute_ivat_kernel_64(&vat[0, 0], n, &vat[0, 0], &argmin_seq[0], nthreads)

    return vat_np, vat_np, argmin_seq_np, p_seq_np


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_ivat_c_32(float[:, ::1] adj):
    """
    Compute IVAT (improved VAT) for float32 distance matrix.

    Returns (ivat_matrix, vat_matrix, argmin_seq, p_seq).
    """
    cdef int n = adj.shape[0]

    # Compute VAT using the C implementation
    vat_np, p_seq_np, q_seq_np = compute_vat_c_32(adj)
    cdef float[:, ::1] vat = vat_np

    argmin_seq_np = np.zeros(n - 1, dtype=np.int32)
    cdef int[:] argmin_seq = argmin_seq_np

    cdef int nthreads = openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nthreads < 2:
        nthreads = 1

    with nogil:
        _compute_ivat_kernel_32(&vat[0, 0], n, &vat[0, 0], &argmin_seq[0], nthreads)

    return vat_np, vat_np, argmin_seq_np, p_seq_np


def compute_ivat_c(adj, inplace=False):
    """
    Compute IVAT (improved VAT) for a distance matrix.
    Automatically dispatches to float32 or float64 based on input dtype.

    Note: The `inplace` parameter is accepted for API compatibility but is ignored
    (the compiled version always returns new matrices).

    Returns (ivat_matrix, vat_matrix, argmin_seq, p_seq).
    """
    if adj.dtype == np.float32:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float32)
        return compute_ivat_c_32(adj_c)
    elif adj.dtype == np.float64:
        adj_c = np.require(adj, requirements=['C_CONTIGUOUS'], dtype=np.float64)
        return compute_ivat_c_64(adj_c)
    else:
        raise TypeError(f"Expected float32 or float64, got {adj.dtype}")
