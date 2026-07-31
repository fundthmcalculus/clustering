# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

"""Cython/OpenMP Lin-Kernighan TSP solver.

Compiled counterpart of ``lk.py`` -- see that module for the algorithm write-up.
The two paths must stay behaviorally equivalent (``CLAUDE.md``). This kernel
adds two things on top of the reference:

* **Fused dtypes.** ``float32`` and ``float64`` variants (``_32``/``_64``) are
  written once each with a Python dispatcher (``lin_kernighan_c``) picking by
  ``data.dtype``. Gains are always accumulated in ``double`` regardless of the
  distance dtype so the positive-gain criterion is not corrupted by float32
  round-off.

* **Multi-threaded local optimization.** The ``n_starts`` independent
  nearest-neighbour restarts are embarrassingly parallel, so each one is
  optimized on its own private scratch buffers inside an OpenMP ``prange``. The
  best tour across restarts is selected serially afterwards. This is the
  "multi-threaded local optimization" path: N local searches run concurrently,
  one per thread, with no shared mutable state.
"""

import numpy as np
cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
cimport openmp


# Gain tolerance: improvements below this are floating-point noise (mirror of
# lk._EPS). Keeping the local search from chasing vanishing deltas guarantees
# termination.
cdef double _EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared helper: in-place segment reversal (dtype-independent, operates on the
# integer tour + position arrays).
# ---------------------------------------------------------------------------
cdef inline void _reverse(int* tour, int* pos, int lo, int hi) noexcept nogil:
    cdef int a, b
    while lo < hi:
        a = tour[lo]
        b = tour[hi]
        tour[lo] = b
        tour[hi] = a
        pos[b] = lo
        pos[a] = hi
        lo += 1
        hi -= 1


# ===========================================================================
# float64 variant
# ===========================================================================
cdef double _tour_length_64(const int* tour, const double* D, int n) noexcept nogil:
    cdef double s = 0.0
    cdef int i, a, b
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        s += D[<Py_ssize_t>a * n + b]
    return s


cdef void _nn_tour_64(
    const double* D, int n, int start, int* tour, char* visited
) noexcept nogil:
    cdef int i, j, current, best
    cdef double bestd, dd
    for i in range(n):
        visited[i] = 0
    current = start
    tour[0] = current
    visited[current] = 1
    for i in range(1, n):
        best = -1
        bestd = INFINITY
        for j in range(n):
            if visited[j]:
                continue
            dd = D[<Py_ssize_t>current * n + j]
            if dd < bestd:
                bestd = dd
                best = j
        tour[i] = best
        visited[best] = 1
        current = best


cdef int _lk_step_64(
    int t1, int* tour, int* pos, const double* D, const int* neigh,
    int k, int n, int max_depth, int* applo, int* apphi
) noexcept nogil:
    """One variable-depth LK improvement anchored at ``t1``.

    Tries both edges incident to ``t1`` (successor and predecessor directions);
    the first direction that nets a gain wins. Applies committed reversals in
    place, records them in ``applo``/``apphi``, and returns how many were kept
    (0 = no improving move; tour restored). Mirrors ``lk._lk_step``.
    """
    cdef int i = pos[t1]  # anchor position; reversals never touch it.
    cdef int napplied, best_len, depth, kk, t2, t3, p3, fe, lo, hi
    cdef int lm1, ll, hh, hp1, chosen_lo, chosen_hi, kidx, direction
    cdef double cum, best_cum, d_t1t2, partial, delta, chosen_delta

    for direction in range(2):  # 0 = successor edge, 1 = predecessor edge
        napplied = 0
        cum = 0.0
        best_cum = 0.0
        best_len = 0

        for depth in range(max_depth):
            if direction == 0:
                if i > n - 2:
                    break
                fe = i + 1
            else:
                if i < 1:
                    break
                fe = i - 1

            t2 = tour[fe]
            d_t1t2 = D[<Py_ssize_t>t1 * n + t2]
            chosen_lo = -1
            chosen_delta = 0.0
            for kk in range(k):
                t3 = neigh[<Py_ssize_t>t2 * k + kk]
                if t3 == t1 or t3 == t2:
                    continue
                partial = d_t1t2 - D[<Py_ssize_t>t2 * n + t3]
                if partial <= _EPS:
                    break
                p3 = pos[t3]
                if direction == 0:
                    hi = p3 - 1 if p3 > 0 else n - 1
                    lo = i + 1
                    if hi < i + 2:
                        continue
                else:
                    lo = p3 + 1 if p3 < n - 1 else 0
                    hi = i - 1
                    if lo > i - 2:
                        continue
                lm1 = tour[(lo - 1 + n) % n]
                ll = tour[lo]
                hh = tour[hi]
                hp1 = tour[(hi + 1) % n]
                delta = (
                    D[<Py_ssize_t>lm1 * n + ll]
                    + D[<Py_ssize_t>hh * n + hp1]
                    - D[<Py_ssize_t>lm1 * n + hh]
                    - D[<Py_ssize_t>ll * n + hp1]
                )
                chosen_lo = lo
                chosen_hi = hi
                chosen_delta = delta
                break

            if chosen_lo == -1:
                break

            _reverse(tour, pos, chosen_lo, chosen_hi)
            applo[napplied] = chosen_lo
            apphi[napplied] = chosen_hi
            napplied += 1
            cum += chosen_delta
            if cum > best_cum + _EPS:
                best_cum = cum
                best_len = napplied

        # Undo reversals past the best-gain prefix.
        for kidx in range(napplied - 1, best_len - 1, -1):
            _reverse(tour, pos, applo[kidx], apphi[kidx])

        if best_cum > _EPS:
            return best_len

    return 0


cdef void _optimize_64(
    int* tour, int* pos, const double* D, const int* neigh,
    int k, int n, int max_depth, char* dont_look, int* applo, int* apphi
) noexcept nogil:
    cdef int c1, m, lo, hi, kept
    cdef bint improved_any = True
    for c1 in range(n):
        dont_look[c1] = 0
    while improved_any:
        improved_any = False
        for c1 in range(n):
            if dont_look[c1]:
                continue
            kept = _lk_step_64(c1, tour, pos, D, neigh, k, n, max_depth, applo, apphi)
            if kept > 0:
                improved_any = True
                for m in range(kept):
                    lo = applo[m]
                    hi = apphi[m]
                    dont_look[tour[(lo - 1 + n) % n]] = 0
                    dont_look[tour[lo]] = 0
                    dont_look[tour[hi]] = 0
                    dont_look[tour[(hi + 1) % n]] = 0
                dont_look[c1] = 0
            else:
                dont_look[c1] = 1


cdef void _run_one_64(
    int s, const double* D, const int* neigh, int k, int n, int max_depth,
    const int* start_cities, int* tours, int* pos_all, char* vis_all,
    char* dl_all, int* lo_all, int* hi_all, double* lengths
) noexcept nogil:
    cdef int* tour = tours + <Py_ssize_t>s * n
    cdef int* pos = pos_all + <Py_ssize_t>s * n
    cdef char* visited = vis_all + <Py_ssize_t>s * n
    cdef char* dont_look = dl_all + <Py_ssize_t>s * n
    cdef int* applo = lo_all + <Py_ssize_t>s * max_depth
    cdef int* apphi = hi_all + <Py_ssize_t>s * max_depth
    cdef int idx
    _nn_tour_64(D, n, start_cities[s], tour, visited)
    for idx in range(n):
        pos[tour[idx]] = idx
    _optimize_64(tour, pos, D, neigh, k, n, max_depth, dont_look, applo, apphi)
    lengths[s] = _tour_length_64(tour, D, n)


def lin_kernighan_c_64(
    double[:, ::1] D,
    int[:, ::1] neigh,
    int[::1] start_cities,
    int max_depth,
    int num_threads,
):
    """float64 multi-start Lin-Kernighan. Returns (tour int32, length float)."""
    cdef int n = D.shape[0]
    cdef int k = neigh.shape[1]
    cdef int n_starts = start_cities.shape[0]

    cdef int nthreads = num_threads
    if nthreads <= 0:
        nthreads = openmp.omp_get_max_threads()
    if nthreads > n_starts:
        nthreads = n_starts
    if nthreads < 1:
        nthreads = 1

    cdef Py_ssize_t nn = <Py_ssize_t>n_starts * n
    cdef Py_ssize_t nd = <Py_ssize_t>n_starts * max_depth
    cdef int* tours = <int*>malloc(nn * sizeof(int))
    cdef int* pos_all = <int*>malloc(nn * sizeof(int))
    cdef char* vis_all = <char*>malloc(nn * sizeof(char))
    cdef char* dl_all = <char*>malloc(nn * sizeof(char))
    cdef int* lo_all = <int*>malloc(nd * sizeof(int))
    cdef int* hi_all = <int*>malloc(nd * sizeof(int))
    cdef double* lengths = <double*>malloc(<Py_ssize_t>n_starts * sizeof(double))

    if (tours == NULL or pos_all == NULL or vis_all == NULL or dl_all == NULL
            or lo_all == NULL or hi_all == NULL or lengths == NULL):
        free(tours); free(pos_all); free(vis_all); free(dl_all)
        free(lo_all); free(hi_all); free(lengths)
        raise MemoryError("could not allocate Lin-Kernighan scratch buffers")

    cdef const double* Dp = &D[0, 0]
    cdef const int* neighp = &neigh[0, 0]
    cdef const int* startp = &start_cities[0]
    cdef int s

    with nogil:
        for s in prange(n_starts, schedule='dynamic', num_threads=nthreads):
            _run_one_64(
                s, Dp, neighp, k, n, max_depth, startp,
                tours, pos_all, vis_all, dl_all, lo_all, hi_all, lengths
            )

    # Select the best restart serially.
    cdef int best_s = 0
    cdef double best_len = lengths[0]
    for s in range(1, n_starts):
        if lengths[s] < best_len:
            best_len = lengths[s]
            best_s = s

    out_np = np.empty(n, dtype=np.int32)
    cdef int[::1] out = out_np
    cdef int i
    for i in range(n):
        out[i] = tours[<Py_ssize_t>best_s * n + i]

    free(tours); free(pos_all); free(vis_all); free(dl_all)
    free(lo_all); free(hi_all); free(lengths)
    return out_np, best_len


# ===========================================================================
# float32 variant (distances stored as float; gains still accumulated double)
# ===========================================================================
cdef double _tour_length_32(const int* tour, const float* D, int n) noexcept nogil:
    cdef double s = 0.0
    cdef int i, a, b
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        s += <double>D[<Py_ssize_t>a * n + b]
    return s


cdef void _nn_tour_32(
    const float* D, int n, int start, int* tour, char* visited
) noexcept nogil:
    cdef int i, j, current, best
    cdef double bestd, dd
    for i in range(n):
        visited[i] = 0
    current = start
    tour[0] = current
    visited[current] = 1
    for i in range(1, n):
        best = -1
        bestd = INFINITY
        for j in range(n):
            if visited[j]:
                continue
            dd = <double>D[<Py_ssize_t>current * n + j]
            if dd < bestd:
                bestd = dd
                best = j
        tour[i] = best
        visited[best] = 1
        current = best


cdef int _lk_step_32(
    int t1, int* tour, int* pos, const float* D, const int* neigh,
    int k, int n, int max_depth, int* applo, int* apphi
) noexcept nogil:
    cdef int i = pos[t1]
    cdef int napplied, best_len, depth, kk, t2, t3, p3, fe, lo, hi
    cdef int lm1, ll, hh, hp1, chosen_lo, chosen_hi, kidx, direction
    cdef double cum, best_cum, d_t1t2, partial, delta, chosen_delta

    for direction in range(2):
        napplied = 0
        cum = 0.0
        best_cum = 0.0
        best_len = 0

        for depth in range(max_depth):
            if direction == 0:
                if i > n - 2:
                    break
                fe = i + 1
            else:
                if i < 1:
                    break
                fe = i - 1

            t2 = tour[fe]
            d_t1t2 = <double>D[<Py_ssize_t>t1 * n + t2]
            chosen_lo = -1
            chosen_delta = 0.0
            for kk in range(k):
                t3 = neigh[<Py_ssize_t>t2 * k + kk]
                if t3 == t1 or t3 == t2:
                    continue
                partial = d_t1t2 - <double>D[<Py_ssize_t>t2 * n + t3]
                if partial <= _EPS:
                    break
                p3 = pos[t3]
                if direction == 0:
                    hi = p3 - 1 if p3 > 0 else n - 1
                    lo = i + 1
                    if hi < i + 2:
                        continue
                else:
                    lo = p3 + 1 if p3 < n - 1 else 0
                    hi = i - 1
                    if lo > i - 2:
                        continue
                lm1 = tour[(lo - 1 + n) % n]
                ll = tour[lo]
                hh = tour[hi]
                hp1 = tour[(hi + 1) % n]
                delta = (
                    <double>D[<Py_ssize_t>lm1 * n + ll]
                    + <double>D[<Py_ssize_t>hh * n + hp1]
                    - <double>D[<Py_ssize_t>lm1 * n + hh]
                    - <double>D[<Py_ssize_t>ll * n + hp1]
                )
                chosen_lo = lo
                chosen_hi = hi
                chosen_delta = delta
                break

            if chosen_lo == -1:
                break

            _reverse(tour, pos, chosen_lo, chosen_hi)
            applo[napplied] = chosen_lo
            apphi[napplied] = chosen_hi
            napplied += 1
            cum += chosen_delta
            if cum > best_cum + _EPS:
                best_cum = cum
                best_len = napplied

        for kidx in range(napplied - 1, best_len - 1, -1):
            _reverse(tour, pos, applo[kidx], apphi[kidx])

        if best_cum > _EPS:
            return best_len

    return 0


cdef void _optimize_32(
    int* tour, int* pos, const float* D, const int* neigh,
    int k, int n, int max_depth, char* dont_look, int* applo, int* apphi
) noexcept nogil:
    cdef int c1, m, lo, hi, kept
    cdef bint improved_any = True
    for c1 in range(n):
        dont_look[c1] = 0
    while improved_any:
        improved_any = False
        for c1 in range(n):
            if dont_look[c1]:
                continue
            kept = _lk_step_32(c1, tour, pos, D, neigh, k, n, max_depth, applo, apphi)
            if kept > 0:
                improved_any = True
                for m in range(kept):
                    lo = applo[m]
                    hi = apphi[m]
                    dont_look[tour[(lo - 1 + n) % n]] = 0
                    dont_look[tour[lo]] = 0
                    dont_look[tour[hi]] = 0
                    dont_look[tour[(hi + 1) % n]] = 0
                dont_look[c1] = 0
            else:
                dont_look[c1] = 1


cdef void _run_one_32(
    int s, const float* D, const int* neigh, int k, int n, int max_depth,
    const int* start_cities, int* tours, int* pos_all, char* vis_all,
    char* dl_all, int* lo_all, int* hi_all, double* lengths
) noexcept nogil:
    cdef int* tour = tours + <Py_ssize_t>s * n
    cdef int* pos = pos_all + <Py_ssize_t>s * n
    cdef char* visited = vis_all + <Py_ssize_t>s * n
    cdef char* dont_look = dl_all + <Py_ssize_t>s * n
    cdef int* applo = lo_all + <Py_ssize_t>s * max_depth
    cdef int* apphi = hi_all + <Py_ssize_t>s * max_depth
    cdef int idx
    _nn_tour_32(D, n, start_cities[s], tour, visited)
    for idx in range(n):
        pos[tour[idx]] = idx
    _optimize_32(tour, pos, D, neigh, k, n, max_depth, dont_look, applo, apphi)
    lengths[s] = _tour_length_32(tour, D, n)


def lin_kernighan_c_32(
    float[:, ::1] D,
    int[:, ::1] neigh,
    int[::1] start_cities,
    int max_depth,
    int num_threads,
):
    """float32 multi-start Lin-Kernighan. Returns (tour int32, length float)."""
    cdef int n = D.shape[0]
    cdef int k = neigh.shape[1]
    cdef int n_starts = start_cities.shape[0]

    cdef int nthreads = num_threads
    if nthreads <= 0:
        nthreads = openmp.omp_get_max_threads()
    if nthreads > n_starts:
        nthreads = n_starts
    if nthreads < 1:
        nthreads = 1

    cdef Py_ssize_t nn = <Py_ssize_t>n_starts * n
    cdef Py_ssize_t nd = <Py_ssize_t>n_starts * max_depth
    cdef int* tours = <int*>malloc(nn * sizeof(int))
    cdef int* pos_all = <int*>malloc(nn * sizeof(int))
    cdef char* vis_all = <char*>malloc(nn * sizeof(char))
    cdef char* dl_all = <char*>malloc(nn * sizeof(char))
    cdef int* lo_all = <int*>malloc(nd * sizeof(int))
    cdef int* hi_all = <int*>malloc(nd * sizeof(int))
    cdef double* lengths = <double*>malloc(<Py_ssize_t>n_starts * sizeof(double))

    if (tours == NULL or pos_all == NULL or vis_all == NULL or dl_all == NULL
            or lo_all == NULL or hi_all == NULL or lengths == NULL):
        free(tours); free(pos_all); free(vis_all); free(dl_all)
        free(lo_all); free(hi_all); free(lengths)
        raise MemoryError("could not allocate Lin-Kernighan scratch buffers")

    cdef const float* Dp = &D[0, 0]
    cdef const int* neighp = &neigh[0, 0]
    cdef const int* startp = &start_cities[0]
    cdef int s

    with nogil:
        for s in prange(n_starts, schedule='dynamic', num_threads=nthreads):
            _run_one_32(
                s, Dp, neighp, k, n, max_depth, startp,
                tours, pos_all, vis_all, dl_all, lo_all, hi_all, lengths
            )

    cdef int best_s = 0
    cdef double best_len = lengths[0]
    for s in range(1, n_starts):
        if lengths[s] < best_len:
            best_len = lengths[s]
            best_s = s

    out_np = np.empty(n, dtype=np.int32)
    cdef int[::1] out = out_np
    cdef int i
    for i in range(n):
        out[i] = tours[<Py_ssize_t>best_s * n + i]

    free(tours); free(pos_all); free(vis_all); free(dl_all)
    free(lo_all); free(hi_all); free(lengths)
    return out_np, best_len


# ---------------------------------------------------------------------------
# Neighbour-list construction + public dispatch
# ---------------------------------------------------------------------------
def _build_neighbor_lists(distances, int k):
    """Each city's ``k`` nearest neighbours, sorted ascending (int32, C order).

    The diagonal is masked to ``+inf`` so a city is never its own neighbour.
    """
    masked = np.array(distances, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(order.astype(np.int32))


def lin_kernighan_c(
    distances,
    n_starts=1,
    max_depth=5,
    neighbors=8,
    num_threads=0,
    seed=None,
):
    """Multi-start Lin-Kernighan (C/OpenMP), dispatched by ``distances.dtype``.

    Parameters
    ----------
    distances : ndarray, shape (n, n)
        Symmetric pairwise-distance matrix (float32 or float64).
    n_starts : int
        Independent nearest-neighbour restarts (optimized in parallel).
    max_depth : int
        Maximum depth of each variable-depth LK chain.
    neighbors : int
        Candidate nearest-neighbour list size per city.
    num_threads : int
        OpenMP threads for the restart loop (<=0 -> OpenMP default, capped at
        ``n_starts``).
    seed : int, optional
        Seed for start-city selection when ``n_starts > n``.

    Returns
    -------
    (tour, length) : (ndarray int32 of shape (n,), float)
    """
    distances = np.asarray(distances)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2-D matrix")

    n = distances.shape[0]
    if n <= 3:
        tour = np.arange(max(n, 0), dtype=np.int32)
        if n < 2:
            return tour, 0.0
        nxt = np.roll(tour, -1)
        return tour, float(distances[tour, nxt].sum())

    n_starts = max(1, int(n_starts))
    max_depth = max(1, int(max_depth))
    k = min(max(1, int(neighbors)), n - 1)
    neigh = _build_neighbor_lists(distances, k)

    rng = np.random.default_rng(seed)
    if n_starts <= n:
        start_cities = np.linspace(0, n - 1, n_starts).astype(np.int32)
    else:
        start_cities = rng.integers(0, n, size=n_starts).astype(np.int32)

    if distances.dtype == np.float32:
        D = np.require(distances, requirements=["C_CONTIGUOUS"], dtype=np.float32)
        return lin_kernighan_c_32(D, neigh, start_cities, max_depth, int(num_threads))
    elif distances.dtype == np.float64:
        D = np.require(distances, requirements=["C_CONTIGUOUS"], dtype=np.float64)
        return lin_kernighan_c_64(D, neigh, start_cities, max_depth, int(num_threads))
    else:
        # Match the fcm/pcvat dispatchers: default numeric input to float64.
        D = np.require(distances, requirements=["C_CONTIGUOUS"], dtype=np.float64)
        return lin_kernighan_c_64(D, neigh, start_cities, max_depth, int(num_threads))
