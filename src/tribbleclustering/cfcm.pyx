# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport sqrt, isnan, isinf, fmax, fabs
from libc.stdint cimport int64_t, int32_t
from dataclasses import dataclass


# Convergence test, kept byte-for-byte equivalent to the
# `np.allclose(c_new, c, rtol=1e-5, atol=1e-8)` in fcm.fuzzy_c_means. The two
# implementations sit behind a silent import-time fallback, so they have to stop
# on the same iteration and return the same centers -- a compiled path with its
# own tolerance quietly returns a different answer than the reference.
DEF _CONV_RTOL = 1e-5
DEF _CONV_ATOL = 1e-8


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_distances_gram_32(
    const float[:, ::1] x,
    const float[:, ::1] c,
    float[::1] x_norm2,
    float[::1] c_norm2,
    float[:, ::1] xc,
    float[:, ::1] distances
) noexcept nogil:
    """Compute distances using gram identity: ||x-c||^2 = ||x||^2 - 2*x*c^T + ||c||^2

    ``xc`` is the raw ``x @ c.T`` GEMM output; the factor of two is applied here
    rather than in numpy so the caller can reuse one preallocated buffer.
    """
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int i, j
    cdef float dist2

    for i in range(n_samples):
        for j in range(n_clusters):
            dist2 = x_norm2[i] + c_norm2[j] - 2.0 * xc[i, j]
            dist2 = fmax(dist2, 0.0)
            distances[i, j] = sqrt(dist2)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_distances_gram_64(
    const double[:, ::1] x,
    const double[:, ::1] c,
    double[::1] x_norm2,
    double[::1] c_norm2,
    double[:, ::1] xc,
    double[:, ::1] distances
) noexcept nogil:
    """Compute distances using gram identity: ||x-c||^2 = ||x||^2 - 2*x*c^T + ||c||^2

    ``xc`` is the raw ``x @ c.T`` GEMM output; the factor of two is applied here
    rather than in numpy so the caller can reuse one preallocated buffer.
    """
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int i, j
    cdef double dist2

    for i in range(n_samples):
        for j in range(n_clusters):
            dist2 = x_norm2[i] + c_norm2[j] - 2.0 * xc[i, j]
            dist2 = fmax(dist2, 0.0)
            distances[i, j] = sqrt(dist2)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_weights_32(
    const float[:, ::1] distances,
    float m,
    float[:, ::1] w_ij
) noexcept nogil:
    """Membership update. Mirrors fcm._get_weights.

    The textbook form sums (d_ij / d_il) ** p over l for every j, which is
    O(n_clusters ** 2) pow() calls per sample. Factoring d_ij out of that sum
    gives the algebraically identical

        w_ij = d_ij ** -p / sum_l d_il ** -p

    at one pow() per (sample, cluster). d_ij ** -p overflows for near-coincident
    points, so the row's smallest distance is used as the reference scale:
    (d_min / d_ij) ** p lies in (0, 1] and cancels out of the normalization.

    w_ij doubles as the per-row scratch buffer, so no allocation is needed.
    """
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, n_zero
    cdef float p = 2.0 / (m - 1.0)
    cdef float d, d_min, denom, val

    for i in range(n_samples):
        # d_min over the strictly positive distances; -1 means "none seen yet".
        d_min = -1.0
        n_zero = 0
        for j in range(n_clusters):
            d = distances[i, j]
            if d == 0.0:
                n_zero += 1
            elif d_min < 0.0 or d < d_min:
                d_min = d

        if n_zero > 0:
            # Coincident with one or more centers: crisp membership split
            # across them, the same convention fcm._get_weights applies.
            val = 1.0 / n_zero
            for j in range(n_clusters):
                w_ij[i, j] = val if distances[i, j] == 0.0 else 0.0
            continue

        # denom >= 1: the cluster attaining d_min contributes exactly 1.0.
        denom = 0.0
        for j in range(n_clusters):
            val = (d_min / distances[i, j]) ** p
            w_ij[i, j] = val
            denom += val

        for j in range(n_clusters):
            val = w_ij[i, j] / denom
            if isnan(val) or isinf(val):
                w_ij[i, j] = 0.0
            else:
                w_ij[i, j] = val


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_weights_64(
    const double[:, ::1] distances,
    double m,
    double[:, ::1] w_ij
) noexcept nogil:
    """Membership update. Mirrors fcm._get_weights.

    The textbook form sums (d_ij / d_il) ** p over l for every j, which is
    O(n_clusters ** 2) pow() calls per sample. Factoring d_ij out of that sum
    gives the algebraically identical

        w_ij = d_ij ** -p / sum_l d_il ** -p

    at one pow() per (sample, cluster). d_ij ** -p overflows for near-coincident
    points, so the row's smallest distance is used as the reference scale:
    (d_min / d_ij) ** p lies in (0, 1] and cancels out of the normalization.

    w_ij doubles as the per-row scratch buffer, so no allocation is needed.
    """
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, n_zero
    cdef double p = 2.0 / (m - 1.0)
    cdef double d, d_min, denom, val

    for i in range(n_samples):
        # d_min over the strictly positive distances; -1 means "none seen yet".
        d_min = -1.0
        n_zero = 0
        for j in range(n_clusters):
            d = distances[i, j]
            if d == 0.0:
                n_zero += 1
            elif d_min < 0.0 or d < d_min:
                d_min = d

        if n_zero > 0:
            # Coincident with one or more centers: crisp membership split
            # across them, the same convention fcm._get_weights applies.
            val = 1.0 / n_zero
            for j in range(n_clusters):
                w_ij[i, j] = val if distances[i, j] == 0.0 else 0.0
            continue

        # denom >= 1: the cluster attaining d_min contributes exactly 1.0.
        denom = 0.0
        for j in range(n_clusters):
            val = (d_min / distances[i, j]) ** p
            w_ij[i, j] = val
            denom += val

        for j in range(n_clusters):
            val = w_ij[i, j] / denom
            if isnan(val) or isinf(val):
                w_ij[i, j] = 0.0
            else:
                w_ij[i, j] = val


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_new_centers_32(
    const float[:, ::1] w_ij,
    const float[:, ::1] x,
    float m,
    float[:, ::1] v_ij
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = w_ij.shape[1]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float wm, w_sum

    for j in range(n_clusters):
        w_sum = 0.0
        for i in range(n_samples):
            wm = w_ij[i, j] ** m
            w_sum += wm
            for k in range(n_features):
                v_ij[j, k] += wm * x[i, k]

        if w_sum > 0.0:
            for k in range(n_features):
                v_ij[j, k] /= w_sum
        else:
            for k in range(n_features):
                v_ij[j, k] = 0.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_new_centers_64(
    const double[:, ::1] w_ij,
    const double[:, ::1] x,
    double m,
    double[:, ::1] v_ij
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = w_ij.shape[1]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef double wm, w_sum

    for j in range(n_clusters):
        w_sum = 0.0
        for i in range(n_samples):
            wm = w_ij[i, j] ** m
            w_sum += wm
            for k in range(n_features):
                v_ij[j, k] += wm * x[i, k]

        if w_sum > 0.0:
            for k in range(n_features):
                v_ij[j, k] /= w_sum
        else:
            for k in range(n_features):
                v_ij[j, k] = 0.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _init_centers_32(
    const float[:, ::1] x,
    int n_clusters,
    const int64_t[::1] indices,
    float[:, ::1] c
) noexcept nogil:
    cdef int n_features = x.shape[1]
    cdef int i, k

    for i in range(n_clusters):
        for k in range(n_features):
            c[i, k] = 0.5 * (x[indices[2*i], k] + x[indices[2*i + 1], k])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _init_centers_64(
    const double[:, ::1] x,
    int n_clusters,
    const int64_t[::1] indices,
    double[:, ::1] c
) noexcept nogil:
    cdef int n_features = x.shape[1]
    cdef int i, k

    for i in range(n_clusters):
        for k in range(n_features):
            c[i, k] = 0.5 * (x[indices[2*i], k] + x[indices[2*i + 1], k])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _fuzzy_c_means_kernel_32(
    float[:, ::1] x,
    int n,
    float m,
    float[:, ::1] c_init,
    int max_iter = 100
):
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef float[:, ::1] c
    cdef float[:, ::1] c_new
    cdef float[:, ::1] w_ij
    cdef float[:, ::1] distances
    cdef float[::1] x_norm2
    cdef float[::1] c_norm2
    cdef float[:, ::1] xc
    cdef int i, j, k, iteration
    cdef bint converged = False
    cdef int n_iter = 0

    c = np.zeros((n, n_features), dtype=np.float32)
    c_new = np.zeros((n, n_features), dtype=np.float32)
    w_ij = np.zeros((n_samples, n), dtype=np.float32)
    distances = np.zeros((n_samples, n), dtype=np.float32)

    for i in range(n):
        for k in range(n_features):
            c[i, k] = c_init[i, k]

    # `c` is mutated in place below, so `c_np` stays a live view of the current
    # centers and the gram buffers can be allocated once for the whole run.
    x_np = np.asarray(x)
    c_np = np.asarray(c)
    # ||x||^2 does not depend on the centers -- computing it per iteration also
    # materialized a full (n_samples, n_features) temporary each time.
    x_norm2_np = np.einsum("ij,ij->i", x_np, x_np)
    c_norm2_np = np.empty(n, dtype=np.float32)  # ||c||^2, shape (n,)
    xc_np = np.empty((n_samples, n), dtype=np.float32)  # x @ c.T, shape (n_samples, n)
    x_norm2 = x_norm2_np
    c_norm2 = c_norm2_np
    xc = xc_np

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Gram components via numpy's BLAS, written into the reused buffers.
        np.einsum("ij,ij->i", c_np, c_np, out=c_norm2_np)
        np.dot(x_np, c_np.T, out=xc_np)

        _compute_distances_gram_32(x, c, x_norm2, c_norm2, xc, distances)
        _compute_weights_32(distances, m, w_ij)

        for i in range(n):
            for k in range(n_features):
                c_new[i, k] = 0.0

        _compute_new_centers_32(w_ij, x, m, c_new)

        converged = True
        for i in range(n):
            for k in range(n_features):
                if fabs(c_new[i, k] - c[i, k]) > _CONV_ATOL + _CONV_RTOL * fabs(c[i, k]):
                    converged = False
                    break
            if not converged:
                break

        if converged:
            break

        # Update centers
        for i in range(n):
            for k in range(n_features):
                c[i, k] = c_new[i, k]

    # Final distance/weight computation with latest centers
    np.einsum("ij,ij->i", c_np, c_np, out=c_norm2_np)
    np.dot(x_np, c_np.T, out=xc_np)
    _compute_distances_gram_32(x, c, x_norm2, c_norm2, xc, distances)
    _compute_weights_32(distances, m, w_ij)

    return (np.asarray(c), np.asarray(w_ij), n_iter, converged)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _fuzzy_c_means_kernel_64(
    double[:, ::1] x,
    int n,
    double m,
    double[:, ::1] c_init,
    int max_iter = 100
):
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef double[:, ::1] c
    cdef double[:, ::1] c_new
    cdef double[:, ::1] w_ij
    cdef double[:, ::1] distances
    cdef double[::1] x_norm2
    cdef double[::1] c_norm2
    cdef double[:, ::1] xc
    cdef int i, j, k, iteration
    cdef bint converged = False
    cdef int n_iter = 0

    c = np.zeros((n, n_features), dtype=np.float64)
    c_new = np.zeros((n, n_features), dtype=np.float64)
    w_ij = np.zeros((n_samples, n), dtype=np.float64)
    distances = np.zeros((n_samples, n), dtype=np.float64)

    for i in range(n):
        for k in range(n_features):
            c[i, k] = c_init[i, k]

    # `c` is mutated in place below, so `c_np` stays a live view of the current
    # centers and the gram buffers can be allocated once for the whole run.
    x_np = np.asarray(x)
    c_np = np.asarray(c)
    # ||x||^2 does not depend on the centers -- computing it per iteration also
    # materialized a full (n_samples, n_features) temporary each time.
    x_norm2_np = np.einsum("ij,ij->i", x_np, x_np)
    c_norm2_np = np.empty(n, dtype=np.float64)  # ||c||^2, shape (n,)
    xc_np = np.empty((n_samples, n), dtype=np.float64)  # x @ c.T, shape (n_samples, n)
    x_norm2 = x_norm2_np
    c_norm2 = c_norm2_np
    xc = xc_np

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Gram components via numpy's BLAS, written into the reused buffers.
        np.einsum("ij,ij->i", c_np, c_np, out=c_norm2_np)
        np.dot(x_np, c_np.T, out=xc_np)

        _compute_distances_gram_64(x, c, x_norm2, c_norm2, xc, distances)
        _compute_weights_64(distances, m, w_ij)

        for i in range(n):
            for k in range(n_features):
                c_new[i, k] = 0.0

        _compute_new_centers_64(w_ij, x, m, c_new)

        converged = True
        for i in range(n):
            for k in range(n_features):
                if fabs(c_new[i, k] - c[i, k]) > _CONV_ATOL + _CONV_RTOL * fabs(c[i, k]):
                    converged = False
                    break
            if not converged:
                break

        if converged:
            break

        # Update centers
        for i in range(n):
            for k in range(n_features):
                c[i, k] = c_new[i, k]

    # Final distance/weight computation with latest centers
    np.einsum("ij,ij->i", c_np, c_np, out=c_norm2_np)
    np.dot(x_np, c_np.T, out=xc_np)
    _compute_distances_gram_64(x, c, x_norm2, c_norm2, xc, distances)
    _compute_weights_64(distances, m, w_ij)

    return (np.asarray(c), np.asarray(w_ij), n_iter, converged)


def fuzzy_c_means_32(
    x,
    int n,
    m = 2.0,
    *,
    max_iter: int = 100,
    indices = None,
    initial_guess = None,
    random_state = None,
) -> tuple:
    x = np.asarray(x, dtype=np.float32)
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef float[:, ::1] c_init
    cdef int i, k
    cdef int64_t[::1] indices_view

    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")

    c_init = np.zeros((n, n_features), dtype=np.float32)

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=np.float32)
        if initial_guess.shape[0] != n or initial_guess.shape[1] != n_features:
            raise ValueError(
                f"initial_guess must have shape ({n}, {n_features})"
            )
        for i in range(n):
            for k in range(n_features):
                c_init[i, k] = initial_guess[i, k]
    elif indices is not None:
        indices_arr = np.asarray(indices, dtype=np.int64)
        if indices_arr.shape[0] == n:
            for i in range(n):
                for k in range(n_features):
                    c_init[i, k] = x[indices_arr[i], k]
        elif indices_arr.shape[0] >= 2 * n:
            indices_view = indices_arr
            _init_centers_32(x, n, indices_view, c_init)
        else:
            raise ValueError(
                f"indices must have exactly {n} elements or at least {2*n} elements, got {indices_arr.shape[0]}"
            )
    else:
        # replace=True only when there aren't 2n distinct rows (avoids ValueError);
        # matches fcm.py. A coincident center is handled by the zero-distance branch.
        rng = np.random.default_rng(random_state)
        indices_arr = rng.choice(n_samples, size=n * 2, replace=2 * n > n_samples).astype(np.int64)
        indices_view = indices_arr
        _init_centers_32(x, n, indices_view, c_init)

    m_float = np.float32(m)
    c, w, n_iter, converged = _fuzzy_c_means_kernel_32(x, n, m_float, c_init, max_iter)
    return (c, w, n_iter, converged)


def fuzzy_c_means_64(
    x,
    int n,
    m = 2.0,
    *,
    max_iter: int = 100,
    indices = None,
    initial_guess = None,
    random_state = None,
) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef double[:, ::1] c_init
    cdef int i, k
    cdef int64_t[::1] indices_view

    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")

    c_init = np.zeros((n, n_features), dtype=np.float64)

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=np.float64)
        if initial_guess.shape[0] != n or initial_guess.shape[1] != n_features:
            raise ValueError(
                f"initial_guess must have shape ({n}, {n_features})"
            )
        for i in range(n):
            for k in range(n_features):
                c_init[i, k] = initial_guess[i, k]
    elif indices is not None:
        indices_arr = np.asarray(indices, dtype=np.int64)
        if indices_arr.shape[0] == n:
            for i in range(n):
                for k in range(n_features):
                    c_init[i, k] = x[indices_arr[i], k]
        elif indices_arr.shape[0] >= 2 * n:
            indices_view = indices_arr
            _init_centers_64(x, n, indices_view, c_init)
        else:
            raise ValueError(
                f"indices must have exactly {n} elements or at least {2*n} elements, got {indices_arr.shape[0]}"
            )
    else:
        # replace=True only when there aren't 2n distinct rows (avoids ValueError);
        # matches fcm.py. A coincident center is handled by the zero-distance branch.
        rng = np.random.default_rng(random_state)
        indices_arr = rng.choice(n_samples, size=n * 2, replace=2 * n > n_samples).astype(np.int64)
        indices_view = indices_arr
        _init_centers_64(x, n, indices_view, c_init)

    m_double = np.float64(m)
    c, w, n_iter, converged = _fuzzy_c_means_kernel_64(x, n, m_double, c_init, max_iter)
    return (c, w, n_iter, converged)


def fuzzy_c_means(
    x,
    int n,
    m = 2.0,
    *,
    max_iter: int = 100,
    indices = None,
    initial_guess = None,
    random_state = None,
) -> tuple:
    """
    Compute the fuzzy c-means clustering algorithm (Cython-optimized).

    :param x: Input data points, shape (n_samples, n_features)
    :param n: Number of clusters
    :param m: Fuzziness parameter, default 2.0
    :param max_iter: Maximum number of iterations, default 100
    :param indices: Indices of initial cluster centers, if provided
    :param initial_guess: Initial cluster centers, if provided
    :param random_state: Seed for the random center initialization -- an int, an
        already-constructed np.random.Generator, or None for fresh OS entropy.
        Only consulted when neither `indices` nor `initial_guess` is given.
        Threaded, like fcm.fuzzy_c_means: neither reads nor reseeds the
        process-global np.random stream.
    :return: FuzzyCMeansResult containing cluster_centers_, membership_matrix_, n_iter_, and converged
    """
    x = np.asarray(x)

    if x.dtype == np.float32:
        return fuzzy_c_means_32(
            x, n, np.float32(m),
            max_iter=max_iter,
            indices=indices,
            initial_guess=initial_guess,
            random_state=random_state
        )
    elif x.dtype == np.float64:
        return fuzzy_c_means_64(
            x, n, np.float64(m),
            max_iter=max_iter,
            indices=indices,
            initial_guess=initial_guess,
            random_state=random_state
        )
    else:
        raise TypeError(
            f"Unsupported dtype {x.dtype}. Expected float32 or float64."
        )
