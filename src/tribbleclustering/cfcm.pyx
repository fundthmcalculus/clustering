# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport sqrt, isnan, isinf
from libc.stdint cimport int64_t, int32_t

ctypedef fused float_type:
    float
    double


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_distances(
    const float_type[:, ::1] x,
    const float_type[:, ::1] c,
    float_type[:, ::1] distances
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float_type d, diff

    for i in range(n_samples):
        for j in range(n_clusters):
            d = 0.0
            for k in range(n_features):
                diff = x[i, k] - c[j, k]
                d += diff * diff
            distances[i, j] = sqrt(d)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_weights(
    const float_type[:, ::1] distances,
    float_type m,
    float_type[:, ::1] w_ij
) noexcept nogil:
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, jj
    cdef float_type denom, dist_ratio, val

    for i in range(n_samples):
        for j in range(n_clusters):
            if distances[i, j] == 0.0:
                w_ij[i, j] = 0.0
                continue

            denom = 0.0
            for jj in range(n_clusters):
                if distances[i, jj] == 0.0:
                    denom = 1.0
                    break
                dist_ratio = distances[i, j] / distances[i, jj]
                denom += dist_ratio ** (2.0 / (m - 1.0))

            if denom > 0.0:
                val = 1.0 / denom
            else:
                val = 0.0

            if isnan(val) or isinf(val):
                w_ij[i, j] = 0.0
            else:
                w_ij[i, j] = val


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_new_centers(
    const float_type[:, ::1] w_ij,
    const float_type[:, ::1] x,
    float_type m,
    float_type[:, ::1] v_ij
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = w_ij.shape[1]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float_type wm, w_sum

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
cdef void _init_centers(
    const float_type[:, ::1] x,
    int n_clusters,
    const int64_t[::1] indices,
    float_type[:, ::1] c
) noexcept nogil:
    cdef int n_features = x.shape[1]
    cdef int i, k

    for i in range(n_clusters):
        for k in range(n_features):
            c[i, k] = 0.5 * (x[indices[2*i], k] + x[indices[2*i + 1], k])


def fuzzy_c_means(
    float_type[:, ::1] x,
    int n,
    float_type m = 2.0,
    *,
    indices = None,
    float_type[:, ::1] initial_guess = None,
) -> tuple:
    """
    Compute the fuzzy c-means clustering algorithm (Cython-optimized).

    :param x: Input data points, shape (n_samples, n_features)
    :param n: Number of clusters
    :param m: Fuzziness parameter, default 2.0
    :param indices: Indices of initial cluster centers, if provided
    :param initial_guess: Initial cluster centers, if provided
    :return: Tuple of cluster centers (shape (n_clusters, n_features)) and membership matrix (shape (n_samples, n_clusters))
    """
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef float_type[:, ::1] c
    cdef float_type[:, ::1] c_new
    cdef float_type[:, ::1] w_ij
    cdef float_type[:, ::1] distances
    cdef int i, j, k, iteration
    cdef float_type delta, max_delta
    cdef int64_t[::1] indices_view

    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")

    c = np.zeros((n, n_features), dtype=x.dtype)
    c_new = np.zeros((n, n_features), dtype=x.dtype)
    w_ij = np.zeros((n_samples, n), dtype=x.dtype)
    distances = np.zeros((n_samples, n), dtype=x.dtype)

    if initial_guess is not None:
        if initial_guess.shape[0] != n or initial_guess.shape[1] != n_features:
            raise ValueError(
                f"initial_guess must have shape ({n}, {n_features})"
            )
        for i in range(n):
            for k in range(n_features):
                c[i, k] = initial_guess[i, k]
    elif indices is not None:
        indices_arr = np.asarray(indices, dtype=np.int64)
        if indices_arr.shape[0] == n:
            # If exactly n indices provided, use them directly
            for i in range(n):
                for k in range(n_features):
                    c[i, k] = x[indices_arr[i], k]
        elif indices_arr.shape[0] >= 2 * n:
            # If n*2 or more indices provided, pair them and combine
            indices_view = indices_arr
            _init_centers(x, n, indices_view, c)
        else:
            raise ValueError(
                f"indices must have exactly {n} elements or at least {2*n} elements, got {indices_arr.shape[0]}"
            )
    else:
        indices_arr = np.random.choice(n_samples, size=n * 2, replace=False).astype(np.int64)
        indices_view = indices_arr
        _init_centers(x, n, indices_view, c)

    for iteration in range(100):
        _compute_distances(x, c, distances)
        _compute_weights(distances, m, w_ij)

        for i in range(n):
            for k in range(n_features):
                c_new[i, k] = 0.0

        _compute_new_centers(w_ij, x, m, c_new)

        max_delta = 0.0
        for i in range(n):
            for k in range(n_features):
                delta = (c_new[i, k] - c[i, k]) ** 2
                if delta > max_delta:
                    max_delta = delta

        if max_delta < 1e-10:
            break

        for i in range(n):
            for k in range(n_features):
                c[i, k] = c_new[i, k]

    _compute_distances(x, c, distances)
    _compute_weights(distances, m, w_ij)

    return np.asarray(c), np.asarray(w_ij)
