# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport sqrt, isnan, isinf
from libc.stdint cimport int64_t, int32_t


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_distances_32(
    const float[:, ::1] x,
    const float[:, ::1] c,
    float[:, ::1] distances
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float d, diff

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
cdef void _compute_distances_64(
    const double[:, ::1] x,
    const double[:, ::1] c,
    double[:, ::1] distances
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef double d, diff

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
cdef void _compute_weights_32(
    const float[:, ::1] distances,
    float m,
    float[:, ::1] w_ij
) noexcept nogil:
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, jj
    cdef float denom, dist_ratio, val

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
cdef void _compute_weights_64(
    const double[:, ::1] distances,
    double m,
    double[:, ::1] w_ij
) noexcept nogil:
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, jj
    cdef double denom, dist_ratio, val

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
    float[:, ::1] c_init
):
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef float[:, ::1] c
    cdef float[:, ::1] c_new
    cdef float[:, ::1] w_ij
    cdef float[:, ::1] distances
    cdef int i, j, k, iteration
    cdef float delta, max_delta

    c = np.zeros((n, n_features), dtype=np.float32)
    c_new = np.zeros((n, n_features), dtype=np.float32)
    w_ij = np.zeros((n_samples, n), dtype=np.float32)
    distances = np.zeros((n_samples, n), dtype=np.float32)

    for i in range(n):
        for k in range(n_features):
            c[i, k] = c_init[i, k]

    for iteration in range(100):
        _compute_distances_32(x, c, distances)
        _compute_weights_32(distances, m, w_ij)

        for i in range(n):
            for k in range(n_features):
                c_new[i, k] = 0.0

        _compute_new_centers_32(w_ij, x, m, c_new)

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

    _compute_distances_32(x, c, distances)
    _compute_weights_32(distances, m, w_ij)

    return np.asarray(c), np.asarray(w_ij)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _fuzzy_c_means_kernel_64(
    double[:, ::1] x,
    int n,
    double m,
    double[:, ::1] c_init
):
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    cdef double[:, ::1] c
    cdef double[:, ::1] c_new
    cdef double[:, ::1] w_ij
    cdef double[:, ::1] distances
    cdef int i, j, k, iteration
    cdef double delta, max_delta

    c = np.zeros((n, n_features), dtype=np.float64)
    c_new = np.zeros((n, n_features), dtype=np.float64)
    w_ij = np.zeros((n_samples, n), dtype=np.float64)
    distances = np.zeros((n_samples, n), dtype=np.float64)

    for i in range(n):
        for k in range(n_features):
            c[i, k] = c_init[i, k]

    for iteration in range(100):
        _compute_distances_64(x, c, distances)
        _compute_weights_64(distances, m, w_ij)

        for i in range(n):
            for k in range(n_features):
                c_new[i, k] = 0.0

        _compute_new_centers_64(w_ij, x, m, c_new)

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

    _compute_distances_64(x, c, distances)
    _compute_weights_64(distances, m, w_ij)

    return np.asarray(c), np.asarray(w_ij)


def fuzzy_c_means_32(
    x,
    int n,
    m = 2.0,
    *,
    indices = None,
    initial_guess = None,
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
        indices_arr = np.random.choice(n_samples, size=n * 2, replace=False).astype(np.int64)
        indices_view = indices_arr
        _init_centers_32(x, n, indices_view, c_init)

    m_float = np.float32(m)
    return _fuzzy_c_means_kernel_32(x, n, m_float, c_init)


def fuzzy_c_means_64(
    x,
    int n,
    m = 2.0,
    *,
    indices = None,
    initial_guess = None,
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
        indices_arr = np.random.choice(n_samples, size=n * 2, replace=False).astype(np.int64)
        indices_view = indices_arr
        _init_centers_64(x, n, indices_view, c_init)

    m_double = np.float64(m)
    return _fuzzy_c_means_kernel_64(x, n, m_double, c_init)


def fuzzy_c_means(
    x,
    int n,
    m = 2.0,
    *,
    indices = None,
    initial_guess = None,
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
    x = np.asarray(x)

    if x.dtype == np.float32:
        return fuzzy_c_means_32(
            x, n, np.float32(m),
            indices=indices,
            initial_guess=initial_guess
        )
    elif x.dtype == np.float64:
        return fuzzy_c_means_64(
            x, n, np.float64(m),
            indices=indices,
            initial_guess=initial_guess
        )
    else:
        raise TypeError(
            f"Unsupported dtype {x.dtype}. Expected float32 or float64."
        )
