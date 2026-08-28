"""Non-Euclidean Relational Fuzzy c-Means (NERFCM).

Hathaway, R. J. and Bezdek, J. C. (1994), "NERF c-means: Non-Euclidean
relational fuzzy clustering," *Pattern Recognition* 27(3):429-437.

NERFCM computes a fuzzy partition directly from an (n, n) dissimilarity
matrix ``R`` -- it never falls back to a coordinate mean. That makes it the
geometry-consistent back end for :class:`~tribbleclustering.IVATMeans`: fed
the (squared) iVAT minimax matrix, it stays in the same minimax/ultrametric
space that the front end used to find the clusters in the first place (see
``docs/novel-niche.md`` and GitHub issue #54).

``R`` must hold **squared** dissimilarities
-------------------------------------------
The relational distance of object ``j`` to a fuzzy cluster with membership
column ``u_i`` is

    d_i(j) = (R v_i)_j - 0.5 * v_i^T R v_i,       v_i = u_i^m / sum(u_i^m)

This is the relational *dual* of the FCM objective (Hathaway, Davenport &
Bezdek, "Relational duals of the c-means clustering algorithms," *Pattern
Recognition* 22(2):205-212, 1989), and the duality holds only when ``R``
holds squared distances: ``d_i(j)`` is then exactly the squared Euclidean
distance from object ``j`` to the FCM centroid of cluster ``i``. Passing
``D`` where ``D ** 2`` is meant does not error -- it silently optimizes in
the flattened geometry of ``sqrt(R)`` instead (GitHub issue #89).

When beta-spread fires, and when it cannot
------------------------------------------
``d_i(j)`` can go negative when ``R`` is not of *negative type* -- Schoenberg's
condition for ``R`` to be realizable as the squared distances between n points
in R^(n-1). The beta-spread correction (Hathaway & Bezdek 1994, Sec. 3) then
adds a constant to every off-diagonal entry of ``R`` until the distances are
non-negative again.

Beta-spread is therefore a safeguard for inputs *outside* that class, and the
iVAT minimax matrix is not one of them. That matrix is the subdominant
ultrametric ``u(D)``, and ultrametrics have strict p-negative type for every
``p >= 0`` (Faver, Kochalski, Murugan, Verheggen, Wesson & Weston, "Roundness
properties of ultrametric spaces," *Glasgow Math. J.* 56(3):519-535, 2014).
Both ``u(D)`` and ``u(D) ** 2`` are admissible, so beta-spread provably never
fires on a minimax input -- the correction is inert on exactly the input this
module was added for. See
``tests/test_ivatmeans_refine.py::TestBetaSpread``.
"""

from typing import Optional

import numpy as np
from numpy import ndarray


def _cluster_relational_distances(r: ndarray, u: ndarray, m: float) -> ndarray:
    """Squared relational distance from every object to every cluster.

    ``r`` is (n, n), ``u`` is (n, n_clusters). Returns an (n, n_clusters)
    array of distances.
    """
    w = u**m
    v = w / np.sum(w, axis=0, keepdims=True)
    rv = r @ v
    quad = np.sum(v * rv, axis=0)
    return rv - 0.5 * quad[np.newaxis, :]


def _get_relational_weights(d: ndarray, m: float) -> ndarray:
    """FCM membership update from relational distances (mirrors fcm._get_weights)."""
    d = np.maximum(d, 0.0)
    d_to_ii = d[:, :, np.newaxis]
    d_to_all = d[:, np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 1.0 / np.sum((d_to_ii / d_to_all) ** (1.0 / (m - 1)), axis=2)
    u = np.where(np.isnan(u) | np.isinf(u), 0.0, u)

    # Objects that land exactly on a prototype (d == 0) get crisp membership
    # split evenly across the (usually singleton) set of zero-distance clusters.
    zero_mask = d <= 1e-12
    has_zero = zero_mask.any(axis=1)
    if np.any(has_zero):
        counts = zero_mask[has_zero].sum(axis=1, keepdims=True)
        u[has_zero] = zero_mask[has_zero] / counts
    return u


def relational_fuzzy_c_means(
    r: ndarray,
    n_clusters: int,
    m: float = 2.0,
    *,
    u_init: Optional[ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    beta_spread: bool = True,
    random_state: Optional[int] = None,
) -> tuple[ndarray, float]:
    """Run NERFCM on a dissimilarity matrix.

    :param r: Symmetric (n, n) matrix of **squared** dissimilarities -- the
        relational dual is defined on squared distances (see the module
        docstring). It need not be of negative type; ``beta_spread`` covers
        the case where it is not.
    :param n_clusters: Number of clusters.
    :param m: Fuzziness parameter, default 2.0. Must be > 1.
    :param u_init: Optional initial (n, n_clusters) membership matrix (columns
        need not be normalized; a hard 0/1 partition from an upstream cut is a
        natural choice).
    :param max_iter: Maximum number of iterations.
    :param tol: Convergence threshold on the largest membership change.
    :param beta_spread: Apply the Hathaway-Bezdek beta-spread correction when
        ``r`` induces negative relational distances -- i.e. when ``r`` is not
        of negative type. It is inert on any ultrametric/minimax input,
        including the iVAT matrix, which is of negative type at every power
        (see the module docstring).
    :return: Tuple of ``(u, beta)`` -- the converged (n, n_clusters) membership
        matrix (rows sum to 1) and the total beta-spread correction applied to
        ``r`` (0.0 if none was needed).
    """
    r = np.asarray(r, dtype=np.float64)
    n = r.shape[0]
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(
            f"r must be a square (n, n) dissimilarity matrix, got {r.shape}"
        )
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if m <= 1.0:
        raise ValueError("m must be greater than 1.0")

    if u_init is not None:
        u = np.array(u_init, dtype=np.float64, copy=True)
        if u.shape != (n, n_clusters):
            raise ValueError(
                f"u_init must have shape ({n}, {n_clusters}), got {u.shape}"
            )
        col_sums = np.sum(u, axis=1, keepdims=True)
        col_sums = np.where(col_sums == 0.0, 1.0, col_sums)
        u = u / col_sums
    else:
        rng = np.random.default_rng(random_state)
        u = rng.dirichlet(np.ones(n_clusters), size=n)

    r_work = np.array(r, copy=True)
    np.fill_diagonal(r_work, 0.0)
    beta_total = 0.0

    for _ in range(max_iter):
        d = _cluster_relational_distances(r_work, u, m)

        if beta_spread:
            min_d = float(d.min())
            if min_d < 0.0:
                beta = -2.0 * min_d
                beta_total += beta
                r_work = r_work + beta * (1.0 - np.eye(n))
                np.fill_diagonal(r_work, 0.0)
                d = _cluster_relational_distances(r_work, u, m)

        u_new = _get_relational_weights(d, m)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u = u_new

    return u, beta_total


def relational_out_of_sample_membership(
    r_new: ndarray,
    r_train: ndarray,
    u_train: ndarray,
    m: float = 2.0,
    beta: float = 0.0,
) -> ndarray:
    """Extend a fitted NERFCM partition to new, out-of-sample dissimilarities.

    Implements the Hathaway-Bezdek out-of-sample formula: a new object is
    never given a coordinate or added to ``r_train``, it is scored against the
    existing cluster weight vectors ``v_i`` derived from ``u_train``.

    :param r_new: (n_new, n_train) **squared** dissimilarities from each new
        object to every training object, in the same units/scale as
        ``r_train`` (for the iVAT/minimax use case, built via the
        single-linkage nearest-neighbor extension -- see
        :mod:`tribbleclustering.ivatmeans`).
    :param r_train: (n_train, n_train) training matrix of squared
        dissimilarities, as originally passed to
        :func:`relational_fuzzy_c_means` (*not* the beta-corrected working
        copy).
    :param u_train: (n_train, n_clusters) converged training membership matrix.
    :param m: Fuzziness parameter used when ``u_train`` was fit.
    :param beta: Beta-spread correction returned by
        :func:`relational_fuzzy_c_means`; applied here for consistency with the
        corrected matrix the training run actually converged against.
    :return: (n_new, n_clusters) membership matrix for the new objects.
    """
    r_new = np.asarray(r_new, dtype=np.float64)
    r_train = np.asarray(r_train, dtype=np.float64)
    n_train = r_train.shape[0]

    r_train_work = r_train
    if beta:
        r_train_work = r_train + beta * (1.0 - np.eye(n_train))
        np.fill_diagonal(r_train_work, 0.0)

    w = u_train**m
    v = w / np.sum(w, axis=0, keepdims=True)
    quad = np.sum(v * (r_train_work @ v), axis=0)

    r_new_work = r_new + beta if beta else r_new
    d = r_new_work @ v - 0.5 * quad[np.newaxis, :]
    return _get_relational_weights(d, m)
