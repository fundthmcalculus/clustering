from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray


@dataclass
class FuzzyCMeansResult:
    """Result of fuzzy c-means clustering."""

    cluster_centers_: ndarray
    membership_matrix_: ndarray
    n_iter_: int
    converged: bool

    def __iter__(self):
        """Support tuple unpacking for backward compatibility: c, w = result"""
        return iter((self.cluster_centers_, self.membership_matrix_))


def _j_w_c(x: np.ndarray, c: np.ndarray, m: float) -> float:
    """Compute the weighted sum of squared distances"""
    w_ij = _get_weights(c, m, x)
    # Compute squared distances using gram identity (same as in _get_weights)
    x_norm2 = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    c_norm2 = np.sum(c**2, axis=1)  # (k,)
    xc = 2 * np.dot(x, c.T)  # (n, k)
    dist2 = np.maximum(x_norm2 + c_norm2 - xc, 0)  # (n, k)
    j_wc = np.sum(w_ij**m * dist2, axis=None)
    return j_wc


def _get_weights(c: ndarray, m: float, x: ndarray) -> ndarray:
    # Compute squared distances using gram identity: ||x-c||^2 = ||x||^2 - 2*x*c^T + ||c||^2
    # ||x||^2: shape (n, 1)
    x_norm2 = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    # ||c||^2: shape (k,)
    c_norm2 = np.sum(c**2, axis=1)  # (k,)
    # -2*x*c^T: shape (n, k) via GEMM
    xc = 2 * np.dot(x, c.T)  # (n, k)

    # distances squared: (n, k)
    dist2 = x_norm2 + c_norm2 - xc
    # Ensure non-negative to avoid numerical issues
    dist2 = np.maximum(dist2, 0)
    distances = np.sqrt(dist2)

    # Compute weights: w_ij = d_ij^(-2/(m-1)) / Σ_l d_il^(-2/(m-1))
    # This avoids the (n,k,k) intermediate tensor
    exp = -2.0 / (m - 1)
    w_power = distances**exp  # (n, k)
    w_ij = w_power / np.sum(w_power, axis=1, keepdims=True)  # (n, k)

    w_ij = np.where(np.isnan(w_ij) | np.isinf(w_ij), 0.0, w_ij)
    return w_ij


def _get_v_ij(w_ij: ndarray, m: float, x: ndarray) -> ndarray:
    # Compute centers: v = (w^m)^T @ x / (w^m).sum(0)[:,None]
    # This avoids the (n,k,d) intermediate tensor by using GEMM
    w_m = w_ij**m  # (n, k)
    denominator = np.sum(w_m, axis=0)  # (k,)
    # GEMM: (k, n) @ (n, d) = (k, d)
    numerator = np.dot(w_m.T, x)  # (k, d)
    v_ij = numerator / denominator[:, np.newaxis]
    return v_ij


def fuzzy_c_means(
    x: np.ndarray,
    n: int,
    m: float = 2.0,
    *,
    max_iter: int = 100,
    indices: Optional[np.ndarray | list[int]] = None,
    initial_guess: Optional[np.ndarray] = None,
) -> FuzzyCMeansResult:
    """
    Compute the fuzzy c-means clustering algorithm.

    :param x: Input data points, shape (n_samples, n_features)
    :param n: Number of clusters
    :param m: Fuzziness parameter, default 2.0
    :param max_iter: Maximum number of iterations, default 100
    :param indices: Indices of initial cluster centers, if provided
    :param initial_guess: Initial cluster centers, if provided
    :return: FuzzyCMeansResult containing cluster_centers_, membership_matrix_,
        n_iter_ (actual iterations), and converged (boolean)
    """
    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")
    # 1. Create the candidate centers
    if indices is not None:
        c = x[indices, :]
    elif initial_guess is not None:
        if initial_guess.shape != (n, x.shape[1]):
            raise ValueError(
                f"initial_guess must have shape ({n}, {x.shape[1]}), "
                f"got {initial_guess.shape}"
            )
        c = initial_guess
    else:
        indices = np.random.choice(x.shape[0], size=n * 2, replace=False)
        c = x[indices, :]
        # Combine every two rows into one so no cluster center exactly matches a data-point
        c = c.reshape(n, 2, x.shape[1]).mean(axis=1)

    # Track convergence
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        w_ij = _get_weights(c, m, x)
        c_new = _get_v_ij(w_ij, m, x)
        n_iter = iteration + 1
        if np.allclose(c_new, c, rtol=1e-5, atol=1e-8):
            converged = True
            break
        c = c_new

    # Calculate final membership matrix
    w_ij = _get_weights(c, m, x)

    return FuzzyCMeansResult(
        cluster_centers_=c,
        membership_matrix_=w_ij,
        n_iter_=n_iter,
        converged=converged,
    )
