from typing import Optional, Literal

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize


def _j_w_c(x: np.ndarray, c: np.ndarray, m: float) -> float:
    """Compute the weighted sum of squared distances"""
    w_ij = _get_weights(c, m, x)
    j_wc = np.sum(
        w_ij**m * np.sum((x[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2.0, axis=2),
        axis=None,
    )

    return j_wc


def _get_weights(c: ndarray, m: float, x: ndarray) -> ndarray:
    distances = np.linalg.norm(x[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
    distances_to_jj = distances[:, :, np.newaxis]
    distances_to_all = distances[:, np.newaxis, :]
    w_ij = 1.0 / np.sum((distances_to_jj / distances_to_all) ** (2.0 / (m - 1)), axis=2)
    w_ij = np.where(np.isnan(w_ij) | np.isinf(w_ij), 0.0, w_ij)
    return w_ij


def _get_v_ij(w_ij: ndarray, m: float, x: ndarray) -> ndarray:
    v_ij = (
        np.sum(w_ij[:, :, np.newaxis] ** m * (x[:, np.newaxis, :]), axis=0)
        / np.sum(w_ij**m, axis=0)[:, np.newaxis]
    )
    return v_ij


def fuzzy_c_means(
    x: np.ndarray,
    n: int,
    m: float = 2.0,
    *,
    method: Literal["gd", "iter"] = "iter",
    indices: Optional[np.ndarray | list[int]] = None,
    initial_guess: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the fuzzy c-means tribbleclustering algorithm.

    :param x: Input data points, shape (n_samples, n_features)
    :param n: Number of clusters
    :param m: Fuzziness parameter, default 2.0
    :param method: Clustering method, either 'gd' (gradient descent) or 'iter' (iterative), default 'iter'
    :param indices: Indices of initial cluster centers, if provided
    :param initial_guess: Initial cluster centers, if provided
    :return: Tuple of membership matrix (shape (n_samples, n_clusters)) and cluster centers (shape (n_clusters, n_features))
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

    # 2. Iteratively refine with a gradient descent method
    def optim_j_w_c(c_opt: np.ndarray) -> float:
        c_reshaped = c_opt.reshape(n, x.shape[1])
        return _j_w_c(x, c_reshaped, m)

    if method == "gd":
        result = minimize(optim_j_w_c, c.flatten(), method="BFGS")
        c = result.x.reshape(n, x.shape[1])
    elif method == "iter":
        # Max of 100 iterations
        for _ in range(100):
            w_ij = _get_weights(c, m, x)
            c_new = _get_v_ij(w_ij, m, x)
            if np.allclose(c_new, c, rtol=1e-5, atol=1e-8):
                break
            c = c_new
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'gd' or 'iter'.")

    # Calculate membership matrix
    w_ij = _get_weights(c, m, x)

    # 3. Return the center-points
    return c, w_ij
