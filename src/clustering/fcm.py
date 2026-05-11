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
    return w_ij


def fuzzy_c_means(
    x: np.ndarray, n: int, m: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the fuzzy c-means"""
    # 1. Create the candidate centers
    indices = np.random.choice(x.shape[0], size=n * 2, replace=False)
    c = x[indices, :]
    # Combine every two rows into one so no cluster center exactly matches a data-point
    c = c.reshape(n, 2, x.shape[1]).mean(axis=1)

    # 2. Iteratively refine with a gradient descent method
    def optim_j_w_c(c_opt: np.ndarray) -> float:
        c_reshaped = c_opt.reshape(n, x.shape[1])
        return _j_w_c(x, c_reshaped, m)

    result = minimize(optim_j_w_c, c.flatten(), method="BFGS")
    c = result.x.reshape(n, x.shape[1])

    # Calculate membership matrix
    w_ij = _get_weights(c, m, x)

    # 3. Return the center-points
    return c, w_ij
