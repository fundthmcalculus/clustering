import numpy as np
from numba import prange, njit


@njit(cache=True, parallel=True, nogil=True)
def pairwise_distances(data: np.ndarray) -> np.ndarray:
    dist_arr = np.zeros((data.shape[0], data.shape[0]), dtype=data.dtype)
    for i in prange(len(data)):
        dist_arr[i, i + 1 :] = np.linalg.norm(data[i, :] - data[i + 1 :, :])
        dist_arr[i + 1 :, i] = dist_arr[i, i + 1 :]
    return dist_arr