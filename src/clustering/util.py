import numpy as np
from numba import prange, njit


@njit(cache=True, parallel=True, nogil=True)
def pairwise_distances(data: np.ndarray) -> np.ndarray:
    dist_arr = np.zeros((data.shape[0], data.shape[0]), dtype=data.dtype)
    for i in prange(len(data)):
        for j in range(i+1,len(data)):
            dist_arr[i, j] = np.linalg.norm(data[i, :] - data[j, :])
            dist_arr[j, i] = dist_arr[i, j]
    return dist_arr