import numpy as np
from numba import prange, njit


@njit(cache=True, parallel=True, nogil=True)
def pairwise_distances(data: np.ndarray, norm_only: bool = False) -> np.ndarray:
    is_1d: bool = data.shape[1] == 1
    if is_1d and not norm_only:
        # Vectorized computation for 1D case
        return np.abs(data.T - data)
    else:
        dist_arr = np.zeros((data.shape[0], data.shape[0]), dtype=data.dtype)
        for i in prange(len(data)):
            for j in range(i + 1, len(data)):
                dist_arr[i, j] = np.linalg.norm(data[i, :] - data[j, :])
                dist_arr[j, i] = dist_arr[i, j]
        return dist_arr


def _random_cities(
    center_x, center_y, n_cities: int = 10, cluster_diameter: float = 3.0
) -> np.ndarray:
    if n_cities == 1:
        return np.array([[center_x, center_y]])
    # Randomly distribute cities in a uniform circle?
    theta = np.linspace(0, 2 * np.pi, n_cities + 1, dtype=np.float32)
    theta = theta[:-1]
    # Add slight random scramble to locations
    scramble = np.random.uniform(
        -cluster_diameter * 0.05, cluster_diameter * 0.05, size=(n_cities, 2)
    )
    city_x = np.cos(theta) * cluster_diameter / 2.0 + center_x + scramble[:, 0]
    city_y = np.sin(theta) * cluster_diameter / 2.0 + center_y + scramble[:, 1]
    return np.c_[city_x, city_y]


def circle_random_clusters(
    n_clusters: int = 10,
    n_cities: int = 10,
    cluster_diameter: float = 2.0,
    cluster_spacing: float = 10.0,
) -> np.ndarray:
    city_locations = np.zeros(shape=(0, 2), dtype=np.float32)
    for theta in np.linspace(0, 2 * np.pi, n_clusters):
        theta *= n_clusters / (n_clusters + 1)
        cx = cluster_spacing * np.cos(theta)
        cy = cluster_spacing * np.sin(theta)
        city_locations = np.concatenate(
            (
                city_locations,
                _random_cities(
                    cx, cy, n_cities=n_cities, cluster_diameter=cluster_diameter
                ),
            ),
            axis=0,
        )
    return city_locations
