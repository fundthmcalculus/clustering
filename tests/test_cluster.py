from dataclasses import dataclass
import time
from typing import Any, Union

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d, QhullError

from clustering.util import pairwise_distances, circle_random_clusters, _random_cities
from src.clustering import vat_prim_mst_seq, compute_ivat, fcm


def _hierarchical_circle_clusters(
        clusters_per_level: list[int],
        diameters_per_level: list[float],
) -> np.ndarray:
    """
    Create hierarchical clusters arranged in circles around circles recursively.

    Args:
        clusters_per_level: Number of clusters at each hierarchical level (e.g., [3, 4, 5] means 3 top-level clusters, 
                           each containing 4 mid-level clusters, each containing 5 leaf clusters)
        diameters_per_level: Diameter for clusters at each level

    Returns:
        Array of point coordinates (N, 2)

    Raises:
        ValueError: If configuration would create more than 16000 points
    """
    if len(clusters_per_level) != len(diameters_per_level):
        raise ValueError("clusters_per_level and diameters_per_level must have the same length")

    # Calculate total number of points
    total_points = np.prod(clusters_per_level)

    if total_points > 16000:
        raise ValueError(
            f"Configuration would create {total_points} points, which exceeds the limit of 16000. "
            f"Reduce clusters_per_level, diameters_per_level, or points_per_leaf."
        )

    def _create_level(
            center_x: float,
            center_y: float,
            level_idx: int,
    ) -> np.ndarray:
        """Recursively create clusters at the current level"""
        n_clusters = clusters_per_level[level_idx]
        diameter = diameters_per_level[level_idx]
        # Last level gets a bit of noise
        if level_idx == len(clusters_per_level)-1:
            # Base case: create leaf points
            return _random_cities(
                center_x, center_y,
                n_cities=n_clusters,
                cluster_diameter=diameter
            )

        all_points = np.zeros(shape=(0, 2), dtype=np.float32)

        for theta in np.linspace(0, 2 * np.pi, n_clusters, endpoint=False):
            # Calculate position of sub-cluster center
            sub_cx = center_x + diameter * np.cos(theta)
            sub_cy = center_y + diameter * np.sin(theta)

            # Recursively create points for this sub-cluster
            sub_points = _create_level(sub_cx, sub_cy,level_idx + 1)

            all_points = np.concatenate((all_points, sub_points), axis=0)

        return all_points

    # Start recursion from the origin
    return _create_level(0.0, 0.0, 0)


def _test_cluster_sequencing():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    # 59 is letter recognition
    # 827 is sepsis survival (allocates 80+ GB RAM)
    # 148 is shuttle stat log (allocates 50 GB RAM)
    letter_recognition = fetch_ucirepo(id=59)

    # data (as pandas dataframes)
    X = np.array(letter_recognition.data.features)

    # metadata
    print(f"Metadata: {letter_recognition.metadata}")

    # variable information
    print(f"Variable Information: {letter_recognition.variables}")

    # Compute the pairwise distances
    t0 = time.time()
    ordered_matrix = vat_prim_mst_seq(X)
    t1 = time.time()

    print(f"Elapsed time for {len(X)} data points: {t1-t0:.02f}")


def test_merge_ivat():
    all_cities = circle_random_clusters(
        n_clusters=10, n_cities=5, cluster_spacing=5.0, cluster_diameter=1
    )
    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]
    matrix_of_pairwise_distance = pairwise_distances(all_cities)

    ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(matrix_of_pairwise_distance)
    plot_vat_ivat(ivat_mst, vat_mst)


def plot_vat_ivat(ivat_mst: np.ndarray, vat_mst: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    im1 = ax1.imshow(vat_mst, cmap="viridis")
    ax1.set_title("VAT Matrix")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ivat_mst, cmap="viridis")
    ax2.set_title("iVAT Matrix")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.show()


def test_fcm_with_center_on_datapoint():
    """Test FCM behavior when a cluster center coincides with a data point"""
    # Create 5 points on a line: (1,0), (3,0), (5,0), (7,0), (9,0)
    data_points = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0], [9.0, 0.0]])

    # Run FCM with 2 clusters
    n_clusters = 2
    for idx0 in range(len(data_points)):
        for idx1 in range(idx0, len(data_points)):
            cluster_centers, membership_weights = fcm.fuzzy_c_means(
                data_points, n_clusters, m=2.0, indices=[idx0, idx1]
            )

            # Verify that we got 2 cluster centers
            assert cluster_centers.shape == (
                n_clusters,
                2,
            ), f"Expected shape {(n_clusters, 2)}, got {cluster_centers.shape}"

            # Verify that membership weights sum to 1 for each data point
            membership_sums = np.sum(membership_weights, axis=1)
            # Verify that membership weights sum to 1 for each data point
            # (or 0 for duplicate points where cluster center coincides with data point)
            membership_sums = np.sum(membership_weights, axis=1)
            expected_values = np.where(
                membership_sums > 0.5, 1.0, 0.0
            )  # Expect 1.0 or 0.0
            np.testing.assert_array_almost_equal(
                membership_sums,
                expected_values,
                err_msg=f"Membership weights should sum to 1 for each data point (or 0 for duplicates): {idx0, idx1}",
            )

            # Verify all membership weights are between 0 and 1
            assert np.all(membership_weights >= 0) and np.all(
                membership_weights <= 1
            ), "All membership weights should be between 0 and 1"


def test_heirarchy_ivat_means():
    """Test hierarchical circle clusters with iVAT and FCM"""
    # Example: 3 top-level clusters, each with 4 mid-level, each with 5 leaf clusters (3*4*5*10 = 600 points)
    all_cities = _hierarchical_circle_clusters(
        clusters_per_level=[3, 4, 5],
        diameters_per_level=[15.0, 5.0, 1.0]
    )

    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]

    print(f"Created {len(all_cities)} hierarchical points")

    # Compute pairwise distances and iVAT
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(matrix_of_pairwise_distance)

    # Get cluster information from iVAT
    res = _get_ivat_means(all_cities, ivat_mst, vat_order, n_levels=3)
    # Run FCM with iVAT-derived initial guess
    n_clusters = len(res[0].initial_centroids)
    meth_c, w_c = fcm.fuzzy_c_means(all_cities, n_clusters, 2, initial_guess=res[0].initial_centroids)

    print(f"Detected {n_clusters} clusters using iVAT")

    # Visualize results
    plot_vat_ivat(ivat_mst, vat_mst)
    ivat_results = [res] if isinstance(res, IvatMeansResult) else res
    for r in ivat_results:
        # plot_membership(all_cities, r.cluster_city_ids, meth_c, w_c)
        plot_diagonal(
            r.diagonal_values,
            [r.max_diff_index],
            r.peak_threshold,
            r.sorted_diagonal,
            r.abrupt_change_indices,
        )
        plot_voronoi(all_cities, r.initial_centroids)
    plt.show()


def plot_voronoi(all_cities, centroids):
    try:
        v = Voronoi(centroids)
        fig = voronoi_plot_2d(v)
        fig.axes[0].set_title("Voronoi plot")
        fig.axes[0].scatter(all_cities[:, 0], all_cities[:, 1])
        fig.show()
    except QhullError:
        pass


def test_multi_dim_pairwise_dist_perf():
    results = []
    # Do 1 pairwise distances to reduce nogil/numba randomness
    pairwise_distances(np.zeros((100, 3)))

    dims = [1, 2]
    sizes = [1000, 2000, 3000, 5000, 8000, 10000, 20000]
    # sizes = [1000, 2000]
    for dim in dims:
        for size in sizes:
            data = np.random.rand(size, dim)
            start = time.time()
            pairwise_distances(data)
            end = time.time()
            elapsed = end - start
            results.append((dim, size, elapsed))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group results by dimension
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    for dim, color in zip(dims, colors):
        dim_results = [(size, time) for d, size, time in results if d == dim]
        sizes, times = zip(*dim_results)
        ax.plot(sizes, times, marker='o', label=f'Dim={dim}', color=color, linewidth=2)

    ax.set_xlabel('Data Size', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Pairwise Distance Computation Performance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_fuzzy_c_means():
    n_total: int = 256
    n_clusters: int = 16
    n_cities: int = n_total // n_clusters
    all_cities = circle_random_clusters(
        n_clusters=n_clusters, n_cities=n_cities, cluster_spacing=5, cluster_diameter=0.5
    )
    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]

    # Time the elbow method (multiple FCM calls with varying cluster counts)
    start_elbow = time.time()
    elbow_results = []
    cluster_range = range(2, n_clusters + 1)
    for k in cluster_range:
        centers, weights = fcm.fuzzy_c_means(all_cities, k, 2)
        elbow_results.append((k, centers, weights))
    end_elbow = time.time()
    elbow_time = end_elbow - start_elbow

    start_ivat = time.time()
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # Compute the IVAT
    ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(matrix_of_pairwise_distance)
    res = _get_ivat_means(all_cities, ivat_mst, vat_order)

    # Time the single FCM call
    start_single = time.time()
    meth_c, w_c = fcm.fuzzy_c_means(all_cities, n_clusters, 2, initial_guess=res.initial_centroids)
    mid_single = time.time()
    _, _ = fcm.fuzzy_c_means(all_cities, n_clusters, 2)
    end_single = time.time()
    _, _ = fcm.fuzzy_c_means(all_cities, n_clusters, 2, method='gd')
    end_gd = time.time()
    smart_fcm_time = mid_single - start_single
    single_fcm_time = end_single - mid_single
    iter_fcm_time = end_gd - end_single
    single_ivat_time = start_single - start_ivat

    # Print performance comparison
    print(f"\n{'=' * 60}")
    print(f"Performance Comparison:")
    print(f"{'=' * 60}")
    print(f"Elbow Method (n=2 to {n_clusters}): {elbow_time:.4f} seconds")
    print(f"Single iter-FCM (n={n_clusters}):     {single_fcm_time:.4f} seconds")
    print(f"Single GD-FCM (n={n_clusters}):     {iter_fcm_time:.4f} seconds")
    print(f"Smart FCM (n={n_clusters}):     {smart_fcm_time:.4f} seconds")
    print(f"IVAT (n={n_clusters}):           {single_ivat_time:.4f} seconds")
    print(f"Time difference:          {elbow_time - single_ivat_time:.4f} seconds")
    print(f"Elbow method is {elbow_time/single_ivat_time:.2f}x slower")
    print(f"{'='*60}\n")

    # Assert that every city has been allocated to a cluster
    all_allocated_cities = np.sort(np.concatenate(res.cluster_city_ids))
    # print(f"All cities:\n{np.r_[0:len(all_cities)]}")
    # print(f"Allocated Cities:\n{all_allocated_cities}")
    assert len(all_allocated_cities) == len(
        all_cities
    ), f"Not all cities allocated: {len(all_allocated_cities)} allocated out of {len(all_cities)} total"
    assert len(np.unique(all_allocated_cities)) == len(
        all_cities
    ), f"Duplicate city allocations detected"

    plot_vat_ivat(ivat_mst, vat_mst)

    plot_diagonal(
        res.diagonal_values,
        [res.max_diff_index],
        res.peak_threshold,
        res.sorted_diagonal,
        res.abrupt_change_indices,
    )

    plot_membership(all_cities, res.cluster_city_ids, meth_c, w_c)
    plt.show()


@dataclass
class IvatMeansResult:
    abrupt_change_indices: ndarray
    cluster_city_ids: list[ndarray]
    diagonal_values: ndarray
    initial_centroids: ndarray
    max_diff_index: int
    peak_threshold: float
    sorted_diagonal: ndarray


def _get_ivat_means(all_cities: ndarray, ivat_mst: ndarray, vat_order: ndarray, n_levels: int = 1) -> Union[IvatMeansResult, list[IvatMeansResult]]:
    # Look down the off-by-1 diagonal and count the number of substantial changes.
    diagonal_values = np.diag(ivat_mst, k=1)
    # Augment back to original size, just prepend the initial value to avoid throwing off the diff fcn
    # Expand this to the original size for convenience.
    diagonal_values = np.concatenate(
        [np.array([diagonal_values[0]]), diagonal_values], axis=0
    )
    # Sort the diagonal values
    sorted_diagonal = np.sort(diagonal_values)
    # Find the maximum difference and the index thereof
    diagonal_diffs = np.diff(sorted_diagonal)
    max_diff_indices = _arg_max(diagonal_diffs, n_levels)
    peaks_threshold = sorted_diagonal[max_diff_indices + 1]

    # Sort peaks_threshold in decreasing order and reorder max_diff_indices accordingly
    sort_order = np.argsort(peaks_threshold)[::-1]
    peaks_threshold = peaks_threshold[sort_order]
    max_diff_indices = max_diff_indices[sort_order]

    results = []
    for index, peak_th in enumerate(peaks_threshold):
        # Prevent weird floating-point comparisons.
        abrupt_change_idx = np.where(diagonal_values >= 0.99*peak_th)[0]

        # Use each section as a cluster endpoint, inclusive.
        cluster_group = np.concatenate([np.array([0]), abrupt_change_idx, np.array([len(all_cities)])])
        cluster_city_indexs = []
        for idx in range(0, len(cluster_group) - 1):
            cg_start = cluster_group[idx]
            cg_end = cluster_group[idx + 1]
            # Use the VAT order to pick out the cities in each cluster
            cluster_city_indexs.append(vat_order[cg_start:cg_end])

        # Compute the initial guess as the centroid of each city cluster
        initial_centroids_item = np.array([
            np.mean(all_cities[cluster_ids], axis=0)
            for cluster_ids in cluster_city_indexs
        ])

        results.append(IvatMeansResult(
            abrupt_change_indices=abrupt_change_idx,
            cluster_city_ids=cluster_city_indexs,
            diagonal_values=diagonal_values,
            initial_centroids=initial_centroids_item,
            max_diff_index=int(max_diff_indices[index]),
            peak_threshold=float(peak_th),
            sorted_diagonal=sorted_diagonal
        ))

    if n_levels == 1:
        return results[0]
    return results


def get_ivat_means(all_cities: ndarray, ivat_mst: ndarray, vat_order: ndarray) -> tuple[ndarray, list[Any], ndarray]:
    res = _get_ivat_means(all_cities, ivat_mst, vat_order)
    return [res.initial_centroids], [res.cluster_city_ids], np.array([res.peak_threshold])


def _arg_max(a: ndarray, n: int = 1) -> ndarray:
    """Get the indexes of the n-largest values in the array."""
    if n >= len(a):
        return np.argsort(a)[::-1]
    # Use argpartition to find the n largest elements efficiently
    partitioned_indices = np.argpartition(a, -n)[-n:]
    # Sort these indices by their corresponding values in descending order
    sorted_indices = partitioned_indices[np.argsort(a[partitioned_indices])[::-1]]
    return sorted_indices


def plot_membership(all_cities: ndarray, cluster_city_ids: list[Any],
                    meth_c: ndarray, w_c: ndarray):
    # Create a color map for clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, meth_c.shape[0]))

    # Create plot
    fig, ax = plt.subplots()

    # Plot each point with blended color based on membership weights
    for i in range(all_cities.shape[0]):
        # Blend colors based on membership weights
        blended_color = np.zeros(4)  # RGBA
        for j in range(meth_c.shape[0]):
            blended_color += w_c[i, j] * colors[j]

        blended_color /= blended_color.max()

        ax.scatter(
            all_cities[i, 0],
            all_cities[i, 1],
            c=[blended_color],
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # Plot cluster city IDs with "*" markers
    ivat_centers = []
    for idx, cluster_ids in enumerate(cluster_city_ids):
        cluster_points = all_cities[cluster_ids]
        cluster_color = colors[idx % len(colors)]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            marker="*",
            edgecolors=cluster_color,
            facecolors="none",
            label=f"Cluster {idx}",
        )
        center = np.mean(cluster_points, axis=0)
        ivat_centers.append(center)
    ivat_centers = np.array(ivat_centers)

    # Plot ivat cluster centers
    ax.scatter(
        ivat_centers[:, 0],
        ivat_centers[:, 1],
        c="red",
        s=150,
        marker="D",
        edgecolors="white",
        label="iVAT Cluster Centers",
    )

    # Plot cluster centers
    ax.scatter(
        meth_c[:, 0],
        meth_c[:, 1],
        c="black",
        s=150,
        marker="X",
        edgecolors="white",
        label="FCM Cluster Centers",
    )

    ax.set_title("Fuzzy C-Means Clustering with Membership-based Colors")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.legend()
    plt.tight_layout()


def plot_diagonal(
    diagonal_values: ndarray,
    max_diff_indices: list[int],
    peaks_threshold: float,
    sorted_diagonal: ndarray,
    abrupt_change_indices: ndarray,
) -> ndarray:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(diagonal_values, marker="o")
    ax1.set_title("Off-by-One Diagonal of iVAT Matrix")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Distance Value")
    ax1.grid(True)

    ax2.plot(sorted_diagonal, marker="o")
    for idx in max_diff_indices:
        ax2.axvline(
            x=idx,
            color="r",
            linestyle="--",
            label=f"Max diff at index {idx}",
        )
    ax2.legend()
    ax2.set_title("Sorted Off-by-One Diagonal of iVAT Matrix")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Distance Value")
    ax2.grid(True)
    plt.tight_layout()

    # Count abrupt size changes using a basic stats test
    ax1.axhline(
        y=peaks_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold: {peaks_threshold:.2f}",
    )
    ax2.text(
        0.02,
        0.98,
        f"Abrupt changes: {len(abrupt_change_indices)}, threshold: {peaks_threshold:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
