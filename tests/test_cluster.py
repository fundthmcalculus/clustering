import time
from typing import Any

import numpy as np

from clustering.util import pairwise_distances
from src.clustering import vat_prim_mst_seq, compute_ivat, fcm
from matplotlib import pyplot as plt
from numpy import ndarray, dtype


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


def _circle_random_clusters(
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
    all_cities = _circle_random_clusters(
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


def test_fuzzy_c_means():
    n_total: int = 1024
    n_clusters: int = 32
    n_cities: int = n_total // n_clusters
    all_cities = _circle_random_clusters(
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
    max_diff_index = np.argmax(diagonal_diffs)
    peaks_threshold = sorted_diagonal[max_diff_index + 1]
    abrupt_change_indices = np.where(diagonal_values >= peaks_threshold)[0]

    # Use each section as a cluster endpoint, inclusive.
    cluster_groups = np.concatenate(
        [np.array([0]), abrupt_change_indices, np.array([len(all_cities)])]
    )
    cluster_city_ids = []
    for idx in range(0, len(cluster_groups) - 1):
        cg_start = cluster_groups[idx]
        cg_end = cluster_groups[idx + 1]
        # Use the VAT order to pick out the cities in each cluster
        cluster_city_ids.append(vat_order[cg_start:cg_end])

    # Compute the initial guess as the centroid of each city cluster
    initial_centroids = np.array([
        np.mean(all_cities[cluster_ids], axis=0)
        for cluster_ids in cluster_city_ids
    ])

    # Time the single FCM call
    start_single = time.time()
    meth_c, w_c = fcm.fuzzy_c_means(all_cities, n_clusters, 2, initial_guess=initial_centroids)
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
    all_allocated_cities = np.concatenate(cluster_city_ids)
    all_allocated_cities = np.sort(all_allocated_cities)
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
        diagonal_values,
        max_diff_index,
        peaks_threshold,
        sorted_diagonal,
        abrupt_change_indices,
    )

    plot_membership(all_cities, cluster_city_ids, meth_c, w_c)
    plt.show()


def plot_membership(all_cities: ndarray[tuple[Any, ...], dtype[Any]], cluster_city_ids: list[Any],
                    meth_c: ndarray[tuple[Any, ...], dtype[Any]], w_c: ndarray[tuple[Any, ...], dtype[Any]]):
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
    max_diff_index: int,
    peaks_threshold,
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
    ax2.axvline(
        x=max_diff_index,
        color="r",
        linestyle="--",
        label=f"Max diff at index {max_diff_index}",
    )
    ax2.plot(
        [max_diff_index, max_diff_index + 1],
        [sorted_diagonal[max_diff_index], sorted_diagonal[max_diff_index + 1]],
        "ro-",
        linewidth=3,
        markersize=8,
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

    return abrupt_change_indices
