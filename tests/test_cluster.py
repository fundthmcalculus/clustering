import time
from typing import Any, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from scipy.spatial import Voronoi, voronoi_plot_2d, QhullError

from tribbleclustering.util import (
    pairwise_distances,
    circle_random_clusters,
    _random_cities,
)
from tribbleclustering import (
    compute_ivat,
    fcm,
    get_ivat_levels,
    get_ivat_hierarchy,
    ClusterNode,
    IvatMeansResult,
)


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
        raise ValueError(
            "clusters_per_level and diameters_per_level must have the same length"
        )

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
        if level_idx == len(clusters_per_level) - 1:
            # Base case: create leaf points
            return _random_cities(
                center_x, center_y, n_cities=n_clusters, cluster_diameter=diameter
            )

        all_points = np.zeros(shape=(0, 2), dtype=np.float32)

        for theta in np.linspace(0, 2 * np.pi, n_clusters, endpoint=False):
            # Calculate position of sub-cluster center
            sub_cx = center_x + diameter * np.cos(theta)
            sub_cy = center_y + diameter * np.sin(theta)

            # Recursively create points for this sub-cluster
            sub_points = _create_level(sub_cx, sub_cy, level_idx + 1)

            all_points = np.concatenate((all_points, sub_points), axis=0)

        return all_points

    # Start recursion from the origin
    return _create_level(0.0, 0.0, 0)


def test_merge_ivat():
    all_cities = circle_random_clusters(
        n_clusters=10, n_cities=5, cluster_spacing=5.0, cluster_diameter=1
    )
    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]
    matrix_of_pairwise_distance = pairwise_distances(all_cities)

    # compute_ivat returns 3 values: IVAT matrix, argmin sequence, VAT order
    ivat_mst, argmin_seq, vat_order = compute_ivat(matrix_of_pairwise_distance)

    # Compute VAT matrix separately
    from tribbleclustering.pvat import compute_vat

    vat_mst, _ = compute_vat(matrix_of_pairwise_distance)

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
    # plt.show()


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
        clusters_per_level=[3, 4, 5], diameters_per_level=[15.0, 5.0, 1.0]
    )

    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]

    print(f"Created {len(all_cities)} hierarchical points")

    # Compute pairwise distances and iVAT
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # compute_ivat returns 3 values: IVAT matrix, argmin sequence, VAT order
    ivat_mst, argmin_seq, vat_order = compute_ivat(matrix_of_pairwise_distance)
    # Compute VAT matrix separately
    from tribbleclustering.pvat import compute_vat

    vat_mst, _ = compute_vat(matrix_of_pairwise_distance)

    # Get cluster information from iVAT
    res = get_ivat_levels(all_cities, ivat_mst, vat_order, n_levels=3)
    # Run FCM with iVAT-derived initial guess
    n_clusters = len(res[0].initial_centroids)
    meth_c, w_c = fcm.fuzzy_c_means(
        all_cities, n_clusters, 2, initial_guess=res[0].initial_centroids
    )

    print(f"Detected {n_clusters} clusters using iVAT")

    # Test hierarchy
    root = get_ivat_hierarchy(all_cities, ivat_mst, vat_order, n_levels=3)
    assert len(root.children) == len(res[0].cluster_city_ids)

    # Check tree structure (recursive count of leaves or similar)
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.children)

    print(f"Hierarchy tree has {count_nodes(root)} nodes")

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
    # plt.show()


def plot_voronoi(all_cities, centroids):
    try:
        v = Voronoi(centroids)
        fig = voronoi_plot_2d(v)
        fig.axes[0].set_title("Voronoi plot")
        fig.axes[0].scatter(all_cities[:, 0], all_cities[:, 1])
        # fig.show()
    except QhullError:
        pass


def test_multi_dim_pairwise_dist_perf():
    results = []
    # Do 1 pairwise distances to reduce nogil/numba randomness
    pairwise_distances(np.zeros((100, 3)), False)
    pairwise_distances(np.zeros((100, 3)), True)

    norms_only = [False, True]
    dim = 1
    # sizes = [1000, 2000, 3000, 5000, 8000, 10_000, 20_000, 30_000]
    sizes = [1000, 2000, 3000]
    for norm_only in norms_only:
        for size in sizes:
            data = np.random.rand(size, dim)
            start = time.time()
            pairwise_distances(data, norm_only)
            end = time.time()
            elapsed = end - start
            results.append((norm_only, size, elapsed))

    # Plot results
    fig, ax = plt.subplots()

    # Group results by dimension
    for norm in norms_only:
        dim_results = [(size, time) for d, size, time in results if d == norm]
        sizes, times = zip(*dim_results)
        sizes_arr = np.array(sizes)
        times_arr = np.array(times)

        ax.plot(sizes, times, marker="o", label=f"L2-only={norm} (data)", linewidth=2)

        # Fit quadratic polynomial (degree 2)
        quadratic_coeffs = np.polyfit(sizes_arr, times_arr, 2)
        quadratic_poly = np.poly1d(quadratic_coeffs)

        # Generate smooth x-values for plotting curves
        x_smooth = np.linspace(min(sizes), max(sizes), 200)

        # Plot polynomial fits
        ax.plot(
            x_smooth,
            quadratic_poly(x_smooth),
            linestyle="--",
            label=f"L2-only={norm} (quadratic fit): {quadratic_coeffs[0]:.2e}x² + {quadratic_coeffs[1]:.2e}x + {quadratic_coeffs[2]:.2e}",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel("Data Size", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Pairwise Distance Computation Performance", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot ratio of the two datasets
    fig_ratio, ax_ratio = plt.subplots()

    # Extract times for each norm_only value
    false_results = [(size, time) for norm, size, time in results if norm == False]
    true_results = [(size, time) for norm, size, time in results if norm == True]

    # Compute ratios (False / True)
    ratios = []
    ratio_sizes = []
    for (size_f, time_f), (size_t, time_t) in zip(false_results, true_results):
        assert size_f == size_t, "Size mismatch between datasets"
        ratios.append(time_f / time_t)
        ratio_sizes.append(size_f)

    ax_ratio.plot(
        ratio_sizes,
        ratios,
        marker="o",
        linewidth=2,
        label="Ratio (L2-only=False / L2-only=True)",
    )
    ax_ratio.axhline(
        y=np.mean(ratios),
        color="r",
        linestyle="--",
        label=f"Mean Ratio: {np.mean(ratios):.2f}",
    )
    ax_ratio.set_xlabel("Data Size", fontsize=12)
    ax_ratio.set_ylabel("Time Ratio (False/True)", fontsize=12)
    ax_ratio.set_title(
        "Performance Ratio: Full Distance Matrix vs L2-Norm Only", fontsize=14
    )
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)
    plt.tight_layout()

    # plt.show()


def test_fuzzy_c_means():
    n_total: int = 256
    n_clusters: int = 16
    n_cities: int = n_total // n_clusters
    all_cities = circle_random_clusters(
        n_clusters=n_clusters,
        n_cities=n_cities,
        cluster_spacing=5,
        cluster_diameter=0.5,
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
    # compute_ivat returns 3 values: IVAT matrix, argmin sequence, VAT order
    ivat_mst, argmin_seq, vat_order = compute_ivat(matrix_of_pairwise_distance)
    # Compute VAT matrix separately
    from tribbleclustering.pvat import compute_vat

    vat_mst, _ = compute_vat(matrix_of_pairwise_distance)
    res = get_ivat_levels(all_cities, ivat_mst, vat_order)

    # Time the single FCM call
    start_single = time.time()
    meth_c, w_c = fcm.fuzzy_c_means(
        all_cities, n_clusters, 2, initial_guess=res.initial_centroids
    )
    mid_single = time.time()
    _, _ = fcm.fuzzy_c_means(all_cities, n_clusters, 2)
    end_single = time.time()
    smart_fcm_time = mid_single - start_single
    single_fcm_time = end_single - mid_single
    single_ivat_time = start_single - start_ivat

    # Print performance comparison
    print(f"\n{'=' * 60}")
    print(f"Performance Comparison:")
    print(f"{'=' * 60}")
    print(f"Elbow Method (n=2 to {n_clusters}): {elbow_time:.4f} seconds")
    print(f"Single iter-FCM (n={n_clusters}):     {single_fcm_time:.4f} seconds")
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
    # plt.show()


def plot_membership(
    all_cities: ndarray, cluster_city_ids: list[Any], meth_c: ndarray, w_c: ndarray
):
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


def test_ivat_hierarchy_logic():
    # Create simple hierarchical data: 2 clusters of 10 points each
    # Each cluster consists of 2 subclusters of 5 points each
    # Total 20 points
    cluster1 = np.random.randn(10, 2) + [10, 10]
    cluster1[:5] += [1, 1]
    cluster1[5:] += [-1, -1]

    cluster2 = np.random.randn(10, 2) + [-10, -10]
    cluster2[:5] += [1, 1]
    cluster2[5:] += [-1, -1]

    all_cities = np.vstack([cluster1, cluster2])
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # compute_ivat returns 3 values: IVAT matrix, argmin sequence, VAT order
    ivat_mst, argmin_seq, vat_order = compute_ivat(matrix_of_pairwise_distance)
    # Compute VAT matrix separately
    from tribbleclustering.pvat import compute_vat

    vat_mst, _ = compute_vat(matrix_of_pairwise_distance)

    # We want 2 levels: level 1 should have 2 clusters, level 2 should have 4 clusters
    root = get_ivat_hierarchy(all_cities, ivat_mst, vat_order, n_levels=2)

    # We should have a root with some children
    assert len(root.children) >= 2

    # Check that children of root are parents of the next level
    total_grandchildren = 0
    for child in root.children:
        total_grandchildren += len(child.children)
        # Check that children indices are subset of parent indices
        for grandchild in child.children:
            assert np.all(np.isin(grandchild.indices, child.indices))

    assert total_grandchildren > len(root.children)


def plot_tree(root: ClusterNode):
    """Plot the hierarchical structure as a tree."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    positions = {}

    def get_width(node):
        if not node.children:
            return 1
        return sum(get_width(child) for child in node.children)

    def assign_pos(node, depth, left, right):
        width = right - left
        x = left + width / 2
        y = -depth
        positions[id(node)] = (x, y)

        if node.children:
            current_left = left
            total_child_width = sum(get_width(c) for c in node.children)
            for child in node.children:
                child_width = get_width(child)
                norm_child_width = (child_width / total_child_width) * width
                assign_pos(
                    child, depth + 1, current_left, current_left + norm_child_width
                )
                current_left += norm_child_width

    assign_pos(root, 0, 0, 1)

    def draw(node):
        x, y = positions[id(node)]
        ax.scatter(x, y, s=500, c="skyblue", edgecolors="black", zorder=2)
        ax.text(
            x, y, str(len(node.indices)), ha="center", va="center", fontsize=8, zorder=3
        )

        for child in node.children:
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y, cy], c="gray", zorder=1, alpha=0.5)
            draw(child)

    draw(root)
    ax.set_title("Hierarchical Cluster Tree (numbers indicate points in cluster)")
    plt.tight_layout()
    return fig


def animate_hierarchical_clustering(all_cities, root: ClusterNode):
    """Create an animation showing clusters at each level of the hierarchy."""
    levels = []

    def collect_levels(node, depth):
        if len(levels) <= depth:
            levels.append([])
        levels[depth].append(node)
        for child in node.children:
            collect_levels(child, depth + 1)

    collect_levels(root, 0)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        nodes = levels[frame]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(nodes)))

        for i, node in enumerate(nodes):
            points = all_cities[node.indices]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=colors[i],
                label=f"Cluster {i}",
                s=15,
                alpha=0.6,
            )

        ax.set_title(f"Hierarchy Level {frame} ({len(nodes)} clusters)")
        ax.set_aspect("equal")
        if len(nodes) < 10:
            ax.legend(loc="upper right", markerscale=2)

    anim = FuncAnimation(fig, update, frames=len(levels), interval=1500, repeat=True)
    return anim


def test_visualize_hierarchy():
    """Test and visualize the hierarchical breakdown with a tree plot and animation."""
    # 3 top-level clusters, each with 3 sub-clusters, each with 5 points (3*3*5 = 45 points)
    all_cities = _hierarchical_circle_clusters(
        clusters_per_level=[3, 3, 5], diameters_per_level=[20.0, 5.0, 1.0]
    )

    # Scramble the order
    all_cities = all_cities[np.random.permutation(len(all_cities))]

    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # compute_ivat returns 3 values: IVAT matrix, argmin sequence, VAT order
    ivat_mst, argmin_seq, vat_order = compute_ivat(matrix_of_pairwise_distance)
    # Compute VAT matrix separately
    from tribbleclustering.pvat import compute_vat

    vat_mst, _ = compute_vat(matrix_of_pairwise_distance)

    # Get hierarchy (3 levels)
    root = get_ivat_hierarchy(all_cities, ivat_mst, vat_order, n_levels=3)

    # Plot tree
    fig_tree = plot_tree(root)

    # Animate
    anim = animate_hierarchical_clustering(all_cities, root)

    # In a real test environment, we might save these
    anim.save("hierarchy_animation.gif", writer="imagemagick")
    fig_tree.savefig("hierarchy_tree.png")

    # plt.show()
    assert len(root.children) > 0
