import time

import numpy as np
from cluster import compute_ordered_dis_njit_merge, vat_prim_mst_seq, compute_ivat, fcm
from matplotlib import pyplot as plt
from numpy import ndarray
from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances
from test_combinatorics import circle_random_clusters


def test_cluster_sequencing():
    print("\n")
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


def test_vat_scaling():
    city_count: list[int] = []
    merge_time: list[float] = []
    lib_time: list[float] = []
    o1 = 7
    o2 = 10
    n = 2 * (o2 - o1 + 1)
    for group_count in np.logspace(o1, o2, n, base=2, dtype="int"):
        city_count.append(group_count)
        # print(f"City count: {group_count}")
        all_cities = circle_random_clusters(n_clusters=group_count, n_cities=1)
        matrix_of_pairwise_distance = pairwise_distances(all_cities)
        # Scramble the order to ensure we sort it!
        cols = np.arange(len(all_cities), dtype="int")
        rand_col_order = np.random.permutation(cols)
        matrix_of_pairwise_distance = matrix_of_pairwise_distance[:, rand_col_order][
            rand_col_order, :
        ]
        # Cluster using our VAT
        t0 = time.time()
        ordered_matrix2, path_merge, path_ivat = compute_ordered_dis_njit_merge(
            matrix_of_pairwise_distance, inplace=False
        )
        t1 = time.time()
        # Cluster using the library VAT
        ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance.copy())
        # ordered_matrix = 0 * ordered_matrix2
        t2 = time.time()

        # Print the results
        merge_time.append(t1 - t0)
        lib_time.append(t2 - t1)

    # Chop the smallest entry off - that has the njit compilation time baked in.
    city_count = city_count[1:]
    merge_time = merge_time[1:]
    lib_time = lib_time[1:]

    # Prepare data for regression
    city_array = np.array(city_count)
    merge_array = np.array(merge_time)
    lib_array = np.array(lib_time)
    city_count_scl = city_array[:] / city_array[0]
    merge_count_scl = merge_array[:] / merge_array[0]
    lib_count_scl = lib_array[:] / lib_array[0]

    # Plot scaling variance
    plt.figure()
    plt.plot(city_count_scl, merge_count_scl, "o-", label="Merge VAT")
    plt.plot(city_count_scl, lib_count_scl, "o-", label="Lib VAT")
    plt.plot(city_count_scl, city_count_scl**2, "-", label="$O(N)=N^2$")
    plt.plot(
        city_count_scl,
        city_count_scl**2 * np.log(city_count_scl + 1),
        "-",
        label="$O(N)=N^2*log(N)$",
    )
    plt.plot(city_count_scl, city_count_scl**3, "-", label="$O(N)=N^3$")
    plt.xlabel("N Scaling")
    plt.ylabel("Time Scaling")
    plt.legend()
    plt.title("VAT Scaling Test")
    # plt.savefig('vat_scaling_comparison.eps', format='eps')
    # plt.close()

    plt.figure()
    plt.loglog(city_count, merge_time, "o", label="Merge VAT")
    plt.loglog(city_count, lib_time, "o", label="Lib VAT")
    plt.xlabel("Number of Cities")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.title("VAT Scaling Test")
    # plt.savefig('vat_scaling_time.eps', format='eps')
    # plt.close()

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"VAT Scaling Test (N={group_count})")
    im1 = ax1.imshow(ordered_matrix, cmap="viridis")
    ax1.set_title(f"Library VAT Result: {t2-t1:.2f} seconds")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ordered_matrix2, cmap="viridis")
    ax2.set_title(f"Merge Sort VAT Result: {t1-t0:.2f} seconds")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    # plt.savefig('vat_comparison.eps', format='eps')
    # plt.close()
    plt.show()


def test_merge_ivat():
    all_cities = circle_random_clusters(n_clusters=10, n_cities=1)
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # Scramble the order to ensure we sort it!
    cols = np.arange(len(all_cities), dtype="int")
    rand_col_order = np.random.permutation(cols)
    matrix_of_pairwise_distance = matrix_of_pairwise_distance[:, rand_col_order][
        rand_col_order, :
    ]
    ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(matrix_of_pairwise_distance)

    plot_vat_ivat(ivat_mst, vat_mst)


def plot_vat_ivat(ivat_mst, vat_mst):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    im1 = ax1.imshow(vat_mst, cmap="viridis")
    ax1.set_title("VAT Matrix")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ivat_mst, cmap="viridis")
    ax2.set_title("iVAT Matrix")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()


def test_plot_scaling():
    n = np.logspace(1, 5.1, 96)
    plt.plot(n, n**3, label="$N^3$")
    plt.plot(n, n**2 * np.log(n), label="$N^2 \log N$")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time Complexity")
    plt.title("Scaling Time Complexity")
    plt.legend()
    plt.show()


def test_show_fibb_bin_heap():
    v = np.logspace(1, 4)
    e = (v**2 - v) // 2
    arr_v = v**2
    bin_h_v = e * np.log2(v)
    fibb_h_v = e + v * np.log2(v)
    plt.plot(v, arr_v, label="Array")
    plt.plot(v, bin_h_v, label="Binary Heap")
    plt.plot(v, fibb_h_v, label="Fibonacci Heap")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time Complexity")
    plt.title("Heap Time Complexity Comparison")
    plt.legend()
    plt.show()


def test_fuzzy_c_means():
    n_clusters = 10
    all_cities = circle_random_clusters(
        n_clusters=n_clusters, n_cities=20, cluster_spacing=5, cluster_diameter=0.5
    )
    # Scramble the order of the cities
    scramble_order = np.random.permutation(len(all_cities))
    all_cities = all_cities[scramble_order]

    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    meth_c, w_c = fcm.fuzzy_c_means(all_cities, n_clusters, 2)

    # Compute the IVAT
    ivat_mst, vat_mst, ivat_order, vat_order = compute_ivat(matrix_of_pairwise_distance)
    # Plot it.
    plot_vat_ivat(ivat_mst, vat_mst)
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
    plot_diagonal(
        diagonal_values,
        max_diff_index,
        peaks_threshold,
        sorted_diagonal,
        abrupt_change_indices,
    )

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

    # Assert that every city has been allocated to a cluster
    all_allocated_cities = np.concatenate(cluster_city_ids)
    all_allocated_cities = np.sort(all_allocated_cities)
    print(f"All cities:\n{np.r_[0:len(all_cities)]}")
    print(f"Allocated Cities:\n{all_allocated_cities}")
    assert len(all_allocated_cities) == len(
        all_cities
    ), f"Not all cities allocated: {len(all_allocated_cities)} allocated out of {len(all_cities)} total"
    assert len(np.unique(all_allocated_cities)) == len(
        all_cities
    ), f"Duplicate city allocations detected"

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
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("fuzzy_c_means_membership.eps", format="eps", bbox_inches="tight")
    plt.show()


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
    # ax2.axhline(y=peaks_threshold, color='r', linestyle='--', label=f'Threshold: {peaks_threshold:.2f}')
    ax2.text(
        0.02,
        0.98,
        f"Abrupt changes: {len(abrupt_change_indices)}, threshold: {peaks_threshold:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return abrupt_change_indices
