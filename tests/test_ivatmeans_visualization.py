import numpy as np
import matplotlib.pyplot as plt
from tribbleclustering import IVATMeans
from tribbleclustering.util import circle_random_clusters


def test_ivatmeans_centroids_with_varying_n_clusters():
    """
    Visualize how IVATMeans centroids change as n_clusters varies.
    Shows the data points and cluster centers for each n_clusters value.
    """
    # Create hierarchical cluster data
    data = circle_random_clusters(
        n_clusters=4, n_cities=15, cluster_spacing=6.0, cluster_diameter=0.8
    )

    n_clusters_values = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, len(n_clusters_values), figsize=(18, 4))

    for idx, n_clusters in enumerate(n_clusters_values):
        ax = axes[idx]

        # Fit IVATMeans
        ivat = IVATMeans(n_clusters=n_clusters, random_state=42)
        ivat.fit(data)

        # Plot data points
        ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30, label="Data points")

        # Plot cluster centers
        centers = ivat.cluster_centers_
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="red",
            marker="X",
            s=300,
            edgecolors="black",
            linewidths=2,
            label=f"Centers ({centers.shape[0]})",
        )

        # Add lines from points to nearest center
        for i, point in enumerate(data):
            nearest_center = centers[ivat.labels_[i]]
            ax.plot(
                [point[0], nearest_center[0]],
                [point[1], nearest_center[1]],
                "gray",
                alpha=0.1,
                linewidth=0.5,
            )

        ax.set_title(f"n_clusters={n_clusters}\n({centers.shape[0]} detected)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("ivatmeans_n_clusters_variation.png", dpi=100, bbox_inches="tight")
    print("Plot saved as 'ivatmeans_n_clusters_variation.png'")
    plt.close()

    # Verify that results are valid
    for n_clusters in n_clusters_values:
        ivat = IVATMeans(n_clusters=n_clusters, random_state=42)
        ivat.fit(data)
        assert ivat.cluster_centers_ is not None
        assert ivat.labels_ is not None
        assert len(ivat.labels_) == len(data)
        assert np.all(ivat.labels_ < ivat.cluster_centers_.shape[0])


def test_ivatmeans_cluster_stability():
    """
    Test that cluster assignments are stable across refits with same data and random_state.
    """
    data = circle_random_clusters(
        n_clusters=3, n_cities=10, cluster_spacing=5.0, cluster_diameter=0.5
    )

    for n_clusters in [1, 2, 3]:
        ivat1 = IVATMeans(n_clusters=n_clusters, random_state=42)
        ivat1.fit(data)
        labels1 = ivat1.labels_.copy()
        centers1 = ivat1.cluster_centers_.copy()

        ivat2 = IVATMeans(n_clusters=n_clusters, random_state=42)
        ivat2.fit(data)
        labels2 = ivat2.labels_
        centers2 = ivat2.cluster_centers_

        # Labels and centers should be identical with same random_state
        assert np.array_equal(labels1, labels2)
        assert np.allclose(centers1, centers2)


if __name__ == "__main__":
    test_ivatmeans_centroids_with_varying_n_clusters()
    test_ivatmeans_cluster_stability()
    print("All visualization tests passed!")
