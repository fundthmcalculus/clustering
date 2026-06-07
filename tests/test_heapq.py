import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
import time
import matplotlib.pyplot as plt

from tribbleclustering import compute_vat
from tribbleclustering.pvat import vat_prim_mst
from tribbleclustering.pqvat import vat_prim_mst_numba
from tribbleclustering.pcvat import vat_prim_mst_c, compute_vat_c


def _bench(fn, arg, n_iter=15, warmup=2):
    """Time `fn(arg)` repeatedly, returning per-call times in milliseconds.

    Runs a few warmup calls (JIT compile / cache / thread-pool spin-up) that
    are excluded from the sample, then collects `n_iter` individual timings so
    the caller can compute a mean and a 2-sigma spread.
    """
    for _ in range(warmup):
        fn(arg)
    samples = np.empty(n_iter, dtype=np.float64)
    for k in range(n_iter):
        t0 = time.perf_counter()
        fn(arg)
        samples[k] = (time.perf_counter() - t0) * 1000.0
    return samples


@pytest.fixture
def small_distance_matrix():
    """Create a small symmetric distance matrix for testing."""
    np.random.seed(42)
    data = np.random.randn(5, 2)
    distances = squareform(pdist(data, metric="euclidean"))
    return distances


@pytest.fixture
def medium_distance_matrix():
    """Create a medium-sized symmetric distance matrix."""
    np.random.seed(42)
    data = np.random.randn(50, 5)
    distances = squareform(pdist(data, metric="euclidean"))
    return distances


@pytest.fixture
def large_distance_matrix():
    """Create a large symmetric distance matrix."""
    np.random.seed(42)
    data = np.random.randn(200, 10)
    distances = squareform(pdist(data, metric="euclidean"))
    return distances


class TestCorrectness:
    """Correctness validation tests."""

    def test_small_matrix_agreement(self, small_distance_matrix):
        """Test that numba version matches original on small matrix."""
        heap_seq_orig, parent_seq_orig = vat_prim_mst(small_distance_matrix)
        heap_seq_numba, parent_seq_numba = vat_prim_mst_numba(small_distance_matrix)

        # Both should traverse all vertices
        assert len(heap_seq_orig) == len(heap_seq_numba)
        assert len(parent_seq_orig) == len(parent_seq_numba)

        # Sequences should match
        np.testing.assert_array_equal(heap_seq_orig, heap_seq_numba)
        np.testing.assert_array_equal(parent_seq_orig, parent_seq_numba)

    def test_medium_matrix_agreement(self, medium_distance_matrix):
        """Test that numba version produces valid MST (sequence may differ with tied weights)."""
        heap_seq_orig, parent_seq_orig = vat_prim_mst(medium_distance_matrix)
        heap_seq_numba, parent_seq_numba = vat_prim_mst_numba(medium_distance_matrix)

        # Both should visit all vertices
        assert len(heap_seq_orig) == len(heap_seq_numba) == len(medium_distance_matrix)
        # Both should visit each vertex exactly once
        assert set(heap_seq_numba) == set(range(len(medium_distance_matrix)))
        assert set(heap_seq_orig) == set(range(len(medium_distance_matrix)))

    def test_heap_seq_is_permutation(self, small_distance_matrix):
        """Test that heap_seq is a valid permutation of all vertices."""
        heap_seq, _ = vat_prim_mst_numba(small_distance_matrix)
        n = len(small_distance_matrix)

        # Should contain all vertices exactly once
        assert len(heap_seq) == n
        assert set(heap_seq) == set(range(n))

    def test_parent_seq_length(self, small_distance_matrix):
        """Test that parent_seq has correct length."""
        _, parent_seq = vat_prim_mst_numba(small_distance_matrix)
        n = len(small_distance_matrix)

        # Parent sequence should have n entries
        assert len(parent_seq) == n

    def test_single_vertex(self):
        """Test with single vertex matrix."""
        dist = np.array([[0.0]])
        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 1
        assert heap_seq[0] == 0
        assert len(parent_seq) == 1

    def test_two_vertices(self):
        """Test with two vertex matrix."""
        dist = np.array([[0.0, 1.5], [1.5, 0.0]])
        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 2
        assert set(heap_seq) == {0, 1}
        assert len(parent_seq) == 2

    def test_symmetric_matrix_handling(self):
        """Test that algorithm handles symmetric matrices correctly."""
        np.random.seed(123)
        data = np.random.randn(10, 3)
        distances = squareform(pdist(data, metric="euclidean"))

        # Both should work on symmetric matrix
        heap_seq_orig, parent_seq_orig = vat_prim_mst(distances)
        heap_seq_numba, parent_seq_numba = vat_prim_mst_numba(distances)

        assert len(heap_seq_orig) == len(heap_seq_numba)
        np.testing.assert_array_equal(heap_seq_orig, heap_seq_numba)

    def test_no_nan_or_inf_in_output(self, small_distance_matrix):
        """Test that outputs don't contain NaN or inf values."""
        heap_seq, parent_seq = vat_prim_mst_numba(small_distance_matrix)

        assert not np.any(np.isnan(heap_seq.astype(float)))
        assert not np.any(np.isinf(heap_seq.astype(float)))
        assert not np.any(np.isnan(parent_seq.astype(float)))
        assert not np.any(np.isinf(parent_seq.astype(float)))

    def test_c_version_small_matrix_agreement(self, small_distance_matrix):
        """Test that C version matches original on small matrix."""
        heap_seq_orig, parent_seq_orig = vat_prim_mst(small_distance_matrix)
        heap_seq_c, parent_seq_c = vat_prim_mst_c(
            small_distance_matrix.astype(np.float64)
        )

        np.testing.assert_array_equal(heap_seq_orig, heap_seq_c)
        np.testing.assert_array_equal(parent_seq_orig, parent_seq_c)

    def test_c_version_medium_matrix_agreement(self, medium_distance_matrix):
        """Test that C version matches original on medium matrix."""
        heap_seq_orig, parent_seq_orig = vat_prim_mst(medium_distance_matrix)
        heap_seq_c, parent_seq_c = vat_prim_mst_c(
            medium_distance_matrix.astype(np.float64)
        )

        np.testing.assert_array_equal(heap_seq_orig, heap_seq_c)
        np.testing.assert_array_equal(parent_seq_orig, parent_seq_c)


class TestPerformance:
    """Performance validation and benchmarking."""

    def test_small_matrix_performance(self, small_distance_matrix):
        """Benchmark on small matrix."""
        # Warm up
        vat_prim_mst_numba(small_distance_matrix)

        # Time numba version
        start = time.time()
        for _ in range(10):
            vat_prim_mst_numba(small_distance_matrix)
        numba_time = time.time() - start

        print(f"\nSmall matrix (n=5): {numba_time/10*1000:.3f}ms per call")
        assert numba_time > 0  # Sanity check

    def test_medium_matrix_performance(self, medium_distance_matrix):
        """Benchmark on medium matrix."""
        # Warm up
        vat_prim_mst_numba(medium_distance_matrix)

        # Time numba version
        start = time.time()
        for _ in range(5):
            vat_prim_mst_numba(medium_distance_matrix)
        numba_time = time.time() - start

        print(f"\nMedium matrix (n=50): {numba_time/5*1000:.3f}ms per call")
        assert numba_time > 0

    def test_large_matrix_performance(self, large_distance_matrix):
        """Benchmark on large matrix."""
        # Warm up
        vat_prim_mst_numba(large_distance_matrix)

        # Time numba version
        start = time.time()
        for _ in range(3):
            vat_prim_mst_numba(large_distance_matrix)
        numba_time = time.time() - start

        print(f"\nLarge matrix (n=200): {numba_time/3*1000:.3f}ms per call")
        assert numba_time > 0

    def test_vat_scaling_behavior(self):
        """Compare full VAT pipeline: compute_vat (heapq) vs compute_vat_c.

        Reports mean +/- 2 sigma (95% spread) in text and as a shaded band on
        the plot, so timing noise (which dominates the small sizes) is visible.
        """
        np.random.seed(42)

        mean_c, two_sig_c = [], []
        mean_orig, two_sig_orig = [], []
        # sizes = [25, 100, 500, 1000, 2000, 5000, 10000, 15000, 25000, 50000]
        sizes = [25, 100, 500, 1000, 2000]

        print("\ncompute_vat vs compute_vat_c  (mean +/- 2 sigma, ms):")
        print(f"{'Size':>6} | {'heapq (ms)':>20} | {'C (ms)':>20} | {'C/heapq':>9}")
        print("-" * 70)

        for size in sizes:
            data = np.random.randn(size, 5)
            distances = squareform(pdist(data, metric="euclidean")).astype(np.float32)

            s_orig = _bench(compute_vat, distances)
            s_c = _bench(compute_vat_c, distances)

            m_o, sd_o = float(s_orig.mean()), float(s_orig.std(ddof=1))
            m_c, sd_c = float(s_c.mean()), float(s_c.std(ddof=1))
            mean_orig.append(m_o)
            two_sig_orig.append(2 * sd_o)
            mean_c.append(m_c)
            two_sig_c.append(2 * sd_c)

            print(
                f"{size:>6} | {m_o:>9.3f} +/- {2*sd_o:>6.3f} | "
                f"{m_c:>9.3f} +/- {2*sd_c:>6.3f} | "
                f"{m_c/m_o:>8.2f}x{'✓' if m_c < m_o else 'X'}"
            )

        mean_orig = np.array(mean_orig)
        two_sig_orig = np.array(two_sig_orig)
        mean_c = np.array(mean_c)
        two_sig_c = np.array(two_sig_c)

        # Plot with 2-sigma shaded bands
        plt.figure()
        (l1,) = plt.plot(
            sizes,
            mean_orig,
            "s-",
            label="compute_vat (heapq)",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            sizes,
            np.maximum(mean_orig - two_sig_orig, 0.0),
            mean_orig + two_sig_orig,
            color=l1.get_color(),
            alpha=0.2,
            label=r"heapq $\pm 2\sigma$",
        )
        (l2,) = plt.plot(
            sizes,
            mean_c,
            "^-",
            label="compute_vat_c (C extension)",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            sizes,
            np.maximum(mean_c - two_sig_c, 0.0),
            mean_c + two_sig_c,
            color=l2.get_color(),
            alpha=0.2,
            label=r"C $\pm 2\sigma$",
        )
        plt.xlabel("Matrix Size (n)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.title("compute_vat vs compute_vat_c (mean $\\pm 2\\sigma$)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("vat_scaling_performance.png", dpi=150)
        print("\nPlot saved to 'vat_scaling_performance.png'")

        # Times should generally increase with size (not a strict requirement, but expected)
        assert mean_c[-1] >= mean_c[-2]

    def test_scaling_behavior(self):
        """Compare MST only: vat_prim_mst (heapq) vs vat_prim_mst_c.

        Reports mean +/- 2 sigma in text and as a shaded band on the plot.
        """
        np.random.seed(42)

        mean_orig, two_sig_orig = [], []
        mean_c, two_sig_c = [], []
        # sizes = [25, 100, 500, 1000, 2000, 5000, 10000, 15000]
        sizes = [25, 100, 500, 1000, 2000]

        print("\nvat_prim_mst vs vat_prim_mst_c  (mean +/- 2 sigma, ms):")
        print(f"{'Size':>6} | {'heapq (ms)':>20} | {'C (ms)':>20} | {'C/heapq':>9}")
        print("-" * 70)

        for size in sizes:
            data = np.random.randn(size, 5)
            distances = squareform(pdist(data, metric="euclidean")).astype(np.float64)

            s_orig = _bench(vat_prim_mst, distances)
            s_c = _bench(vat_prim_mst_c, distances)

            m_o, sd_o = float(s_orig.mean()), float(s_orig.std(ddof=1))
            m_c, sd_c = float(s_c.mean()), float(s_c.std(ddof=1))
            mean_orig.append(m_o)
            two_sig_orig.append(2 * sd_o)
            mean_c.append(m_c)
            two_sig_c.append(2 * sd_c)

            print(
                f"{size:>6} | {m_o:>9.3f} +/- {2*sd_o:>6.3f} | "
                f"{m_c:>9.3f} +/- {2*sd_c:>6.3f} | "
                f"{m_c/m_o:>8.2f}x{'✓' if m_c < m_o else 'X'}"
            )

        mean_orig = np.array(mean_orig)
        two_sig_orig = np.array(two_sig_orig)
        mean_c = np.array(mean_c)
        two_sig_c = np.array(two_sig_c)

        # Plot with 2-sigma shaded bands
        plt.figure()
        (l1,) = plt.plot(
            sizes,
            mean_orig,
            "s-",
            label="vat_prim_mst (heapq)",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            sizes,
            np.maximum(mean_orig - two_sig_orig, 0.0),
            mean_orig + two_sig_orig,
            color=l1.get_color(),
            alpha=0.2,
            label=r"heapq $\pm 2\sigma$",
        )
        (l2,) = plt.plot(
            sizes,
            mean_c,
            "^-",
            label="vat_prim_mst_c (C extension)",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            sizes,
            np.maximum(mean_c - two_sig_c, 0.0),
            mean_c + two_sig_c,
            color=l2.get_color(),
            alpha=0.2,
            label=r"C $\pm 2\sigma$",
        )
        plt.xlabel("Matrix Size (n)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.title("vat_prim_mst vs vat_prim_mst_c (mean $\\pm 2\\sigma$)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("scaling_performance.png", dpi=150)
        print("\nPlot saved to 'scaling_performance.png'")

        # Times should generally increase with size (not a strict requirement, but expected)
        assert mean_c[-1] >= mean_c[-2]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_distances(self):
        """Test with uniform distance matrix."""
        dist = np.ones((5, 5)) * 2.0
        np.fill_diagonal(dist, 0.0)

        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 5
        assert len(parent_seq) == 5
        assert set(heap_seq) == set(range(5))

    def test_very_small_distances(self):
        """Test with very small distance values."""
        dist = np.eye(10) * 1e-10
        for i in range(10):
            for j in range(i + 1, 10):
                dist[i, j] = dist[j, i] = 1e-10 + 1e-12 * (i + j)

        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 10
        assert set(heap_seq) == set(range(10))

    def test_large_distance_values(self):
        """Test with very large distance values."""
        dist = np.eye(8) * 1e10
        for i in range(8):
            for j in range(i + 1, 8):
                dist[i, j] = dist[j, i] = 1e10 + 1e8 * (i + j)

        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 8
        assert set(heap_seq) == set(range(8))

    def test_multiple_equal_max_values(self):
        """Test when distance matrix has multiple equal maximum values."""
        dist = np.array(
            [[0.0, 5.0, 3.0], [5.0, 0.0, 5.0], [3.0, 5.0, 0.0]], dtype=np.float64
        )

        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 3
        assert set(heap_seq) == {0, 1, 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
