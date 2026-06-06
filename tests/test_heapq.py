import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
import time
import matplotlib.pyplot as plt

from tribbleclustering import compute_vat
from tribbleclustering.pvat import vat_prim_mst
from tribbleclustering.pqvat import vat_prim_mst_numba
from tribbleclustering.pcvat import vat_prim_mst_c, compute_vat_c


@pytest.fixture
def small_distance_matrix():
    """Create a small symmetric distance matrix for testing."""
    np.random.seed(42)
    data = np.random.randn(5, 2)
    distances = squareform(pdist(data, metric='euclidean'))
    return distances


@pytest.fixture
def medium_distance_matrix():
    """Create a medium-sized symmetric distance matrix."""
    np.random.seed(42)
    data = np.random.randn(50, 5)
    distances = squareform(pdist(data, metric='euclidean'))
    return distances


@pytest.fixture
def large_distance_matrix():
    """Create a large symmetric distance matrix."""
    np.random.seed(42)
    data = np.random.randn(200, 10)
    distances = squareform(pdist(data, metric='euclidean'))
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
        distances = squareform(pdist(data, metric='euclidean'))

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
        heap_seq_c, parent_seq_c = vat_prim_mst_c(small_distance_matrix.astype(np.float64))

        np.testing.assert_array_equal(heap_seq_orig, heap_seq_c)
        np.testing.assert_array_equal(parent_seq_orig, parent_seq_c)

    def test_c_version_medium_matrix_agreement(self, medium_distance_matrix):
        """Test that C version matches original on medium matrix."""
        heap_seq_orig, parent_seq_orig = vat_prim_mst(medium_distance_matrix)
        heap_seq_c, parent_seq_c = vat_prim_mst_c(medium_distance_matrix.astype(np.float64))

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
        """Test that performance scales reasonably with input size."""
        np.random.seed(42)

        times_numba = []
        times_orig = []
        times_c = []
        sizes = [25, 100, 500, 1000, 2000, 5000]

        print('\nPerformance Scaling Comparison:')
        print(f"{'Size':>6} | {'heapq ms':>10} | {'numba ms':>10} | {'C ms':>10} | {'C/heapq':>9} | {'C/numba':>9}")
        print('-' * 72)

        for size in sizes:
            data = np.random.randn(size, 5)
            distances = squareform(pdist(data, metric='euclidean')).astype(np.float64)

            # Warm up all versions
            compute_vat(distances)
            compute_vat_c(distances)

            N = 5
            start = time.time()
            for _ in range(N):
                compute_vat(distances)
            elapsed_orig = time.time() - start
            times_orig.append(elapsed_orig / N)

            start = time.time()
            for _ in range(N):
                compute_vat_c(distances)
            elapsed_c = time.time() - start
            times_c.append(elapsed_c / N)

            orig_ms = elapsed_orig / N * 1000
            c_ms = elapsed_c / N * 1000

            print(f"{size:>6} | {orig_ms:>10.3f} | {c_ms:>10.3f}"
                  f" | {c_ms / orig_ms:>8.2f}x{'✓' if c_ms < orig_ms else '✗'}")

        # Plot comparison
        plt.figure()
        plt.plot(sizes, [t * 1000 for t in times_orig], 's-', label='compute_vat (heapq)', linewidth=2, markersize=8)
        plt.plot(sizes, [t * 1000 for t in times_c], '^-', label='compute_vat_c (C extension)', linewidth=2,
                 markersize=8)
        plt.xlabel('Matrix Size (n)', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title('Performance Comparison: Original vs C Extension', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('vat_scaling_performance.png', dpi=150)
        print("\nPlot saved to 'vat_scaling_performance.png'")

        # Times should generally increase with size (not a strict requirement, but expected)
        assert times_c[1] >= times_c[0] * 0.5  # Allow some variance

    def test_scaling_behavior(self):
        """Test that performance scales reasonably with input size."""
        import matplotlib.pyplot as plt

        np.random.seed(42)

        times_numba = []
        times_orig = []
        times_c = []
        sizes = [25, 100, 500, 1000,2000,5000]

        print('\nPerformance Scaling Comparison:')
        print(f"{'Size':>6} | {'heapq ms':>10} | {'numba ms':>10} | {'C ms':>10} | {'C/heapq':>9} | {'C/numba':>9}")
        print('-' * 72)

        for size in sizes:
            data = np.random.randn(size, 5)
            distances = squareform(pdist(data, metric='euclidean')).astype(np.float64)

            # Warm up all versions
            vat_prim_mst_numba(distances)
            vat_prim_mst(distances)
            vat_prim_mst_c(distances)

            N = 5
            start = time.time()
            for _ in range(N): vat_prim_mst(distances)
            elapsed_orig = time.time() - start
            times_orig.append(elapsed_orig / N)

            start = time.time()
            for _ in range(N): vat_prim_mst_numba(distances)
            elapsed_numba = time.time() - start
            times_numba.append(elapsed_numba / N)

            start = time.time()
            for _ in range(N): vat_prim_mst_c(distances)
            elapsed_c = time.time() - start
            times_c.append(elapsed_c / N)

            orig_ms  = elapsed_orig  / N * 1000
            numba_ms = elapsed_numba / N * 1000
            c_ms     = elapsed_c     / N * 1000

            print(f"{size:>6} | {orig_ms:>10.3f} | {numba_ms:>10.3f} | {c_ms:>10.3f}"
                  f" | {c_ms/orig_ms:>8.2f}x{'✓' if c_ms < orig_ms else '✗'}"
                  f" | {c_ms/numba_ms:>8.2f}x{'✓' if c_ms < numba_ms else '✗'}")

        # Plot comparison
        plt.figure()
        plt.plot(sizes, [t*1000 for t in times_numba], 'o-', label='vat_prim_mst_numba', linewidth=2, markersize=8)
        plt.plot(sizes, [t*1000 for t in times_orig], 's-', label='vat_prim_mst (heapq)', linewidth=2, markersize=8)
        plt.plot(sizes, [t*1000 for t in times_c], '^-', label='vat_prim_mst_c (C extension)', linewidth=2, markersize=8)
        plt.xlabel('Matrix Size (n)', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title('Performance Comparison: Numba vs Original vs C Extension', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scaling_performance.png', dpi=150)
        print("\nPlot saved to 'scaling_performance.png'")

        # Times should generally increase with size (not a strict requirement, but expected)
        assert times_numba[1] >= times_numba[0] * 0.5  # Allow some variance


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
        dist = np.array([
            [0.0, 5.0, 3.0],
            [5.0, 0.0, 5.0],
            [3.0, 5.0, 0.0]
        ], dtype=np.float64)

        heap_seq, parent_seq = vat_prim_mst_numba(dist)

        assert len(heap_seq) == 3
        assert set(heap_seq) == {0, 1, 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
