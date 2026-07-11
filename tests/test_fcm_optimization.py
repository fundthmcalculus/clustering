"""
Test and benchmark the fuzzy c-means implementations (Python vs Cython).
"""

import numpy as np
import pytest
import time
from numpy.testing import assert_allclose

from tribbleclustering.fcm import fuzzy_c_means as fuzzy_c_means_python

try:
    from tribbleclustering.cfcm import fuzzy_c_means as fuzzy_c_means_cython

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.fixture
def synthetic_data():
    """Generate synthetic clustering data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    n_clusters = 3

    cluster_centers = np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 3.0]])

    x = np.vstack(
        [
            cluster_centers[0] + np.random.randn(n_samples // 3, n_features) * 0.5,
            cluster_centers[1] + np.random.randn(n_samples // 3, n_features) * 0.5,
            cluster_centers[2]
            + np.random.randn(n_samples - 2 * (n_samples // 3), n_features) * 0.5,
        ]
    ).astype(np.float64)

    return x, n_clusters


class TestFCMCorrectness:
    """Test correctness of FCM implementations."""

    def test_python_basic(self, synthetic_data):
        """Test Python implementation with basic input."""
        x, n_clusters = synthetic_data
        c, w = fuzzy_c_means_python(x, n_clusters, m=2.0)

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(w >= 0)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_basic(self, synthetic_data):
        """Test Cython implementation with basic input."""
        x, n_clusters = synthetic_data
        c, w = fuzzy_c_means_cython(x, n_clusters, m=2.0)

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(w >= 0)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_matches_python(self, synthetic_data):
        """Test that Cython and Python implementations converge with same initialization."""
        x, n_clusters = synthetic_data

        # Test with initial_guess to avoid randomness differences during convergence
        np.random.seed(42)
        initial_guess = x[:n_clusters].copy()

        c_py, w_py = fuzzy_c_means_python(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_cy, w_cy = fuzzy_c_means_cython(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        # With same initial guess, results should match closely
        # Note: differences may occur due to floating-point rounding and convergence path
        # differences from distance caching optimization
        assert_allclose(c_py, c_cy, rtol=1e-3, atol=1e-5)
        assert_allclose(w_py, w_cy, rtol=1e-3, atol=1e-5)

    def test_python_with_initial_guess(self, synthetic_data):
        """Test Python implementation with initial cluster centers."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        c, w = fuzzy_c_means_python(x, n_clusters, m=2.0, initial_guess=initial_guess)

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_with_initial_guess(self, synthetic_data):
        """Test Cython implementation with initial cluster centers."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        c, w = fuzzy_c_means_cython(x, n_clusters, m=2.0, initial_guess=initial_guess)

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_initial_guess_matches(self, synthetic_data):
        """Test that Cython and Python give same results with initial_guess."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        c_py, w_py = fuzzy_c_means_python(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_cy, w_cy = fuzzy_c_means_cython(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        # Tolerances account for floating-point rounding differences and convergence path
        # differences from distance caching optimization
        assert_allclose(c_py, c_cy, rtol=1e-3, atol=1e-5)
        assert_allclose(w_py, w_cy, rtol=1e-3, atol=1e-5)

    def test_python_error_both_indices_and_guess(self, synthetic_data):
        """Test that Python raises error when both indices and initial_guess are provided."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()
        indices = np.arange(n_clusters * 2)

        with pytest.raises(
            ValueError, match="initial_guess and indices cannot both be provided"
        ):
            fuzzy_c_means_python(
                x, n_clusters, initial_guess=initial_guess, indices=indices
            )

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_error_both_indices_and_guess(self, synthetic_data):
        """Test that Cython raises error when both indices and initial_guess are provided."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].astype(np.float64).copy()
        indices = np.arange(n_clusters * 2, dtype=np.int64)

        with pytest.raises(
            ValueError, match="initial_guess and indices cannot both be provided"
        ):
            fuzzy_c_means_cython(
                x, n_clusters, initial_guess=initial_guess, indices=indices
            )

    def test_python_different_m_values(self, synthetic_data):
        """Test Python implementation with different fuzziness parameters."""
        x, n_clusters = synthetic_data

        for m in [1.5, 2.0, 3.0]:
            c, w = fuzzy_c_means_python(x, n_clusters, m=m)
            assert c.shape == (n_clusters, x.shape[1])
            assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_different_m_values(self, synthetic_data):
        """Test Cython implementation with different fuzziness parameters."""
        x, n_clusters = synthetic_data

        for m in [1.5, 2.0, 3.0]:
            c, w = fuzzy_c_means_cython(x, n_clusters, m=m)
            assert c.shape == (n_clusters, x.shape[1])
            assert w.shape == (x.shape[0], n_clusters)


class TestFCMPerformance:
    """Benchmark the implementations."""

    @pytest.mark.benchmark
    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_performance_small_dataset(self):
        """Benchmark on small dataset (100 samples)."""
        np.random.seed(42)
        x = np.random.randn(100, 5).astype(np.float64)
        n_clusters = 3

        time_py = self._time_implementation(fuzzy_c_means_python, x, n_clusters)
        time_cy = self._time_implementation(fuzzy_c_means_cython, x, n_clusters)

        print("\nSmall dataset (100 samples, 5 features, 3 clusters):")
        print(f"  Python: {time_py:.4f}s")
        print(f"  Cython: {time_cy:.4f}s")
        print(f"  Speedup: {time_py / time_cy:.2f}x")

        assert time_cy <= time_py * 2.0, "Cython should not be significantly slower"

    @pytest.mark.benchmark
    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_performance_medium_dataset(self):
        """Benchmark on medium dataset (1000 samples)."""
        np.random.seed(42)
        x = np.random.randn(1000, 10).astype(np.float64)
        n_clusters = 5

        time_py = self._time_implementation(
            fuzzy_c_means_python, x, n_clusters, iterations=1
        )
        time_cy = self._time_implementation(
            fuzzy_c_means_cython, x, n_clusters, iterations=1
        )

        print("\nMedium dataset (1000 samples, 10 features, 5 clusters):")
        print(f"  Python: {time_py:.4f}s")
        print(f"  Cython: {time_cy:.4f}s")
        print(f"  Speedup: {time_py / time_cy:.2f}x")

        assert time_cy <= time_py * 2.0, "Cython should not be significantly slower"

    @pytest.mark.benchmark
    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_performance_large_dataset(self):
        """Benchmark on larger dataset (5000 samples)."""
        np.random.seed(42)
        x = np.random.randn(5000, 10).astype(np.float64)
        n_clusters = 8

        time_py = self._time_implementation(
            fuzzy_c_means_python, x, n_clusters, iterations=1
        )
        time_cy = self._time_implementation(
            fuzzy_c_means_cython, x, n_clusters, iterations=1
        )

        print("\nLarge dataset (5000 samples, 10 features, 8 clusters):")
        print(f"  Python: {time_py:.4f}s")
        print(f"  Cython: {time_cy:.4f}s")
        print(f"  Speedup: {time_py / time_cy:.2f}x")

        assert time_cy <= time_py * 2.0, "Cython should not be significantly slower"

    @staticmethod
    def _time_implementation(func, x, n_clusters, iterations=3):
        """Time a function over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.time()
            func(x.copy(), n_clusters)
            times.append(time.time() - start)
        return min(times)


class TestFCMNumericalStability:
    """Test numerical stability of implementations."""

    def test_python_handles_zero_distances(self):
        """Test Python implementation with duplicate points."""
        x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1]])
        c, w = fuzzy_c_means_python(x, 2)

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_handles_zero_distances(self):
        """Test Cython implementation with duplicate points."""
        x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float64)
        c, w = fuzzy_c_means_cython(x, 2)

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    def test_python_large_m_value(self):
        """Test Python with large fuzziness parameter."""
        np.random.seed(42)
        x = np.random.randn(50, 3)
        c, w = fuzzy_c_means_python(x, 3, m=10.0)

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_large_m_value(self):
        """Test Cython with large fuzziness parameter."""
        np.random.seed(42)
        x = np.random.randn(50, 3).astype(np.float64)
        c, w = fuzzy_c_means_cython(x, 3, m=10.0)

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
