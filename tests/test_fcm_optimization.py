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
        result = fuzzy_c_means_python(x, n_clusters, m=2.0)
        c, w = result  # Test unpacking support

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(w >= 0)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_basic(self, synthetic_data):
        """Test Cython implementation with basic input."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means_cython(x, n_clusters, m=2.0)
        # Cython version returns (c, w, n_iter, converged)
        c, w, n_iter, converged = result

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(w >= 0)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert isinstance(n_iter, int)
        assert isinstance(converged, bool)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_matches_python(self, synthetic_data):
        """Test that Cython and Python implementations converge with same initialization."""
        x, n_clusters = synthetic_data

        # Test with initial_guess to avoid randomness differences during convergence
        np.random.seed(42)
        initial_guess = x[:n_clusters].copy()

        result_py = fuzzy_c_means_python(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_py, w_py = result_py  # Python version returns FuzzyCMeansResult with __iter__

        result_cy = fuzzy_c_means_cython(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_cy, w_cy, _, _ = result_cy  # Cython returns 4-tuple

        # Same start, same convergence test (issue #100), so the two paths run
        # the same iterations and agree to BLAS rounding -- the tolerance only
        # absorbs a differently-ordered dot product.
        assert_allclose(c_py, c_cy, rtol=1e-6, atol=1e-8)
        assert_allclose(w_py, w_cy, rtol=1e-6, atol=1e-8)

    def test_python_with_initial_guess(self, synthetic_data):
        """Test Python implementation with initial cluster centers."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        result = fuzzy_c_means_python(x, n_clusters, m=2.0, initial_guess=initial_guess)
        c, w = result

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_with_initial_guess(self, synthetic_data):
        """Test Cython implementation with initial cluster centers."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        result = fuzzy_c_means_cython(x, n_clusters, m=2.0, initial_guess=initial_guess)
        c, w, _, _ = result  # Cython returns 4-tuple

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_initial_guess_matches(self, synthetic_data):
        """Test that Cython and Python give same results with initial_guess."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        result_py = fuzzy_c_means_python(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_py, w_py = result_py

        result_cy = fuzzy_c_means_cython(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_cy, w_cy, _, _ = result_cy  # Cython returns 4-tuple

        # See test_cython_matches_python: agreement is to BLAS rounding.
        assert_allclose(c_py, c_cy, rtol=1e-6, atol=1e-8)
        assert_allclose(w_py, w_cy, rtol=1e-6, atol=1e-8)

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
            result = fuzzy_c_means_python(x, n_clusters, m=m)
            c, w = result
            assert c.shape == (n_clusters, x.shape[1])
            assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_different_m_values(self, synthetic_data):
        """Test Cython implementation with different fuzziness parameters."""
        x, n_clusters = synthetic_data

        for m in [1.5, 2.0, 3.0]:
            result = fuzzy_c_means_cython(x, n_clusters, m=m)
            c, w, _, _ = result  # Cython returns 4-tuple
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
        start = self._shared_start(x, n_clusters)

        time_py = self._time_implementation(
            fuzzy_c_means_python, x, n_clusters, initial_guess=start
        )
        time_cy = self._time_implementation(
            fuzzy_c_means_cython, x, n_clusters, initial_guess=start
        )

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
        start = self._shared_start(x, n_clusters)

        time_py = self._time_implementation(
            fuzzy_c_means_python, x, n_clusters, iterations=1, initial_guess=start
        )
        time_cy = self._time_implementation(
            fuzzy_c_means_cython, x, n_clusters, iterations=1, initial_guess=start
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
        start = self._shared_start(x, n_clusters)

        time_py = self._time_implementation(
            fuzzy_c_means_python, x, n_clusters, iterations=1, initial_guess=start
        )
        time_cy = self._time_implementation(
            fuzzy_c_means_cython, x, n_clusters, iterations=1, initial_guess=start
        )

        print("\nLarge dataset (5000 samples, 10 features, 8 clusters):")
        print(f"  Python: {time_py:.4f}s")
        print(f"  Cython: {time_cy:.4f}s")
        print(f"  Speedup: {time_py / time_cy:.2f}x")

        assert time_cy <= time_py * 2.0, "Cython should not be significantly slower"

    @staticmethod
    def _time_implementation(func, x, n_clusters, iterations=3, initial_guess=None):
        """Time a function over multiple iterations.

        ``initial_guess`` is not optional in spirit: without it each
        implementation draws its own random centers, converges in its own
        number of iterations, and the ratio below measures the luck of two
        draws rather than the speed of two kernels. Issue #100 was diagnosed
        through exactly that confound.
        """
        times = []
        for _ in range(iterations):
            start = time.time()
            func(x.copy(), n_clusters, initial_guess=initial_guess)
            times.append(time.time() - start)
        return min(times)

    @staticmethod
    def _shared_start(x, n_clusters, seed=0):
        """Centers both implementations start from, drawn away from the data."""
        rng = np.random.default_rng(seed)
        return x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()


class TestFCMNumericalStability:
    """Test numerical stability of implementations."""

    def test_python_handles_zero_distances(self):
        """Test Python implementation with duplicate points."""
        x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1]])
        result = fuzzy_c_means_python(x, 2)
        c, w = result

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_handles_zero_distances(self):
        """Test Cython implementation with duplicate points."""
        x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float64)
        result = fuzzy_c_means_cython(x, 2)
        c, w, _, _ = result  # Cython returns 4-tuple

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_coincident_point_membership_matches_python(self):
        """A point sitting exactly on a center takes crisp membership (issue #100).

        The compiled kernel used to hand the coincident cluster a membership of
        0 and every *other* cluster a membership of 1 -- the inverse of the
        intended convention, and a silently different answer from
        ``fcm._get_weights`` on the same input.
        """
        x = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1], [5.0, 5.0]],
            dtype=np.float64,
        )
        initial_guess = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])

        _, w_py = fuzzy_c_means_python(x, 3, m=2.0, initial_guess=initial_guess)
        _, w_cy, _, _ = fuzzy_c_means_cython(x, 3, m=2.0, initial_guess=initial_guess)

        # Rows 0, 1 and 4 sit on a center; each must be crisp on that center.
        assert_allclose(w_cy.sum(axis=1), 1.0, atol=1e-12)
        assert_allclose(w_cy, w_py, rtol=1e-6, atol=1e-9)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    @pytest.mark.parametrize(
        "n_samples, n_features, n_clusters", [(100, 5, 3), (500, 15, 5), (1000, 10, 5)]
    )
    def test_cython_stops_on_the_same_iteration_as_python(
        self, n_samples, n_features, n_clusters
    ):
        """Both paths share fcm.py's rtol=1e-5/atol=1e-8 convergence test.

        They sit behind a silent import-time fallback, so a compiled kernel
        with its own stopping rule returns a different answer than the
        reference for the same call -- and makes every speed comparison
        between them a comparison of unequal work.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n_samples, n_features)) * 10.0
        initial_guess = x[rng.choice(n_samples, size=n_clusters, replace=False)].copy()

        res_py = fuzzy_c_means_python(x, n_clusters, m=2.0, initial_guess=initial_guess)
        c_cy, w_cy, n_iter_cy, _ = fuzzy_c_means_cython(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        assert n_iter_cy == res_py.n_iter_
        assert_allclose(c_cy, res_py.cluster_centers_, rtol=1e-6, atol=1e-8)
        assert_allclose(w_cy, res_py.membership_matrix_, rtol=1e-6, atol=1e-8)

    def test_python_large_m_value(self):
        """Test Python with large fuzziness parameter."""
        np.random.seed(42)
        x = np.random.randn(50, 3)
        result = fuzzy_c_means_python(x, 3, m=10.0)
        c, w = result

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_cython_large_m_value(self):
        """Test Cython with large fuzziness parameter."""
        np.random.seed(42)
        x = np.random.randn(50, 3).astype(np.float64)
        result = fuzzy_c_means_cython(x, 3, m=10.0)
        c, w, _, _ = result  # Cython returns 4-tuple

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
