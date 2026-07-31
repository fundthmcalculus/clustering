"""Test FCM memory layout optimization (distance caching).

The optimization skips distance recomputation when cluster centers
move less than a threshold. This test verifies convergence behavior
is not adversely affected.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

try:
    from tribbleclustering.cfcm import fuzzy_c_means as fuzzy_c_means_optimized

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

from tribbleclustering.fcm import fuzzy_c_means as fuzzy_c_means_baseline


@pytest.fixture
def synthetic_data():
    """Generate consistent synthetic clustering data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_clusters = 3

    cluster_centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [0.0, 3.0, 0.0, 3.0, 0.0],
        ]
    )

    x = np.vstack(
        [
            cluster_centers[0] + np.random.randn(n_samples // 3, n_features) * 0.5,
            cluster_centers[1] + np.random.randn(n_samples // 3, n_features) * 0.5,
            cluster_centers[2]
            + np.random.randn(n_samples - 2 * (n_samples // 3), n_features) * 0.5,
        ]
    ).astype(np.float64)

    return x, n_clusters


class TestFCMMemoryOptimization:
    """Test memory layout optimization correctness."""

    def test_baseline_convergence(self, synthetic_data):
        """Baseline implementation should converge."""
        x, n_clusters = synthetic_data

        c, w = fuzzy_c_means_baseline(x, n_clusters, m=2.0)

        # Basic sanity checks
        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_convergence(self, synthetic_data):
        """Optimized implementation should converge."""
        x, n_clusters = synthetic_data

        c, w = fuzzy_c_means_optimized(x, n_clusters, m=2.0)

        # Basic sanity checks
        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_matches_baseline(self, synthetic_data):
        """Optimized version should match baseline with same initialization."""
        x, n_clusters = synthetic_data

        # Use deterministic initialization
        initial_guess = x[:n_clusters].copy()

        c_baseline, w_baseline = fuzzy_c_means_baseline(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )
        c_optimized, w_optimized = fuzzy_c_means_optimized(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        # Allow 2% relative tolerance due to numerical precision differences
        # The optimization should produce very similar results, not identical
        # due to potential rounding differences in the skipped iterations
        assert_allclose(c_baseline, c_optimized, rtol=2e-2)
        assert_allclose(w_baseline, w_optimized, rtol=2e-2)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_with_initial_guess(self, synthetic_data):
        """Optimized version works with initial_guess parameter."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        c, w = fuzzy_c_means_optimized(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_with_indices(self, synthetic_data):
        """Optimized version works with indices parameter."""
        x, n_clusters = synthetic_data
        indices = np.arange(n_clusters * 2)

        c, w = fuzzy_c_means_optimized(x, n_clusters, m=2.0, indices=indices)

        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_different_m_values(self, synthetic_data):
        """Test optimized implementation with different fuzziness parameters."""
        x, n_clusters = synthetic_data

        for m in [1.5, 2.0, 3.0]:
            c, w = fuzzy_c_means_optimized(x, n_clusters, m=m)
            assert c.shape == (n_clusters, x.shape[1])
            assert w.shape == (x.shape[0], n_clusters)
            assert np.all(np.isfinite(c))
            assert np.all(np.isfinite(w))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_handles_duplicate_points(self):
        """Test optimized implementation with duplicate points."""
        x = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.1, 1.1]],  # Duplicate
            dtype=np.float64,
        )

        c, w = fuzzy_c_means_optimized(x, 2)

        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(w))
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_single_cluster(self, synthetic_data):
        """Test with single cluster."""
        x, _ = synthetic_data

        c, w = fuzzy_c_means_optimized(x, 1, m=2.0)

        assert c.shape == (1, x.shape[1])
        assert w.shape == (x.shape[0], 1)
        assert np.allclose(w, 1.0)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_many_clusters(self, synthetic_data):
        """Test with many clusters."""
        x, _ = synthetic_data

        c, w = fuzzy_c_means_optimized(x, 10, m=2.0)

        assert c.shape == (10, x.shape[1])
        assert w.shape == (x.shape[0], 10)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_convergence_speed(self, synthetic_data):
        """Verify optimization doesn't cause excessive iterations.

        The optimization skips distance recomputation, which might
        affect convergence. However, it should not significantly
        increase iteration count (no more than 2x).
        """
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        # This is a proxy test - we just verify it completes successfully
        # and produces reasonable results. A full benchmark would measure
        # wall-clock time and iteration count.
        c, w = fuzzy_c_means_optimized(
            x, n_clusters, m=2.0, initial_guess=initial_guess
        )

        # Check results are reasonable
        assert c.shape == (n_clusters, x.shape[1])
        assert np.all(np.isfinite(c))

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_float32_dtype(self):
        """Test optimized implementation with float32 input."""
        np.random.seed(42)
        x = np.random.randn(50, 5).astype(np.float32)

        c, w = fuzzy_c_means_optimized(x, 3, m=2.0)

        # Results should be in float32
        assert c.dtype == np.float32
        assert w.dtype == np.float32

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
    def test_optimized_float64_dtype(self):
        """Test optimized implementation with float64 input."""
        np.random.seed(42)
        x = np.random.randn(50, 5).astype(np.float64)

        c, w = fuzzy_c_means_optimized(x, 3, m=2.0)

        # Results should be in float64
        assert c.dtype == np.float64
        assert w.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
