"""
Test convergence status reporting for fuzzy c-means.
"""

import numpy as np
import pytest

from tribbleclustering import fuzzy_c_means, FuzzyMeansResult
from tribbleclustering.fuzzycmeans import FuzzyCMeans


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


class TestConvergenceStatus:
    """Test convergence status reporting."""

    def test_return_type_is_fuzzy_means_result(self, synthetic_data):
        """Test that fuzzy_c_means returns FuzzyMeansResult."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        assert isinstance(result, FuzzyMeansResult)
        assert hasattr(result, "cluster_centers_")
        assert hasattr(result, "membership_matrix_")
        assert hasattr(result, "n_iter_")
        assert hasattr(result, "converged")

    def test_convergence_flag_present(self, synthetic_data):
        """Test that convergence flag is always present."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iter_, int)
        assert result.n_iter_ > 0

    def test_converged_on_small_problem(self, synthetic_data):
        """Test that algorithm converges on small problem."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters, max_iter=1000)

        assert result.converged is True
        assert result.n_iter_ < 1000

    def test_max_iter_parameter_respected(self, synthetic_data):
        """Test that max_iter parameter limits iterations."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters, max_iter=5)

        assert result.n_iter_ <= 5

    def test_iteration_count_reasonable(self, synthetic_data):
        """Test that iteration count is within expected range."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        assert 1 <= result.n_iter_ <= 100

    def test_tuple_unpacking_backwards_compatibility(self, synthetic_data):
        """Test that result can be unpacked like a tuple."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        # Should support tuple unpacking
        c, w = result
        assert c.shape == (n_clusters, x.shape[1])
        assert w.shape == (x.shape[0], n_clusters)

    def test_sklearn_wrapper_convergence_attributes(self, synthetic_data):
        """Test that FuzzyCMeans wrapper exposes convergence attributes."""
        x, n_clusters = synthetic_data
        clf = FuzzyCMeans(n_clusters, max_iter=100, random_state=42)
        clf.fit(x)

        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "converged")
        assert clf.n_iter_ is not None
        assert clf.converged is not None

    def test_sklearn_wrapper_max_iter_parameter(self, synthetic_data):
        """Test that FuzzyCMeans respects max_iter parameter."""
        x, n_clusters = synthetic_data
        clf = FuzzyCMeans(n_clusters, max_iter=5, random_state=42)
        clf.fit(x)

        assert clf.n_iter_ <= 5

    def test_different_data_different_convergence(self):
        """Test that different data can have different convergence behavior."""
        np.random.seed(42)

        # Well-separated clusters (should converge quickly)
        x_easy = np.vstack(
            [
                np.random.randn(50, 2) * 0.1 + [0, 0],
                np.random.randn(50, 2) * 0.1 + [10, 10],
            ]
        ).astype(np.float64)

        # Overlapping clusters (may take more iterations)
        x_hard = np.vstack(
            [
                np.random.randn(50, 2) * 1.0 + [0, 0],
                np.random.randn(50, 2) * 1.0 + [1.5, 1.5],
            ]
        ).astype(np.float64)

        result_easy = fuzzy_c_means(x_easy, 2)
        result_hard = fuzzy_c_means(x_hard, 2)

        # Both should converge eventually
        assert result_easy.converged
        assert result_hard.converged

    def test_convergence_with_high_max_iter(self, synthetic_data):
        """Test convergence status with high iteration limit."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters, max_iter=1000)

        # With sufficient iterations, should converge
        assert result.converged is True

    def test_convergence_reproducibility(self, synthetic_data):
        """Test that same seed produces same convergence result."""
        x, n_clusters = synthetic_data

        np.random.seed(42)
        result1 = fuzzy_c_means(x, n_clusters, initial_guess=x[:n_clusters])

        np.random.seed(42)
        result2 = fuzzy_c_means(x, n_clusters, initial_guess=x[:n_clusters])

        assert result1.converged == result2.converged
        assert result1.n_iter_ == result2.n_iter_

    def test_result_has_correct_shapes(self, synthetic_data):
        """Test that result arrays have correct shapes."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        assert result.cluster_centers_.shape == (n_clusters, x.shape[1])
        assert result.membership_matrix_.shape == (x.shape[0], n_clusters)

    def test_membership_matrix_normalized(self, synthetic_data):
        """Test that membership matrix is properly normalized."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        # Each row should sum to 1
        row_sums = np.sum(result.membership_matrix_, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestMaxIterParameter:
    """Test max_iter parameter functionality."""

    def test_max_iter_default_value(self, synthetic_data):
        """Test that default max_iter is 100."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters)

        # Should have converged with default 100 iterations
        assert result.n_iter_ <= 100

    def test_max_iter_small_value(self, synthetic_data):
        """Test behavior with very small max_iter."""
        x, n_clusters = synthetic_data
        result = fuzzy_c_means(x, n_clusters, max_iter=1)

        assert result.n_iter_ == 1
        # May or may not have converged
        assert isinstance(result.converged, bool)

    def test_max_iter_zero_raises_error_or_returns(self, synthetic_data):
        """Test behavior with max_iter=0."""
        x, n_clusters = synthetic_data

        # This should either raise an error or return with n_iter_=0
        # The range(0) will produce an empty loop
        result = fuzzy_c_means(x, n_clusters, max_iter=0)
        assert result.n_iter_ == 0
        assert result.converged is False


class TestConvergenceWithInitialGuess:
    """Test convergence with different initialization methods."""

    def test_convergence_with_initial_guess(self, synthetic_data):
        """Test convergence when initial_guess is provided."""
        x, n_clusters = synthetic_data
        initial_guess = x[:n_clusters].copy()

        result = fuzzy_c_means(x, n_clusters, initial_guess=initial_guess)

        assert isinstance(result, FuzzyMeansResult)
        assert result.converged is not None

    def test_convergence_with_indices(self, synthetic_data):
        """Test convergence when indices are provided."""
        x, n_clusters = synthetic_data
        indices = np.arange(n_clusters)

        result = fuzzy_c_means(x, n_clusters, indices=indices)

        assert isinstance(result, FuzzyMeansResult)
        assert result.converged is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
