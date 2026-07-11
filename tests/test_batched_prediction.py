"""Test batched prediction optimization for IVATMeans and FuzzyCMeans.

Batched prediction processes large test sets in memory-efficient chunks
to avoid allocating huge temporary arrays.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tribbleclustering import IVATMeans, FuzzyCMeans


@pytest.fixture
def trained_ivatmeans():
    """Train an IVATMeans model for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float64)
    model = IVATMeans(n_clusters=3)
    model.fit(X)
    return model


@pytest.fixture
def trained_fuzzy_cmeans():
    """Train a FuzzyCMeans model for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float64)
    model = FuzzyCMeans(n_clusters=3)
    model.fit(X)
    return model


class TestIVATMeansBatchedPrediction:
    """Test batched prediction for IVATMeans."""

    def test_small_batch_prediction(self, trained_ivatmeans):
        """Test prediction with small test set (no batching)."""
        X_test = np.random.randn(50, 5).astype(np.float64)
        labels = trained_ivatmeans.predict(X_test)

        assert labels.shape == (50,)
        assert np.all((labels >= 0) & (labels < 3))

    def test_large_batch_prediction(self, trained_ivatmeans):
        """Test prediction with large test set (batching occurs)."""
        np.random.seed(123)
        X_test = np.random.randn(50000, 5).astype(np.float64)
        labels = trained_ivatmeans.predict(X_test)

        assert labels.shape == (50000,)
        assert np.all((labels >= 0) & (labels < 3))

    def test_batch_consistency(self, trained_ivatmeans):
        """Verify batched prediction matches non-batched prediction."""
        np.random.seed(123)
        X_test = np.random.randn(5000, 5).astype(np.float64)

        # Predict without batching (all at once)
        labels_direct = trained_ivatmeans.predict(X_test, batch_size=X_test.shape[0])

        # Predict with batching
        labels_batched = trained_ivatmeans.predict(X_test, batch_size=1000)

        # Should be identical
        assert_array_equal(labels_direct, labels_batched)

    def test_batch_size_parameter(self, trained_ivatmeans):
        """Test different batch sizes produce same results."""
        np.random.seed(123)
        X_test = np.random.randn(2500, 5).astype(np.float64)

        # Test multiple batch sizes
        batch_sizes = [100, 500, 1000, 5000]
        predictions = []

        for batch_size in batch_sizes:
            labels = trained_ivatmeans.predict(X_test, batch_size=batch_size)
            predictions.append(labels)

        # All should be identical
        for i in range(1, len(predictions)):
            assert_array_equal(predictions[0], predictions[i])

    def test_prediction_with_single_sample(self, trained_ivatmeans):
        """Test prediction with single sample."""
        X_test = np.random.randn(1, 5).astype(np.float64)
        labels = trained_ivatmeans.predict(X_test)

        assert labels.shape == (1,)
        assert 0 <= labels[0] < 3

    def test_prediction_dtype(self, trained_ivatmeans):
        """Test prediction output dtype."""
        X_test = np.random.randn(100, 5).astype(np.float64)
        labels = trained_ivatmeans.predict(X_test)

        # Output should be integer type (int32 or int64)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_error_on_unfitted_model(self):
        """Test error when predicting with unfitted model."""
        model = IVATMeans(n_clusters=3)
        X_test = np.random.randn(100, 5).astype(np.float64)

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(X_test)

    def test_invalid_input_dimensions(self, trained_ivatmeans):
        """Test error with invalid input dimensions."""
        X_test = np.random.randn(100)  # 1D, should be 2D

        with pytest.raises(ValueError, match="2-dimensional"):
            trained_ivatmeans.predict(X_test)


class TestFuzzyCMeansBatchedPrediction:
    """Test batched prediction for FuzzyCMeans."""

    def test_small_batch_prediction(self, trained_fuzzy_cmeans):
        """Test prediction with small test set (no batching)."""
        X_test = np.random.randn(50, 5).astype(np.float64)
        labels = trained_fuzzy_cmeans.predict(X_test)

        assert labels.shape == (50,)
        assert np.all((labels >= 0) & (labels < 3))

    def test_large_batch_prediction(self, trained_fuzzy_cmeans):
        """Test prediction with large test set (batching occurs)."""
        np.random.seed(123)
        X_test = np.random.randn(50000, 5).astype(np.float64)
        labels = trained_fuzzy_cmeans.predict(X_test)

        assert labels.shape == (50000,)
        assert np.all((labels >= 0) & (labels < 3))

    def test_batch_consistency(self, trained_fuzzy_cmeans):
        """Verify batched prediction matches non-batched prediction."""
        np.random.seed(123)
        X_test = np.random.randn(5000, 5).astype(np.float64)

        # Predict without batching (all at once)
        labels_direct = trained_fuzzy_cmeans.predict(X_test, batch_size=X_test.shape[0])

        # Predict with batching
        labels_batched = trained_fuzzy_cmeans.predict(X_test, batch_size=1000)

        # Should be identical
        assert_array_equal(labels_direct, labels_batched)

    def test_batch_size_parameter(self, trained_fuzzy_cmeans):
        """Test different batch sizes produce same results."""
        np.random.seed(123)
        X_test = np.random.randn(2500, 5).astype(np.float64)

        # Test multiple batch sizes
        batch_sizes = [100, 500, 1000, 5000]
        predictions = []

        for batch_size in batch_sizes:
            labels = trained_fuzzy_cmeans.predict(X_test, batch_size=batch_size)
            predictions.append(labels)

        # All should be identical
        for i in range(1, len(predictions)):
            assert_array_equal(predictions[0], predictions[i])

    def test_prediction_with_single_sample(self, trained_fuzzy_cmeans):
        """Test prediction with single sample."""
        X_test = np.random.randn(1, 5).astype(np.float64)
        labels = trained_fuzzy_cmeans.predict(X_test)

        assert labels.shape == (1,)
        assert 0 <= labels[0] < 3

    def test_prediction_dtype(self, trained_fuzzy_cmeans):
        """Test prediction output dtype."""
        X_test = np.random.randn(100, 5).astype(np.float64)
        labels = trained_fuzzy_cmeans.predict(X_test)

        # Output should be integer type (int32 or int64)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_error_on_unfitted_model(self):
        """Test error when predicting with unfitted model."""
        model = FuzzyCMeans(n_clusters=3)
        X_test = np.random.randn(100, 5).astype(np.float64)

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(X_test)

    def test_invalid_input_dimensions(self, trained_fuzzy_cmeans):
        """Test error with invalid input dimensions."""
        X_test = np.random.randn(100)  # 1D, should be 2D

        with pytest.raises(ValueError, match="2-dimensional"):
            trained_fuzzy_cmeans.predict(X_test)


class TestBatchedPredictionMemoryUsage:
    """Test memory efficiency of batched prediction."""

    def test_memory_efficiency_ivatmeans(self):
        """IVATMeans batched prediction uses less memory than direct."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50).astype(np.float64)
        X_test = np.random.randn(100000, 50).astype(np.float64)

        model = IVATMeans(n_clusters=5)
        model.fit(X_train)

        # Batched prediction should complete without memory error
        labels = model.predict(X_test, batch_size=5000)
        assert labels.shape == (100000,)

    def test_memory_efficiency_fuzzy_cmeans(self):
        """FuzzyCMeans batched prediction uses less memory than direct."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50).astype(np.float64)
        X_test = np.random.randn(100000, 50).astype(np.float64)

        model = FuzzyCMeans(n_clusters=5)
        model.fit(X_train)

        # Batched prediction should complete without memory error
        labels = model.predict(X_test, batch_size=5000)
        assert labels.shape == (100000,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
