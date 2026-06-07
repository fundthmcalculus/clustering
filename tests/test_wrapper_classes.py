import numpy as np
import pytest
from tribbleclustering import FuzzyCMeans, IVATMeans
from tribbleclustering.util import circle_random_clusters


@pytest.fixture
def simple_data():
    """Create simple circular cluster data for testing."""
    return circle_random_clusters(
        n_clusters=3,
        n_cities=10,
        cluster_spacing=5.0,
        cluster_diameter=0.5,
    )


@pytest.fixture
def single_cluster_data():
    """Create data with a single cluster."""
    return np.random.randn(20, 2) + [0, 0]


class TestFuzzyCMeans:
    def test_init_default(self):
        """Test FuzzyCMeans initialization with defaults."""
        fcm = FuzzyCMeans(n_clusters=3)
        assert fcm.n_clusters == 3
        assert fcm.m == 2.0
        assert fcm.random_state is None
        assert fcm.cluster_centers_ is None
        assert fcm.labels_ is None
        assert fcm.membership_matrix_ is None

    def test_init_custom_m(self):
        """Test FuzzyCMeans initialization with custom m parameter."""
        fcm = FuzzyCMeans(n_clusters=2, m=1.5)
        assert fcm.m == 1.5

    def test_init_with_random_state(self):
        """Test FuzzyCMeans initialization with random_state."""
        fcm = FuzzyCMeans(n_clusters=2, random_state=42)
        assert fcm.random_state == 42

    def test_fit(self, simple_data):
        """Test FuzzyCMeans fit method."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        result = fcm.fit(simple_data)

        assert result is fcm, "fit should return self"
        assert fcm.cluster_centers_ is not None
        assert fcm.labels_ is not None
        assert fcm.membership_matrix_ is not None
        assert fcm.cluster_centers_.shape == (3, 2)
        assert fcm.labels_.shape == (30,)
        assert fcm.membership_matrix_.shape == (30, 3)

    def test_fit_invalid_data_shape(self, simple_data):
        """Test fit with invalid data shape."""
        fcm = FuzzyCMeans(n_clusters=3)
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            fcm.fit(simple_data.flatten())

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        fcm = FuzzyCMeans(n_clusters=3)
        with pytest.raises(ValueError, match="Model has not been fitted"):
            fcm.predict(np.random.rand(5, 2))

    def test_predict(self, simple_data):
        """Test FuzzyCMeans predict method."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        fcm.fit(simple_data)

        predictions = fcm.predict(simple_data)
        assert predictions.shape == (30,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_predict_new_data(self, simple_data):
        """Test predicting on new data."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        fcm.fit(simple_data)

        new_data = np.random.rand(5, 2) * 3
        predictions = fcm.predict(new_data)
        assert predictions.shape == (5,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_fit_predict(self, simple_data):
        """Test FuzzyCMeans fit_predict method."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        labels = fcm.fit_predict(simple_data)

        assert labels.shape == (30,)
        assert np.array_equal(labels, fcm.labels_)

    def test_get_soft_labels_before_fit(self):
        """Test that get_soft_labels raises error before fit."""
        fcm = FuzzyCMeans(n_clusters=3)
        with pytest.raises(ValueError, match="Model has not been fitted"):
            fcm.get_soft_labels()

    def test_get_soft_labels(self, simple_data):
        """Test FuzzyCMeans get_soft_labels method."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        fcm.fit(simple_data)

        soft_labels = fcm.get_soft_labels()
        assert soft_labels.shape == (30, 3)
        assert np.allclose(soft_labels.sum(axis=1), 1.0)
        assert np.all(soft_labels >= 0)
        assert np.all(soft_labels <= 1)

    def test_different_n_clusters(self, simple_data):
        """Test FuzzyCMeans with different cluster numbers."""
        for n_clusters in [1, 2, 4, 5]:
            fcm = FuzzyCMeans(n_clusters=n_clusters, random_state=42)
            fcm.fit(simple_data)
            assert fcm.cluster_centers_.shape == (n_clusters, 2)
            assert fcm.membership_matrix_.shape == (30, n_clusters)

    def test_reproducibility_with_random_state(self, simple_data):
        """Test that same random_state produces same results."""
        fcm1 = FuzzyCMeans(n_clusters=3, random_state=42)
        fcm1.fit(simple_data.copy())
        labels1 = fcm1.labels_.copy()

        fcm2 = FuzzyCMeans(n_clusters=3, random_state=42)
        fcm2.fit(simple_data.copy())
        labels2 = fcm2.labels_

        assert np.array_equal(labels1, labels2)


class TestIVATMeans:
    def test_init_default(self):
        """Test IVATMeans initialization with defaults."""
        ivat = IVATMeans()
        assert ivat.n_clusters == 2
        assert ivat.random_state is None
        assert ivat.cluster_centers_ is None
        assert ivat.labels_ is None

    def test_init_custom_n_clusters(self):
        """Test IVATMeans initialization with custom n_clusters."""
        ivat = IVATMeans(n_clusters=5)
        assert ivat.n_clusters == 5

    def test_init_with_random_state(self):
        """Test IVATMeans initialization with random_state."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        assert ivat.random_state == 42

    def test_fit(self, simple_data):
        """Test IVATMeans fit method."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        result = ivat.fit(simple_data)

        assert result is ivat, "fit should return self"
        assert ivat.cluster_centers_ is not None
        assert ivat.labels_ is not None
        assert ivat.cluster_centers_.shape[1] == 2  # same feature dimension
        assert ivat.labels_.shape == (30,)

    def test_fit_invalid_data_shape(self, simple_data):
        """Test fit with invalid data shape."""
        ivat = IVATMeans(n_clusters=3)
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            ivat.fit(simple_data.flatten())

    def test_fit_with_varying_n_clusters(self, simple_data):
        """Test that fitting with varying n_clusters produces valid results."""
        for n_clusters in [1, 2, 3]:
            ivat = IVATMeans(n_clusters=n_clusters, random_state=42)
            ivat.fit(simple_data)
            # n_clusters controls hierarchical levels, not final cluster count
            # The actual number of clusters depends on the peaks detected in the data
            assert ivat.cluster_centers_.shape[0] >= 1
            assert len(ivat.labels_) == len(simple_data)

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        ivat = IVATMeans(n_clusters=3)
        with pytest.raises(ValueError, match="Model has not been fitted"):
            ivat.predict(np.random.rand(5, 2))

    def test_predict(self, simple_data):
        """Test IVATMeans predict method."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        ivat.fit(simple_data)

        predictions = ivat.predict(simple_data)
        assert predictions.shape == (30,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_predict_new_data(self, simple_data):
        """Test predicting on new data."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        ivat.fit(simple_data)

        new_data = np.random.rand(5, 2) * 3
        predictions = ivat.predict(new_data)
        assert predictions.shape == (5,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_fit_predict(self, simple_data):
        """Test IVATMeans fit_predict method."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        labels = ivat.fit_predict(simple_data)

        assert labels.shape == (30,)
        assert np.array_equal(labels, ivat.labels_)

    def test_single_cluster(self, single_cluster_data):
        """Test IVATMeans with n_clusters=1."""
        ivat = IVATMeans(n_clusters=1, random_state=42)
        ivat.fit(single_cluster_data)

        # With n_clusters=1, we extract one level from the hierarchy
        assert ivat.cluster_centers_.shape[0] >= 1
        assert len(ivat.labels_) == len(single_cluster_data)
        # All labels should be valid cluster indices
        assert np.all(ivat.labels_ < ivat.cluster_centers_.shape[0])

    def test_labels_within_bounds(self, simple_data):
        """Test that all labels are within valid cluster bounds."""
        ivat = IVATMeans(n_clusters=3, random_state=42)
        ivat.fit(simple_data)

        max_label = np.max(ivat.labels_)
        assert max_label < ivat.n_clusters

    def test_reproducibility_with_random_state(self, simple_data):
        """Test that same random_state produces same results."""
        ivat1 = IVATMeans(n_clusters=3, random_state=42)
        ivat1.fit(simple_data.copy())
        labels1 = ivat1.labels_.copy()

        ivat2 = IVATMeans(n_clusters=3, random_state=42)
        ivat2.fit(simple_data.copy())
        labels2 = ivat2.labels_

        assert np.array_equal(labels1, labels2)


class TestComparisonFuzzyCMeansVsIVATMeans:
    def test_both_algorithms_produce_valid_labels(self, simple_data):
        """Test that both algorithms produce valid cluster labels."""
        fcm = FuzzyCMeans(n_clusters=3, random_state=42)
        ivat = IVATMeans(n_clusters=3, random_state=42)

        fcm.fit(simple_data)
        ivat.fit(simple_data)

        # Both should have same data size and cluster range
        assert fcm.labels_.shape == ivat.labels_.shape
        assert np.all(fcm.labels_ < 3)
        assert np.all(ivat.labels_ < 3)

    def test_both_algorithms_have_correct_cluster_centers(self, simple_data):
        """Test that both algorithms have valid cluster centers."""
        n_clusters = 3
        fcm = FuzzyCMeans(n_clusters=n_clusters, random_state=42)
        ivat = IVATMeans(n_clusters=n_clusters, random_state=42)

        fcm.fit(simple_data)
        ivat.fit(simple_data)

        # FuzzyCMeans should have exactly n_clusters
        assert fcm.cluster_centers_.shape[0] == n_clusters
        # IVATMeans may have different number of clusters depending on detected peaks
        assert ivat.cluster_centers_.shape[0] >= 1
        # Both should have correct feature dimension
        assert fcm.cluster_centers_.shape[1] == simple_data.shape[1]
        assert ivat.cluster_centers_.shape[1] == simple_data.shape[1]
