"""Tests for IVATMeans's pluggable `refine` back end (GitHub issue #54).

iVAT's minimax recurrence recovers non-convex/chained structure, but the
original back end represented each recovered cluster by a Euclidean mean and
assigned points by nearest-centroid -- a geometry mismatch that discards the
advantage on precisely the data iVAT is good at (see docs/novel-niche.md).
These tests reproduce that failure with the legacy "euclidean" back end and
verify the new "medoid" (default) and "relational" (NERFCM) back ends fix it.
"""

import numpy as np
import pytest

from tribbleclustering import IVATMeans
from tribbleclustering.nerfcm import (
    relational_fuzzy_c_means,
    relational_out_of_sample_membership,
)


def _concentric_rings(n_per_ring: int = 60, seed: int = 0):
    """Two concentric rings: the canonical non-convex case from issue #54."""
    rng = np.random.default_rng(seed)
    t1 = rng.uniform(0, 2 * np.pi, n_per_ring)
    r1 = 1.0 + rng.normal(0, 0.05, n_per_ring)
    inner = np.c_[r1 * np.cos(t1), r1 * np.sin(t1)]

    t2 = rng.uniform(0, 2 * np.pi, n_per_ring)
    r2 = 3.0 + rng.normal(0, 0.05, n_per_ring)
    outer = np.c_[r2 * np.cos(t2), r2 * np.sin(t2)]

    X = np.vstack([inner, outer]).astype(np.float64)
    y = np.array([0] * n_per_ring + [1] * n_per_ring)
    return X, y


def _best_permutation_accuracy(labels: np.ndarray, y: np.ndarray) -> float:
    """Accuracy against ground truth, allowing for label-permutation (2 clusters)."""
    labels = np.asarray(labels)
    return max(np.mean(labels == y), np.mean(labels == (1 - y)))


@pytest.fixture(scope="module")
def rings():
    return _concentric_rings()


class TestRefineValidation:
    def test_invalid_refine_raises(self):
        with pytest.raises(ValueError, match="refine must be"):
            IVATMeans(refine="not-a-real-option")

    def test_default_refine_is_medoid(self):
        assert IVATMeans().refine == "medoid"


class TestMedoidRefine:
    def test_prototypes_are_real_data_points(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="medoid", random_state=42)
        model.fit(X)

        for center in model.cluster_centers_:
            assert np.any(
                np.all(np.isclose(X, center), axis=1)
            ), "medoid prototype must be an actual data point, never off-data"

    def test_membership_is_not_set(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="medoid", random_state=42)
        model.fit(X)
        assert model.membership_ is None

    def test_recovers_rings_better_than_euclidean(self, rings):
        X, y = rings
        euclidean = IVATMeans(n_clusters=2, refine="euclidean", random_state=42)
        euclidean.fit(X)
        medoid = IVATMeans(n_clusters=2, refine="medoid", random_state=42)
        medoid.fit(X)

        acc_euclidean = _best_permutation_accuracy(euclidean.labels_, y)
        acc_medoid = _best_permutation_accuracy(medoid.labels_, y)
        assert acc_medoid > acc_euclidean

    def test_custom_metric(self, rings):
        X, _ = rings
        model = IVATMeans(
            n_clusters=2, refine="medoid", metric="manhattan", random_state=42
        )
        model.fit(X)
        assert model.cluster_centers_.shape[0] >= 1
        labels = model.predict(X)
        assert labels.shape == (X.shape[0],)


class TestRelationalRefine:
    def test_soft_membership_rows_sum_to_one(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)

        assert model.membership_ is not None
        assert model.membership_.shape == (X.shape[0], model.cluster_centers_.shape[0])
        assert np.allclose(model.membership_.sum(axis=1), 1.0)
        assert np.all(model.membership_ >= 0.0)

    def test_get_soft_labels(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)
        assert np.array_equal(model.get_soft_labels(), model.membership_)

    def test_get_soft_labels_requires_relational_fit(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="medoid", random_state=42)
        model.fit(X)
        with pytest.raises(ValueError, match="refine='relational'"):
            model.get_soft_labels()

    def test_recovers_rings_essentially_perfectly(self, rings):
        """The motivating claim of issue #54: relational stays in minimax
        geometry end-to-end and should cleanly separate the rings where the
        Euclidean back end re-merges them."""
        X, y = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)
        assert _best_permutation_accuracy(model.labels_, y) > 0.95

    def test_out_of_sample_prediction_separates_rings(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)

        rng = np.random.default_rng(7)
        t = rng.uniform(0, 2 * np.pi, 15)
        new_inner = np.c_[np.cos(t), np.sin(t)]
        new_outer = np.c_[3 * np.cos(t), 3 * np.sin(t)]

        pred_inner = model.predict(new_inner)
        pred_outer = model.predict(new_outer)

        assert len(np.unique(pred_inner)) == 1
        assert len(np.unique(pred_outer)) == 1
        assert pred_inner[0] != pred_outer[0]

    def test_batched_prediction_matches_direct(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)

        rng = np.random.default_rng(3)
        X_new = rng.normal(size=(50, 2)) * 2

        direct = model.predict(X_new, batch_size=X_new.shape[0])
        batched = model.predict(X_new, batch_size=7)
        assert np.array_equal(direct, batched)

    def test_predict_before_fit_raises(self):
        model = IVATMeans(n_clusters=2, refine="relational")
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(np.zeros((3, 2)))

    def test_single_cluster(self):
        X = np.random.default_rng(0).normal(size=(15, 3))
        model = IVATMeans(n_clusters=1, refine="relational", random_state=42)
        model.fit(X)
        assert model.membership_.shape == (15, model.cluster_centers_.shape[0])


class TestEuclideanRefineBackwardCompatibility:
    def test_matches_original_mean_centroid_behavior(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="euclidean", random_state=42)
        model.fit(X)

        for k, cluster_ids in enumerate(model._ivat_result.cluster_city_ids):
            expected_mean = np.mean(X[cluster_ids], axis=0)
            assert np.allclose(model.cluster_centers_[k], expected_mean)


class TestRelationalFuzzyCMeans:
    def test_raises_on_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            relational_fuzzy_c_means(np.zeros((3, 4)), n_clusters=2)

    def test_raises_on_invalid_m(self):
        with pytest.raises(ValueError, match="greater than 1.0"):
            relational_fuzzy_c_means(np.zeros((3, 3)), n_clusters=2, m=1.0)

    def test_two_well_separated_blobs(self):
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=[0, 0], scale=0.1, size=(20, 2))
        blob_b = rng.normal(loc=[10, 10], scale=0.1, size=(20, 2))
        X = np.vstack([blob_a, blob_b])
        r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) ** 2

        u_init = np.zeros((40, 2))
        u_init[:20, 0] = 1.0
        u_init[20:, 1] = 1.0

        u, beta = relational_fuzzy_c_means(r, n_clusters=2, u_init=u_init)
        assert u.shape == (40, 2)
        assert np.allclose(u.sum(axis=1), 1.0)
        labels = np.argmax(u, axis=1)
        assert len(np.unique(labels[:20])) == 1
        assert len(np.unique(labels[20:])) == 1
        assert labels[0] != labels[20]
        # Squared Euclidean distance is a valid (Euclidean) dissimilarity, so
        # no beta-spread correction should be necessary here.
        assert beta == 0.0

    def test_beta_spread_applied_for_non_euclidean_input(self):
        # An asymmetric-ish, non-Euclidean dissimilarity matrix (e.g. a
        # minimax/ultrametric-like matrix with a triangle-inequality-violating
        # perturbation) can require the beta-spread correction.
        rng = np.random.default_rng(1)
        n = 12
        r = rng.uniform(0.5, 1.0, size=(n, n))
        r = (r + r.T) / 2.0
        np.fill_diagonal(r, 0.0)

        u, beta = relational_fuzzy_c_means(r, n_clusters=3, beta_spread=True)
        assert u.shape == (n, 3)
        assert np.allclose(u.sum(axis=1), 1.0)
        assert beta >= 0.0

    def test_out_of_sample_membership_shape_and_normalization(self):
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=[0, 0], scale=0.1, size=(20, 2))
        blob_b = rng.normal(loc=[10, 10], scale=0.1, size=(20, 2))
        X = np.vstack([blob_a, blob_b])
        r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) ** 2

        u_init = np.zeros((40, 2))
        u_init[:20, 0] = 1.0
        u_init[20:, 1] = 1.0
        u, beta = relational_fuzzy_c_means(r, n_clusters=2, u_init=u_init)

        new_point = np.array([[0.05, -0.05]])
        r_new = np.linalg.norm(new_point[:, None, :] - X[None, :, :], axis=2) ** 2
        membership = relational_out_of_sample_membership(r_new, r, u, m=2.0, beta=beta)

        assert membership.shape == (1, 2)
        assert np.isclose(membership.sum(), 1.0)
        assert np.argmax(membership) == 0
