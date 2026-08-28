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

# Import the VAT/iVAT front end through ivatmeans so these tests exercise
# whichever kernel the class itself picked (compiled pcvat or pure numba).
from tribbleclustering.ivatmeans import _compute_ivat, _pairwise_distances
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
        # Beta-spread fires exactly when `r` is NOT of negative type, i.e. when
        # no point configuration has `r` as its squared distances. A near-clique
        # with one hugely stretched edge is the simplest such matrix.
        #
        # The matrix this test used to build -- uniform(0.5, 1.0), symmetrized,
        # hollow -- is of negative type (min eigenvalue of -0.5 J r J measured
        # at +2.4e-16), so beta was always 0.0 and the only assertion made on
        # it, `beta >= 0.0`, could not fail. See issue #89.
        n = 6
        r = np.ones((n, n)) - np.eye(n)
        r[0, 1] = r[1, 0] = 20.0
        assert not TestBetaSpread.is_of_negative_type(r)

        u, beta = relational_fuzzy_c_means(
            r, n_clusters=2, beta_spread=True, random_state=0
        )
        assert u.shape == (n, 2)
        assert np.allclose(u.sum(axis=1), 1.0)
        assert beta > 0.0

        # Flag plumbing only: `beta_total` is incremented solely inside the
        # `if beta_spread:` branch, so this can never fail. It documents the
        # opt-out, it is not a second control.
        _, beta_off = relational_fuzzy_c_means(
            r, n_clusters=2, beta_spread=False, random_state=0
        )
        assert beta_off == 0.0

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


class TestBetaSpread:
    """Issue #89: the beta-spread docstring stated its trigger condition backwards.

    It claimed the correction was needed whenever ``r`` induces negative
    relational distances, "always true for a genuinely non-Euclidean ``r``,
    such as the iVAT minimax matrix". The iVAT matrix is the subdominant
    ultrametric u(D), and ultrametrics have strict p-negative type for every
    p >= 0 (Faver, Kochalski, Murugan, Verheggen, Wesson & Weston, *Roundness
    properties of ultrametric spaces*, Glasgow Math. J. 56(3):519-535, 2014),
    so it is the one input on which beta-spread can never fire.
    """

    @staticmethod
    def is_of_negative_type(r: np.ndarray) -> bool:
        """Schoenberg's criterion: a hollow symmetric ``r`` is of negative type
        iff ``-0.5 J r J`` is PSD -- equivalently, iff some point configuration
        has ``r`` as its matrix of squared distances."""
        n = r.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * centering @ r @ centering
        eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2.0)
        tolerance = 1e-10 * max(float(np.abs(eigenvalues).max()), 1.0)
        return bool(eigenvalues.min() > -tolerance)

    @staticmethod
    def minimax_matrix(X: np.ndarray) -> np.ndarray:
        ivat_matrix, _, _ = _compute_ivat(_pairwise_distances(X))
        r = np.asarray(ivat_matrix, dtype=np.float64)
        r = (r + r.T) / 2.0
        np.fill_diagonal(r, 0.0)
        return r

    def test_ivat_matrix_is_an_ultrametric(self, rings):
        X, _ = rings
        r = self.minimax_matrix(X)
        for k in range(0, r.shape[0], 7):  # strided; the full check is O(n^3)
            bound = np.maximum(r[:, k][:, np.newaxis], r[k, :][np.newaxis, :])
            assert np.all(
                r <= bound + 1e-9
            ), "minimax matrix violates u[i,j] <= max(u[i,k], u[k,j])"

    def test_minimax_matrix_is_of_negative_type_at_both_powers(self, rings):
        X, _ = rings
        r = self.minimax_matrix(X)
        assert self.is_of_negative_type(r)
        assert self.is_of_negative_type(r**2)

    def test_beta_spread_never_fires_on_a_minimax_matrix(self, rings):
        X, _ = rings
        r = self.minimax_matrix(X)
        for label, matrix in (("u(D)", r), ("u(D)**2", r**2)):
            _, beta = relational_fuzzy_c_means(
                matrix, n_clusters=2, beta_spread=True, random_state=0
            )
            assert beta == 0.0, f"beta-spread fired on {label}, of negative type"

    def test_ivatmeans_relational_fit_never_needs_beta_spread(self, rings):
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)
        assert model._relational_beta == 0.0


class TestRelationalOutOfSample:
    """The out-of-sample extension must agree with the partition it was fit on."""

    def test_predict_on_training_data_reproduces_the_fitted_labels(self, rings):
        """The single-linkage insertion of a training point is that point's own
        row, so out-of-sample scoring must reproduce the in-sample partition.
        This pins the fit and predict sides to the same units: any rescaling of
        the training matrix that is not mirrored in the inserted edge breaks it.
        """
        X, _ = rings
        model = IVATMeans(n_clusters=2, refine="relational", random_state=42)
        model.fit(X)
        assert np.array_equal(model.predict(X), model.labels_)
