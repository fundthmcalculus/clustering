"""Tests for the ConiVAT (constraint-based iVAT) implementation."""

import numpy as np
import pytest

from tribbleclustering import (
    ConiVAT,
    compute_conivat,
    compute_ivat,
    expand_constraints,
    generate_constraints_from_labels,
    learn_metric,
    pairwise_distances,
    transform_with_metric,
)
from tribbleclustering.util import circle_random_clusters

try:
    from tribbleclustering.pcvat import compute_ivat_c  # noqa: F401

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.fixture
def blobs():
    """Three well-separated 2D Gaussian blobs with ground-truth labels."""
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.standard_normal((25, 2)) + [0.0, 0.0],
            rng.standard_normal((25, 2)) + [12.0, 0.0],
            rng.standard_normal((25, 2)) + [0.0, 12.0],
        ]
    ).astype(np.float64)
    y = np.array([0] * 25 + [1] * 25 + [2] * 25)
    return X, y


@pytest.fixture
def simple_data():
    """Three tight circular clusters (shared with the wrapper-class tests)."""
    return circle_random_clusters(
        n_clusters=3, n_cities=10, cluster_spacing=5.0, cluster_diameter=0.5
    )


class TestConstraintGeneration:
    def test_labels_split_into_must_and_cannot(self, blobs):
        _, y = blobs
        ml, cl = generate_constraints_from_labels(y, n_constraints=40, random_state=1)
        # Every generated pair is honestly typed against the labels.
        assert all(y[i] == y[j] for i, j in ml)
        assert all(y[i] != y[j] for i, j in cl)
        assert len(ml) + len(cl) <= 40

    def test_reproducible_with_seed(self, blobs):
        _, y = blobs
        a = generate_constraints_from_labels(y, 30, random_state=7)
        b = generate_constraints_from_labels(y, 30, random_state=7)
        assert a == b

    def test_degenerate_tiny_input(self):
        ml, cl = generate_constraints_from_labels(np.array([0]), 10)
        assert ml == [] and cl == []


class TestExpandConstraints:
    def test_transitive_closure_forms_clique(self):
        # 0-1 and 1-2 must-link => the whole {0,1,2} clique is must-link.
        ml, _ = expand_constraints([(0, 1), (1, 2)], [], n_samples=10)
        assert set(ml) == {(0, 1), (0, 2), (1, 2)}

    def test_cannot_link_propagates_across_components(self):
        # {0,1} vs {8}: one cannot-link becomes every cross pair.
        _, cl = expand_constraints([(0, 1)], [(1, 8)], n_samples=10)
        assert set(cl) == {(0, 8), (1, 8)}

    def test_inconsistent_constraint_dropped(self):
        # (0,1) is must-link AND cannot-link => the cannot-link is inconsistent.
        ml, cl = expand_constraints([(0, 1)], [(0, 1)], n_samples=5)
        assert set(ml) == {(0, 1)}
        assert cl == []

    def test_pairs_normalized_and_sorted(self):
        ml, _ = expand_constraints([(2, 0)], [], n_samples=5)
        assert ml == [(0, 2)]


class TestMetricLearning:
    def test_identity_without_constraints(self, blobs):
        X, _ = blobs
        A = learn_metric(X, [], [])
        assert np.allclose(A, np.eye(X.shape[1]))

    def test_learned_metric_is_psd_and_symmetric(self, blobs):
        X, y = blobs
        ml, cl = generate_constraints_from_labels(y, 40, random_state=2)
        A = learn_metric(X, ml, cl)
        assert A.shape == (X.shape[1], X.shape[1])
        assert np.allclose(A, A.T)
        assert np.all(np.linalg.eigvalsh(A) >= -1e-8)

    def test_transform_matches_mahalanobis(self, blobs):
        X, y = blobs
        ml, cl = generate_constraints_from_labels(y, 40, random_state=3)
        A = learn_metric(X, ml, cl)
        Xt = transform_with_metric(X, A)
        # Euclidean distance in the transformed space == d_A in the original.
        v = X[0] - X[5]
        d_A = np.sqrt(v @ A @ v)
        d_euclid = np.linalg.norm(Xt[0] - Xt[5])
        assert d_A == pytest.approx(d_euclid, rel=1e-6)


class TestComputeConivat:
    def test_reduces_to_ivat_without_constraints(self, blobs):
        X, _ = blobs
        D = pairwise_distances(np.ascontiguousarray(X.astype(np.float64)))
        ivat_matrix, _, ivat_order = compute_ivat(D.copy(), inplace=False)
        cv_matrix, _, cv_order = compute_conivat(
            X, metric_learning=False, backend="python"
        )
        assert np.allclose(ivat_matrix, cv_matrix)
        assert np.array_equal(ivat_order, cv_order)

    def test_invalid_backend_raises(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="backend must be"):
            compute_conivat(X, metric_learning=False, backend="nope")


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
class TestComputeConivatCython:
    def test_cython_matches_python(self, blobs):
        # The compiled and pure paths must stay behaviorally equivalent.
        X, y = blobs
        ml, cl = generate_constraints_from_labels(y, 30, random_state=5)
        py_matrix, _, py_order = compute_conivat(
            X, must_link=ml, cannot_link=cl, backend="python"
        )
        cy_matrix, _, cy_order = compute_conivat(
            X, must_link=ml, cannot_link=cl, backend="cython"
        )
        assert np.allclose(py_matrix, cy_matrix, rtol=1e-5, atol=1e-6)
        assert np.array_equal(py_order, cy_order)

    def test_cython_reduces_to_compiled_ivat(self, blobs):
        # With no constraints / no metric learning, compiled ConiVAT == the
        # optimized compiled iVAT it delegates to.
        from tribbleclustering.pcvat import (
            pairwise_distances_c,
            compute_ivat_c as _civat,
        )

        X, _ = blobs
        Xc = np.ascontiguousarray(X.astype(np.float64))
        iv_matrix, _, iv_order = _civat(pairwise_distances_c(Xc), inplace=False)
        cv_matrix, _, cv_order = compute_conivat(
            X, metric_learning=False, backend="cython"
        )
        assert np.allclose(iv_matrix, cv_matrix)
        assert np.array_equal(iv_order, cv_order)

    def test_output_shapes(self, blobs):
        X, y = blobs
        matrix, argmin_seq, order = compute_conivat(X, labels=y, random_state=1)
        n = X.shape[0]
        assert matrix.shape == (n, n)
        assert len(argmin_seq) == n - 1
        assert order.shape == (n,)
        # A valid VAT ordering is a permutation of all objects.
        assert np.array_equal(np.sort(order), np.arange(n))

    def test_must_link_pairs_are_zeroed(self, blobs):
        # Forcing two far-apart points "similar" collapses their transitive
        # (minimax) dissimilarity to zero along the connecting path.
        X, _ = blobs
        matrix, _, order = compute_conivat(
            X, must_link=[(0, 50)], cannot_link=[], metric_learning=False
        )
        pos = {int(o): p for p, o in enumerate(order)}
        assert matrix[pos[0], pos[50]] == pytest.approx(0.0, abs=1e-12)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            compute_conivat(np.arange(10))

    def test_inplace_matches_copy(self, blobs):
        X, y = blobs
        ml, cl = generate_constraints_from_labels(y, 30, random_state=5)
        a, _, _ = compute_conivat(X, must_link=ml, cannot_link=cl, inplace=False)
        b, _, _ = compute_conivat(X, must_link=ml, cannot_link=cl, inplace=True)
        assert np.allclose(a, b)


class TestConiVATClass:
    def test_init_defaults(self):
        c = ConiVAT()
        assert c.n_clusters == 2
        assert c.metric_learning is True
        assert c.cluster_centers_ is None
        assert c.labels_ is None

    def test_fit_returns_self_and_sets_state(self, blobs):
        X, y = blobs
        c = ConiVAT(n_clusters=3, random_state=1)
        result = c.fit(X, y=y)
        assert result is c
        assert c.cluster_centers_ is not None
        assert c.labels_ is not None
        assert c.cluster_centers_.shape[1] == X.shape[1]
        assert c.labels_.shape == (len(X),)

    def test_fit_invalid_shape(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            ConiVAT().fit(X.ravel())

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match="Model has not been fitted"):
            ConiVAT().predict(np.random.rand(5, 2))

    def test_fit_predict_matches_labels(self, simple_data):
        c = ConiVAT(n_clusters=3, random_state=1)
        labels = c.fit_predict(simple_data)
        assert labels.shape == (len(simple_data),)
        assert np.array_equal(labels, c.labels_)

    def test_recovers_well_separated_blobs(self, blobs):
        # With honest constraints ConiVAT should cleanly separate 3 blobs.
        X, y = blobs
        labels = ConiVAT(n_clusters=3, random_state=1).fit_predict(X, y=y)
        # Each true cluster maps to a single predicted label (purity == 1).
        for cls in np.unique(y):
            assert len(np.unique(labels[y == cls])) == 1

    def test_explicit_constraints_without_labels(self, blobs):
        X, _ = blobs
        c = ConiVAT(
            n_clusters=3,
            must_link=[(0, 1), (25, 26)],
            cannot_link=[(0, 25)],
            random_state=1,
        )
        c.fit(X)
        assert c.labels_.shape == (len(X),)

    def test_reproducible_with_random_state(self, blobs):
        X, y = blobs
        a = ConiVAT(n_clusters=3, random_state=42).fit(X.copy(), y=y).labels_.copy()
        b = ConiVAT(n_clusters=3, random_state=42).fit(X.copy(), y=y).labels_
        assert np.array_equal(a, b)
