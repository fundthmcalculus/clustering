"""Certificate-and-filter MST (dlmst / cdlmst) against dense Prim.

With distinct weights the MST is unique and Prim's visiting order is fixed by
it, so the VAT order ``p``, the parent positions ``q`` and the IVAT matrix
must match the existing implementations exactly, not approximately.
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

import tribbleclustering as tc
from tribbleclustering import dlmst
from tribbleclustering.pvat import compute_ivat, vat_prim_mst
from tribbleclustering.util import pairwise_distances

try:
    from tribbleclustering import cdlmst

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

needs_c = pytest.mark.skipif(not CYTHON_AVAILABLE, reason="cdlmst not built")


def _blobs(n, d=4, c=5, seed=0, dtype=np.float64, spread=1.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-30, 30, size=(c, d))
    X = centers[rng.integers(0, c, n)] + rng.standard_normal((n, d)) * spread
    return np.ascontiguousarray(X.astype(dtype))


def _mst_total(R):
    return float(minimum_spanning_tree(R.astype(np.float64)).sum())


CASES = [
    (2, 2, 1),
    (3, 2, 1),
    (40, 2, 3),
    (300, 5, 7),
    (900, 3, 40),  # many clusters -> several components after the k-NN pass
]


@pytest.mark.parametrize("n,d,c", CASES)
@pytest.mark.parametrize("from_points", [False, True])
def test_py_order_matches_prim(n, d, c, from_points):
    X = _blobs(n, d, c, seed=n)
    R = pairwise_distances(X)
    p0, q0 = vat_prim_mst(R)
    p, q, _ = dlmst.vat_order_py(X if from_points else R, from_points=from_points)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)


@pytest.mark.parametrize("k", [1, 2, 4, 16])
def test_py_mst_weight_matches_scipy(k):
    X = _blobs(400, 3, 12, seed=k)
    R = pairwise_distances(X)
    _, _, w = dlmst.mst_dlmst_py(R, k=k)
    assert w.sum() == pytest.approx(_mst_total(R), rel=1e-12)


def test_py_ivat_matches_compute_ivat():
    X = _blobs(250, 4, 6, seed=3)
    R = pairwise_distances(X)
    D0, a0, p0 = compute_ivat(R.copy())
    D1, a1, p1 = dlmst.compute_ivat_dlmst_py(R)
    np.testing.assert_array_equal(p1, p0)
    assert a1 == list(a0)
    np.testing.assert_allclose(D1, D0, rtol=0, atol=0)


def test_ties_still_give_a_minimum_tree():
    # Integer grid: massive weight ties. The tree may differ from Prim's,
    # but its weight must be minimal.
    g = np.stack(np.meshgrid(np.arange(12), np.arange(9)), -1).reshape(-1, 2)
    R = pairwise_distances(g.astype(np.float64))
    _, _, w = dlmst.mst_dlmst_py(R)
    assert w.sum() == pytest.approx(_mst_total(R))
    if CYTHON_AVAILABLE:
        _, _, wc, _, _ = cdlmst.mst_dlmst_c(R)
        assert wc.sum() == pytest.approx(_mst_total(R))


@needs_c
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n,d,c", CASES + [(3000, 6, 9)])
def test_c_matches_prim(n, d, c, dtype):
    X = _blobs(n, d, c, seed=n, dtype=dtype)
    R = pairwise_distances(X.astype(np.float64)).astype(dtype)
    from tribbleclustering.pcvat import compute_ivat_c, vat_prim_mst_c

    p0, q0 = vat_prim_mst_c(R)
    p, q, _ = cdlmst.vat_order_dlmst_c(R)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)
    D0, a0, _ = compute_ivat_c(R.copy())
    D1, a1, p1 = cdlmst.compute_ivat_dlmst_c(R)
    np.testing.assert_array_equal(p1, p0)
    np.testing.assert_array_equal(a1, a0)
    np.testing.assert_array_equal(D1, D0)


@needs_c
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_c_points_match_matrix(dtype):
    X = _blobs(1500, 5, 8, seed=7, dtype=dtype)
    R = pairwise_distances(X.astype(np.float64))
    p0, q0 = vat_prim_mst(R)
    p, q, _ = cdlmst.vat_order_dlmst_c(X, from_points=True)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)
    D, _, _ = cdlmst.compute_ivat_dlmst_c(X, from_points=True)
    D0, _, _ = compute_ivat(R.copy())
    tol = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(D, D0, rtol=tol, atol=tol)


@needs_c
def test_c_many_components_uses_boruvka_fallback():
    # 600 tight, well-separated clusters with k=1: far more components than
    # the contraction table allows, so the Borůvka merge path runs.
    X = _blobs(2400, 3, 600, seed=11, spread=0.01)
    R = pairwise_distances(X)
    st: dict = {}
    _, _, w, _, _ = cdlmst.mst_dlmst_c(R, k=1, stats=st)
    assert w.sum() == pytest.approx(_mst_total(R), rel=1e-12)
    assert st["passes"] >= 3
    p0, q0 = vat_prim_mst(R)
    p, q, _ = cdlmst.vat_order_dlmst_c(R, k=1)
    np.testing.assert_array_equal(p, p0)


@needs_c
def test_c_thread_count_does_not_change_result():
    X = _blobs(2000, 4, 6, seed=5)
    a = cdlmst.vat_order_dlmst_c(X, from_points=True, nthreads=1)
    b = cdlmst.vat_order_dlmst_c(X, from_points=True, nthreads=4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@needs_c
def test_c_rejects_bad_input():
    with pytest.raises(TypeError):
        cdlmst.vat_order_dlmst_c(np.zeros((4, 4), dtype=np.int64))
    with pytest.raises(ValueError):
        cdlmst.vat_order_dlmst_c(np.zeros((4, 3)))


def test_public_dispatch():
    X = _blobs(200, 3, 4, seed=2)
    R = pairwise_distances(X)
    V0, p0 = tc.compute_vat(R)
    V1, p1 = tc.compute_vat_dlmst(R)
    np.testing.assert_array_equal(p1, p0)
    np.testing.assert_array_equal(V1, V0)
    D0, _, _ = tc.compute_ivat(R.copy())
    D1, _, p2 = tc.compute_ivat_dlmst(X, from_points=True)
    np.testing.assert_array_equal(p2, p0)
    np.testing.assert_allclose(D1, D0, atol=1e-10)
    p3, q3, _ = tc.vat_order_dlmst(X, from_points=True)
    np.testing.assert_array_equal(p3, p0)
