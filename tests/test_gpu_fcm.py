"""Tests for the optional CuPy GPU Fuzzy C-Means path.

Skipped when no CUDA device / CuPy is available.
"""
import numpy as np
import pytest

from tribbleclustering import gpu
from tribbleclustering.fcm import fuzzy_c_means as fcm_cpu
from tribbleclustering.fuzzycmeans import FuzzyCMeans

pytestmark = pytest.mark.skipif(
    not gpu.is_available(), reason="no CUDA device / CuPy available"
)


def _blobs(n, d, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20, 20, (k, d))
    lbl = rng.integers(0, k, n)
    return np.ascontiguousarray(rng.standard_normal((n, d)) * 1.5 + centers[lbl])


@pytest.mark.parametrize("n,d,k", [(2000, 8, 5), (20000, 16, 8)])
def test_gpu_fcm_matches_cpu_fixed_point(n, d, k):
    """Same initial centers -> same converged partition (to fp tolerance)."""
    X = _blobs(n, d, k, seed=1)
    ig = X[np.random.default_rng(0).choice(n, k, replace=False)].copy()
    cc, wc = fcm_cpu(X, k, m=2.0, initial_guess=ig.copy())
    cg, wg = gpu.fuzzy_c_means_gpu(X, k, m=2.0, initial_guess=ig.copy())
    # centers align (identical init, identical update order)
    assert np.max(np.abs(np.sort(cc, axis=0) - np.sort(cg, axis=0))) < 1e-3
    assert np.max(np.abs(wc - wg)) < 1e-3
    # memberships are a valid partition of unity
    assert np.allclose(wg.sum(axis=1), 1.0, atol=1e-5)


def test_gpu_fcm_shapes_and_indices_init():
    X = _blobs(3000, 6, 4, seed=2)
    c, w = gpu.fuzzy_c_means_gpu(X, 4, m=2.0, indices=[0, 1, 2, 3])
    assert c.shape == (4, 6)
    assert w.shape == (3000, 4)


def test_gpu_fcm_rejects_conflicting_init():
    X = _blobs(500, 4, 3)
    with pytest.raises(ValueError):
        gpu.fuzzy_c_means_gpu(X, 3, initial_guess=X[:3].copy(), indices=[0, 1, 2])


def test_fuzzycmeans_wrapper_gpu_matches_cpu_labels():
    X = _blobs(20000, 12, 6, seed=3)
    # identical random_state -> identical initial centers on both backends, so
    # cluster indices align and hard labels can be compared directly.
    cpu = FuzzyCMeans(6, random_state=0, use_gpu=False).fit(X)
    g = FuzzyCMeans(6, random_state=0, use_gpu=True).fit(X)
    agreement = np.mean(cpu.labels_ == g.labels_)
    assert agreement > 0.99, f"label agreement {agreement:.4f}"
