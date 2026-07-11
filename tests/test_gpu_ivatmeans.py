"""IVATMeans GPU distance-backend routing.

Skipped when no CUDA device / CuPy is available.
"""
import numpy as np
import pytest

from tribbleclustering import gpu
from tribbleclustering.ivatmeans import IVATMeans

pytestmark = pytest.mark.skipif(
    not gpu.is_available(), reason="no CUDA device / CuPy available"
)


def _blobs(n, d, k, dtype, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-30, 30, (k, d))
    lbl = rng.integers(0, k, n)
    X = rng.standard_normal((n, d)) * 2.0 + centers[lbl]
    return np.ascontiguousarray(X.astype(dtype))


def test_gpu_cpu_identical_labels_f32():
    """GPU float32 distances are bit-identical to the CPU kernel (both
    accumulate in double), so the whole VAT ordering and the resulting labels
    must match exactly."""
    X = _blobs(3000, 96, 6, np.float32, seed=1)  # d>=64 -> auto picks GPU
    cpu = IVATMeans(n_clusters=6, distance_backend="cpu").fit(X)
    g = IVATMeans(n_clusters=6, distance_backend="gpu").fit(X)
    assert np.array_equal(cpu.labels_, g.labels_)
    assert np.array_equal(cpu.cluster_centers_, g.cluster_centers_)


def test_auto_uses_gpu_for_highdim_f32_only():
    X_hi = _blobs(500, 96, 4, np.float32)
    X_lo = _blobs(500, 8, 4, np.float32)
    X_f64 = _blobs(500, 96, 4, np.float64)
    assert gpu.gpu_pairwise_beneficial(X_hi) is True
    assert gpu.gpu_pairwise_beneficial(X_lo) is False   # low dimension
    assert gpu.gpu_pairwise_beneficial(X_f64) is False  # float64 loses on GPU


def test_auto_matches_cpu_labels():
    X = _blobs(2500, 128, 5, np.float32, seed=2)
    auto = IVATMeans(n_clusters=5, distance_backend="auto").fit(X)  # -> GPU
    cpu = IVATMeans(n_clusters=5, distance_backend="cpu").fit(X)
    assert np.array_equal(auto.labels_, cpu.labels_)


def test_bad_backend_rejected():
    X = _blobs(200, 8, 3, np.float32)
    with pytest.raises(ValueError):
        IVATMeans(n_clusters=3, distance_backend="tpu").fit(X)
