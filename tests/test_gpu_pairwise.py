"""Tests for the optional CuPy GPU pairwise-distance path.

Skipped entirely when no CUDA device / CuPy is available, so the suite still
passes on CPU-only machines.
"""
import numpy as np
import pytest

from tribbleclustering import gpu
from tribbleclustering.pcvat import pairwise_distances_c_32, pairwise_distances_c_64

pytestmark = pytest.mark.skipif(
    not gpu.is_available(), reason="no CUDA device / CuPy available"
)


def _blobs(n, d, dtype, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.standard_normal((n, d)) * 3.0).astype(dtype))


@pytest.mark.parametrize("n,d", [(1, 3), (2, 1), (300, 8), (1000, 32)])
def test_gpu_matches_cpu_f64_exact(n, d):
    X = _blobs(n, d, np.float64)
    Dg = gpu.pairwise_distances_gpu(X)
    Dc = pairwise_distances_c_64(X)
    # float64 direct accumulation reproduces the CPU kernel to rounding.
    assert np.max(np.abs(Dg - Dc)) < 1e-10
    assert np.all(np.diag(Dg) == 0)
    assert np.array_equal(Dg, Dg.T)


@pytest.mark.parametrize("n,d", [(300, 8), (1500, 16)])
def test_gpu_matches_cpu_f32(n, d):
    X = _blobs(n, d, np.float32)
    Dg = gpu.pairwise_distances_gpu(X)  # high_precision -> double accum
    Dc = pairwise_distances_c_32(X)
    assert np.max(np.abs(Dg.astype(np.float64) - Dc.astype(np.float64))) < 1e-3
    assert Dg.dtype == np.float32


def test_tiling_matches_single_shot():
    X = _blobs(1500, 10, np.float64, seed=2)
    full = gpu.pairwise_distances_gpu(X)
    tiled = gpu.pairwise_distances_gpu(X, tile_rows=37)  # awkward tile height
    assert np.array_equal(full, tiled)


def test_fast_f32_is_close():
    X = _blobs(1200, 20, np.float32, seed=3)
    Dc = pairwise_distances_c_32(X).astype(np.float64)
    fast = gpu.pairwise_distances_gpu(X, high_precision=False).astype(np.float64)
    assert np.max(np.abs(fast - Dc)) < 1e-2  # native-precision accum, looser


def test_dispatcher_backends_agree():
    X = _blobs(800, 10, np.float64, seed=4)
    cpu = gpu.pairwise_distances(X, backend="cpu")
    g = gpu.pairwise_distances(X, backend="gpu")
    assert np.max(np.abs(cpu - g)) < 1e-10


def test_bad_dtype_rejected():
    with pytest.raises(TypeError):
        gpu.pairwise_distances_gpu(np.zeros((4, 3), dtype=np.int32))
