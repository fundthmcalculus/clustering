"""Tests for the fully on-device GPU VAT front-end (distances -> Boruvka -> order).

Skipped when no CUDA device / CuPy is available.
"""

import heapq

import numpy as np
import pytest

from tribbleclustering import gpu, gpu_vat
from tribbleclustering.pcvat import compute_ivat_c, pairwise_distances_c_64

pytestmark = pytest.mark.skipif(
    not gpu.is_available(), reason="no CUDA device / CuPy available"
)


def _blobs(n, d, k, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-40, 40, (k, d))
    lbl = rng.integers(0, k, n)
    X = rng.standard_normal((n, d)) * 2.0 + centers[lbl]
    return np.ascontiguousarray(X.astype(dtype))


def _ivat_from_order(D, order):
    n = len(order)
    V = D[np.ix_(order, order)].copy()
    I = np.zeros_like(V)
    for r in range(1, n):
        jj = int(np.argmin(V[r, :r]))
        mn = V[r, jj]
        for c in range(r):
            if c == jj:
                I[r, c] = mn
            else:
                cur = I[jj, c] if jj > c else I[c, jj]
                I[r, c] = max(mn, cur)
    return I + I.T


@pytest.mark.parametrize("n,d,k", [(300, 6, 5), (1500, 8, 8)])
def test_gpu_vat_order_matches_serial(n, d, k):
    X = _blobs(n, d, k, dtype=np.float32, seed=1)
    order, parent = gpu_vat.vat_gpu(X)
    D = pairwise_distances_c_64(X.astype(np.float64))
    _, _, p_serial = compute_ivat_c(D.copy(), inplace=False)
    assert np.array_equal(order, p_serial)  # exact VAT ordering
    # and the iVAT image built from it is identical to the serial engine's
    img_gpu = _ivat_from_order(D, order.astype(np.int64))
    img_ser, _, _ = compute_ivat_c(D.copy(), inplace=False)
    assert np.max(np.abs(img_gpu - img_ser)) == 0.0


def test_vat_gpu_is_a_permutation():
    X = _blobs(800, 8, 6, seed=2)
    order, parent = gpu_vat.vat_gpu(X)
    assert np.array_equal(np.sort(order), np.arange(len(order)))
    assert parent[order[0]] == -1  # the seed has no parent


def test_return_distances_resident():
    import cupy as cp

    X = _blobs(500, 6, 4, seed=3)
    order, parent, Dg = gpu_vat.vat_gpu(X, return_distances=True)
    assert isinstance(Dg, cp.ndarray)  # matrix stayed on device
    assert Dg.shape == (500, 500)
    # resident matrix matches the CPU distances
    Dc = pairwise_distances_c_64(X)
    assert np.max(np.abs(cp.asnumpy(Dg) - Dc)) < 1e-9


def test_boruvka_device_is_valid_mst():
    import cupy as cp

    X = _blobs(1000, 8, 6, seed=4)
    Dg = gpu.pairwise_distances_device(X)
    mu, mv = gpu_vat.boruvka_mst_device(Dg)
    assert mu.shape[0] == 999  # spanning tree edge count
    # MST weight equals scipy's single-linkage total (unique MST)
    from scipy.sparse.csgraph import minimum_spanning_tree

    Dc = pairwise_distances_c_64(X)
    ref_w = minimum_spanning_tree(Dc).sum()
    got_w = float(cp.asnumpy(Dg[mu, mv]).sum())
    assert abs(got_w - ref_w) < 1e-6 * abs(ref_w)
