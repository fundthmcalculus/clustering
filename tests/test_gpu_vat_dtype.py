"""Tests for the multi-precision (f16/f32/f64) on-device VAT/iVAT additions:
templated Boruvka, the fully on-device iVAT recurrence, and the IVATMeans dtype
policy. Skipped when no CUDA device / CuPy is available.
"""

import warnings

import numpy as np
import pytest

import tribbleclustering as tc
from tribbleclustering import gpu, gpu_vat
from tribbleclustering.pcvat import pairwise_distances_c_64, compute_ivat_c

pytestmark = pytest.mark.skipif(
    not gpu.is_available(), reason="no CUDA device / CuPy available"
)


def _blobs(n_per, d, k, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-30, 30, size=(k, d))
    X = np.vstack([rng.standard_normal((n_per, d)) + centers[i] for i in range(k)])
    return np.ascontiguousarray(X.astype(dtype))


def _prim_mst_weight(D):
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    best = D[0].copy()
    best[0] = np.inf
    in_tree[0] = True
    total = 0.0
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        total += float(best[j])
        in_tree[j] = True
        best = np.minimum(best, D[j])
    return total


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_boruvka_device_valid_mst_all_dtypes(dtype):
    import cupy as cp

    X = _blobs(400, 8, 5, seed=1)
    Dg = gpu.pairwise_distances_device(X, dtype=dtype)
    assert cp.dtype(Dg.dtype).name == dtype
    mu, mv = gpu_vat.boruvka_mst_device(Dg)
    n = X.shape[0]
    assert mu.shape[0] == n - 1
    # connected spanning tree
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(cp.asnumpy(mu).tolist(), cp.asnumpy(mv).tolist()):
        parent[find(a)] = find(b)
    assert len({find(i) for i in range(n)}) == 1


def test_boruvka_f32_weight_matches_reference():
    import cupy as cp

    X = _blobs(500, 8, 6, seed=2)
    Dg = gpu.pairwise_distances_device(X, dtype="float32")
    mu, mv = gpu_vat.boruvka_mst_device(Dg)
    got = float(cp.asnumpy(Dg[mu, mv]).astype(np.float64).sum())
    ref = _prim_mst_weight(pairwise_distances_c_64(X))
    assert np.isclose(got, ref, rtol=1e-4)


def test_ivat_image_device_bit_exact_f64():
    """Fed the same f64 matrix, the on-device iVAT recurrence is bit-identical
    to the CPU engine."""
    X = _blobs(400, 8, 5, seed=3)
    Dc = pairwise_distances_c_64(X)
    iv_cpu, _, order = compute_ivat_c(Dc.copy(), inplace=False)
    V = gpu_vat.ivat_image_device(Dc, order, v_dtype="float64")
    assert np.max(np.abs(iv_cpu - np.asarray(V.get()))) == 0.0


def test_ivat_gpu_default_is_exact_and_device_recurrence():
    """ivat_gpu() with no dtype keeps input precision and closes the loop on the
    device; the image matches the CPU engine to fp rounding."""
    X = _blobs(400, 8, 5, seed=4)  # float64
    iv_cpu, _, order_cpu = compute_ivat_c(pairwise_distances_c_64(X).copy(), False)
    iv_gpu, order_gpu = gpu_vat.ivat_gpu(X)  # dtype=None -> f64, device recurrence
    assert np.array_equal(order_gpu, order_cpu)
    assert np.max(np.abs(iv_cpu - iv_gpu)) < 1e-9


def test_ivat_gpu_f32_matches_order():
    X = _blobs(400, 8, 5, seed=5)
    _, _, order_cpu = compute_ivat_c(pairwise_distances_c_64(X).copy(), False)
    iv_gpu, order_gpu = gpu_vat.ivat_gpu(X, dtype="float32")
    assert np.array_equal(order_gpu, order_cpu)  # f32 order is exact here
    assert iv_gpu.dtype == np.float32


def test_ivatmeans_dtype_float64_warns():
    X = _blobs(200, 6, 4, seed=6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tc.IVATMeans(n_clusters=4, on_device=True, dtype="float64").fit(X)
    assert any("float32" in str(rec.message) for rec in w)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_ivatmeans_dtype_supported_no_warning(dtype):
    X = _blobs(200, 6, 4, seed=7)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tc.IVATMeans(n_clusters=4, on_device=True, dtype=dtype).fit(X)
    assert not any(
        "converted" in str(rec.message) and "float32" in str(rec.message) for rec in w
    )


def test_ivatmeans_on_device_f32_matches_cpu_partition():
    X = _blobs(400, 6, 4, seed=8)
    dev = tc.IVATMeans(n_clusters=4, on_device=True, dtype="float32").fit_predict(X)
    cpu = tc.IVATMeans(n_clusters=4, on_device=False).fit_predict(X)

    from collections import Counter

    def agreement(a, b):
        s = 0
        for lab in set(a.tolist()):
            members = np.where(a == lab)[0]
            if len(members):
                maj = Counter(b[members].tolist()).most_common(1)[0][0]
                s += int(np.sum(b[members] == maj))
        return s / len(a)

    assert agreement(dev, cpu) >= 0.99
