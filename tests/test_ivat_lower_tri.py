"""
Correctness and performance tests for the lower-triangular VAT and IVAT kernels.

Methodology under test
-----------------------
VAT gather: instead of reading adj[P[i],:] sequentially and scattering to
out[i,:] via an inverse permutation, fill only out[i,j] for j<=i via a direct
gather (out[i,j] = adj[P[i],P[j]]), then back-copy lower → upper. Eliminates
the invp array entirely.

IVAT kernel: only the lower triangle is written during the sequential O(n^2)
construction loop; reads of previous IVAT values are canonicalised to the lower
triangle. A separate parallelisable back-copy mirrors lower → upper.

Reference: the Python/Numba implementations in pvat, which produce ground truth.
"""

import time

import numpy as np
import pytest

from tribbleclustering.pvat import (
    compute_ivat as compute_ivat_py,
    compute_vat as compute_vat_py,
)
from tribbleclustering.pcvat import (
    compute_ivat_c_64,
    compute_ivat_c_32,
    compute_vat_c_64,
    compute_vat_c_32,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_sym_dist(n, seed=0, dtype=np.float64):
    """Return a random symmetric distance matrix with zero diagonal."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n)).astype(dtype)
    m = (raw + raw.T) / 2
    np.fill_diagonal(m, 0.0)
    return np.ascontiguousarray(m)


def _py_ivat(dist):
    """Python reference: returns (ivat, argmin_seq, p_seq)."""
    return compute_ivat_py(dist.astype(np.float64))


def _py_vat(dist):
    """Python reference: returns (vat, p_seq)."""
    return compute_vat_py(dist.astype(np.float64))


# ---------------------------------------------------------------------------
# VAT correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_vat_f64_matches_python(n):
    dist = _random_sym_dist(n, seed=n)
    vat_ref, p_ref = _py_vat(dist)

    vat_c, p_c, _ = compute_vat_c_64(np.ascontiguousarray(dist, dtype=np.float64))

    np.testing.assert_allclose(
        vat_c, vat_ref, rtol=1e-10, atol=1e-12, err_msg=f"VAT mismatch (n={n}, float64)"
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_vat_f32_matches_python(n):
    dist = _random_sym_dist(n, seed=n, dtype=np.float32)
    vat_ref, p_ref = _py_vat(dist)

    vat_c, p_c, _ = compute_vat_c_32(np.ascontiguousarray(dist, dtype=np.float32))

    np.testing.assert_allclose(
        vat_c.astype(np.float64),
        vat_ref,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"VAT mismatch (n={n}, float32)",
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


def test_vat_f64_symmetric():
    dist = _random_sym_dist(100, seed=42)
    vat, _, _ = compute_vat_c_64(np.ascontiguousarray(dist))
    np.testing.assert_allclose(
        vat, vat.T, atol=0, err_msg="VAT matrix is not symmetric"
    )


def test_vat_f32_symmetric():
    dist = _random_sym_dist(100, seed=42, dtype=np.float32)
    vat, _, _ = compute_vat_c_32(np.ascontiguousarray(dist))
    np.testing.assert_allclose(
        vat, vat.T, atol=0, err_msg="VAT matrix is not symmetric (float32)"
    )


# ---------------------------------------------------------------------------
# IVAT correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_ivat_f64_matches_python(n):
    dist = _random_sym_dist(n, seed=n)
    ivat_ref, _, p_ref = _py_ivat(dist)

    ivat_c, _, p_c = compute_ivat_c_64(np.ascontiguousarray(dist, dtype=np.float64))

    np.testing.assert_allclose(
        ivat_c,
        ivat_ref,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"IVAT mismatch (n={n}, float64)",
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_ivat_f32_matches_python(n):
    dist = _random_sym_dist(n, seed=n, dtype=np.float32)
    ivat_ref, _, p_ref = _py_ivat(dist)

    ivat_c, _, p_c = compute_ivat_c_32(np.ascontiguousarray(dist, dtype=np.float32))

    # float32 kernel accumulates in float32, so tolerance is wider.
    np.testing.assert_allclose(
        ivat_c.astype(np.float64),
        ivat_ref,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"IVAT mismatch (n={n}, float32)",
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


def test_ivat_f64_symmetric():
    """IVAT result must be symmetric (lower-tri back-copy correctness)."""
    dist = _random_sym_dist(100, seed=42)
    ivat, _, _ = compute_ivat_c_64(np.ascontiguousarray(dist))
    np.testing.assert_allclose(
        ivat, ivat.T, atol=0, err_msg="IVAT matrix is not symmetric"
    )


def test_ivat_f32_symmetric():
    dist = _random_sym_dist(100, seed=42, dtype=np.float32)
    ivat, _, _ = compute_ivat_c_32(np.ascontiguousarray(dist))
    np.testing.assert_allclose(
        ivat, ivat.T, atol=0, err_msg="IVAT matrix is not symmetric (float32)"
    )


def test_ivat_n2():
    """Minimal case: 2×2 matrix."""
    dist = np.array([[0.0, 3.14], [3.14, 0.0]], dtype=np.float64)
    ivat, argmin, perm = compute_ivat_c_64(dist)
    assert ivat.shape == (2, 2)
    np.testing.assert_allclose(ivat, ivat.T, atol=0)


# ---------------------------------------------------------------------------
# Performance benchmark (not a pass/fail test — prints timings)
# ---------------------------------------------------------------------------


def _bench(fn, n, dtype, repeats=3):
    dist = _random_sym_dist(n, seed=7, dtype=dtype)
    dist_c = np.ascontiguousarray(dist)
    fn(dist_c)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(dist_c)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000  # ms


def test_performance(capsys):
    """Print timing for VAT and IVAT across matrix sizes."""
    sizes = [100, 300, 500, 1000]
    header = f"{'n':>6}  {'VAT f64':>10}  {'VAT f32':>10}  {'IVAT f64':>10}  {'IVAT f32':>10}"
    sep = "-" * len(header)

    with capsys.disabled():
        print("\n\nLower-tri VAT + IVAT performance (ms, best of 3):")
        print(header)
        print(sep)
        for n in sizes:
            tv64 = _bench(compute_vat_c_64, n, np.float64)
            tv32 = _bench(compute_vat_c_32, n, np.float32)
            ti64 = _bench(compute_ivat_c_64, n, np.float64)
            ti32 = _bench(compute_ivat_c_32, n, np.float32)
            print(f"{n:>6}  {tv64:>10.2f}  {tv32:>10.2f}  {ti64:>10.2f}  {ti32:>10.2f}")
