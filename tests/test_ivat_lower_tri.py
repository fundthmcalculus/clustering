"""
Correctness and performance tests for the lower-triangular IVAT kernel.

Methodology under test
-----------------------
Instead of writing both ivat[r,c] and ivat[c,r] on every cell fill, the
kernel writes only the lower triangle during the O(n^2) construction loop,
then back-copies it to the upper triangle in a single (parallelisable) pass.
Reads of previously-computed IVAT entries are canonicalised to the lower
triangle (swap indices when best_jj < c).

The reference is the Python/Numba implementation in pvat.compute_ivat(), which
produces the ground-truth result.
"""
import time

import numpy as np
import pytest

from tribbleclustering.pvat import compute_ivat as compute_ivat_py
from tribbleclustering.pcvat import compute_ivat_c_64, compute_ivat_c_32


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
    """Python reference: returns (ivat, vat, argmin_seq, p_seq)."""
    return compute_ivat_py(dist.astype(np.float64))


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_ivat_f64_matches_python(n):
    dist = _random_sym_dist(n, seed=n)
    ivat_ref, vat_ref, _, p_ref = _py_ivat(dist)

    ivat_c, vat_c, _, p_c = compute_ivat_c_64(np.ascontiguousarray(dist, dtype=np.float64))

    np.testing.assert_allclose(
        ivat_c, ivat_ref, rtol=1e-10, atol=1e-12,
        err_msg=f"IVAT mismatch (n={n}, float64)"
    )
    np.testing.assert_allclose(
        vat_c, vat_ref, rtol=1e-10, atol=1e-12,
        err_msg=f"VAT mismatch (n={n}, float64)"
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


@pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 148])
def test_ivat_f32_matches_python(n):
    dist = _random_sym_dist(n, seed=n, dtype=np.float32)
    ivat_ref, vat_ref, _, p_ref = _py_ivat(dist)

    ivat_c, vat_c, _, p_c = compute_ivat_c_32(np.ascontiguousarray(dist, dtype=np.float32))

    # float32 kernel accumulates in float32, so tolerance is wider.
    np.testing.assert_allclose(
        ivat_c.astype(np.float64), ivat_ref, rtol=1e-5, atol=1e-6,
        err_msg=f"IVAT mismatch (n={n}, float32)"
    )
    np.testing.assert_array_equal(p_c, p_ref, err_msg=f"Permutation mismatch (n={n})")


def test_ivat_f64_symmetric():
    """IVAT result must be symmetric (lower-tri back-copy correctness)."""
    dist = _random_sym_dist(100, seed=42)
    ivat, _, _, _ = compute_ivat_c_64(np.ascontiguousarray(dist))
    np.testing.assert_allclose(ivat, ivat.T, atol=0,
                               err_msg="IVAT matrix is not symmetric")


def test_ivat_f32_symmetric():
    dist = _random_sym_dist(100, seed=42, dtype=np.float32)
    ivat, _, _, _ = compute_ivat_c_32(np.ascontiguousarray(dist))
    np.testing.assert_allclose(ivat, ivat.T, atol=0,
                               err_msg="IVAT matrix is not symmetric (float32)")


def test_ivat_n2():
    """Minimal case: 2×2 matrix."""
    dist = np.array([[0.0, 3.14], [3.14, 0.0]], dtype=np.float64)
    ivat, vat, argmin, perm = compute_ivat_c_64(dist)
    assert ivat.shape == (2, 2)
    np.testing.assert_allclose(ivat, ivat.T, atol=0)


# ---------------------------------------------------------------------------
# Performance benchmark (not a pass/fail test — prints timings)
# ---------------------------------------------------------------------------

def _bench_ivat(n, dtype, repeats=3):
    dist = _random_sym_dist(n, seed=7, dtype=dtype)
    dist_c = np.ascontiguousarray(dist)

    fn = compute_ivat_c_64 if dtype == np.float64 else compute_ivat_c_32

    # Warm-up
    fn(dist_c)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(dist_c)
        times.append(time.perf_counter() - t0)
    return min(times)


def test_ivat_performance(capsys):
    """Print timing comparison across matrix sizes."""
    sizes = [100, 300, 500, 1000]
    print("\n\nIVAT kernel performance (lower-tri write + back-copy):")
    print(f"{'n':>6}  {'f64 (ms)':>10}  {'f32 (ms)':>10}")
    print("-" * 32)
    for n in sizes:
        t64 = _bench_ivat(n, np.float64) * 1000
        t32 = _bench_ivat(n, np.float32) * 1000
        print(f"{n:>6}  {t64:>10.2f}  {t32:>10.2f}")

    with capsys.disabled():
        print("\nIVAT kernel performance (lower-tri write + back-copy):")
        print(f"{'n':>6}  {'f64 (ms)':>10}  {'f32 (ms)':>10}")
        print("-" * 32)
        for n in sizes:
            t64 = _bench_ivat(n, np.float64) * 1000
            t32 = _bench_ivat(n, np.float32) * 1000
            print(f"{n:>6}  {t64:>10.2f}  {t32:>10.2f}")
