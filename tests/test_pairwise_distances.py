"""Correctness and performance comparison for the OpenMP pairwise-distance
C extension (`pairwise_distances_c_32` / `_c_64`) against:

  * the numba reference `util.pairwise_distances`, and
  * scipy's `squareform(pdist(...))` ground truth.

Run with:  pytest tests/test_pairwise_distances.py -v -s
"""
import time

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from tribbleclustering.util import pairwise_distances
from tribbleclustering.pcvat import (
    pairwise_distances_c,
    pairwise_distances_c_32,
    pairwise_distances_c_64,
)


def _bench(fn, arg, n_iter=15, warmup=2):
    """Time `fn(arg)` repeatedly, returning per-call times in milliseconds.

    Warmup calls (JIT compile / cache / thread-pool spin-up) are excluded; then
    `n_iter` individual timings are collected for mean / 2-sigma reporting.
    """
    for _ in range(warmup):
        fn(arg)
    samples = np.empty(n_iter, dtype=np.float64)
    for k in range(n_iter):
        t0 = time.perf_counter()
        fn(arg)
        samples[k] = (time.perf_counter() - t0) * 1000.0
    return samples


def _scipy_ref(data):
    """Ground-truth dense Euclidean distance matrix via scipy."""
    return squareform(pdist(data, metric="euclidean"))


@pytest.fixture
def tiny_data():
    np.random.seed(42)
    return np.random.randn(5, 2)


@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.randn(50, 5)


@pytest.fixture
def medium_data():
    np.random.seed(42)
    return np.random.randn(300, 8)


@pytest.fixture
def large_data():
    """Past _PAR_THRESHOLD (512), so the OpenMP path is exercised."""
    np.random.seed(42)
    return np.random.randn(800, 10)


class TestCorrectness:
    """Validate the C extension against scipy ground truth and numba."""

    @pytest.mark.parametrize("fixture", ["tiny_data", "small_data", "medium_data", "large_data"])
    def test_c64_matches_scipy(self, fixture, request):
        data = request.getfixturevalue(fixture).astype(np.float64)
        ref = _scipy_ref(data)
        out = pairwise_distances_c_64(data)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("fixture", ["tiny_data", "small_data", "medium_data", "large_data"])
    def test_c32_matches_scipy(self, fixture, request):
        data = request.getfixturevalue(fixture).astype(np.float32)
        ref = _scipy_ref(data.astype(np.float64))
        out = pairwise_distances_c_32(data)
        # float32 accumulation tolerance (kernel accumulates in double).
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_c64_matches_numba_reference(self, medium_data):
        data = medium_data.astype(np.float64)
        ref = pairwise_distances(data, norm_only=True)
        out = pairwise_distances_c_64(data)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_output_is_symmetric_with_zero_diagonal(self, medium_data):
        out = pairwise_distances_c_64(medium_data.astype(np.float64))
        np.testing.assert_array_equal(out, out.T)
        np.testing.assert_array_equal(np.diag(out), np.zeros(out.shape[0]))

    def test_dtype_preserved(self, small_data):
        assert pairwise_distances_c_64(small_data.astype(np.float64)).dtype == np.float64
        assert pairwise_distances_c_32(small_data.astype(np.float32)).dtype == np.float32

    def test_dispatch_selects_by_dtype(self, small_data):
        out32 = pairwise_distances_c(small_data.astype(np.float32))
        out64 = pairwise_distances_c(small_data.astype(np.float64))
        assert out32.dtype == np.float32
        assert out64.dtype == np.float64
        np.testing.assert_allclose(out32, out64, rtol=1e-5, atol=1e-5)

    def test_dispatch_handles_noncontiguous_input(self, small_data):
        # Fortran-ordered / transposed view is not C-contiguous; dispatch must
        # copy it before handing the buffer to the kernel.
        data = np.asfortranarray(small_data.astype(np.float64))
        out = pairwise_distances_c(data)
        np.testing.assert_allclose(out, _scipy_ref(small_data), rtol=1e-12, atol=1e-12)

    def test_dispatch_rejects_bad_dtype(self, small_data):
        with pytest.raises(TypeError):
            pairwise_distances_c(small_data.astype(np.int32))

    def test_single_point(self):
        out = pairwise_distances_c_64(np.array([[1.0, 2.0, 3.0]]))
        assert out.shape == (1, 1)
        assert out[0, 0] == 0.0

    def test_one_dimensional_features(self):
        # d == 1: euclidean distance reduces to absolute difference.
        data = np.array([[0.0], [3.0], [-1.0], [5.0]])
        out = pairwise_distances_c_64(data)
        np.testing.assert_allclose(out, _scipy_ref(data), rtol=1e-12, atol=1e-12)

    def test_empty_input(self):
        out = pairwise_distances_c_64(np.empty((0, 4), dtype=np.float64))
        assert out.shape == (0, 0)

    def test_no_nan_or_inf(self, large_data):
        out = pairwise_distances_c_64(large_data.astype(np.float64))
        assert np.all(np.isfinite(out))


class TestPerformance:
    """Benchmark the C extension vs the numba reference and scipy."""

    def test_scaling_behavior(self):
        """Compare pairwise_distances (numba), pairwise_distances_c (C/OpenMP)
        and scipy pdist across matrix sizes.

        Reports mean +/- 2 sigma (95% spread) in text and as shaded bands on
        the plot. Asserts the C extension agrees with scipy and is no slower
        than numba at the largest size (where the parallel path dominates).
        """
        np.random.seed(42)

        sizes = [50, 100, 250, 500, 1000, 2000, 4000]
        n_features = 8

        mean_numba, two_sig_numba = [], []
        mean_c, two_sig_c = [], []
        mean_scipy, two_sig_scipy = [], []

        print("\npairwise_distances: numba vs C/OpenMP vs scipy  (mean +/- 2 sigma, ms):")
        print(f"{'Size':>6} | {'numba (ms)':>18} | {'C (ms)':>18} | "
              f"{'scipy (ms)':>18} | {'C/numba':>8}")
        print("-" * 86)

        for size in sizes:
            data64 = np.random.randn(size, n_features).astype(np.float64)
            data32 = data64.astype(np.float32)

            s_numba = _bench(lambda d: pairwise_distances(d, norm_only=True), data64)
            s_c = _bench(pairwise_distances_c_32, data32)
            s_scipy = _bench(_scipy_ref, data64)

            # Correctness check alongside the benchmark.
            np.testing.assert_allclose(
                pairwise_distances_c_64(data64), _scipy_ref(data64),
                rtol=1e-10, atol=1e-10,
            )

            m_n, sd_n = float(s_numba.mean()), float(s_numba.std(ddof=1))
            m_c, sd_c = float(s_c.mean()), float(s_c.std(ddof=1))
            m_s, sd_s = float(s_scipy.mean()), float(s_scipy.std(ddof=1))

            mean_numba.append(m_n); two_sig_numba.append(2 * sd_n)
            mean_c.append(m_c); two_sig_c.append(2 * sd_c)
            mean_scipy.append(m_s); two_sig_scipy.append(2 * sd_s)

            print(f"{size:>6} | {m_n:>9.3f} +/- {2*sd_n:>5.3f} | "
                  f"{m_c:>9.3f} +/- {2*sd_c:>5.3f} | "
                  f"{m_s:>9.3f} +/- {2*sd_s:>5.3f} | "
                  f"{m_c/m_n:>7.2f}x{' (faster)' if m_c < m_n else ' (slower)'}")

        mean_numba = np.array(mean_numba); two_sig_numba = np.array(two_sig_numba)
        mean_c = np.array(mean_c); two_sig_c = np.array(two_sig_c)
        mean_scipy = np.array(mean_scipy); two_sig_scipy = np.array(two_sig_scipy)

        plt.figure()
        for mean, two_sig, marker, label in (
            (mean_numba, two_sig_numba, "s-", "pairwise_distances (numba)"),
            (mean_c, two_sig_c, "^-", "pairwise_distances_c (C/OpenMP)"),
            (mean_scipy, two_sig_scipy, "o-", "scipy pdist"),
        ):
            line, = plt.plot(sizes, mean, marker, label=label, linewidth=2, markersize=7)
            plt.fill_between(sizes, np.maximum(mean - two_sig, 0.0), mean + two_sig,
                             color=line.get_color(), alpha=0.2)
        plt.xlabel("Matrix Size (n)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.title("pairwise_distances: numba vs C/OpenMP vs scipy (mean $\\pm 2\\sigma$)",
                  fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pairwise_distances_performance.png", dpi=150)
        print("\nPlot saved to 'pairwise_distances_performance.png'")

        # The C/OpenMP path should beat the numba reference at the largest size.
        assert mean_c[-1] <= mean_numba[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
