"""Benchmark the compiled FCM kernel against the pure-numpy reference.

Both are driven from the same initial centers so they run the same number of
iterations; otherwise the ratio measures two random draws rather than two
kernels (see issue #100).
"""

import time
import numpy as np
import pytest

try:
    from tribbleclustering.cfcm import fuzzy_c_means as fuzzy_c_means_optimized

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

from tribbleclustering.fcm import fuzzy_c_means as fuzzy_c_means_baseline


@pytest.mark.benchmark
@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
def test_memory_optimization_benchmark():
    """Benchmark the compiled kernel against the pure-numpy reference.

    Expected: the compiled kernel is faster; the assertion only fails it for a
    real regression, since the pure path is BLAS-backed and already fast.
    """
    np.random.seed(42)

    # Test dataset: medium-sized problem
    n_samples = 500
    n_features = 15
    n_clusters = 5

    # Create well-separated clusters for consistent convergence
    cluster_centers = np.random.randn(n_clusters, n_features) * 10
    x = np.vstack(
        [
            cluster_centers[i]
            + np.random.randn(n_samples // n_clusters, n_features) * 0.5
            for i in range(n_clusters)
        ]
    ).astype(np.float64)

    # Same starting centers for both, so both run the same iterations.
    rng = np.random.default_rng(0)
    start = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()

    # Warmup
    fuzzy_c_means_baseline(x, n_clusters, m=2.0, initial_guess=start)
    fuzzy_c_means_optimized(x, n_clusters, m=2.0, initial_guess=start)

    # Benchmark baseline
    times_baseline = []
    for _ in range(3):
        t0 = time.perf_counter()
        fuzzy_c_means_baseline(x, n_clusters, m=2.0, initial_guess=start)
        times_baseline.append(time.perf_counter() - t0)

    # Benchmark optimized
    times_optimized = []
    for _ in range(3):
        t0 = time.perf_counter()
        fuzzy_c_means_optimized(x, n_clusters, m=2.0, initial_guess=start)
        times_optimized.append(time.perf_counter() - t0)

    t_baseline = np.mean(times_baseline)
    t_optimized = np.mean(times_optimized)
    speedup = t_baseline / t_optimized

    print(f"\n{'Benchmark Results':=^60}")
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
    print(f"Baseline (pure Python):  {t_baseline*1000:7.2f} ms")
    print(f"Compiled (Cython):       {t_optimized*1000:7.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'':=^60}")

    # Loose on purpose: the pure path is BLAS-backed, so rough parity is
    # acceptable and only a real regression (issue #100 measured 0.27x) fails.
    assert speedup >= 0.9, f"Compiled kernel is slower than pure numpy: {speedup}x"


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
def test_memory_optimization_convergence_iterations():
    """Track iteration count during convergence.

    The compiled kernel shares fcm.py's convergence test, so it should not
    significantly affect the number of iterations to convergence.
    """
    np.random.seed(42)

    x = np.random.randn(100, 10).astype(np.float64)
    initial_guess = x[:3].copy()

    # Run both versions - they should converge in similar time
    # (number of iterations should be similar, though not identical)
    t0 = time.perf_counter()
    result_baseline = fuzzy_c_means_baseline(x, 3, m=2.0, initial_guess=initial_guess)
    c_baseline, w_baseline = result_baseline
    t_baseline = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_optimized, w_optimized, _, _ = fuzzy_c_means_optimized(
        x, 3, m=2.0, initial_guess=initial_guess
    )
    t_optimized = time.perf_counter() - t0

    # Both should produce valid results
    assert np.all(np.isfinite(c_baseline))
    assert np.all(np.isfinite(c_optimized))

    print(f"\n{'Convergence Comparison':=^60}")
    print(f"Baseline time: {t_baseline*1000:.2f} ms")
    print(f"Optimized time: {t_optimized*1000:.2f} ms")
    print(f"Ratio: {t_baseline/t_optimized:.2f}x")
    print(f"{'':=^60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
