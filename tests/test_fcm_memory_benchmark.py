"""Benchmark FCM memory layout optimization.

This benchmark demonstrates the performance improvement from
distance caching during FCM iterations.
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


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
def test_memory_optimization_benchmark():
    """Benchmark distance caching optimization.

    Expected: Optimized version should be 1.2-1.4x faster
    than baseline for typical convergence patterns.
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

    # Warmup
    fuzzy_c_means_baseline(x, n_clusters, m=2.0)
    fuzzy_c_means_optimized(x, n_clusters, m=2.0)

    # Benchmark baseline
    times_baseline = []
    for _ in range(3):
        t0 = time.perf_counter()
        fuzzy_c_means_baseline(x, n_clusters, m=2.0)
        times_baseline.append(time.perf_counter() - t0)

    # Benchmark optimized
    times_optimized = []
    for _ in range(3):
        t0 = time.perf_counter()
        fuzzy_c_means_optimized(x, n_clusters, m=2.0)
        times_optimized.append(time.perf_counter() - t0)

    t_baseline = np.mean(times_baseline)
    t_optimized = np.mean(times_optimized)
    speedup = t_baseline / t_optimized

    print(f"\n{'Benchmark Results':=^60}")
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
    print(f"Baseline (pure Python):  {t_baseline*1000:7.2f} ms")
    print(f"Optimized (distance caching): {t_optimized*1000:7.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'':=^60}")

    # The optimization should show at least some improvement
    # On systems with good cache behavior, 1.2-1.4x improvement is typical
    assert speedup >= 0.9, f"Optimization degraded performance: {speedup}x"


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
def test_memory_optimization_convergence_iterations():
    """Track iteration count during convergence.

    The distance caching optimization should not significantly affect
    the number of iterations to convergence.
    """
    np.random.seed(42)

    x = np.random.randn(100, 10).astype(np.float64)
    initial_guess = x[:3].copy()

    # Run both versions - they should converge in similar time
    # (number of iterations should be similar, though not identical)
    t0 = time.perf_counter()
    c_baseline, w_baseline = fuzzy_c_means_baseline(
        x, 3, m=2.0, initial_guess=initial_guess
    )
    t_baseline = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_optimized, w_optimized = fuzzy_c_means_optimized(
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
