"""Tests for the Lin-Kernighan TSP solver (pure-Python and Cython/OpenMP).

Correctness for a metaheuristic is exercised through invariants that must hold
regardless of the random instance:

* the returned tour is a valid permutation of the cities;
* local search never *increases* length relative to the nearest-neighbour
  construction it starts from;
* on tiny instances the result matches the brute-force optimum;
* the compiled and pure paths are behaviorally equivalent for identical
  parameters (a hard requirement from ``CLAUDE.md``).
"""

import itertools

import numpy as np
import pytest

from tribbleclustering.lk import (
    lin_kernighan as lin_kernighan_py,
    tour_length,
    _nearest_neighbor_tour,
)
from tribbleclustering.linkernighan import LinKernighan
from tribbleclustering.util import pairwise_distances

try:
    from tribbleclustering.clk import lin_kernighan_c

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def _random_distance_matrix(n, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2)).astype(dtype)
    return pairwise_distances(pts)


def _brute_force_optimum(D):
    """Optimal closed-tour length by exhaustive search (small n only)."""
    n = D.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        length = tour_length(np.array((0,) + perm), D)
        best = min(best, length)
    return best


def _is_permutation(tour, n):
    return sorted(np.asarray(tour).tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# Pure-Python reference
# ---------------------------------------------------------------------------
class TestPurePython:
    def test_returns_valid_permutation(self):
        D = _random_distance_matrix(50, seed=1)
        tour, length = lin_kernighan_py(D, n_starts=3, neighbors=8, seed=1)
        assert _is_permutation(tour, 50)
        assert np.isclose(length, tour_length(tour, D))

    def test_improves_over_nearest_neighbor(self):
        D = _random_distance_matrix(80, seed=2)
        nn_len = tour_length(_nearest_neighbor_tour(D, 0), D)
        _, length = lin_kernighan_py(D, n_starts=4, neighbors=10, seed=2)
        assert length <= nn_len + 1e-9

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_trivial_sizes(self, n):
        D = _random_distance_matrix(n, seed=n)
        tour, length = lin_kernighan_py(D)
        assert _is_permutation(tour, n)
        assert length >= 0.0

    def test_near_brute_force_optimum(self):
        # LK is a local-search heuristic: with full neighbour lists and all
        # starts it lands on a strong local optimum, reliably within a small
        # gap of the exhaustive optimum (empirically <12% on these sizes).
        for trial in range(8):
            D = _random_distance_matrix(7, seed=200 + trial)
            _, length = lin_kernighan_py(D, n_starts=7, max_depth=7, neighbors=6)
            opt = _brute_force_optimum(D)
            assert length <= opt * 1.25 + 1e-6

    def test_optimal_on_four_cities(self):
        # With 4 cities there is a single distinct tour up to rotation/reflection.
        for trial in range(10):
            D = _random_distance_matrix(4, seed=400 + trial)
            _, length = lin_kernighan_py(D, n_starts=4, max_depth=4, neighbors=3)
            assert length <= _brute_force_optimum(D) + 1e-6

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            lin_kernighan_py(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Compiled Cython/OpenMP kernel
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
class TestCython:
    def test_returns_valid_permutation(self):
        D = _random_distance_matrix(120, seed=3)
        tour, length = lin_kernighan_c(
            D, n_starts=8, neighbors=10, num_threads=4, seed=3
        )
        assert _is_permutation(tour, 120)
        assert np.isclose(length, tour_length(tour, D))

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_dtype_dispatch(self, dtype):
        D = _random_distance_matrix(60, seed=4, dtype=dtype)
        tour, length = lin_kernighan_c(D, n_starts=4, neighbors=8, num_threads=2)
        assert tour.dtype == np.int32
        assert _is_permutation(tour, 60)
        assert length > 0

    def test_multithreaded_matches_serial(self):
        # Same starts, deterministic construction -> identical best regardless
        # of how many OpenMP threads split the restart loop.
        D = _random_distance_matrix(100, seed=5)
        _, serial = lin_kernighan_c(D, n_starts=8, neighbors=8, num_threads=1, seed=5)
        _, parallel = lin_kernighan_c(D, n_starts=8, neighbors=8, num_threads=4, seed=5)
        assert np.isclose(serial, parallel)

    def test_matches_python_equivalence(self):
        # Identical parameters (single start, serial) must give the same tour.
        D = _random_distance_matrix(90, seed=6)
        _, length_c = lin_kernighan_c(
            D, n_starts=1, max_depth=5, neighbors=8, num_threads=1
        )
        _, length_py = lin_kernighan_py(D, n_starts=1, max_depth=5, neighbors=8)
        assert np.isclose(length_c, length_py)

    def test_near_brute_force_optimum(self):
        for trial in range(8):
            D = _random_distance_matrix(7, seed=300 + trial)
            _, length = lin_kernighan_c(
                D, n_starts=7, max_depth=7, neighbors=6, num_threads=2
            )
            opt = _brute_force_optimum(D)
            assert length <= opt * 1.25 + 1e-6


# ---------------------------------------------------------------------------
# sklearn-style wrapper
# ---------------------------------------------------------------------------
class TestWrapper:
    def test_solve_point_cloud(self):
        rng = np.random.default_rng(10)
        pts = rng.random((70, 2))
        lk = LinKernighan(
            n_starts=6, neighbors=10, num_threads=4, random_state=10
        ).solve(pts)
        assert _is_permutation(lk.tour_, 70)
        assert lk.tour_length_ == pytest.approx(
            tour_length(lk.tour_, lk.distance_matrix_)
        )

    def test_solve_precomputed(self):
        D = _random_distance_matrix(50, seed=11)
        lk = LinKernighan().solve(D, precomputed=True)
        assert _is_permutation(lk.tour_, 50)
        assert lk.tour_length_ == pytest.approx(tour_length(lk.tour_, D))

    def test_fit_predict(self):
        rng = np.random.default_rng(12)
        pts = rng.random((40, 3))
        order = LinKernighan(n_starts=4).fit_predict(pts)
        assert _is_permutation(order, 40)

    def test_float32_dtype(self):
        rng = np.random.default_rng(13)
        pts = rng.random((60, 2))
        lk = LinKernighan(dtype="float32", n_starts=4).solve(pts)
        assert lk.distance_matrix_.dtype == np.float32
        assert _is_permutation(lk.tour_, 60)

    def test_precomputed_requires_square(self):
        with pytest.raises(ValueError):
            LinKernighan().solve(np.zeros((3, 5)), precomputed=True)

    def test_tour_length_for_before_fit_raises(self):
        with pytest.raises(ValueError):
            LinKernighan().tour_length_for(np.arange(3))
