"""Regression tests for the reverse-delete dense-graph / TSP spike.

The spike lives in ``experiments/`` (research area, not shipped code), so we put
the repo root on ``sys.path`` to import it. These tests guard the two claims the
findings rest on: the ``m=1`` MST duality and ``m=2`` tour validity.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rd = pytest.importorskip("experiments.reverse_delete_tsp")


def _small_D(n: int, seed: int) -> np.ndarray:
    return rd.distance_matrix(rd.uniform_cities(n, seed=seed))


@pytest.mark.parametrize("n", [8, 12, 20])
def test_reverse_delete_m1_is_the_mst(n: int) -> None:
    """m=1 reverse-delete == Kruskal MST (edge set) == Prim MST (weight)."""
    D = _small_D(n, seed=n)
    rd_adj, _ = rd.reverse_delete(D, min_degree=1)
    rd_edges = rd.edge_set(rd_adj)
    kruskal = rd.kruskal_mst_edges(D)
    prim = rd.prim_mst_edges(D)

    # A spanning tree has exactly n-1 edges and every vertex present.
    assert len(rd_edges) == n - 1
    # Generic Euclidean points have unique nearest neighbours -> identical edges.
    assert rd_edges == kruskal
    # MST weight is unique even under any tie-breaking.
    w = rd.total_weight
    assert w(D, rd_edges) == pytest.approx(w(D, kruskal), rel=1e-5)
    assert w(D, prim) == pytest.approx(w(D, kruskal), rel=1e-5)


@pytest.mark.parametrize("n", [10, 15, 25])
def test_reverse_delete_tour_is_a_valid_permutation(n: int) -> None:
    """m=2 pipeline always returns a valid Hamiltonian tour (with repair)."""
    D = _small_D(n, seed=n)
    tour, converged = rd.reverse_delete_tour(D, repair=True)
    assert sorted(tour) == list(range(n))  # each city exactly once
    if converged:
        # A converged run means the survivor was already a single 2-regular cycle.
        adj, _ = rd.reverse_delete(D, min_degree=2)
        assert all(len(adj[i]) == 2 for i in adj)


def test_m2_survivor_is_connected_min_degree_two() -> None:
    """The core invariant: the m=2 survivor is an island-free 2-core."""
    n = 30
    D = _small_D(n, seed=1)
    adj, _ = rd.reverse_delete(D, min_degree=2)
    assert all(len(adj[i]) >= 2 for i in adj)  # no vertex below the floor
    assert rd._reachable(adj, 0, n - 1)  # still one connected piece
    # It is a strict subgraph of the dense graph (edges were actually deleted).
    assert len(rd.edges_of(adj)) < n * (n - 1) // 2


def test_duality_weight_is_the_invariant_under_ties() -> None:
    """On a graph with many tied distances the MST is non-unique, so edge-set
    equality is not guaranteed — but total MST *weight* always is. Assert the
    robust invariant (weight), not the fragile one (edge set)."""
    grid = np.array([[x, y] for x in range(5) for y in range(5)], dtype=np.float32)
    D = rd.distance_matrix(grid)
    rd_edges = rd.edge_set(rd.reverse_delete(D, min_degree=1)[0])
    kruskal = rd.kruskal_mst_edges(D)
    assert len(rd_edges) == len(grid) - 1
    assert rd.total_weight(D, rd_edges) == pytest.approx(
        rd.total_weight(D, kruskal), rel=1e-5
    )


def test_two_opt_never_lengthens() -> None:
    D = _small_D(20, seed=7)
    nn = rd.nearest_neighbour_tour(D)
    assert rd.tour_length(D, rd.two_opt(D, nn)) <= rd.tour_length(D, nn) + 1e-6
