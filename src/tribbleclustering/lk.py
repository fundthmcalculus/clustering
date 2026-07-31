"""Pure-numpy Lin-Kernighan TSP solver (reference / fallback path).

This is the readable reference implementation of the Lin-Kernighan (LK) local
search for the (symmetric) Travelling Salesman Problem. The compiled
``clk.pyx`` kernel mirrors this algorithm move-for-move and adds OpenMP
multi-threaded multi-start; the two paths must stay behaviorally equivalent
(see ``CLAUDE.md``).

The LK move implemented here is the classic *variable-depth* sequential edge
exchange restricted to breadth 1 (one candidate per level), realised as a chain
of 2-opt reversals:

1. Fix an anchor city ``t1`` and remove the edge ``(t1, t2)`` where
   ``t2 = succ(t1)``.
2. Add a candidate edge ``(t2, t3)`` to a near neighbour ``t3`` of ``t2``. The
   only reconnection that keeps a valid tour removes ``(t4, t3)`` with
   ``t4 = pred(t3)`` and re-adds ``(t1, t4)`` -- i.e. reversing the tour
   segment ``t2..t4``. This is a single 2-opt move; its gain is tracked.
3. After the reversal ``t1``'s successor becomes ``t4``, so the chain can
   continue from the new free end up to ``max_depth`` levels, letting the
   cumulative gain dip negative before recovering (the hallmark of LK's
   variable depth).
4. The prefix of reversals with the best cumulative gain is kept; the rest are
   undone. Nothing is committed unless the tour strictly improves, so the
   optimizer never increases tour length.

Candidate neighbours are precomputed nearest-neighbour lists (sorted ascending
by distance) so the positive-gain pruning ``g1 = d(t1,t2) - d(t2,t3) > 0`` lets
us ``break`` as soon as a neighbour is too far. Don't-look bits skip anchors
whose neighbourhood has not changed since they last failed to improve.
"""

from typing import Optional

import numpy as np
from numpy import ndarray

# Gain tolerance: improvements below this are treated as floating-point noise so
# the local search terminates instead of chasing vanishing deltas.
_EPS = 1e-9


def tour_length(tour: ndarray, distances: ndarray) -> float:
    """Total length of a closed ``tour`` under the ``distances`` matrix.

    Parameters
    ----------
    tour : ndarray of shape (n_cities,)
        A permutation of ``range(n_cities)`` giving the visit order.
    distances : ndarray of shape (n_cities, n_cities)
        Symmetric pairwise-distance (dissimilarity) matrix.

    Returns
    -------
    float
        Sum of edge lengths including the closing edge back to the start.
    """
    tour = np.asarray(tour)
    if tour.shape[0] < 2:
        return 0.0
    nxt = np.roll(tour, -1)
    return float(distances[tour, nxt].sum())


def _build_neighbor_lists(distances: ndarray, k: int) -> ndarray:
    """Return each city's ``k`` nearest neighbours, sorted ascending.

    The diagonal is masked to ``+inf`` so a city is never its own neighbour.
    Ascending order is what lets the LK step prune on gain.
    """
    masked = distances.astype(np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    # argsort gives ascending distance; take the closest k columns.
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(order.astype(np.int32))


def _nearest_neighbor_tour(distances: ndarray, start: int) -> ndarray:
    """Greedy nearest-neighbour construction from ``start`` city."""
    n = distances.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    current = start
    tour[0] = current
    visited[current] = True
    for i in range(1, n):
        row = distances[current].copy()
        row[visited] = np.inf
        current = int(np.argmin(row))
        tour[i] = current
        visited[current] = True
    return tour


def _reverse(tour: ndarray, pos: ndarray, lo: int, hi: int) -> None:
    """Reverse ``tour[lo:hi+1]`` in place, keeping ``pos`` (city->index) valid."""
    while lo < hi:
        a = tour[lo]
        b = tour[hi]
        tour[lo] = b
        tour[hi] = a
        pos[b] = lo
        pos[a] = hi
        lo += 1
        hi -= 1


def _lk_step(
    t1: int,
    tour: ndarray,
    pos: ndarray,
    distances: ndarray,
    neigh: ndarray,
    max_depth: int,
) -> list:
    """Attempt one variable-depth LK improvement anchored at city ``t1``.

    Both edges incident to ``t1`` are tried as the first broken edge (the
    successor and predecessor directions); the first direction that yields a
    net improvement wins. Each direction runs an independent chain of 2-opt
    reversals up to ``max_depth`` and keeps the prefix with the best cumulative
    gain. Returns the list of committed ``(lo, hi)`` reversals (empty if no
    improving move was found). ``tour``/``pos`` are mutated in place; a failed
    direction is rolled back before the next is tried.
    """
    n = tour.shape[0]
    i = int(pos[t1])  # anchor position; reversals never touch it, so it holds.

    for direction in (0, 1):  # 0 = successor edge, 1 = predecessor edge
        applied: list = []
        cum = 0.0
        best_cum = 0.0
        best_len = 0

        for _ in range(max_depth):
            if direction == 0:
                if i > n - 2:
                    break
                fe = i + 1  # free-end position: current successor of t1
            else:
                if i < 1:
                    break
                fe = i - 1  # free-end position: current predecessor of t1

            t2 = int(tour[fe])
            d_t1t2 = distances[t1, t2]
            chosen = None
            for t3 in neigh[t2]:
                t3 = int(t3)
                if t3 == t1 or t3 == t2:
                    continue
                partial = d_t1t2 - distances[t2, t3]
                if partial <= _EPS:
                    # neigh[t2] is sorted ascending: no farther neighbour can
                    # give a positive g1 either.
                    break
                p3 = int(pos[t3])
                if direction == 0:
                    # Reverse [i+1, j]; j == pred(t3), wrapping when t3 is first.
                    j = p3 - 1 if p3 > 0 else n - 1
                    lo, hi = i + 1, j
                    if hi < i + 2:  # need a segment of length >= 2 after t1
                        continue
                else:
                    # Reverse [a, i-1]; a == succ(t3), wrapping when t3 is last.
                    a = p3 + 1 if p3 < n - 1 else 0
                    lo, hi = a, i - 1
                    if lo > i - 2:  # need a segment of length >= 2 before t1
                        continue
                # Generic 2-opt delta for reversing [lo, hi]: removes the edges
                # just outside the segment and adds the two crossing edges.
                lm1 = int(tour[(lo - 1) % n])
                ll = int(tour[lo])
                hh = int(tour[hi])
                hp1 = int(tour[(hi + 1) % n])
                delta = (
                    distances[lm1, ll]
                    + distances[hh, hp1]
                    - distances[lm1, hh]
                    - distances[ll, hp1]
                )
                chosen = (lo, hi, delta)
                break  # LK breadth 1: take the first (nearest) gaining candidate.

            if chosen is None:
                break

            lo, hi, delta = chosen
            _reverse(tour, pos, lo, hi)
            applied.append((lo, hi))
            cum += delta
            if cum > best_cum + _EPS:
                best_cum = cum
                best_len = len(applied)

        # Roll back everything past the best-gain prefix.
        for k in range(len(applied) - 1, best_len - 1, -1):
            lo, hi = applied[k]
            _reverse(tour, pos, lo, hi)

        if best_cum > _EPS:
            return applied[:best_len]

    return []


def _optimize(
    tour: ndarray,
    pos: ndarray,
    distances: ndarray,
    neigh: ndarray,
    max_depth: int,
) -> None:
    """Drive ``_lk_step`` to a local optimum using don't-look bits."""
    n = tour.shape[0]
    dont_look = np.zeros(n, dtype=bool)
    improved_any = True
    while improved_any:
        improved_any = False
        for c1 in range(n):
            if dont_look[c1]:
                continue
            applied = _lk_step(c1, tour, pos, distances, neigh, max_depth)
            if applied:
                improved_any = True
                # Re-activate anchors whose incident edges just changed.
                for lo, hi in applied:
                    for p in (lo - 1, lo, hi, (hi + 1) % n):
                        dont_look[tour[p]] = False
                dont_look[c1] = False
            else:
                dont_look[c1] = True


def lin_kernighan(
    distances: ndarray,
    n_starts: int = 1,
    max_depth: int = 5,
    neighbors: int = 8,
    seed: Optional[int] = None,
) -> tuple[ndarray, float]:
    """Solve a symmetric TSP with multi-start Lin-Kernighan local search.

    Parameters
    ----------
    distances : ndarray of shape (n_cities, n_cities)
        Symmetric pairwise-distance (dissimilarity) matrix.
    n_starts : int, optional
        Number of nearest-neighbour starting tours to optimize; the best
        result is returned. Default 1.
    max_depth : int, optional
        Maximum depth of each variable-depth LK chain. Default 5.
    neighbors : int, optional
        Size of the candidate nearest-neighbour list per city. Default 8.
    seed : int, optional
        Seed for choosing distinct starting cities when ``n_starts`` exceeds the
        obvious deterministic set.

    Returns
    -------
    tour : ndarray of shape (n_cities,)
        Best tour found, as a permutation of ``range(n_cities)``.
    length : float
        Length of the returned tour.
    """
    distances = np.ascontiguousarray(distances)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2-D matrix")

    n = distances.shape[0]
    if n <= 3:
        # Any permutation is optimal (or trivial) for n <= 3.
        tour = np.arange(max(n, 0), dtype=np.int32)
        return tour, tour_length(tour, distances)

    n_starts = max(1, int(n_starts))
    k = min(max(1, int(neighbors)), n - 1)
    neigh = _build_neighbor_lists(distances, k)

    rng = np.random.default_rng(seed)
    if n_starts <= n:
        # Evenly spaced distinct start cities for reproducible diversity.
        start_cities = np.linspace(0, n - 1, n_starts, dtype=np.int64)
    else:
        start_cities = rng.integers(0, n, size=n_starts)

    best_tour: Optional[ndarray] = None
    best_len = np.inf
    for start in start_cities:
        tour = _nearest_neighbor_tour(distances, int(start))
        pos = np.empty(n, dtype=np.int32)
        pos[tour] = np.arange(n, dtype=np.int32)
        _optimize(tour, pos, distances, neigh, max_depth)
        length = tour_length(tour, distances)
        if length < best_len:
            best_len = length
            best_tour = tour.copy()

    assert best_tour is not None  # n_starts >= 1
    return best_tour, float(best_len)
