"""OPT 4 "done right": drive the #45 Lin-Kernighan solver from a reverse-delete
candidate list instead of its internal pure-kNN list.

The LK solver in PR #45 (`tribbleclustering.lk`) builds its candidate neighbour
lists internally from k-nearest-neighbours; there is no hook to inject an
external candidate set. The single upstream change this experiment proposes is a
`candidates=` parameter on `lin_kernighan(...)`:

    def lin_kernighan(distances, candidates=None, ...):
        if candidates is not None:
            neigh = _sort_candidates(distances, candidates)   # <- new
        else:
            neigh = _build_neighbor_lists(distances, k)       # existing kNN

Everything else (`_lk_step`, `_optimize`, don't-look bits, the variable-depth
chain) is untouched — `neigh[t2]` just iterates a per-city candidate list, so a
ragged Python list works in place of the fixed-width kNN array.

Because #45 is not merged yet, the LK core below is **vendored verbatim** from
that branch's `src/tribbleclustering/lk.py` (attribution above) with only the
`candidates=` hook added. When #45 lands, delete this copy and pass the same
candidate lists straight into `tribbleclustering.lin_kernighan`.

This lets us ask the OPT-4 question properly: does a reverse-delete / Delaunay
candidate set beat #45's plain kNN candidates for LK? Run:
    python -m experiments.reverse_delete_lk
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from numpy import ndarray

from experiments.reverse_delete_tsp import (  # noqa: E402
    Adj,
    distance_matrix,
    nearest_neighbour_tour,
    tour_length as rd_tour_length,
    two_opt,
    uniform_cities,
)
from experiments.reverse_delete_opt import (  # noqa: E402
    delaunay_candidate_graph,
    knn_candidate_graph,
    reachable_bidirectional,
    reverse_delete_on,
)

_EPS = 1e-9


# ===========================================================================
# Vendored LK core (from PR #45 src/tribbleclustering/lk.py) + candidates hook
# ===========================================================================
def tour_length(tour: ndarray, distances: ndarray) -> float:
    tour = np.asarray(tour)
    if tour.shape[0] < 2:
        return 0.0
    return float(distances[tour, np.roll(tour, -1)].sum())


def _build_neighbor_lists(distances: ndarray, k: int) -> ndarray:
    masked = distances.astype(np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(order.astype(np.int32))


def _sort_candidates(distances: ndarray, candidates: Sequence[Sequence[int]]) -> list:
    """NEW hook: turn an external candidate set into per-city lists sorted
    ascending by distance (LK's positive-gain break requires ascending order)."""
    neigh = []
    for i, cand in enumerate(candidates):
        c = [j for j in cand if j != i]
        c.sort(key=lambda j: distances[i, j])
        neigh.append(c)
    return neigh


def _nearest_neighbor_tour(distances: ndarray, start: int) -> ndarray:
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
    while lo < hi:
        a, b = tour[lo], tour[hi]
        tour[lo], tour[hi] = b, a
        pos[b], pos[a] = lo, hi
        lo += 1
        hi -= 1


def _lk_step(t1, tour, pos, distances, neigh, max_depth) -> list:
    n = tour.shape[0]
    i = int(pos[t1])
    for direction in (0, 1):
        applied: list = []
        cum = 0.0
        best_cum = 0.0
        best_len = 0
        for _ in range(max_depth):
            if direction == 0:
                if i > n - 2:
                    break
                fe = i + 1
            else:
                if i < 1:
                    break
                fe = i - 1
            t2 = int(tour[fe])
            d_t1t2 = distances[t1, t2]
            chosen = None
            for t3 in neigh[t2]:
                t3 = int(t3)
                if t3 == t1 or t3 == t2:
                    continue
                if d_t1t2 - distances[t2, t3] <= _EPS:
                    break
                p3 = int(pos[t3])
                if direction == 0:
                    j = p3 - 1 if p3 > 0 else n - 1
                    lo, hi = i + 1, j
                    if hi < i + 2:
                        continue
                else:
                    a = p3 + 1 if p3 < n - 1 else 0
                    lo, hi = a, i - 1
                    if lo > i - 2:
                        continue
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
                break
            if chosen is None:
                break
            lo, hi, delta = chosen
            _reverse(tour, pos, lo, hi)
            applied.append((lo, hi))
            cum += delta
            if cum > best_cum + _EPS:
                best_cum = cum
                best_len = len(applied)
        for kk in range(len(applied) - 1, best_len - 1, -1):
            lo, hi = applied[kk]
            _reverse(tour, pos, lo, hi)
        if best_cum > _EPS:
            return applied[:best_len]
    return []


def _optimize(tour, pos, distances, neigh, max_depth) -> None:
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
                for lo, hi in applied:
                    for p in (lo - 1, lo, hi, (hi + 1) % n):
                        dont_look[tour[p]] = False
                dont_look[c1] = False
            else:
                dont_look[c1] = True


def lin_kernighan(
    distances: ndarray,
    candidates: Optional[Sequence[Sequence[int]]] = None,
    n_starts: int = 1,
    max_depth: int = 5,
    neighbors: int = 8,
    seed: Optional[int] = None,
) -> tuple[ndarray, float]:
    """#45's LK with the one added `candidates=` hook (see module docstring)."""
    # Force float64: the gain test uses _EPS=1e-9, so a float32 matrix (~1e-5
    # precision) makes LK chase rounding noise and never terminate. (#45's
    # compiled path accumulates gain in double for the same reason; the pure
    # path should cast too — worth upstreaming.)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    n = distances.shape[0]
    if n <= 3:
        tour = np.arange(max(n, 0), dtype=np.int32)
        return tour, tour_length(tour, distances)

    if candidates is not None:
        neigh: object = _sort_candidates(distances, candidates)
    else:
        k = min(max(1, int(neighbors)), n - 1)
        neigh = _build_neighbor_lists(distances, k)

    rng = np.random.default_rng(seed)
    n_starts = max(1, int(n_starts))
    starts = (
        np.linspace(0, n - 1, n_starts, dtype=np.int64)
        if n_starts <= n
        else rng.integers(0, n, size=n_starts)
    )
    best_tour: Optional[ndarray] = None
    best_len = np.inf
    for start in starts:
        tour = _nearest_neighbor_tour(distances, int(start))
        pos = np.empty(n, dtype=np.int32)
        pos[tour] = np.arange(n, dtype=np.int32)
        _optimize(tour, pos, distances, neigh, max_depth)
        length = tour_length(tour, distances)
        if length < best_len:
            best_len = length
            best_tour = tour.copy()
    assert best_tour is not None
    return best_tour, float(best_len)


# ===========================================================================
# Candidate sets for LK
# ===========================================================================
def _adj_union_knn(adj: Adj, cities: ndarray, extra_k: int) -> list[list[int]]:
    """Union an adjacency with each city's `extra_k` nearest neighbours."""
    D = distance_matrix(cities)
    n = len(cities)
    out: list[set] = [set(adj[i]) for i in range(n)]
    if extra_k > 0:
        for i in range(n):
            for j in np.argsort(D[i])[1 : extra_k + 1]:
                out[i].add(int(j))
    return [sorted(s) for s in out]


def knn_candidates(cities: ndarray, k: int = 8) -> list[list[int]]:
    """#45's default: pure k-NN candidate lists."""
    D = distance_matrix(cities)
    return [list(np.argsort(D[i])[1 : k + 1]) for i in range(len(cities))]


def reverse_delete_candidates(
    cities: ndarray, seed_k: int = 10, extra_k: int = 5
) -> list[list[int]]:
    """Reverse-delete m=2 "2-core" (built on a kNN seed graph, per OPT 1+3),
    unioned with a few nearest neighbours so LK has enough breadth."""
    D = distance_matrix(cities)
    seed = knn_candidate_graph(cities, seed_k)
    core = reverse_delete_on(D, seed, min_degree=2, reach=reachable_bidirectional)
    return _adj_union_knn(core, cities, extra_k)


def delaunay_candidates(cities: ndarray, extra_k: int = 3) -> list[list[int]]:
    """Delaunay edges unioned with a few nearest neighbours."""
    dg = delaunay_candidate_graph(cities)
    return _adj_union_knn(dg, cities, extra_k)


def _avg_width(cand: list[list[int]]) -> float:
    return float(np.mean([len(c) for c in cand]))


# ===========================================================================
# Benchmark
# ===========================================================================
def bench(sizes=(50, 100, 200), seeds=5, n_starts=3) -> None:
    print("Wiring the reverse-delete candidate list into #45's Lin-Kernighan")
    print("=" * 92)
    print(
        f"Tour length as ratio to per-instance best; time = mean s/instance. "
        f"LK n_starts={n_starts}."
    )
    print(
        f"{'n':>4} | {'full2opt':>8} {'s':>6} | {'LK-kNN8':>8} {'s':>6} "
        f"| {'LK-revdel':>9} {'s':>6} {'|cand|':>6} | {'LK-delaunay':>11} {'s':>6}"
    )
    print("-" * 92)
    for n in sizes:
        f2, lkk, lkr, lkd = [], [], [], []
        tf = tk = tr = td = 0.0
        widths = []
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            # One float64 matrix for every method, so lengths are comparable.
            D = np.ascontiguousarray(distance_matrix(cities), dtype=np.float64)

            t0 = time.perf_counter()
            f2t = two_opt(D, nearest_neighbour_tour(D))
            tf += time.perf_counter() - t0
            f2_len = rd_tour_length(D, f2t)

            ck = knn_candidates(cities, 8)
            t0 = time.perf_counter()
            _, lk_k = lin_kernighan(D, candidates=ck, n_starts=n_starts)
            tk += time.perf_counter() - t0

            cr = reverse_delete_candidates(cities)
            widths.append(_avg_width(cr))
            t0 = time.perf_counter()
            _, lk_r = lin_kernighan(D, candidates=cr, n_starts=n_starts)
            tr += time.perf_counter() - t0

            cd = delaunay_candidates(cities)
            t0 = time.perf_counter()
            _, lk_d = lin_kernighan(D, candidates=cd, n_starts=n_starts)
            td += time.perf_counter() - t0

            best = min(f2_len, lk_k, lk_r, lk_d)
            f2.append(f2_len / best)
            lkk.append(lk_k / best)
            lkr.append(lk_r / best)
            lkd.append(lk_d / best)

        def m(a):
            return float(np.mean(a))

        print(
            f"{n:>4} | {m(f2):>8.3f} {tf / seeds:>6.3f} | {m(lkk):>8.3f} "
            f"{tk / seeds:>6.3f} | {m(lkr):>9.3f} {tr / seeds:>6.3f} "
            f"{m(widths):>6.1f} | {m(lkd):>11.3f} {td / seeds:>6.3f}"
        )
    print("-" * 92)
    print("Reading: 1.000 = best method on that instance. The three LK columns")
    print("share one engine; only the candidate list differs (kNN vs reverse-delete")
    print("2-core vs Delaunay). full2opt is the OPT-4 baseline from the review.")


if __name__ == "__main__":
    bench()
