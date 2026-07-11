"""Spike: a reverse-delete architecture for dense graphs — from MST to TSP.

    "Recursively delete the largest edge that does not create an island."

Start from a *dense* (complete) weighted graph and repeatedly remove the
heaviest edge whose removal keeps the graph connected — i.e. never let any
vertex or subgraph become an "island". This is Kruskal's less-famous twin, the
**reverse-delete algorithm**, and it terminates at the Minimum Spanning Tree:
the exact same tree the package's Prim kernel (`vat_prim_mst`) builds by *adding*
the lightest edges. Since VAT's output depends only on the MST, reverse-delete
is a second, dual route to the same VAT ordering — we verify that here.

The interesting part is one extra rule. "Does not create an island" forbids
*degree-0* vertices; strengthen it to forbid dropping any vertex below degree
`m` and the same recursion generalizes along a single knob:

    m = 1  ->  connectivity floor only  ->  Minimum Spanning Tree  (a tree)
    m = 2  ->  no leaves allowed         ->  a 2-regular connected graph,
                                             i.e. a single Hamiltonian cycle
                                             = a travelling-salesman tour

So the *same* "delete the largest safe edge" recursion sweeps from the MST the
clustering side already uses (m=1) up to a TSP tour for the routing side (m=2).
The heaviest edges are shed first, so what survives are the short local hops a
good tour wants — a purely subtractive cousin of the greedy-edge and
nearest-neighbour constructions.

This script:
  1. implements the recursion (`reverse_delete`) for any minimum-degree floor;
  2. proves the m=1 result equals the package's Prim MST (edge set + weight);
  3. benchmarks the m=2 tour vs nearest-neighbour, 2-opt polish, the MST lower
     bound, and brute-force optimal (small n);
  4. renders the process and the quality artifacts.

Cost note: reverse-delete is O(E) connectivity probes, each an O(V+E) reach
test, so ~O(n^4) on a dense graph — clean to reason about but quadratically
slower to *build* than the package's Prim MST. The value here is the unifying
m-knob framing, not the construction speed; Prim remains the production path.

Run:  python -m experiments.reverse_delete_tsp
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.util import (  # noqa: E402
    circle_random_clusters,
    pairwise_distances,
)

FIG_DIR = Path(__file__).parent / "figures"

Edge = tuple[int, int]
Adj = dict[int, set[int]]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def complete_graph(n: int) -> Adj:
    """Adjacency of the dense (complete) graph K_n as vertex -> neighbour set."""
    return {i: set(j for j in range(n) if j != i) for i in range(n)}


def edges_of(adj: Adj) -> list[Edge]:
    """Undirected edge list (u < v) of an adjacency map."""
    seen: set[Edge] = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            seen.add((u, v) if u < v else (v, u))
    return list(seen)


def _reachable(adj: Adj, src: int, dst: int) -> bool:
    """True iff `dst` is reachable from `src` in `adj` (iterative DFS)."""
    seen = {src}
    stack = [src]
    while stack:
        x = stack.pop()
        if x == dst:
            return True
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


# ---------------------------------------------------------------------------
# The reverse-delete recursion (the whole idea)
# ---------------------------------------------------------------------------
def reverse_delete(D: np.ndarray, min_degree: int = 1) -> tuple[Adj, list[Edge]]:
    """Recursively delete the largest edge that does not create an island.

    Consider every edge once, heaviest first. Delete it iff (a) neither endpoint
    would fall below ``min_degree`` and (b) the two endpoints stay connected
    without it (removing a bridge would split off an island, so it is kept).

    ``min_degree=1`` gives the Minimum Spanning Tree; ``min_degree=2`` drives the
    graph toward a 2-regular connected graph, i.e. a Hamiltonian TSP tour.

    Returns the surviving adjacency and the list of deleted edges in order.
    """
    n = D.shape[0]
    adj = complete_graph(n)
    # One fixed descending pass over the original edge set — the classic
    # reverse-delete order. Ties broken by endpoint index for determinism.
    order = sorted(edges_of(adj), key=lambda e: (D[e[0], e[1]], e), reverse=True)

    removed: list[Edge] = []
    for u, v in order:
        if len(adj[u]) <= min_degree or len(adj[v]) <= min_degree:
            continue  # would create a vertex below the degree floor
        adj[u].discard(v)
        adj[v].discard(u)
        if _reachable(adj, u, v):
            removed.append((u, v))  # safe: an alternate path remains
        else:
            adj[u].add(v)  # bridge — putting it back avoids an island
            adj[v].add(u)
    return adj, removed


# ---------------------------------------------------------------------------
# Tours
# ---------------------------------------------------------------------------
def tour_from_adj(adj: Adj) -> list[int] | None:
    """Walk a 2-regular connected graph into a vertex cycle, else return None.

    Returns None when the reverse-delete recursion stalled above degree 2
    (some vertex still has >2 edges) or fragmented into multiple cycles.
    """
    n = len(adj)
    if any(len(adj[i]) != 2 for i in adj):
        return None
    start = 0
    prev, cur = start, next(iter(adj[start]))
    tour = [start]
    while cur != start:
        tour.append(cur)
        nxt = adj[cur] - {prev}
        if not nxt:  # dead end (should not happen when 2-regular)
            return None
        prev, cur = cur, next(iter(nxt))
    return tour if len(tour) == n else None  # len<n => several disjoint cycles


def shortcut_tour(D: np.ndarray, adj: Adj, start: int = 0) -> list[int]:
    """Complete a stalled reverse-delete subgraph into a Hamiltonian cycle.

    The m=2 recursion usually stalls above degree 2 (see the findings), leaving
    a *connected, min-degree-2* sparse subgraph rather than a finished cycle. We
    finish it in-theme: a depth-first walk that always steps to the nearest
    unvisited neighbour *along a surviving edge*, then shortcuts repeats. On the
    complete graph every shortcut is a legal hop, so the visit order is a valid
    tour — the classic MST/2-factor shortcutting trick, applied to the pruned
    graph. Follow with ``two_opt`` to remove the shortcut crossings.
    """
    seen = {start}
    order = [start]
    stack = [start]
    while stack:
        u = stack[-1]
        nxt = [v for v in adj[u] if v not in seen]
        if nxt:
            v = min(nxt, key=lambda w: D[u, w])
            seen.add(v)
            order.append(v)
            stack.append(v)
        else:
            stack.pop()
    return order


def reverse_delete_tour(D: np.ndarray, repair: bool = True) -> tuple[list[int], bool]:
    """End-to-end m=2 tour: sparsify by reverse-delete, then complete.

    Returns (tour, converged) where ``converged`` is True iff the recursion
    reached a pure 2-regular graph on its own (no repair needed).
    """
    adj, _ = reverse_delete(D, min_degree=2)
    tour = tour_from_adj(adj)
    if tour is not None:
        return tour, True
    if not repair:
        return [], False
    return shortcut_tour(D, adj), False


def tour_length(D: np.ndarray, tour: list[int]) -> float:
    idx = np.asarray(tour)
    return float(np.sum(D[idx, np.roll(idx, -1)]))


def nearest_neighbour_tour(D: np.ndarray, start: int = 0) -> list[int]:
    """Classic constructive baseline: always hop to the closest unvisited city."""
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        unvisited.remove(nxt)
        tour.append(nxt)
        cur = nxt
    return tour


def two_opt(D: np.ndarray, tour: list[int], max_pass: int = 40) -> list[int]:
    """2-opt local search: reverse segments while it shortens the tour."""
    t = list(tour)
    n = len(t)
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            for k in range(i + 2, n):
                c = t[k]
                d = t[(k + 1) % n]
                if d == a:
                    continue
                # gain from replacing (a,b)+(c,d) with (a,c)+(b,d)
                if D[a, c] + D[b, d] < D[a, b] - 1e-12 + D[c, d]:
                    t[i + 1 : k + 1] = t[i + 1 : k + 1][::-1]
                    improved = True
                    a, b = t[i], t[i + 1]
    return t


def brute_force_optimal(D: np.ndarray) -> tuple[list[int], float]:
    """Exact optimum by fixing city 0 and permuting the rest. Use for n <= ~10."""
    n = D.shape[0]
    best_tour = list(range(n))
    best_len = tour_length(D, best_tour)
    for perm in itertools.permutations(range(1, n)):
        cand = [0, *perm]
        length = tour_length(D, cand)
        if length < best_len:
            best_len = length
            best_tour = cand
    return best_tour, best_len


# ---------------------------------------------------------------------------
# Duality: reverse-delete MST == package Prim MST
# ---------------------------------------------------------------------------
def kruskal_mst_edges(D: np.ndarray) -> set[frozenset[int]]:
    """Reference MST (Kruskal + union-find) — unambiguous ground truth."""
    n = D.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted(
        ((D[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda t: t[0],
    )
    mst: set[frozenset[int]] = set()
    for _, i, j in order:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.add(frozenset((i, j)))
    return mst


def prim_mst_edges(D: np.ndarray) -> set[frozenset[int]]:
    """MST edge set reconstructed from the package's Prim/VAT ordering.

    ``vat_prim_mst`` returns the order in which Prim admits vertices. Prim's
    invariant is that each newly admitted vertex joins via its nearest
    already-admitted vertex, so the MST parent of ``seq[k]`` is
    ``argmin_{j<k} D[seq[k], seq[j]]`` — reconstructed here from the ordering.
    """
    from tribbleclustering.pvat import vat_prim_mst

    seq_arr, _ = vat_prim_mst(np.ascontiguousarray(D))
    seq = [int(x) for x in seq_arr]
    edges: set[frozenset[int]] = set()
    for k in range(1, len(seq)):
        v = seq[k]
        parent = min(seq[:k], key=lambda p: D[v, p])
        edges.add(frozenset((v, parent)))
    return edges


def edge_set(adj: Adj) -> set[frozenset[int]]:
    return {frozenset(e) for e in edges_of(adj)}


def total_weight(D: np.ndarray, edges: set[frozenset[int]]) -> float:
    return float(sum(D[tuple(e)] for e in edges))


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def uniform_cities(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(n, 2)).astype(np.float32)


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    return pairwise_distances(np.ascontiguousarray(cities))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _draw_edges(ax, cities, edges, **kw):
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        ax.plot(
            [cities[u, 0], cities[v, 0]],
            [cities[u, 1], cities[v, 1]],
            **kw,
        )


def process_figure() -> Path:
    """Visual story: dense K_n -> MST (m=1) -> m=2 sparse subgraph -> tour."""
    cities = circle_random_clusters(n_clusters=5, n_cities=6, cluster_spacing=12.0)
    cities = np.ascontiguousarray(cities.astype(np.float32))
    D = distance_matrix(cities)
    n = len(cities)
    n_dense = n * (n - 1) // 2

    dense = edges_of(complete_graph(n))
    mst_adj, _ = reverse_delete(D, min_degree=1)
    sparse_adj, _ = reverse_delete(D, min_degree=2)
    tour, converged = reverse_delete_tour(D)
    tour = two_opt(D, tour)
    n_sparse = len(edges_of(sparse_adj))
    tour_edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    panels = [
        (f"dense graph K_n\n{n_dense} edges", dense, "0.75", 0.2),
        (
            f"m=1 -> MST\n{len(edges_of(mst_adj))} edges (tree)",
            edges_of(mst_adj),
            "tab:blue",
            1.4,
        ),
        (
            f"m=2 -> sparse 2-core\n{n_sparse} edges "
            f"({100 * n_sparse / n_dense:.0f}% of dense)",
            edges_of(sparse_adj),
            "tab:green",
            1.4,
        ),
        (
            f"shortcut + 2-opt tour\nlen={tour_length(D, tour):.1f}"
            + ("  (converged)" if converged else "  (repaired)"),
            tour_edges,
            "tab:red",
            1.6,
        ),
    ]
    for ax, (title, edges, color, lw) in zip(axes, panels):
        _draw_edges(ax, cities, edges, color=color, lw=lw, alpha=0.9, zorder=1)
        ax.scatter(cities[:, 0], cities[:, 1], s=28, c="black", zorder=2)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "One recursion, one knob: 'delete the largest edge that does not create "
        "an island' — the minimum-degree floor m sweeps MST -> 2-core -> tour.",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "reverse_delete_process.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def quality_figure(
    small=(6, 7, 8, 9, 10),
    scale=(10, 20, 40, 80, 120),
    n_seeds: int = 12,
) -> Path:
    """Left: tour quality vs optimum (small n). Right: dense-graph pruning."""
    labels = ["RD + shortcut", "RD + shortcut + 2-opt", "NN", "NN + 2-opt"]
    ratios: dict[str, list[float]] = {k: [] for k in labels}
    for n in small:
        buckets: dict[str, list[float]] = {k: [] for k in labels}
        for seed in range(n_seeds):
            D = distance_matrix(uniform_cities(n, seed=seed))
            _, opt = brute_force_optimal(D)
            tour, _ = reverse_delete_tour(D)
            buckets["RD + shortcut"].append(tour_length(D, tour) / opt)
            buckets["RD + shortcut + 2-opt"].append(
                tour_length(D, two_opt(D, tour)) / opt
            )
            nn = nearest_neighbour_tour(D)
            buckets["NN"].append(tour_length(D, nn) / opt)
            buckets["NN + 2-opt"].append(tour_length(D, two_opt(D, nn)) / opt)
        for k in labels:
            ratios[k].append(float(np.mean(buckets[k])))

    retained: list[float] = []
    converged: list[float] = []
    scale_seeds = 5
    for n in scale:
        rr, cc = [], 0
        for seed in range(scale_seeds):
            D = distance_matrix(uniform_cities(n, seed=seed))
            adj, _ = reverse_delete(D, min_degree=2)
            rr.append(len(edges_of(adj)) / (n * (n - 1) / 2))
            if tour_from_adj(adj) is not None:
                cc += 1
        retained.append(float(np.mean(rr)))
        converged.append(cc / scale_seeds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(small))
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:cyan"]
    for k, c in zip(labels, colors):
        ax1.plot(x, ratios[k], "o-", color=c, label=k)
    ax1.axhline(1.0, color="k", ls="--", lw=0.8, label="optimum")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in small])
    ax1.set_xlabel("cities n")
    ax1.set_ylabel("tour length / optimum")
    ax1.set_title(f"Tour quality vs optimum (mean over {n_seeds} seeds)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    xs = np.arange(len(scale))
    ax2.plot(xs, [100 * r for r in retained], "s-", color="tab:green")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(s) for s in scale])
    ax2.set_xlabel("cities n")
    ax2.set_ylabel("edges retained by m=2 (% of dense)", color="tab:green")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_title("Reverse-delete prunes the dense graph to a sparse 2-core")
    ax2.grid(alpha=0.3)
    for xi, r in zip(xs, retained):
        ax2.annotate(
            f"{100 * r:.1f}%",
            (xi, 100 * r),
            fontsize=8,
            ha="center",
            va="bottom",
            color="tab:green",
        )

    fig.suptitle(
        "Reverse-delete TSP (m=2): near-optimal tours from a heavily pruned "
        "dense graph",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "reverse_delete_tsp_quality.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def duality_report() -> None:
    print("\n=== m=1 duality: reverse-delete MST == Prim MST == Kruskal MST ===")
    print(f"{'n':>5} {'RD==Kruskal':>12} {'RD==Prim':>10} {'weight_match':>13}")
    for n in (8, 16, 32, 64, 100):
        cities = uniform_cities(n, seed=n)
        D = distance_matrix(cities)
        rd_adj, _ = reverse_delete(D, min_degree=1)
        rd = edge_set(rd_adj)
        kr = kruskal_mst_edges(D)
        pr = prim_mst_edges(D)
        # MST weight is unique even when tie-breaking picks different edges.
        wmatch = (
            abs(total_weight(D, rd) - total_weight(D, kr)) < 1e-3
            and abs(total_weight(D, pr) - total_weight(D, kr)) < 1e-3
        )
        print(f"{n:>5} {str(rd == kr):>12} {str(rd == pr):>10} {str(wmatch):>13}")


def tsp_report(sizes=(6, 8, 10), n_seeds: int = 20) -> None:
    print("\n=== m=2 tour quality (mean length / brute-force optimum) ===")
    print(
        f"{'n':>4} {'conv%':>6} {'RD+sc':>7} {'RD+sc+2opt':>11} "
        f"{'NN':>7} {'NN+2opt':>9} {'MST/opt':>9}"
    )
    for n in sizes:
        rd, rd2, nn, nn2, mstb = [], [], [], [], []
        n_conv = 0
        for seed in range(n_seeds):
            D = distance_matrix(uniform_cities(n, seed=seed))
            _, opt = brute_force_optimal(D)
            tour, converged = reverse_delete_tour(D)
            n_conv += int(converged)
            rd.append(tour_length(D, tour) / opt)
            rd2.append(tour_length(D, two_opt(D, tour)) / opt)
            nnt = nearest_neighbour_tour(D)
            nn.append(tour_length(D, nnt) / opt)
            nn2.append(tour_length(D, two_opt(D, nnt)) / opt)
            mstb.append(total_weight(D, kruskal_mst_edges(D)) / opt)

        def m(a):
            return float(np.mean(a))

        print(
            f"{n:>4} {100 * n_conv / n_seeds:>5.0f}% {m(rd):>7.3f} "
            f"{m(rd2):>11.3f} {m(nn):>7.3f} {m(nn2):>9.3f} {m(mstb):>9.3f}"
        )


if __name__ == "__main__":
    print("Reverse-delete architecture for dense graphs: MST <-> TSP")
    print("=========================================================")
    duality_report()
    tsp_report()
    print()
    print(f"wrote {process_figure()}")
    print(f"wrote {quality_figure()}")
