"""Optimizations for the reverse-delete dense-graph / TSP spike (follow-up to
`reverse_delete_tsp.py`, prompted by the adversarial review of PR #46).

Each section implements one proposed optimization and benchmarks it against the
baseline. Run:  python -m experiments.reverse_delete_opt

  OPT 1  sparsify first (kNN / Delaunay candidate graph) before reverse-delete
  OPT 2  skip reverse-delete for m=1; use the package's Prim MST
  OPT 3  cheaper connectivity test (bidirectional BFS + early exit)
  OPT 4  use the reverse-delete 2-core as a candidate list for local search
  OPT 5  principled degree-2: additive greedy-edge tour + exact min 2-factor
  OPT 6  robust duality assertion (weight, not edge-set, is the invariant)

Metrics are printed as comparison tables; nothing here is shipped in the wheel.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from tribbleclustering.pvat import vat_prim_mst  # noqa: E402
from experiments.reverse_delete_tsp import (  # noqa: E402
    Adj,
    complete_graph,
    distance_matrix,
    edge_set,
    edges_of,
    kruskal_mst_edges,
    nearest_neighbour_tour,
    prim_mst_edges,
    reverse_delete,
    shortcut_tour,
    total_weight,
    tour_from_adj,
    tour_length,
    two_opt,
    uniform_cities,
)

try:
    from scipy.spatial import Delaunay

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False


# ===========================================================================
# Connectivity tests (OPT 3)
# ===========================================================================
def reachable_dfs(adj: Adj, src: int, dst: int) -> bool:
    """Baseline: single-source DFS (what `reverse_delete_tsp` ships)."""
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


def reachable_bidirectional(adj: Adj, src: int, dst: int) -> bool:
    """Grow frontiers from both ends; stop when they meet. Much smaller
    explored set on the common case where src/dst are a few hops apart."""
    if src == dst:
        return True
    fwd = {src}
    bwd = {dst}
    front_f = [src]
    front_b = [dst]
    while front_f and front_b:
        # expand the smaller frontier
        if len(front_f) <= len(front_b):
            nxt = []
            for x in front_f:
                for y in adj[x]:
                    if y in bwd:
                        return True
                    if y not in fwd:
                        fwd.add(y)
                        nxt.append(y)
            front_f = nxt
        else:
            nxt = []
            for x in front_b:
                for y in adj[x]:
                    if y in fwd:
                        return True
                    if y not in bwd:
                        bwd.add(y)
                        nxt.append(y)
            front_b = nxt
    return False


def reverse_delete_on(
    D: np.ndarray,
    init_adj: Adj,
    min_degree: int = 1,
    reach: Callable[[Adj, int, int], bool] = reachable_dfs,
) -> Adj:
    """Reverse-delete restricted to a given candidate edge set (OPT 1/OPT 3).

    Identical policy to `reverse_delete`, but starts from `init_adj` instead of
    the complete graph and takes a pluggable connectivity test.
    """
    adj: Adj = {u: set(vs) for u, vs in init_adj.items()}
    order = sorted(edges_of(adj), key=lambda e: (D[e[0], e[1]], e), reverse=True)
    for u, v in order:
        if len(adj[u]) <= min_degree or len(adj[v]) <= min_degree:
            continue
        adj[u].discard(v)
        adj[v].discard(u)
        if not reach(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
    return adj


# ===========================================================================
# Candidate graphs (OPT 1)
# ===========================================================================
def knn_candidate_graph(cities: np.ndarray, k: int) -> Adj:
    """Symmetric k-nearest-neighbour graph (undirected union)."""
    D = distance_matrix(cities)
    n = len(cities)
    adj: Adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in np.argsort(D[i])[1 : k + 1]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj


def delaunay_candidate_graph(cities: np.ndarray) -> Adj:
    """Delaunay triangulation edges. For 2-D Euclidean points this graph
    provably contains the MST (and nearly all optimal-tour edges)."""
    n = len(cities)
    adj: Adj = {i: set() for i in range(n)}
    tri = Delaunay(np.asarray(cities, dtype=np.float64))
    for simplex in tri.simplices:
        for a in range(len(simplex)):
            for b in range(a + 1, len(simplex)):
                u, v = int(simplex[a]), int(simplex[b])
                adj[u].add(v)
                adj[v].add(u)
    return adj


def n_edges(adj: Adj) -> int:
    return sum(len(v) for v in adj.values()) // 2


# ===========================================================================
# Candidate-restricted 2-opt (OPT 4)
# ===========================================================================
def neighbour_lists(cities: np.ndarray, k: int) -> list[list[int]]:
    """Per-city candidate neighbours, ascending by distance."""
    D = distance_matrix(cities)
    return [list(np.argsort(D[i])[1 : k + 1]) for i in range(len(cities))]


def two_opt_neighbour(
    D: np.ndarray,
    tour: list[int],
    neigh: list[list[int]],
    max_pass: int = 40,
) -> list[int]:
    """Neighbour-list 2-opt for the edge (a,b)=(t[i],t[i+1]). Two anchors, both
    with the standard sorted-candidate break so each scan is ~O(k):

      A) new edge (a, c) with c near a   -> segment i+1 .. pos[c]
      B) new edge (b, c) with c near b   -> segment i+1 .. pos[c]-1

    Anchor B is essential: NN tours already have short successor edges, so
    anchor A alone prunes out immediately and cannot repair the few long edges.
    """
    t = list(tour)
    n = len(t)
    pos = {c: i for i, c in enumerate(t)}

    def do_move(i: int, k: int) -> None:
        t[i + 1 : k + 1] = t[i + 1 : k + 1][::-1]
        for idx in range(i + 1, k + 1):
            pos[t[idx]] = idx

    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(n - 1):
            a = t[i]
            b = t[i + 1]
            dab = D[a, b]
            moved = False
            # Anchor A: reconnect a to a nearer neighbour c = t[k].
            for c in neigh[a]:
                if D[a, c] >= dab:
                    break
                k = pos[c]
                if k <= i + 1:
                    continue
                d = t[(k + 1) % n]
                if d != a and D[a, c] + D[b, d] < dab + D[c, d] - 1e-12:
                    do_move(i, k)
                    improved = moved = True
                    break
            if moved:
                continue
            # Anchor B: reconnect b to a nearer neighbour c = t[k+1].
            for c in neigh[b]:
                if D[b, c] >= dab:
                    break
                k = pos[c] - 1
                if k <= i + 1:
                    continue
                cc = t[k]
                if cc != b and D[a, cc] + D[b, c] < dab + D[cc, c] - 1e-12:
                    do_move(i, k)
                    improved = True
                    break
    return t


# ===========================================================================
# Principled degree-2 constructions (OPT 5)
# ===========================================================================
def greedy_edge_tour(D: np.ndarray) -> list[int]:
    """Additive dual of reverse-delete: add the *cheapest* edge that keeps every
    degree <= 2 and forms no premature subtour. Always converges to a tour."""
    n = D.shape[0]
    order = sorted(
        ((D[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda t: t[0],
    )
    deg = [0] * n
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: Adj = {i: set() for i in range(n)}
    count = 0
    for _, u, v in order:
        if deg[u] < 2 and deg[v] < 2 and find(u) != find(v):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            parent[find(u)] = find(v)
            count += 1
            if count == n - 1:
                break
    ends = [i for i in range(n) if deg[i] < 2]  # the two path endpoints
    adj[ends[0]].add(ends[1])
    adj[ends[1]].add(ends[0])
    return tour_from_adj(adj) or list(range(n))


def subtours_of(adj: Adj) -> list[list[int]]:
    """Decompose a 2-regular graph into its cycles."""
    n = len(adj)
    seen = [False] * n
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        cyc = [s]
        seen[s] = True
        prev, cur = s, next(iter(adj[s]))
        while cur != s:
            seen[cur] = True
            cyc.append(cur)
            nxt = adj[cur] - {prev}
            prev, cur = cur, next(iter(nxt))
        cycles.append(cyc)
    return cycles


def patch_subtours(D: np.ndarray, cycles: list[list[int]]) -> list[int]:
    """Merge cycles into one tour by repeatedly concatenating the two whose
    nearest endpoints are closest (cheap deterministic stitch)."""
    segs = [list(c) for c in cycles]
    while len(segs) > 1:
        best = (0, 1, np.inf, False)
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                for flip in (False, True):
                    tail = segs[a][-1]
                    head = segs[b][0] if not flip else segs[b][-1]
                    d = D[tail, head]
                    if d < best[2]:
                        best = (a, b, d, flip)
        a, b, _, flip = best
        segb = segs[b][::-1] if flip else segs[b]
        segs[a] = segs[a] + segb
        segs.pop(b)
    return segs[0]


def min_weight_2factor(D: np.ndarray, cand: Adj) -> Adj | None:
    """Exact minimum-weight 2-factor of the candidate graph via Tutte's
    reduction to minimum-weight perfect matching (Edmonds blossom, networkx).

    Gadget: each edge e=(u,v) -> port nodes a_e(near u), b_e(near v) joined by
    an edge of weight w(e); each vertex v of degree d gets d-2 zero-weight
    "core" nodes complete-bipartite to its ports. A perfect matching leaves
    exactly 2 ports per vertex for their edge-edges -> a 2-factor of weight =
    sum of chosen edges. Returns None if no 2-factor exists in `cand`.
    """
    if not _HAS_NX:
        return None
    if any(len(cand[v]) < 2 for v in cand):
        return None
    G = nx.Graph()
    ports: dict[tuple[int, int], tuple] = {}
    for u, v in edges_of(cand):
        au = ("port", u, v, u)
        bv = ("port", u, v, v)
        ports[(u, v, u)] = au
        ports[(u, v, v)] = bv
        G.add_edge(au, bv, weight=float(D[u, v]))
    for v in cand:
        pv = [ports[(min(v, w), max(v, w), v)] for w in cand[v]]
        cores = [("core", v, i) for i in range(len(cand[v]) - 2)]
        for c in cores:
            for p in pv:
                G.add_edge(c, p, weight=0.0)
    matching = nx.min_weight_matching(G)
    n = len(cand)
    adj: Adj = {i: set() for i in range(n)}
    for x, y in matching:
        if x[0] == "port" and y[0] == "port":
            u, v = x[1], x[2]
            adj[u].add(v)
            adj[v].add(u)
    if any(len(adj[i]) != 2 for i in adj):  # gadget/degeneracy guard
        return None
    return adj


# ===========================================================================
# Benchmarks
# ===========================================================================
def _ref_length(D: np.ndarray, candidates: list[list[int]]) -> float:
    """Best tour length found across all methods (per-instance reference)."""
    return min(tour_length(D, t) for t in candidates)


def bench_opt1_sparsify(sizes=(60, 120, 200), k=10, seeds=3) -> None:
    print("\n" + "=" * 78)
    print("OPT 1  Sparsify before reverse-delete (m=1 MST) — build time & MST match")
    print("=" * 78)
    print(
        f"{'n':>4} {'full_s':>8} {'kNN_s':>8} {'delaunay_s':>11} "
        f"{'full=MST':>9} {'kNN=MST':>8} {'del=MST':>8} {'E_full':>7} {'E_kNN':>6}"
    )
    for n in sizes:
        tf = tk = td = 0.0
        ok_f = ok_k = ok_d = True
        ef = ek = 0
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            D = distance_matrix(cities)
            mst = kruskal_mst_edges(D)

            t0 = time.perf_counter()
            full = reverse_delete(D, 1)[0]
            tf += time.perf_counter() - t0
            ef = n_edges(complete_graph(n))

            kg = knn_candidate_graph(cities, k)
            ek = n_edges(kg)
            t0 = time.perf_counter()
            kn = reverse_delete_on(D, kg, 1)
            tk += time.perf_counter() - t0

            dg = delaunay_candidate_graph(cities)
            t0 = time.perf_counter()
            dn = reverse_delete_on(D, dg, 1)
            td += time.perf_counter() - t0

            ok_f &= edge_set(full) == mst
            ok_k &= edge_set(kn) == mst
            ok_d &= edge_set(dn) == mst
        print(
            f"{n:>4} {tf / seeds:>8.3f} {tk / seeds:>8.4f} {td / seeds:>11.4f} "
            f"{str(ok_f):>9} {str(ok_k):>8} {str(ok_d):>8} {ef:>7} {ek:>6}"
        )
    print("  (speedup = full_s / kNN_s; MST columns must all be True)")


def bench_opt2_mst_constructor(sizes=(100, 200, 400, 800), seeds=3) -> None:
    print("\n" + "=" * 78)
    print("OPT 2  m=1 constructor: reverse-delete vs Prim (package) vs Kruskal")
    print("=" * 78)
    print(
        f"{'n':>4} {'revdel_s':>9} {'prim_s':>9} {'kruskal_s':>10} "
        f"{'prim=Krusk_wt':>14} {'speedup(rd/prim)':>17}"
    )
    for n in sizes:
        trd = tpr = tkr = 0.0
        wok = True
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            D = distance_matrix(cities)
            if n <= 200:
                t0 = time.perf_counter()
                reverse_delete(D, 1)
                trd += time.perf_counter() - t0
            t0 = time.perf_counter()
            pr = prim_mst_edges(D)
            tpr += time.perf_counter() - t0
            t0 = time.perf_counter()
            kr = kruskal_mst_edges(D)
            tkr += time.perf_counter() - t0
            wok &= abs(total_weight(D, pr) - total_weight(D, kr)) < 1e-2
        rd = trd / seeds if n <= 200 else float("nan")
        sp = f"{rd / (tpr / seeds):.0f}x" if n <= 200 else "n/a (rd too slow)"
        print(
            f"{n:>4} {rd:>9.3f} {tpr / seeds:>9.4f} {tkr / seeds:>10.4f} "
            f"{str(wok):>14} {sp:>17}"
        )


def bench_opt3_connectivity(sizes=(120, 200, 300), k=12, seeds=3) -> None:
    print("\n" + "=" * 78)
    print("OPT 3  Connectivity test: single DFS vs bidirectional (m=1 on kNN)")
    print("=" * 78)
    print(f"{'n':>4} {'dfs_s':>9} {'bidir_s':>9} {'speedup':>9} {'same_result':>12}")
    for n in sizes:
        td = tb = 0.0
        same = True
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            D = distance_matrix(cities)
            kg = knn_candidate_graph(cities, k)
            t0 = time.perf_counter()
            a1 = reverse_delete_on(D, kg, 1, reach=reachable_dfs)
            td += time.perf_counter() - t0
            t0 = time.perf_counter()
            a2 = reverse_delete_on(D, kg, 1, reach=reachable_bidirectional)
            tb += time.perf_counter() - t0
            same &= edge_set(a1) == edge_set(a2)
        print(
            f"{n:>4} {td / seeds:>9.4f} {tb / seeds:>9.4f} "
            f"{td / tb:>8.2f}x {str(same):>12}"
        )


def bench_opt4_candidate_2opt(sizes=(100, 200, 400, 800), k=10, seeds=5) -> None:
    print("\n" + "=" * 78)
    print("OPT 4  Full O(n^2) 2-opt vs neighbour-list 2-opt (kNN candidates)")
    print("=" * 78)
    print(
        f"{'n':>4} {'full_len':>9} {'full_s':>8} {'cand_len':>9} {'cand_s':>8} "
        f"{'len_ratio':>10} {'speedup':>8}"
    )
    for n in sizes:
        fl = cl = 0.0
        ft = ct = 0.0
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            D = distance_matrix(cities)
            nn = nearest_neighbour_tour(D)
            neigh = neighbour_lists(cities, k)
            t0 = time.perf_counter()
            tf = two_opt(D, nn)
            ft += time.perf_counter() - t0
            fl += tour_length(D, tf)
            t0 = time.perf_counter()
            tc = two_opt_neighbour(D, nn, neigh)
            ct += time.perf_counter() - t0
            cl += tour_length(D, tc)
        print(
            f"{n:>4} {fl / seeds:>9.1f} {ft / seeds:>8.3f} {cl / seeds:>9.1f} "
            f"{ct / seeds:>8.3f} {cl / fl:>10.3f} {ft / ct:>7.1f}x"
        )
    print("  (len_ratio ~1.0 => same quality; speedup = full_s / cand_s)")


def bench_opt5_degree2(sizes=(20, 50, 100), seeds=10) -> None:
    print("\n" + "=" * 78)
    print("OPT 5  Degree-2 construction: reverse-delete m=2 vs additive greedy-edge")
    print("       vs exact min-2-factor. Tour len / per-instance best; +2-opt.")
    print("=" * 78)
    print(
        f"{'n':>4} {'RDm2_conv%':>10} {'RD+2o':>7} {'greedyE':>8} {'greedyE+2o':>11} "
        f"{'2factor_sub':>12} {'2fac+patch+2o':>14}"
    )
    for n in sizes:
        rd_conv = 0
        rd2, ge, ge2, f2, f2patch = [], [], [], [], []
        sub_counts = []
        for s in range(seeds):
            cities = uniform_cities(n, seed=s)
            D = distance_matrix(cities)

            adj, _ = reverse_delete(D, 2)
            t = tour_from_adj(adj)
            if t is not None:
                rd_conv += 1
            else:
                t = shortcut_tour(D, adj)
            rd_tour2 = two_opt(D, t)

            g = greedy_edge_tour(D)
            g2 = two_opt(D, g)

            cand = knn_candidate_graph(cities, min(12, n - 1))
            fac = min_weight_2factor(D, cand) if n <= 60 else None
            if fac is not None:
                cyc = subtours_of(fac)
                sub_counts.append(len(cyc))
                fac_tour = two_opt(D, patch_subtours(D, cyc))
            else:
                fac_tour = None

            ref_cands = [rd_tour2, g, g2]
            if fac_tour is not None:
                ref_cands.append(fac_tour)
            ref = _ref_length(D, ref_cands)
            rd2.append(tour_length(D, rd_tour2) / ref)
            ge.append(tour_length(D, g) / ref)
            ge2.append(tour_length(D, g2) / ref)
            if fac_tour is not None:
                f2patch.append(tour_length(D, fac_tour) / ref)

        def m(a):
            return f"{np.mean(a):.3f}" if a else "  n/a"

        sub = f"{np.mean(sub_counts):.1f}" if sub_counts else "n/a"
        print(
            f"{n:>4} {100 * rd_conv / seeds:>9.0f}% {m(rd2):>7} {m(ge):>8} "
            f"{m(ge2):>11} {sub:>12} {m(f2patch):>14}"
        )
    print("  (ratios to per-instance best-found; 1.000 = best; conv% = pure m=2)")


def bench_opt6_robustness() -> None:
    print("\n" + "=" * 78)
    print("OPT 6  Duality assertion robustness: edge-set vs weight under ties")
    print("=" * 78)
    # A grid graph has many equal distances -> the MST is non-unique, so the
    # edge SET can differ between algorithms even though the WEIGHT is identical.
    g = np.array([[x, y] for x in range(5) for y in range(5)], dtype=np.float32)
    D = distance_matrix(g)
    rd = edge_set(reverse_delete(D, 1)[0])
    kr = kruskal_mst_edges(D)
    print(f"  25-node unit grid (many tied distances):")
    print(f"    reverse-delete edge-set == Kruskal edge-set : {rd == kr}")
    print(
        f"    reverse-delete weight    == Kruskal weight   : "
        f"{abs(total_weight(D, rd) - total_weight(D, kr)) < 1e-4}"
    )
    print("  => assert WEIGHT equality (invariant); edge-set only in general position.")


if __name__ == "__main__":
    print("Reverse-delete optimizations — benchmark suite")
    print(f"(scipy={_HAS_SCIPY}, networkx={_HAS_NX})")
    bench_opt1_sparsify()
    bench_opt2_mst_constructor()
    bench_opt3_connectivity()
    bench_opt4_candidate_2opt()
    bench_opt5_degree2()
    bench_opt6_robustness()
