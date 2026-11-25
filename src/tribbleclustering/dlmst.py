"""Certificate-and-filter MST for VAT/IVAT (pure numpy reference).

This module follows the *executable skeleton* of Algorithm 2 in Khanna,
Putterman & Song, "A Deterministic Linear Time Algorithm for Minimum Spanning
Tree" (arXiv:2611.31187, 2026):

1. Borůvka steps -- each vertex keeps its lightest edge (cut property).
2. A *certificate* forest ``F`` that supports most edges.
3. The filter: drop every edge ``e`` not in ``F`` whose endpoints are joined
   by ``F_{<= w(e)}`` (cycle property) and keep ``U(F)``, the unsupported ones.
4. Solve ``MSF(F ∪ U(F))`` exactly.

What is *not* reproduced, and why
---------------------------------
The paper's certificate procedure (its Section 3) picks a schedule and a
"universal" query set by enumerating every possible schedule against every
possible execution history. The authors note this "may incur enormous
computational overhead, [but] requires no comparisons": the result bounds the
*comparison* count, and running time follows only through the
Pettie–Ramachandran optimal algorithm, which brute-forces optimal decision
trees for tiny sub-instances. Neither step is runnable at any useful size.

The certificate used here is a practical deterministic stand-in with the same
contract (a forest ``F`` plus the edges it does not support):

* ``F0 = MSF(k lightest edges per vertex)`` -- the first of these per vertex
  is exactly one Borůvka step, so step 1 is subsumed;
* Borůvka merge passes over the components of ``F0`` until ``F`` spans.

Relevance to VAT (read before expecting a speedup)
--------------------------------------------------
VAT/IVAT run on the *complete* graph, ``m = n(n-1)/2``. There, the array form
of Prim already runs in ``O(n^2) = O(m)`` -- linear -- as the paper itself
notes. So the theorem gives VAT no asymptotic gain. What the skeleton does
offer is structure: every pass here is independent per row (parallel), while
Prim's round loop is inherently serial; and the filter runs from a forest
alone, so IVAT can be built from the MST with no VAT matrix at all.

The filter uses a Kruskal reconstruction tree (KRT) of ``F``: when Kruskal
joins components ``A`` and ``B`` at weight ``w``, every pair ``(a, b)`` has
``F``-minimax distance exactly ``w``, so the block ``R[A, B]`` is compared to
``w`` in one vectorized step. Blocks partition the ``n(n-1)/2`` pairs, so the
filter makes exactly one comparison per edge.

Ties: with tied weights the returned MST may differ from Prim's choice (both
are minimum); the VAT order can then differ between equal-weight options.
"""

from __future__ import annotations

import heapq
import numpy as np

try:
    from .cdlmst import compute_ivat_dlmst_c as _ivat_c
    from .cdlmst import compute_vat_dlmst_c as _vat_c
    from .cdlmst import vat_order_dlmst_c as _order_c

    _has_compiled_dlmst = True
except ImportError:
    _has_compiled_dlmst = False

# Rows per block when scanning an n x n matrix or computing a distance block.
# Bounds the temporary to about _BLOCK_ELEMS elements.
_BLOCK_ELEMS = 1 << 22


# ---------------------------------------------------------------------------
# Distance oracles: a dense matrix, or points with on-the-fly L2 distances.
# ---------------------------------------------------------------------------
class _Oracle:
    n: int
    dtype: np.dtype

    def block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def row_range(self, i0: int, i1: int) -> np.ndarray:
        return self.block(np.arange(i0, i1), np.arange(self.n))


class _MatrixOracle(_Oracle):
    def __init__(self, R: np.ndarray):
        self.R = R
        self.n = R.shape[0]
        self.dtype = R.dtype

    def block(self, rows, cols):
        return self.R[np.ix_(rows, cols)]

    def row_range(self, i0, i1):
        return self.R[i0:i1]


class _PointsOracle(_Oracle):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.n = X.shape[0]
        self.dtype = X.dtype
        self.sq = np.einsum("ij,ij->i", X, X)

    def block(self, rows, cols):
        A, B = self.X[rows], self.X[cols]
        d2 = self.sq[rows][:, None] + self.sq[cols][None, :] - 2.0 * (A @ B.T)
        np.maximum(d2, 0, out=d2)
        return np.sqrt(d2, out=d2)


def _row_chunks(n: int) -> list[tuple[int, int]]:
    step = max(1, _BLOCK_ELEMS // max(n, 1))
    return [(i, min(i + step, n)) for i in range(0, n, step)]


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------
def _find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _kruskal(n: int, eu: np.ndarray, ev: np.ndarray, ew: np.ndarray) -> np.ndarray:
    """Indices (into the edge arrays) of a minimum spanning forest."""
    order = np.lexsort((np.minimum(eu, ev), np.maximum(eu, ev), ew))
    parent = np.arange(n)
    keep = []
    for i in order:
        a, b = _find(parent, int(eu[i])), _find(parent, int(ev[i]))
        if a != b:
            parent[a] = b
            keep.append(i)
            if len(keep) == n - 1:
                break
    return np.asarray(keep, dtype=np.int64)


def _labels(n: int, eu: np.ndarray, ev: np.ndarray) -> np.ndarray:
    parent = np.arange(n)
    for a, b in zip(eu.tolist(), ev.tolist()):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[ra] = rb
    return np.array([_find(parent, i) for i in range(n)])


# ---------------------------------------------------------------------------
# Passes over the complete graph
# ---------------------------------------------------------------------------
def _k_lightest(o: _Oracle, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each vertex's k lightest incident edges (one Borůvka step when k=1)."""
    n = o.n
    k = min(k, n - 1)
    us, vs, ws = [], [], []
    for i0, i1 in _row_chunks(n):
        blk = np.array(o.row_range(i0, i1), copy=True)
        rows = np.arange(i0, i1)
        blk[rows - i0, rows] = np.inf
        idx = np.argpartition(blk, k - 1, axis=1)[:, :k]
        us.append(np.repeat(rows, k))
        vs.append(idx.ravel())
        ws.append(np.take_along_axis(blk, idx, axis=1).ravel())
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


def _boruvka_merge(o: _Oracle, comp: np.ndarray):
    """One Borůvka step over the components ``comp``: each component's
    lightest outgoing edge."""
    n = o.n
    best_v = np.empty(n, dtype=np.int64)
    best_w = np.empty(n, dtype=o.dtype)
    for i0, i1 in _row_chunks(n):
        blk = np.array(o.row_range(i0, i1), copy=True)
        blk[comp[i0:i1, None] == comp[None, :]] = np.inf
        j = np.argmin(blk, axis=1)
        best_v[i0:i1] = j
        best_w[i0:i1] = blk[np.arange(i1 - i0), j]
    # Lightest per component (ties -> lowest vertex id).
    order = np.lexsort((np.arange(n), best_w, comp))
    first = np.ones(n, dtype=bool)
    first[1:] = comp[order[1:]] != comp[order[:-1]]
    u = order[first]
    return u, best_v[u], best_w[u]


def certificate_forest(
    o: _Oracle, k: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Spanning tree ``F`` that supports most edges, plus the pass count."""
    n = o.n
    eu, ev, ew = _k_lightest(o, k)
    keep = _kruskal(n, eu, ev, ew)
    fu, fv, fw = eu[keep], ev[keep], ew[keep]
    passes = 1
    while len(fu) < n - 1:
        comp = _labels(n, fu, fv)
        mu, mv, mw = _boruvka_merge(o, comp)
        passes += 1
        cu = np.concatenate([fu, mu])
        cv = np.concatenate([fv, mv])
        cw = np.concatenate([fw, mw])
        keep = _kruskal(n, cu, cv, cw)
        fu, fv, fw = cu[keep], cv[keep], cw[keep]
    return fu, fv, fw, passes


def _krt(n: int, fu: np.ndarray, fv: np.ndarray, fw: np.ndarray):
    """Kruskal reconstruction tree of the spanning tree F.

    Returns ``leaf_order`` and, per merge, ``(lo, mid, hi, w)`` such that the
    merge joined ``leaf_order[lo:mid]`` with ``leaf_order[mid:hi]`` at ``w``.
    """
    order = np.argsort(fw, kind="stable")
    parent = np.arange(n)
    node = np.arange(n)  # component root -> KRT node id
    left = np.full(2 * n - 1, -1, dtype=np.int64)
    right = np.full(2 * n - 1, -1, dtype=np.int64)
    wt = np.zeros(2 * n - 1, dtype=fw.dtype)
    nxt = n
    for i in order:
        a, b = _find(parent, int(fu[i])), _find(parent, int(fv[i]))
        left[nxt], right[nxt], wt[nxt] = node[a], node[b], fw[i]
        parent[a] = b
        node[b] = nxt
        nxt += 1
    root = nxt - 1
    # Iterative post-order to lay leaves out contiguously.
    leaf_order = np.empty(n, dtype=np.int64)
    lo = np.zeros(2 * n - 1, dtype=np.int64)
    hi = np.zeros(2 * n - 1, dtype=np.int64)
    pos = 0
    stack = [(root, False)]
    while stack:
        x, done = stack.pop()
        if x < n:
            leaf_order[pos] = x
            lo[x], hi[x] = pos, pos + 1
            pos += 1
        elif done:
            lo[x], hi[x] = lo[left[x]], hi[right[x]]
        else:
            stack.append((x, True))
            stack.append((right[x], False))
            stack.append((left[x], False))
    merges = [(lo[x], hi[left[x]], hi[x], wt[x]) for x in range(n, root + 1)]
    return leaf_order, merges


def unsupported_edges(o: _Oracle, fu, fv, fw):
    """Edges ``(u, v)`` lighter than the ``F``-path maximum between them."""
    n = o.n
    leaf_order, merges = _krt(n, fu, fv, fw)
    us, vs, ws = [], [], []
    for lo, mid, hi, w in merges:
        A, B = leaf_order[lo:mid], leaf_order[mid:hi]
        # Large blocks are cut into row strips to bound the temporary.
        step = max(1, _BLOCK_ELEMS // max(len(B), 1))
        for s in range(0, len(A), step):
            a = A[s : s + step]
            blk = o.block(a, B)
            ia, ib = np.nonzero(blk < w)
            if len(ia):
                us.append(a[ia])
                vs.append(B[ib])
                ws.append(blk[ia, ib])
    if not us:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=o.dtype)
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


def mst_dlmst_py(data: np.ndarray, from_points: bool = False, k: int = 4):
    """Exact MST ``(u, v, w)`` of the complete graph on ``data`` (pure numpy)."""
    o: _Oracle = _PointsOracle(data) if from_points else _MatrixOracle(data)
    return _mst(o, k)


def _mst(o: _Oracle, k: int = 4, stats: dict | None = None):
    """Exact MST of the complete graph via certificate + filter."""
    fu, fv, fw, passes = certificate_forest(o, k)
    uu, uv, uw = unsupported_edges(o, fu, fv, fw)
    cu = np.concatenate([fu, uu])
    cv = np.concatenate([fv, uv])
    cw = np.concatenate([fw, uw])
    keep = _kruskal(o.n, cu, cv, cw)
    if stats is not None:
        stats.update(passes=passes + 1, unsupported=len(uu), forest=len(fu))
    return cu[keep], cv[keep], cw[keep]


# ---------------------------------------------------------------------------
# MST -> VAT ordering, VAT and IVAT
# ---------------------------------------------------------------------------
def _seed(o: _Oracle) -> tuple[int, int]:
    """Row-major first global maximum, matching ``vat_prim_mst``'s seed."""
    best, bi, bj = -np.inf, 0, 0
    for i0, i1 in _row_chunks(o.n):
        blk = o.row_range(i0, i1)
        f = int(np.argmax(blk))
        r, c = divmod(f, o.n)
        if blk[r, c] > best:
            best, bi, bj = blk[r, c], i0 + r, c
    return bi, bj


def mst_to_vat_order(
    n: int, mu, mv, mw, src: int, src_partner: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prim's visiting order run on the MST alone.

    With distinct weights the lightest edge leaving the visited set is always
    an MST edge (cut property), so this reproduces dense Prim's order in
    O(n log n) instead of O(n^2).

    Returns ``(p, q, w)``: ``p`` the order, ``q`` the parent's 1-based position
    in ``p`` (``q[0]`` is the seed partner, as in ``vat_prim_mst``), and ``w``
    the MST edge weight into each vertex (``w[0] = 0``).
    """
    adj: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for a, b, w in zip(mu.tolist(), mv.tolist(), mw.tolist()):
        adj[a].append((w, b))
        adj[b].append((w, a))
    p = np.empty(n, dtype=np.int32)
    q = np.empty(n, dtype=np.int32)
    wout = np.zeros(n, dtype=mw.dtype if len(mw) else np.float64)
    pos = np.full(n, -1, dtype=np.int64)
    heap: list[tuple[float, int, int]] = [(0.0, src, -1)]
    r = 0
    while heap:
        w, v, par = heapq.heappop(heap)
        if pos[v] >= 0:
            continue
        pos[v] = r
        p[r] = v
        q[r] = src_partner if par < 0 else pos[par] + 1
        wout[r] = w
        r += 1
        for wn, x in adj[v]:
            if pos[x] < 0:
                heapq.heappush(heap, (wn, x, v))
    return p, q, wout


def vat_order_py(data: np.ndarray, from_points: bool = False, k: int = 4):
    """VAT order ``(p, q, w)`` from a distance matrix or from points."""
    o: _Oracle = _PointsOracle(data) if from_points else _MatrixOracle(data)
    if o.n == 1:
        return np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1)
    mu, mv, mw = _mst(o, k)
    si, sj = _seed(o)
    return mst_to_vat_order(o.n, mu, mv, mw, si, sj)


def compute_vat_dlmst_py(R: np.ndarray, k: int = 4):
    """Drop-in for ``compute_vat``: ``(vat_matrix, p)``."""
    p, _, _ = vat_order_py(R, k=k)
    return R[np.ix_(p, p)], p


def ivat_from_order(p: np.ndarray, q: np.ndarray, w: np.ndarray, dtype=np.float64):
    """IVAT matrix from the MST alone -- no VAT or distance matrix needed.

    Row ``r`` (in VAT order) is ``max(w[r], row of its parent)`` over the
    earlier columns -- the same recurrence ``compute_ivat`` uses, but the
    parent and its weight come from the MST rather than an argmin scan.
    """
    n = len(p)
    D = np.zeros((n, n), dtype=dtype)
    for r in range(1, n):
        pr = int(q[r]) - 1
        row = D[r, :r]
        # D[pr, c] for c < r: row pr holds c < pr; column pr holds pr < c < r.
        row[:pr] = D[pr, :pr]
        row[pr + 1 :] = D[pr + 1 : r, pr]
        np.maximum(row, w[r], out=row)
        row[pr] = w[r]
    iu = np.triu_indices(n, 1)
    D[iu] = D.T[iu]
    return D


def compute_ivat_dlmst_py(data: np.ndarray, from_points: bool = False, k: int = 4):
    """Drop-in for ``compute_ivat``: ``(ivat, argmin_seq, p)``.

    With ``from_points=True`` no ``n x n`` matrix other than the output is
    ever allocated.
    """
    p, q, w = vat_order_py(data, from_points=from_points, k=k)
    D = ivat_from_order(p, q, w, dtype=data.dtype)
    return D, (q[1:] - 1).tolist(), p


# ---------------------------------------------------------------------------
# Public API: compiled kernels when built, the numpy reference otherwise.
# ---------------------------------------------------------------------------
def vat_order_dlmst(
    data: np.ndarray, from_points: bool = False, k: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VAT order via certificate-and-filter MST.

    Parameters
    ----------
    data : ndarray
        ``(n, n)`` dissimilarity matrix, or ``(n, d)`` points when
        ``from_points=True`` (L2 distances computed on the fly; no ``n x n``
        buffer is allocated).
    from_points : bool
        Interpret ``data`` as points.
    k : int
        Lightest edges per vertex sampled for the certificate forest.

    Returns
    -------
    p : ndarray of int32
        VAT order (identical to ``vat_prim_mst`` for distinct distances).
    q : ndarray of int32
        1-based position in ``p`` of each vertex's MST parent; ``q[0]`` is the
        seed's partner, as in ``vat_prim_mst``.
    w : ndarray
        MST edge weight into each vertex (``w[0] = 0``).
    """
    if _has_compiled_dlmst and data.dtype in (np.float32, np.float64):
        return _order_c(data, from_points, k)
    return vat_order_py(data, from_points, k)


def compute_vat_dlmst(R: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Same result as ``compute_vat``: ``(vat_matrix, p)``."""
    if _has_compiled_dlmst and R.dtype in (np.float32, np.float64):
        V, p, _ = _vat_c(R, k)
        return V, p
    return compute_vat_dlmst_py(R, k)


def compute_ivat_dlmst(
    data: np.ndarray, from_points: bool = False, k: int = 4
) -> tuple[np.ndarray, list, np.ndarray]:
    """Same result as ``compute_ivat``: ``(ivat, argmin_seq, p)``.

    The IVAT matrix is built from the MST alone (row-parallel when compiled),
    so no VAT matrix is needed; with ``from_points=True`` the output is the
    only ``n x n`` allocation.
    """
    if _has_compiled_dlmst and data.dtype in (np.float32, np.float64):
        D, a, p = _ivat_c(data, from_points, k)
        return D, a.tolist(), p
    return compute_ivat_dlmst_py(data, from_points, k)
