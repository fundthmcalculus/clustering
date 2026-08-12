import heapq
from numba import njit, prange
import numpy as np
from numpy import ndarray

try:
    from .pcvat import vat_prim_mst_seq_c as _vat_prim_mst_seq_c

    _has_compiled_vat_prim_mst_seq = True
except ImportError:
    _has_compiled_vat_prim_mst_seq = False


@njit(cache=True, nogil=True)
def _ivat_pathmax_kernel(d_star: np.ndarray, d_p_star: np.ndarray) -> np.ndarray:
    """
    Compute IVAT path-max values in place.

    For each row r from 1 to n-1, find the minimum value in columns [0, r),
    then fill the row with minimax distances: max(min_val, d_p_star[best_jj, c]).
    Returns the argmin sequence.
    """
    n = d_star.shape[0]
    argmin_seq = np.zeros(n - 1, dtype=np.int32)

    for r in range(1, n):
        # Find minimum distance in columns [0, r)
        min_val = d_star[r, 0]
        best_jj = 0
        for c in range(1, r):
            if d_star[r, c] < min_val:
                min_val = d_star[r, c]
                best_jj = c

        argmin_seq[r - 1] = best_jj

        # Fill row r with minimax values
        d_p_star[r, best_jj] = min_val
        d_p_star[best_jj, r] = min_val
        for c in range(r):
            if c != best_jj:
                max_val = max(min_val, d_p_star[best_jj, c])
                d_p_star[c, r] = max_val
                d_p_star[r, c] = max_val

    return argmin_seq


def compute_ivat(
    matrix_of_pairwise_distance: np.ndarray, inplace: bool = False
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Computes the improved VAT (IVAT) for the provided dissimilarity (distance) matrix
    :param matrix_of_pairwise_distance: dissimilarity matrix, typically an
        L2-norm matrix, it must be symmetric and positive semi-definite
    :param inplace: whether to perform the computation in-place on the input matrix
    :return: tuple of the IVAT matrix, the sequence of IVAT (argmin) indices,
        and the permutation (VAT) sequence
    """
    d_star, p_seq, as_seq = compute_ordered_dis_njit_merge(
        matrix_of_pairwise_distance, inplace=inplace
    )
    if not inplace:
        d_p_star = np.zeros(d_star.shape, dtype=d_star.dtype)
    else:
        d_p_star = d_star

    argmin_seq = _ivat_pathmax_kernel(d_star, d_p_star)

    return d_p_star, argmin_seq.tolist(), p_seq


def compute_vat(
    matrix_of_pairwise_distance: np.ndarray, inplace: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the visualization assessment of cluster tendency (VAT) for the provided dissimilarity (distance) matrix
    :param matrix_of_pairwise_distance: dissimilarity matrix, typically an
        L2-norm matrix, it must be symmetric and positive semi-definite
    :param inplace: whether to perform the computation in-place on the input matrix
    :return: tuple of the permuted distance (VAT) matrix and the permutation (VAT) sequence
    """
    d_star, p_seq, as_seq = compute_ordered_dis_njit_merge(
        matrix_of_pairwise_distance, inplace=inplace
    )
    return d_star, p_seq


@njit(cache=True, parallel=True, nogil=True)
def compute_ordered_dis_njit_merge(
    matrix_of_pairwise_distance: np.ndarray,
    inplace: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = matrix_of_pairwise_distance.shape[0]
    if inplace:
        ordered_matrix = matrix_of_pairwise_distance
    else:
        ordered_matrix = np.zeros(
            matrix_of_pairwise_distance.shape, dtype=matrix_of_pairwise_distance.dtype
        )
    p, q = vat_prim_mst(matrix_of_pairwise_distance)

    if inplace:
        # Reorder in place as P·M·Pᵀ (see _permute_sym_inplace). Serial by
        # nature, so the round-count is reported in one progress step.
        _permute_sym_inplace(ordered_matrix, p)
    else:
        for ij in prange(n):
            for jk in range(ij, n):
                ordered_matrix[ij, jk] = ordered_matrix[jk, ij] = (
                    matrix_of_pairwise_distance[p[ij], p[jk]]
                )

    return ordered_matrix, p, q


@njit(cache=True, nogil=True)
def _permute_sym_inplace(M: ndarray, p: ndarray) -> None:
    """Reorder a symmetric matrix in place so that M[i, j] <- M[p[i], p[j]].

    Applied as P·M·Pᵀ in two independent 1-D cycle-following passes — permute
    rows, then permute columns within each row. Each pass reads only the
    not-yet-visited "next" element of a cycle before overwriting the current
    one, so it is exact; workspace is O(n).

    This replaces the earlier cycle-following-on-cell-pairs routine, which was
    incorrect: it wrote both M[r, c] and its mirror M[c, r] in one step, and a
    mirror-written cell could then be read as another cycle's "next" after it
    had already received its final value, corrupting O(n) cells.
    """
    n = M.shape[0]
    seen = np.zeros(n, dtype=np.bool_)
    tmp = np.empty(n, dtype=M.dtype)

    # Phase 1: permute rows so new row i = old row p[i].
    for start in range(n):
        if seen[start] or p[start] == start:
            seen[start] = True
            continue
        tmp[:] = M[start, :]
        i = start
        while True:
            seen[i] = True
            nxt = p[i]
            if nxt == start:
                M[i, :] = tmp
                break
            M[i, :] = M[nxt, :]
            i = nxt

    # Phase 2: permute columns within each row: new[i, j] = cur[i, p[j]].
    for i in range(n):
        tmp[:] = M[i, :]
        for j in range(n):
            M[i, j] = tmp[p[j]]


@njit(cache=True, nogil=True)
def vat_prim_mst(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n: int = len(adj)

    # Find the column of the maximum value.
    max_adj: np.signedinteger = np.argmax(adj)
    src_i: np.signedinteger = max_adj // n
    src_j: np.signedinteger = max_adj % n
    src_key = adj[src_i, src_j]

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(n, np.inf, dtype=adj.dtype)

    # To store the parent array which, in turn, stores MST
    parent: np.ndarray = np.full(n, -1, dtype=np.int32)

    # To keep track of vertices included in MST
    in_mst: np.ndarray = np.full(n, False, dtype=np.bool_)

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, np.signedinteger, np.signedinteger]] = [
        (src_key, src_i, src_j)
    ]  # Priority queue to store vertices that are being processed
    key[src_i] = src_key

    # The final sequence of vertices in MST
    heap_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    heap_seq_idx: int = 0

    # Parent sequences of vertices in MST (for iVAT)
    parent_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    parent_seq_idx: int = 0

    # Preallocated
    vertices: np.ndarray = np.arange(n)

    # Loop until the priority queue becomes empty
    while pq:
        # The first vertex in the pair is the minimum key vertex
        # Extract it from the priority queue
        # The vertex label is stored in the second of the pair
        w, u, v0 = heapq.heappop(pq)

        # Different key values for the same vertex may exist in the priority queue.
        # The one with the least key value is always processed first.
        # Therefore, ignore the rest.
        if in_mst[u]:
            continue

        in_mst[u] = True  # Include the vertex in MST
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        parent_seq[parent_seq_idx] = v0
        parent_seq_idx += 1

        # Iterate through all adjacent vertices of a vertex
        # Parallel processing of adjacent vertices
        mask = (vertices != u) & ~in_mst & (key[vertices] >= adj[u, vertices])
        key[mask] = adj[u, mask]
        for v in vertices[mask]:
            # Heterogeneous heap-key tuple (numpy int vs Python int); mypy cannot
            # infer the element type but numba handles it at runtime.
            heapq.heappush(pq, (key[v], v, heap_seq_idx))  # type: ignore[misc]
            parent[v] = u

    return heap_seq, parent_seq


@njit(cache=True, nogil=True)
def _vat_prim_mst_seq_python(samples: np.ndarray) -> np.ndarray:
    n = len(samples)

    # Find the pair of points with maximum distance.
    max_dist = -np.inf
    src_vertex = 0
    for ij in range(n):
        for jk in range(ij + 1, n):
            cur_dist = _get_dist(samples, ij, jk)
            if cur_dist > max_dist:
                max_dist = cur_dist
                src_vertex = ij

    src_key = max_dist

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(n, np.inf)

    # To keep track of vertices included in MST
    in_mst = np.full(n, False)

    # Insert the source itself into the priority queue and initialize its key
    pq: list[tuple[float, int]] = [(src_key, src_vertex)]
    key[src_vertex] = src_key

    # The final sequence of vertices in MST
    heap_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    heap_seq_idx = 0

    # Loop until the priority queue becomes empty
    while pq:
        w, u = heapq.heappop(pq)

        # Different key values for the same vertex may exist in the priority queue.
        # The one with the least key value is always processed first.
        # Therefore, ignore the rest.
        if in_mst[u]:
            continue

        in_mst[u] = True  # Include the vertex in MST
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        # Iterate through all adjacent vertices of a vertex
        # Compute distances to all non-MST vertices and update keys
        for v in range(n):
            if v != u and not in_mst[v]:
                dist_uv = _get_dist(samples, u, v)
                if dist_uv < key[v]:
                    key[v] = dist_uv
                    heapq.heappush(pq, (key[v], v))

    return heap_seq[:heap_seq_idx]


def vat_prim_mst_seq(samples: np.ndarray) -> np.ndarray:
    """
    Compute VAT ordering directly from samples using Prim's algorithm.

    Dispatches to the compiled Cython implementation if available,
    otherwise falls back to the pure Python/numba implementation.

    Args:
        samples: (n, d) array of samples (float32 or float64)

    Returns:
        VAT ordering sequence (n,) int32 array
    """
    if _has_compiled_vat_prim_mst_seq:
        return _vat_prim_mst_seq_c(samples)
    else:
        return _vat_prim_mst_seq_python(samples)


@njit(cache=True)
def _get_dist(samples: np.ndarray, idx1: int, idx2: int) -> float:
    diff = samples[idx1, :] - samples[idx2, :]
    return np.sqrt(np.sum(np.square(diff)))
