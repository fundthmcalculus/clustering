import heapq

from numba import njit, prange
import numpy as np
from numba_progress import ProgressBar
from numpy import ndarray


def compute_ivat(
    matrix_of_pairwise_distance: np.ndarray, inplace: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_star, p_seq, as_seq = compute_ordered_dis_njit_merge(
        matrix_of_pairwise_distance, inplace=inplace
    )
    N = d_star.shape[0]
    if not inplace:
        d_p_star = np.zeros(d_star.shape, dtype=d_star.dtype)
    else:
        d_p_star = d_star
    argmin_seq = []
    for r in range(1, N):
        jj = np.argmin(d_star[r, :r])
        # TODO - Get from the prim-mst sequence?
        # jj = as_seq[r-1]
        argmin_seq.append(jj)

        # TODO - Handle doing just upper-triangular matrix for memory savings?
        d_p_star[r, jj] = d_star[r, jj]
        d_p_star[jj, r] = d_star[r, jj]
        for c in range(r):
            if c != jj:
                d_p_star[c, r] = d_p_star[r, c] = max(d_star[r, jj], d_p_star[jj, c])

    return d_p_star, d_star, argmin_seq, p_seq


@njit(cache=True, parallel=True, nogil=True)
def compute_ordered_dis_njit_merge(
    matrix_of_pairwise_distance: np.ndarray,
    inplace: bool = False,
    progress_bar: ProgressBar | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    N = matrix_of_pairwise_distance.shape[0]
    if inplace:
        ordered_matrix = matrix_of_pairwise_distance
    else:
        ordered_matrix: np.ndarray = np.zeros(
            matrix_of_pairwise_distance.shape, dtype=matrix_of_pairwise_distance.dtype
        )
    p, q = vat_prim_mst(matrix_of_pairwise_distance, progress_bar=progress_bar)
    # Step 3 - since this is symmetric, we only have to do half
    n_bit_mask = int(np.ceil(N / 8))
    # Boolean is stored as a byte, so this is smaller
    visited = np.zeros((N, n_bit_mask), dtype=np.uint8)

    if progress_bar is not None:
        progress_bar.set(0)

    if inplace:
        # Due to loop-walking, we cannot use the parallel operations since we cannot know a-priori which loops are different.
        for ij in range(N):
            shuffle_ordered_column(N, ij, ordered_matrix, p, visited)
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        for ij in prange(N):
            for jk in range(ij, N):
                ordered_matrix[ij, jk] = ordered_matrix[jk, ij] = (
                    matrix_of_pairwise_distance[p[ij], p[jk]]
                )
                if progress_bar is not None:
                    progress_bar.update(1)

    # Step 4 - since this is symmetric, we only have to do half
    return ordered_matrix, p, q


@njit(cache=True)
def shuffle_ordered_column(
    n: int, ij: int, ordered_matrix: ndarray, p: ndarray, visited: ndarray
):
    for jk in range(ij, n):
        if _get_bit(visited, ij, jk):
            continue
        # Walk this loop, and store which visited
        r0, c0 = ij, jk
        r1, c1 = -1, -1
        p0 = ordered_matrix[r0, c0]
        while r1 != ij or c1 != jk:
            r1, c1 = p[r0], p[c0]
            _set_bit(visited, r0, c0)
            _set_bit(visited, c0, r0)
            ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = ordered_matrix[r1, c1]
            # Next step!
            r0, c0 = r1, c1
        # Close the final block
        ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = p0
        _set_bit(visited, r0, c0)
        _set_bit(visited, c0, r0)


@njit(cache=True)
def _set_bit(bitmask: np.ndarray, row: int, col: int) -> None:
    bitmask[row, col // 8] |= 1 << (col % 8)


@njit(cache=True)
def _get_bit(bitmask: np.ndarray, row: int, col: int) -> int:
    return (bitmask[row, col // 8] >> (col % 8)) & 1


@njit(cache=True)
def vat_prim_mst(
    adj: np.ndarray, progress_bar: ProgressBar | None = None
) -> np.ndarray:
    n: int = len(adj)

    # Find the column of the maximum value.
    max_adj = np.argmax(adj)
    src_i: int = max_adj // n
    src_j: int = max_adj % n
    src_key = adj[src_i, src_j]

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(n, np.inf, dtype=adj.dtype)

    # To store the parent array which, in turn, stores MST
    parent: np.ndarray = np.full(n, -1, dtype=np.int32)

    # To keep track of vertices included in MST
    in_mst: np.ndarray = np.full(n, False, dtype=np.bool_)

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, int, int]] = [
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

        if progress_bar is not None:
            progress_bar.update(1)

        # Iterate through all adjacent vertices of a vertex
        # Parallel processing of adjacent vertices
        mask = (vertices != u) & ~in_mst & (key[vertices] >= adj[u, vertices])
        key[mask] = adj[u, mask]
        for v in vertices[mask]:
            heapq.heappush(pq, (key[v], v, heap_seq_idx))
            parent[v] = u

    return heap_seq, parent_seq


@njit(cache=True)
def vat_prim_mst_seq(samples: np.ndarray) -> np.ndarray:
    n = len(samples)

    # Find the column of the maximum value.
    max_adj = -np.inf
    max_idx = (-1, -1)
    for ij in range(n):
        for jk in range(ij, n):
            cur_dist = _get_dist(samples, ij, jk)
            if cur_dist > max_adj:
                max_adj = cur_dist
                max_idx = (ij, jk)

    src = max_idx[0]
    src_key = max_adj

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(n, float("inf"))

    # To store the parent array which, in turn, stores MST
    parent: np.ndarray = np.full(n, -1)

    # To keep track of vertices included in MST
    in_mst = np.full(n, False)

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, int]] = [
        (src_key, src)
    ]  # Priority queue to store vertices that are being processed
    key[src] = src_key

    # The final sequence of vertices in MST
    heap_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    heap_seq_idx = 0

    # Preallocated
    vertices = np.arange(n)

    # Loop until the priority queue becomes empty
    while pq:
        # The first vertex in the pair is the minimum key vertex
        # Extract it from the priority queue
        # The vertex label is stored in the second of the pair
        u = heapq.heappop(pq)[1]

        # Different key values for the same vertex may exist in the priority queue.
        # The one with the least key value is always processed first.
        # Therefore, ignore the rest.
        if in_mst[u]:
            continue

        in_mst[u] = True  # Include the vertex in MST
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        # Iterate through all adjacent vertices of a vertex
        # Parallel processing of adjacent vertices

        mask = (
            (vertices != u)
            & ~in_mst
            & (key[vertices] > _get_dist(samples, u, vertices))
        )
        key[mask] = _get_dist(samples, u, vertices[mask])
        for v in vertices[mask]:
            heapq.heappush(pq, (key[v], v))
            parent[v] = u

    return heap_seq


@njit(cache=True)
def _get_dist(samples: np.ndarray, idx1: int, idx2: int) -> float:
    diff = samples[idx1, :] - samples[idx2, :]
    return np.sqrt(np.sum(np.square(diff)))
