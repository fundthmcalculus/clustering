import heapq
from dataclasses import dataclass, field
from typing import Union

from numba import njit, prange
import numpy as np
from numba_progress import ProgressBar
from numpy import ndarray


def compute_ivat(
    matrix_of_pairwise_distance: np.ndarray, inplace: bool = False
) -> tuple[np.ndarray, list[int], list[int]]:
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
    n = d_star.shape[0]
    if not inplace:
        d_p_star = np.zeros(d_star.shape, dtype=d_star.dtype)
    else:
        d_p_star = d_star
    argmin_seq = []
    for r in range(1, n):
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

    return d_p_star, argmin_seq, p_seq


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
    progress_bar: ProgressBar | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    n = matrix_of_pairwise_distance.shape[0]
    if inplace:
        ordered_matrix = matrix_of_pairwise_distance
    else:
        ordered_matrix = np.zeros(
            matrix_of_pairwise_distance.shape, dtype=matrix_of_pairwise_distance.dtype
        )
    p, q = vat_prim_mst(matrix_of_pairwise_distance, progress_bar=progress_bar)
    # Step 3 - since this is symmetric, we only have to do half
    n_bit_mask = int(np.ceil(n / 8))
    # Boolean is stored as a byte, so this is smaller
    visited = np.zeros((n, n_bit_mask), dtype=np.uint8)

    if progress_bar is not None:
        progress_bar.set(0)

    if inplace:
        # Due to loop-walking, we cannot use the parallel operations since we
        # cannot know a-priori which loops are different.
        for ij in range(n):
            shuffle_ordered_column(n, ij, ordered_matrix, p, visited)
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        for ij in prange(n):
            for jk in range(ij, n):
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


@njit(cache=True, nogil=True)
def _set_bit(bitmask: np.ndarray, row: int, col: int) -> None:
    bitmask[row, col // 8] |= 1 << (col % 8)


@njit(cache=True, nogil=True)
def _get_bit(bitmask: np.ndarray, row: int, col: int) -> int:
    return (bitmask[row, col // 8] >> (col % 8)) & 1


@njit(cache=True, nogil=True)
def vat_prim_mst(
    adj: np.ndarray, progress_bar: ProgressBar | None = None
) -> tuple[np.ndarray, np.ndarray]:
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


@njit(cache=True, nogil=True)
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


@dataclass
class IvatMeansResult:
    abrupt_change_indices: ndarray
    cluster_city_ids: list[ndarray]
    diagonal_values: ndarray
    initial_centroids: ndarray
    max_diff_index: int
    peak_threshold: float
    sorted_diagonal: ndarray


@dataclass
class ClusterNode:
    indices: ndarray
    centroid: ndarray
    children: list["ClusterNode"] = field(default_factory=list)


def _arg_max(a: ndarray, n: int = 1) -> ndarray:
    """Get the indexes of the n-largest values in the array."""
    if n >= len(a):
        return np.argsort(a)[::-1]
    # Use argpartition to find the n largest elements efficiently
    partitioned_indices = np.argpartition(a, -n)[-n:]
    # Sort these indices by their corresponding values in descending order
    sorted_indices = partitioned_indices[np.argsort(a[partitioned_indices])[::-1]]
    return sorted_indices


def get_ivat_levels(
    all_cities: ndarray,
    ivat_mst: ndarray,
    vat_order: ndarray,
    n_levels: int = 1,
    n_clusters: int = -1,
) -> Union[IvatMeansResult, list[IvatMeansResult]]:
    """
    Extract multiple levels of clusterings from the iVAT matrix.

    Args:
        all_cities: Original data points (N, D)
        ivat_mst: iVAT distance matrix
        vat_order: Permutation indices from VAT/iVAT
        n_levels: Number of hierarchical levels to extract (exclusive with n_clusters)
        n_clusters: Exact number of clusters to consider. If -1, use all possible clusters.

    Returns:
        A single IvatMeansResult if n_levels=1, or a list of them if n_levels > 1.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1")
    if n_clusters != -1 and n_levels != 1:
        raise ValueError("n_levels and n_clusters cannot be used together")
    if n_clusters != -1 and n_clusters < 1:
        raise ValueError(
            "n_clusters must be at least 1 or -1 for all possible clusters"
        )

    # Look down the off-by-1 diagonal and count the number of substantial changes.
    diagonal_values = np.diag(ivat_mst, k=1)
    # Augment back to original size, just prepend the initial value to avoid throwing off the diff fcn
    # Expand this to the original size for convenience.
    diagonal_values = np.concatenate(
        [np.array([diagonal_values[0]]), diagonal_values], axis=0
    )
    # Sort the diagonal values
    sorted_diagonal = np.sort(diagonal_values)
    if n_clusters == -1:
        # Find the maximum difference and the index thereof
        diagonal_diffs = np.diff(sorted_diagonal)
        max_diff_indices = _arg_max(diagonal_diffs, n_levels)
        peaks_threshold = sorted_diagonal[max_diff_indices + 1]

        # Sort peaks_threshold in decreasing order and reorder max_diff_indices accordingly
        sort_order = np.argsort(peaks_threshold)[::-1]
        peaks_threshold = peaks_threshold[sort_order]
        max_diff_indices = max_diff_indices[sort_order]
    elif n_clusters == 1:
        # Pick higher than the highest value
        peaks_threshold = [sorted_diagonal[-1] * 1.1]
        max_diff_indices = [-1]
    else:
        # Since #clusters = #peaks+1, adjust indexing.
        peaks_threshold = sorted_diagonal[-(n_clusters - 1) :]
        max_diff_indices = [-1] * (n_clusters - 1)

    results = []
    for index, peak_th in enumerate(peaks_threshold):
        # Prevent weird floating-point comparisons.
        abrupt_change_idx = np.where(diagonal_values >= peak_th)[0]

        # Use each section as a cluster endpoint, inclusive.
        cluster_group = np.concatenate(
            [np.array([0]), abrupt_change_idx, np.array([len(all_cities)])]
        )
        cluster_city_indexs = []
        for idx, cg_start in enumerate(cluster_group[:-1]):
            cg_end = cluster_group[idx + 1]
            if cg_start < cg_end:
                # Use the VAT order to pick out the cities in each cluster
                cluster_city_indexs.append(vat_order[cg_start:cg_end])

        # Compute the initial guess as the centroid of each city cluster
        initial_centroids_item = np.array(
            [
                np.mean(all_cities[cluster_ids], axis=0)
                for cluster_ids in cluster_city_indexs
            ]
        )

        results.append(
            IvatMeansResult(
                abrupt_change_indices=abrupt_change_idx,
                cluster_city_ids=cluster_city_indexs,
                diagonal_values=diagonal_values,
                initial_centroids=initial_centroids_item,
                max_diff_index=int(max_diff_indices[index]),
                peak_threshold=float(peak_th),
                sorted_diagonal=sorted_diagonal,
            )
        )

    if n_levels == 1:
        return results[0]
    return results


def get_ivat_hierarchy(
    all_cities: ndarray, ivat_mst: ndarray, vat_order: ndarray, n_levels: int = 1
) -> ClusterNode:
    """
    Build a hierarchical tree structure from iVAT results.

    Args:
        all_cities: Original data points (N, D)
        ivat_mst: iVAT distance matrix
        vat_order: Permutation indices from VAT/iVAT
        n_levels: Number of levels to include in the hierarchy

    Returns:
        Root ClusterNode of the hierarchy
    """
    raw_results = get_ivat_levels(all_cities, ivat_mst, vat_order, n_levels=n_levels)
    # get_ivat_levels returns a single result for n_levels=1, else a list.
    levels_results: list[IvatMeansResult] = (
        raw_results if isinstance(raw_results, list) else [raw_results]
    )

    # Root node contains everything
    root = ClusterNode(
        indices=np.arange(len(all_cities)), centroid=np.mean(all_cities, axis=0)
    )

    # current_level_nodes starts with root
    current_level_nodes = [root]

    for level_res in levels_results:
        next_level_nodes = []
        for cluster_indices in level_res.cluster_city_ids:
            new_node = ClusterNode(
                indices=cluster_indices,
                centroid=np.mean(all_cities[cluster_indices], axis=0),
            )
            # Find parent in current_level_nodes
            # Since it's a strict hierarchy, any point in the cluster will be in its parent node.
            # We use the first index for efficiency.
            target_idx = cluster_indices[0]
            found_parent = False
            for parent in current_level_nodes:
                # We can use a faster check since we know parent.indices contains target_idx
                # if it's the right parent.
                if target_idx in parent.indices:
                    parent.children.append(new_node)
                    found_parent = True
                    break

            if not found_parent:
                # Fallback, should not happen if results are hierarchical
                root.children.append(new_node)

            next_level_nodes.append(new_node)
        current_level_nodes = next_level_nodes

    return root
