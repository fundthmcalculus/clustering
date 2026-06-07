import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def vat_prim_mst_numba(
    adj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Highly optimized numba-compiled Prim's MST using inline heap operations.
    Beats heapq through JIT compilation and aggressive inlining.

    Args:
        adj: Adjacency/distance matrix (N, N)

    Returns:
        Tuple of (heap_seq, parent_seq)
    """
    n: int = len(adj)

    # Find the maximum value
    max_adj: float = -np.inf
    src_i: int = 0
    src_j: int = 0
    for i in range(n):
        for j in range(n):
            if adj[i, j] > max_adj:
                max_adj = adj[i, j]
                src_i = i
                src_j = j

    # Preallocate heap with conservative estimate for worst case
    # Each vertex can have up to n edges, but we process them selectively
    heap_size_max = n * (n - 1)  # Absolute worst case
    heap_keys = np.empty(heap_size_max, dtype=adj.dtype)
    heap_u = np.empty(heap_size_max, dtype=np.int32)
    heap_v = np.empty(heap_size_max, dtype=np.int32)

    # Initialize
    key: np.ndarray = np.full(n, np.inf, dtype=adj.dtype)
    in_mst: np.ndarray = np.zeros(n, dtype=np.bool_)

    # Insert source
    heap_keys[0] = max_adj
    heap_u[0] = src_i
    heap_v[0] = src_j
    key[src_i] = max_adj
    heap_len = 1

    # Output arrays
    heap_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    parent_seq: np.ndarray = np.zeros(n, dtype=np.int32)
    heap_seq_idx: int = 0
    parent_seq_idx: int = 0

    vertices: np.ndarray = np.arange(n)

    # Main loop
    while heap_len > 0:
        # Inline heappop: swap last with first, then bubble down
        heap_len -= 1
        w = heap_keys[0]
        u = int(heap_u[0])
        v0 = int(heap_v[0])

        if heap_len > 0:
            # Move last element to root
            heap_keys[0] = heap_keys[heap_len]
            heap_u[0] = heap_u[heap_len]
            heap_v[0] = heap_v[heap_len]

            # Bubble down inline
            idx = 0
            key_at_idx = heap_keys[0]
            u_at_idx = heap_u[0]
            v_at_idx = heap_v[0]

            while True:
                smallest = idx
                left = 2 * idx + 1
                right = 2 * idx + 2

                if left < heap_len:
                    if heap_keys[left] < key_at_idx or (
                        heap_keys[left] == key_at_idx and heap_u[left] < u_at_idx
                    ):
                        smallest = left
                if right < heap_len:
                    if heap_keys[right] < heap_keys[smallest] or (
                        heap_keys[right] == heap_keys[smallest]
                        and heap_u[right] < heap_u[smallest]
                    ):
                        smallest = right

                if smallest != idx:
                    heap_keys[idx] = heap_keys[smallest]
                    heap_u[idx] = heap_u[smallest]
                    heap_v[idx] = heap_v[smallest]
                    idx = smallest
                else:
                    break

            heap_keys[idx] = key_at_idx
            heap_u[idx] = u_at_idx
            heap_v[idx] = v_at_idx

        # Skip if already visited
        if in_mst[u]:
            continue

        in_mst[u] = True
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1
        parent_seq[parent_seq_idx] = v0
        parent_seq_idx += 1

        # Update keys using vectorized mask (very efficient in numpy)
        mask = (vertices != u) & ~in_mst & (key[vertices] >= adj[u, vertices])
        key[mask] = adj[u, mask]

        # Push updated vertices to heap
        for v in vertices[mask]:
            # Inline heappush with bubble up
            idx = heap_len
            heap_keys[idx] = key[v]
            heap_u[idx] = v
            heap_v[idx] = heap_seq_idx
            heap_len += 1

            # Bubble up inline with tiebreakers
            saved_key = key[v]
            saved_u = v
            saved_v = heap_seq_idx

            while idx > 0:
                parent = (idx - 1) // 2
                parent_key = heap_keys[parent]
                parent_u = heap_u[parent]
                if saved_key < parent_key or (
                    saved_key == parent_key and saved_u < parent_u
                ):
                    heap_keys[idx] = parent_key
                    heap_u[idx] = parent_u
                    heap_v[idx] = heap_v[parent]
                    idx = parent
                else:
                    break

            heap_keys[idx] = saved_key
            heap_u[idx] = saved_u
            heap_v[idx] = saved_v

    return heap_seq, parent_seq
