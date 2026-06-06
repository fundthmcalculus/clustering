# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport INFINITY
from cpython.mem cimport PyMem_Malloc, PyMem_Free


@cython.boundscheck(False)
@cython.wraparound(False)
def vat_prim_mst_c(double[:, ::1] adj):
    """
    Optimized C Prim's MST using a decrease-key binary heap.
    Heap is bounded to O(n) entries (no stale duplicates), keeping it
    in L1/L2 cache vs the previous O(n^2) pre-allocation.
    """
    cdef int n = adj.shape[0]
    cdef int i, j, u, v, v_at_idx
    cdef int src_i = 0, src_j = 0
    cdef int heap_size = 0
    cdef int heap_seq_idx = 0
    cdef int idx, pidx, left, right, smallest
    cdef double max_val, adj_uv, saved_key, tmp_key
    cdef int tmp_vert

    # Output arrays
    cdef int[:] heap_seq = np.empty(n, dtype=np.int32)
    cdef int[:] parent_seq = np.empty(n, dtype=np.int32)

    # O(n) allocations — heap fits in L1/L2 cache
    cdef double* key      = <double*>PyMem_Malloc(n * sizeof(double))
    cdef double* h_keys   = <double*>PyMem_Malloc(n * sizeof(double))
    cdef int*    h_verts  = <int*>PyMem_Malloc(n * sizeof(int))
    cdef int*    h_pos    = <int*>PyMem_Malloc(n * sizeof(int))   # h_pos[v] = heap index of v, -1 if absent
    cdef int*    h_par    = <int*>PyMem_Malloc(n * sizeof(int))   # parent_seq value recorded at push/decrease-key
    cdef char*   in_mst   = <char*>PyMem_Malloc(n * sizeof(char))

    if not (key and h_keys and h_verts and h_pos and h_par and in_mst):
        if key:    PyMem_Free(key)
        if h_keys: PyMem_Free(h_keys)
        if h_verts: PyMem_Free(h_verts)
        if h_pos:  PyMem_Free(h_pos)
        if h_par:  PyMem_Free(h_par)
        if in_mst: PyMem_Free(in_mst)
        raise MemoryError()

    try:
        # Init
        for i in range(n):
            key[i]   = INFINITY
            h_pos[i] = -1
            in_mst[i] = 0

        # Find global max to seed source
        max_val = -INFINITY
        for i in range(n):
            for j in range(n):
                if adj[i, j] > max_val:
                    max_val = adj[i, j]
                    src_i = i
                    src_j = j

        # Push source vertex
        key[src_i] = max_val
        h_keys[0]  = max_val
        h_verts[0] = src_i
        h_pos[src_i] = 0
        h_par[src_i] = src_j
        heap_size = 1

        with nogil:
            while heap_size > 0:
                # Pop minimum-key vertex
                u = h_verts[0]
                heap_seq[heap_seq_idx]   = u
                parent_seq[heap_seq_idx] = h_par[u]
                heap_seq_idx += 1

                in_mst[u] = 1
                h_pos[u]  = -1
                heap_size -= 1

                if heap_size > 0:
                    # Move last element to root and bubble down
                    v_at_idx = h_verts[heap_size]
                    h_keys[0]  = h_keys[heap_size]
                    h_verts[0] = v_at_idx
                    h_pos[v_at_idx] = 0

                    idx = 0
                    while True:
                        smallest = idx
                        left  = 2 * idx + 1
                        right = left + 1
                        if left  < heap_size and h_keys[left]  < h_keys[smallest]:
                            smallest = left
                        if right < heap_size and h_keys[right] < h_keys[smallest]:
                            smallest = right
                        if smallest == idx:
                            break
                        # Swap idx <-> smallest, maintaining h_pos
                        tmp_key              = h_keys[idx]
                        tmp_vert             = h_verts[idx]
                        h_keys[idx]          = h_keys[smallest]
                        h_verts[idx]         = h_verts[smallest]
                        h_pos[h_verts[idx]]  = idx
                        h_keys[smallest]     = tmp_key
                        h_verts[smallest]    = tmp_vert
                        h_pos[tmp_vert]      = smallest
                        idx = smallest

                # Scan neighbors and decrease-key / push
                for v in range(n):
                    if in_mst[v]:
                        continue
                    adj_uv = adj[u, v]
                    if adj_uv >= key[v]:
                        continue
                    # New shorter path to v found
                    key[v]   = adj_uv
                    h_par[v] = heap_seq_idx

                    if h_pos[v] == -1:
                        # Insert new vertex at end of heap
                        idx          = heap_size
                        h_keys[idx]  = adj_uv
                        h_verts[idx] = v
                        h_pos[v]     = idx
                        heap_size   += 1
                    else:
                        # Decrease-key: vertex already in heap
                        idx         = h_pos[v]
                        h_keys[idx] = adj_uv

                    # Bubble up from idx
                    saved_key = adj_uv
                    while idx > 0:
                        pidx = (idx - 1) >> 1
                        if h_keys[pidx] <= saved_key:
                            break
                        h_keys[idx]          = h_keys[pidx]
                        h_verts[idx]         = h_verts[pidx]
                        h_pos[h_verts[idx]]  = idx
                        idx = pidx
                    h_keys[idx]  = saved_key
                    h_verts[idx] = v
                    h_pos[v]     = idx

        return (np.asarray(heap_seq), np.asarray(parent_seq))

    finally:
        PyMem_Free(key)
        PyMem_Free(h_keys)
        PyMem_Free(h_verts)
        PyMem_Free(h_pos)
        PyMem_Free(h_par)
        PyMem_Free(in_mst)
