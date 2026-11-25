# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython/OpenMP certificate-and-filter MST for VAT/IVAT.

Compiled counterpart of ``dlmst.py`` (see its docstring for the algorithm,
what is and is not taken from Khanna, Putterman & Song, arXiv:2611.31187, and
why no asymptotic gain is possible on VAT's complete graph). Every O(n^2)
pass here is independent per row and runs under ``prange``; only O(n) or
O(n log n) bookkeeping is serial. That is the structural difference from
dense Prim, whose round loop is inherently serial.

Each kernel works on either a dense distance matrix (``R``) or on points
(``X``, L2 distances computed on the fly, O(n) memory). Kernels are written
once over Cython's ``floating`` fused type, which yields the f32 and f64
variants the rest of the package writes out by hand.

Pipeline: ``_knn_seed`` (k lightest per row + VAT seed) -> Kruskal ->
``_merge_pass`` until the certificate spans -> ``_krt_order`` ->
``_filter`` (one comparison per pair) -> Kruskal on F ∪ U(F) -> ``_tree_prim``
(VAT order from the MST) -> optional ``_ivat_rows`` (parallel IVAT).
"""

import numpy as np
cimport cython
from cython cimport floating
from cython.parallel cimport prange, parallel, threadid
cimport openmp
from libc.math cimport sqrt, INFINITY
from libc.stdlib cimport malloc, realloc, free

# Below this many vertices, OpenMP fork/join overhead dominates.
cdef int _PAR_THRESHOLD = 512
# Row chunk for the k-lightest prefilter.
cdef enum:
    _CHUNK = 64


cdef int _nthreads(int n, int requested):
    cdef int nt = requested if requested > 0 else openmp.omp_get_max_threads()
    if n < _PAR_THRESHOLD or nt < 1:
        return 1
    return nt


# ---------------------------------------------------------------------------
# Distance access: a row of R, or a row of on-the-fly L2 distances from X.
# ---------------------------------------------------------------------------
cdef inline floating _dist(
    const floating* R, const floating* X, Py_ssize_t n, int d, Py_ssize_t i, Py_ssize_t j
) noexcept nogil:
    cdef floating s, diff
    cdef int k
    if R != NULL:
        return R[i * n + j]
    s = 0
    for k in range(d):
        diff = X[i * d + k] - X[j * d + k]
        s = s + diff * diff
    return sqrt(s)


cdef inline const floating* _row(
    const floating* R, const floating* X, Py_ssize_t n, int d, Py_ssize_t i, floating* buf
) noexcept nogil:
    cdef Py_ssize_t j
    cdef int k
    cdef floating s, diff
    cdef const floating* xi
    cdef const floating* xj
    if R != NULL:
        return R + i * n
    xi = X + i * d
    for j in range(n):
        xj = X + j * d
        s = 0
        for k in range(d):
            diff = xi[k] - xj[k]
            s = s + diff * diff
        buf[j] = sqrt(s)
    return buf


# ---------------------------------------------------------------------------
# Pass 1: k lightest incident edges per vertex, fused with the VAT seed scan.
# ---------------------------------------------------------------------------
cdef void _knn_seed(
    const floating* R, const floating* X, int n, int d, int k,
    int* ev, floating* ew, floating* rmax, int* rarg, int nt
) noexcept nogil:
    cdef int i, j, t, j0, j1, jc
    cdef int nch = (n + _CHUNK - 1) // _CHUNK
    cdef floating v, mx, cmin, cmax
    cdef int am
    cdef floating* buf = NULL
    cdef const floating* row
    cdef floating* kw
    cdef int* kv
    with parallel(num_threads=nt):
        buf = <floating*>malloc(n * sizeof(floating)) if R == NULL else NULL
        for i in prange(n, schedule='static'):
            row = _row(R, X, n, d, i, buf)
            kw = ew + <Py_ssize_t>i * k
            kv = ev + <Py_ssize_t>i * k
            for t in range(k):
                kw[t] = INFINITY
                kv[t] = -1
            mx = -INFINITY
            am = 0
            # Chunked: a branch-free min/max over CHUNK entries (vectorizes)
            # gates the scalar insertion and argmax scans, which most chunks
            # skip once the k-list has filled.
            for jc in range(nch):
                j0 = jc * _CHUNK
                j1 = j0 + _CHUNK
                if j1 > n:
                    j1 = n
                cmin = row[j0]
                cmax = row[j0]
                for j in range(j0 + 1, j1):
                    v = row[j]
                    cmin = v if v < cmin else cmin
                    cmax = v if v > cmax else cmax
                if cmax > mx:
                    for j in range(j0, j1):
                        if row[j] > mx:
                            mx = row[j]
                            am = j
                if cmin >= kw[k - 1] and not (j0 <= i < j1):
                    continue
                for j in range(j0, j1):
                    v = row[j]
                    if j == i or v >= kw[k - 1]:
                        continue
                    t = k - 1
                    while t > 0 and kw[t - 1] > v:
                        kw[t] = kw[t - 1]
                        kv[t] = kv[t - 1]
                        t = t - 1
                    kw[t] = v
                    kv[t] = j
            rmax[i] = mx
            rarg[i] = am
        free(buf)


# ---------------------------------------------------------------------------
# Pass 2: lightest edge from each vertex to a different component.
# ---------------------------------------------------------------------------
cdef void _merge_pass(
    const floating* R, const floating* X, int n, int d, const int* comp,
    int* bv, floating* bw, int nt
) noexcept nogil:
    cdef int i, j, c, bj
    cdef floating best, v
    cdef floating* buf = NULL
    cdef const floating* row
    with parallel(num_threads=nt):
        buf = <floating*>malloc(n * sizeof(floating)) if R == NULL else NULL
        for i in prange(n, schedule='static'):
            c = comp[i]
            best = INFINITY
            bj = -1
            if R != NULL:
                row = R + <Py_ssize_t>i * n
                for j in range(n):
                    v = row[j]
                    if v < best and comp[j] != c:
                        best = v
                        bj = j
            else:
                for j in range(n):
                    if comp[j] != c:
                        v = _dist(R, X, n, d, i, j)
                        if v < best:
                            best = v
                            bj = j
            bv[i] = bj
            bw[i] = best
        free(buf)


# ---------------------------------------------------------------------------
# Pass 2b: contraction. With c components (compact ids), one upper-triangle
# pass finds the lightest edge between every pair of components -- the
# contracted graph -- so a single pass replaces log2(c) Borůvka merges.
# Per-thread c x c tables, reduced serially.
# ---------------------------------------------------------------------------
cdef void _contract_pass(
    const floating* R, const floating* X, int n, int d, const int* comp, int c,
    floating* bw, int* bu, int* bv, int nt
) noexcept nogil:
    cdef Py_ssize_t cc = <Py_ssize_t>c * c
    cdef floating* tw = <floating*>malloc(nt * cc * sizeof(floating))
    cdef int* tu = <int*>malloc(nt * cc * sizeof(int))
    cdef int* tv = <int*>malloc(nt * cc * sizeof(int))
    cdef Py_ssize_t t, idx, base
    cdef int i, j, ci, cj, tid
    cdef floating v
    cdef const floating* row
    cdef floating* mw
    for t in range(nt * cc):
        tw[t] = INFINITY
        tu[t] = -1
        tv[t] = -1
    for tid in range(nt):
        for ci in range(c):
            tw[<Py_ssize_t>tid * cc + <Py_ssize_t>ci * c + ci] = -INFINITY
    with parallel(num_threads=nt):
        tid = threadid()
        base = <Py_ssize_t>tid * cc
        for i in prange(n - 1, schedule='dynamic', chunksize=16):
            ci = comp[i]
            mw = tw + base + <Py_ssize_t>ci * c
            if R != NULL:
                row = R + <Py_ssize_t>i * n
            if R != NULL:
                # mw[ci] is pinned to -inf, so same-component pairs never win
                # and the loop body has a single, almost-never-taken branch.
                for j in range(i + 1, n):
                    cj = comp[j]
                    v = row[j]
                    if v < mw[cj]:
                        mw[cj] = v
                        idx = base + <Py_ssize_t>ci * c + cj
                        tu[idx] = i
                        tv[idx] = j
                continue
            for j in range(i + 1, n):
                cj = comp[j]
                if cj == ci:
                    continue
                v = _dist(R, X, n, d, i, j)
                if v < mw[cj]:
                    mw[cj] = v
                    idx = base + <Py_ssize_t>ci * c + cj
                    tu[idx] = i
                    tv[idx] = j
    for t in range(cc):
        bw[t] = INFINITY
        bu[t] = -1
        bv[t] = -1
    for tid in range(nt):
        for t in range(cc):
            idx = <Py_ssize_t>tid * cc + t
            if tu[idx] >= 0 and tw[idx] < bw[t]:
                bw[t] = tw[idx]
                bu[t] = tu[idx]
                bv[t] = tv[idx]
    free(tw); free(tu); free(tv)


# ---------------------------------------------------------------------------
# Pass 3: the filter. In the leaf order of F's Kruskal reconstruction tree,
# the F-minimax distance between leaves i < j is max(g[i..j-1]), so row i is
# one running max -- one comparison per pair, row-parallel.
# ---------------------------------------------------------------------------
cdef Py_ssize_t _filter(
    const floating* R, const floating* Xp, int n, int d,
    const int* order, const floating* g,
    int** tu, int** tv, floating** tw, Py_ssize_t* tcnt, Py_ssize_t* tcap, int nt
) noexcept nogil:
    cdef int i, j, tid, oi
    cdef floating mm, v
    cdef const floating* row
    cdef Py_ssize_t c
    for tid in range(nt):
        tcnt[tid] = 0
    with parallel(num_threads=nt):
        tid = threadid()
        for i in prange(n - 1, schedule='dynamic', chunksize=16):
            oi = order[i]
            mm = -INFINITY
            if R != NULL:
                row = R + <Py_ssize_t>oi * n
            for j in range(i + 1, n):
                if g[j - 1] > mm:
                    mm = g[j - 1]
                if R != NULL:
                    v = row[order[j]]
                else:
                    v = _dist(<const floating*>NULL, Xp, n, d, i, j)
                if v < mm:
                    c = tcnt[tid]
                    if c == tcap[tid]:
                        tcap[tid] = tcap[tid] * 2 + 64
                        tu[tid] = <int*>realloc(tu[tid], tcap[tid] * sizeof(int))
                        tv[tid] = <int*>realloc(tv[tid], tcap[tid] * sizeof(int))
                        tw[tid] = <floating*>realloc(tw[tid], tcap[tid] * sizeof(floating))
                    tu[tid][c] = oi
                    tv[tid][c] = order[j]
                    tw[tid][c] = v
                    tcnt[tid] = c + 1
    return 0


# ---------------------------------------------------------------------------
# Serial O(n α(n)) helpers
# ---------------------------------------------------------------------------
cdef inline int _find(int* par, int x) noexcept nogil:
    cdef int r = x, nx
    while par[r] != r:
        r = par[r]
    while par[x] != r:
        nx = par[x]
        par[x] = r
        x = nx
    return r


cdef int _kruskal_sorted(
    int n, const int* eu, const int* ev, const long long* order, Py_ssize_t m,
    int* out_idx
) noexcept nogil:
    """Scan edges in the given order; write chosen edge indices to out_idx."""
    cdef int* par = <int*>malloc(n * sizeof(int))
    cdef int i, a, b, cnt = 0
    cdef Py_ssize_t t
    cdef long long e
    for i in range(n):
        par[i] = i
    for t in range(m):
        e = order[t]
        a = _find(par, eu[e])
        b = _find(par, ev[e])
        if a != b:
            par[a] = b
            out_idx[cnt] = <int>e
            cnt += 1
            if cnt == n - 1:
                break
    free(par)
    return cnt


cdef void _labels(int n, const int* eu, const int* ev, int m, int* comp) noexcept nogil:
    cdef int* par = <int*>malloc(n * sizeof(int))
    cdef int i, a, b
    for i in range(n):
        par[i] = i
    for i in range(m):
        a = _find(par, eu[i])
        b = _find(par, ev[i])
        if a != b:
            par[a] = b
    for i in range(n):
        comp[i] = _find(par, i)
    free(par)


cdef void _krt_order(
    int n, const int* eu, const int* ev, const floating* ew, const long long* srt,
    int* order, floating* g
) noexcept nogil:
    """Leaf order of the Kruskal reconstruction tree of a spanning tree, via
    linked-list concatenation; g[t] is the merge weight between order[t] and
    order[t+1]. ``srt`` lists the n-1 tree edges by increasing weight."""
    cdef int* par = <int*>malloc(n * sizeof(int))
    cdef int* head = <int*>malloc(n * sizeof(int))
    cdef int* tail = <int*>malloc(n * sizeof(int))
    cdef int* nxt = <int*>malloc(n * sizeof(int))
    cdef floating* gw = <floating*>malloc(n * sizeof(floating))
    cdef int i, a, b, x, t
    cdef long long e
    for i in range(n):
        par[i] = i
        head[i] = i
        tail[i] = i
        nxt[i] = -1
    for t in range(n - 1):
        e = srt[t]
        a = _find(par, eu[e])
        b = _find(par, ev[e])
        # list(a) ++ list(b), rooted at b
        nxt[tail[a]] = head[b]
        gw[tail[a]] = ew[e]
        head[b] = head[a]
        par[a] = b
    x = head[_find(par, 0)]
    for t in range(n):
        order[t] = x
        if t < n - 1:
            g[t] = gw[x]
        x = nxt[x]
    free(par); free(head); free(tail); free(nxt); free(gw)


cdef void _tree_prim(
    int n, const int* mu, const int* mv, const floating* mw, int src, int partner,
    int* p, int* q, floating* wout
) noexcept nogil:
    """Prim's visiting order on the MST (binary heap, O(n log n))."""
    cdef int* deg = <int*>malloc((n + 1) * sizeof(int))
    cdef int* adj = <int*>malloc(2 * (n - 1) * sizeof(int) + sizeof(int))
    cdef floating* aw = <floating*>malloc(2 * (n - 1) * sizeof(floating) + sizeof(floating))
    cdef int* pos = <int*>malloc(n * sizeof(int))
    cdef floating* hk = <floating*>malloc(n * sizeof(floating))
    cdef int* hv = <int*>malloc(n * sizeof(int))
    cdef int* hp = <int*>malloc(n * sizeof(int))
    cdef int i, a, b, hs, r, c, s, v, par, x, t
    cdef floating w, tk
    cdef int tvv, tpp
    for i in range(n + 1):
        deg[i] = 0
    for i in range(n - 1):
        deg[mu[i] + 1] += 1
        deg[mv[i] + 1] += 1
    for i in range(n):
        deg[i + 1] += deg[i]
        pos[i] = -1
    # fill CSR using pos[] as a temporary cursor offset
    for i in range(n):
        hv[i] = deg[i]
    for i in range(n - 1):
        a = mu[i]
        b = mv[i]
        adj[hv[a]] = b
        aw[hv[a]] = mw[i]
        hv[a] += 1
        adj[hv[b]] = a
        aw[hv[b]] = mw[i]
        hv[b] += 1
    hs = 1
    hk[0] = 0
    hv[0] = src
    hp[0] = -1
    r = 0
    while hs > 0:
        w = hk[0]
        v = hv[0]
        par = hp[0]
        hs -= 1
        hk[0] = hk[hs]
        hv[0] = hv[hs]
        hp[0] = hp[hs]
        # sift down
        i = 0
        while True:
            c = 2 * i + 1
            if c >= hs:
                break
            if c + 1 < hs and (hk[c + 1] < hk[c] or (hk[c + 1] == hk[c] and hv[c + 1] < hv[c])):
                c = c + 1
            if hk[c] < hk[i] or (hk[c] == hk[i] and hv[c] < hv[i]):
                tk = hk[c]; hk[c] = hk[i]; hk[i] = tk
                tvv = hv[c]; hv[c] = hv[i]; hv[i] = tvv
                tpp = hp[c]; hp[c] = hp[i]; hp[i] = tpp
                i = c
            else:
                break
        if pos[v] >= 0:
            continue
        pos[v] = r
        p[r] = v
        q[r] = partner if par < 0 else pos[par] + 1
        wout[r] = w
        r += 1
        for t in range(deg[v], deg[v + 1]):
            x = adj[t]
            if pos[x] < 0:
                # push + sift up
                i = hs
                hs += 1
                hk[i] = aw[t]
                hv[i] = x
                hp[i] = v
                while i > 0:
                    s = (i - 1) // 2
                    if hk[i] < hk[s] or (hk[i] == hk[s] and hv[i] < hv[s]):
                        tk = hk[s]; hk[s] = hk[i]; hk[i] = tk
                        tvv = hv[s]; hv[s] = hv[i]; hv[i] = tvv
                        tpp = hp[s]; hp[s] = hp[i]; hp[i] = tpp
                        i = s
                    else:
                        break
    free(deg); free(adj); free(aw); free(pos); free(hk); free(hv); free(hp)


cdef void _ivat_rows(
    int n, const int* p, const int* order, const floating* g, floating* D, int nt
) noexcept nogil:
    """IVAT rows from the MST's KRT: D[r, c] = minimax(p[r], p[c]), each row a
    running max left and right of the vertex's leaf -- row-parallel."""
    cdef int* lpos = <int*>malloc(n * sizeof(int))
    cdef int* col = <int*>malloc(n * sizeof(int))
    cdef int* vpos = <int*>malloc(n * sizeof(int))
    cdef int r, j, i
    cdef floating mm
    cdef floating* row
    for r in range(n):
        vpos[p[r]] = r
    for j in range(n):
        lpos[order[j]] = j
        col[j] = vpos[order[j]]
    for r in prange(n, schedule='static', num_threads=nt):
        row = D + <Py_ssize_t>r * n
        i = lpos[p[r]]
        row[r] = 0
        mm = -INFINITY
        for j in range(i + 1, n):
            if g[j - 1] > mm:
                mm = g[j - 1]
            row[col[j]] = mm
        mm = -INFINITY
        for j in range(i - 1, -1, -1):
            if g[j] > mm:
                mm = g[j]
            row[col[j]] = mm
    free(lpos); free(col); free(vpos)


cdef void _gather(const floating* R, int n, const int* p, floating* out, int nt) noexcept nogil:
    cdef int i, j
    cdef const floating* src
    cdef floating* dst
    for i in prange(n, schedule='static', num_threads=nt):
        src = R + <Py_ssize_t>p[i] * n
        dst = out + <Py_ssize_t>i * n
        for j in range(n):
            dst[j] = src[p[j]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _sorted_idx(w, u, v):
    # (weight, max endpoint, min endpoint) -- the same tie order as dlmst.py
    return np.ascontiguousarray(
        np.lexsort((np.minimum(u, v), np.maximum(u, v), w)), dtype=np.int64
    )


def _kruskal(int n, int[::1] u, int[::1] v, w):
    cdef long long[::1] srt = _sorted_idx(w, np.asarray(u), np.asarray(v))
    cdef int[::1] out = np.empty(max(n - 1, 1), dtype=np.int32)
    cdef int cnt
    with nogil:
        cnt = _kruskal_sorted(n, &u[0], &v[0], &srt[0], srt.shape[0], &out[0])
    return np.asarray(out[:cnt])


def _krt(int n, int[::1] u, int[::1] v, floating[::1] w):
    cdef long long[::1] srt = np.ascontiguousarray(
        np.argsort(np.asarray(w), kind="stable"), dtype=np.int64)
    cdef int[::1] order = np.empty(n, dtype=np.int32)
    gnp = np.zeros(max(n - 1, 1), dtype=np.asarray(w).dtype)
    cdef floating[::1] g = gnp
    with nogil:
        _krt_order(n, &u[0], &v[0], &w[0], &srt[0], &order[0], &g[0])
    return np.asarray(order), gnp


def _mst_impl(const floating[:, ::1] A, bint pts, int k, int nthreads, stats, int max_contract=256):
    cdef int n = A.shape[0]
    cdef int d = A.shape[1] if pts else 0
    cdef const floating* Rp = NULL if pts else &A[0, 0]
    cdef const floating* Xp = &A[0, 0] if pts else NULL
    dt = np.asarray(A).dtype
    cdef int nt = _nthreads(n, nthreads)
    cdef int i, ncomp, tid
    cdef double t0 = openmp.omp_get_wtime()
    tm = {}
    k = max(1, min(k, n - 1))

    # Pass 1: certificate sample + seed
    evn = np.empty(<Py_ssize_t>n * k, dtype=np.int32)
    ewn = np.empty(<Py_ssize_t>n * k, dtype=dt)
    rmaxn = np.empty(n, dtype=dt)
    rargn = np.empty(n, dtype=np.int32)
    cdef int[::1] ev = evn
    cdef floating[::1] ew = ewn
    cdef floating[::1] rmax = rmaxn
    cdef int[::1] rarg = rargn
    with nogil:
        _knn_seed(Rp, Xp, n, d, k, &ev[0], &ew[0], &rmax[0], &rarg[0], nt)
    tm['knn'] = openmp.omp_get_wtime() - t0
    seed_i = int(np.argmax(rmaxn))
    seed_j = int(rargn[seed_i])
    eun = np.repeat(np.arange(n, dtype=np.int32), k)
    ok = evn >= 0
    eun, evn, ewn = eun[ok], evn[ok], ewn[ok]
    keep = _kruskal(n, eun, evn, ewn)
    fu, fv, fw = eun[keep], evn[keep], ewn[keep]
    passes = 1

    # Pass 2: Borůvka merges until F spans
    compn = np.empty(n, dtype=np.int32)
    bvn = np.empty(n, dtype=np.int32)
    bwn = np.empty(n, dtype=dt)
    cdef int[::1] comp = compn
    cdef int[::1] bv = bvn
    cdef floating[::1] bw = bwn
    cdef int[::1] fu_v
    cdef int[::1] fv_v
    cdef int c
    cdef floating[::1] cbw
    cdef int[::1] cbu
    cdef int[::1] cbv
    while fu.shape[0] < n - 1:
        fu = np.ascontiguousarray(fu, dtype=np.int32)
        fv = np.ascontiguousarray(fv, dtype=np.int32)
        fu_v = fu
        fv_v = fv
        with nogil:
            _labels(n, &fu_v[0] if fu_v.shape[0] else NULL,
                    &fv_v[0] if fv_v.shape[0] else NULL, fu_v.shape[0], &comp[0])
        _, compact = np.unique(compn, return_inverse=True)
        c = n - fu.shape[0]
        passes += 1
        if c <= max_contract:
            compn[:] = compact
            cbwn = np.empty(<Py_ssize_t>c * c, dtype=dt)
            cbun = np.empty(<Py_ssize_t>c * c, dtype=np.int32)
            cbvn = np.empty(<Py_ssize_t>c * c, dtype=np.int32)
            cbw = cbwn
            cbu = cbun
            cbv = cbvn
            with nogil:
                _contract_pass(Rp, Xp, n, d, &comp[0], c, &cbw[0], &cbu[0], &cbv[0], nt)
            ok = cbun >= 0
            mu, mv, mw = cbun[ok], cbvn[ok], cbwn[ok]
        else:
            with nogil:
                _merge_pass(Rp, Xp, n, d, &comp[0], &bv[0], &bw[0], nt)
            mu, mv, mw = np.arange(n, dtype=np.int32), bvn, bwn
        cu = np.concatenate([fu, mu])
        cv = np.concatenate([fv, mv])
        cw = np.concatenate([fw, mw])
        keep = _kruskal(n, cu, cv, cw)
        fu, fv, fw = cu[keep], cv[keep], cw[keep]
    fu = np.ascontiguousarray(fu, dtype=np.int32)
    fv = np.ascontiguousarray(fv, dtype=np.int32)
    fw = np.ascontiguousarray(fw, dtype=dt)

    tm['merge+kruskal'] = openmp.omp_get_wtime() - t0 - tm['knn']
    # Pass 3: filter
    ordern, gn = _krt(n, fu, fv, fw)
    cdef int[::1] order = ordern
    cdef floating[::1] g = gn
    cdef const floating[:, ::1] Xperm
    cdef const floating* Xpp = NULL
    if pts:
        Xperm = np.ascontiguousarray(np.asarray(A)[ordern])
        Xpp = &Xperm[0, 0]
    cdef int ntf = nt
    cdef int** tu = <int**>malloc(ntf * sizeof(int*))
    cdef int** tv = <int**>malloc(ntf * sizeof(int*))
    cdef floating** tw = <floating**>malloc(ntf * sizeof(floating*))
    cdef Py_ssize_t* tcnt = <Py_ssize_t*>malloc(ntf * sizeof(Py_ssize_t))
    cdef Py_ssize_t* tcap = <Py_ssize_t*>malloc(ntf * sizeof(Py_ssize_t))
    for tid in range(ntf):
        tu[tid] = NULL; tv[tid] = NULL; tw[tid] = NULL; tcap[tid] = 0; tcnt[tid] = 0
    t1 = openmp.omp_get_wtime()
    with nogil:
        _filter(Rp, Xpp, n, d, &order[0], &g[0], tu, tv, tw, tcnt, tcap, ntf)
    tm['filter'] = openmp.omp_get_wtime() - t1
    parts_u, parts_v, parts_w = [fu], [fv], [fw]
    nuns = 0
    for tid in range(ntf):
        if tcnt[tid]:
            parts_u.append(np.array(<int[:tcnt[tid]]>tu[tid]))
            parts_v.append(np.array(<int[:tcnt[tid]]>tv[tid]))
            parts_w.append(np.array(<floating[:tcnt[tid]]>tw[tid]))
            nuns += tcnt[tid]
        free(tu[tid]); free(tv[tid]); free(tw[tid])
    free(tu); free(tv); free(tw); free(tcnt); free(tcap)
    cu = np.ascontiguousarray(np.concatenate(parts_u), dtype=np.int32)
    cv = np.ascontiguousarray(np.concatenate(parts_v), dtype=np.int32)
    cw = np.ascontiguousarray(np.concatenate(parts_w), dtype=dt)
    keep = _kruskal(n, cu, cv, cw)
    if stats is not None:
        stats.update(passes=passes + 1, unsupported=nuns, forest=int(fu.shape[0]),
                     threads=nt, times=tm, total=openmp.omp_get_wtime() - t0)
    return (np.ascontiguousarray(cu[keep]), np.ascontiguousarray(cv[keep]),
            np.ascontiguousarray(cw[keep]), seed_i, seed_j)


def _order_impl(int n, int[::1] mu, int[::1] mv, floating[::1] mw, int si, int sj):
    pn = np.empty(n, dtype=np.int32)
    qn = np.empty(n, dtype=np.int32)
    wn = np.zeros(n, dtype=np.asarray(mw).dtype)
    cdef int[::1] p = pn
    cdef int[::1] q = qn
    cdef floating[::1] w = wn
    with nogil:
        _tree_prim(n, &mu[0], &mv[0], &mw[0], si, sj, &p[0], &q[0], &w[0])
    return pn, qn, wn


def _prep(data, from_points):
    if data.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected float32 or float64, got {data.dtype}")
    a = np.ascontiguousarray(data)
    if a.ndim != 2 or (not from_points and a.shape[0] != a.shape[1]):
        raise ValueError("expected an (n, n) matrix, or (n, d) points with from_points=True")
    return a


def mst_dlmst_c(data, from_points=False, int k=4, int nthreads=0, stats=None):
    """Exact MST of the complete graph on ``data``.

    Returns ``(u, v, w, seed_i, seed_j)``; the seed is VAT's starting pair.
    """
    a = _prep(data, from_points)
    return _mst_impl(a, from_points, k, nthreads, stats)


def vat_order_dlmst_c(data, from_points=False, int k=4, int nthreads=0, stats=None):
    """VAT order ``(p, q, w)`` -- same convention as ``vat_prim_mst_c``'s
    ``(p, q)``, plus ``w``, the MST weight into each vertex."""
    a = _prep(data, from_points)
    n = a.shape[0]
    if n == 1:
        return np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, a.dtype)
    mu, mv, mw, si, sj = mst_dlmst_c(a, from_points, k, nthreads, stats)
    return _order_impl(n, mu, mv, mw, si, sj)


def _gather_impl(const floating[:, ::1] R, int[::1] p, int nt):
    n = R.shape[0]
    outn = np.empty((n, n), dtype=np.asarray(R).dtype)
    cdef floating[:, ::1] out = outn
    with nogil:
        _gather(&R[0, 0], R.shape[0], &p[0], &out[0, 0], nt)
    return outn


def compute_vat_dlmst_c(adj, int k=4, int nthreads=0):
    """Drop-in for ``compute_vat_c``: ``(vat_matrix, p, q)``."""
    a = _prep(adj, False)
    p, q, _ = vat_order_dlmst_c(a, False, k, nthreads)
    nt = _nthreads(a.shape[0], nthreads)
    return _gather_impl(a, p, nt), p, q


def _ivat_impl(int n, int[::1] p, int[::1] mu, int[::1] mv, floating[::1] mw, int nt):
    ordern, gn = _krt(n, mu, mv, mw)
    cdef int[::1] order = ordern
    cdef floating[::1] g = gn
    Dn = np.empty((n, n), dtype=np.asarray(mw).dtype)
    cdef floating[:, ::1] D = Dn
    with nogil:
        _ivat_rows(n, &p[0], &order[0], &g[0], &D[0, 0], nt)
    return Dn


def compute_ivat_dlmst_c(data, from_points=False, int k=4, int nthreads=0):
    """Drop-in for ``compute_ivat_c``: ``(ivat, argmin_seq, p)``.

    The IVAT matrix is built from the MST alone, row-parallel. With
    ``from_points=True`` the output is the only ``n x n`` allocation.
    """
    a = _prep(data, from_points)
    n = a.shape[0]
    if n == 1:
        return np.zeros((1, 1), a.dtype), np.zeros(0, np.int32), np.zeros(1, np.int32)
    mu, mv, mw, si, sj = mst_dlmst_c(a, from_points, k, nthreads)
    nt = _nthreads(n, nthreads)
    p, q, _ = _order_impl(n, mu, mv, mw, si, sj)
    D = _ivat_impl(n, p, mu, mv, mw, nt)
    return D, (q[1:] - 1).astype(np.int32), p
