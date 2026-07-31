"""Spike: 2D divide-and-conquer VAT ("2D merge-sort").

Explores Scott's idea of recursively bisecting the VAT problem into sub-blocks
that can be ordered in parallel and then merged — the 2D analogue of
merge-sort. Two angles are prototyped and measured against the exact serial
iVAT:

  Approach A — recursive bisection + block merge (APPROXIMATE, O(n^2), parallel)
    Partition the points into two groups by a cheap diameter split, recursively
    VAT-order each group (independently, in parallel), then merge the two
    ordered blocks end-to-end in the orientation whose touching endpoints are
    most similar. This is the literal "bisect into two 64x64 sub-blocks, process
    in parallel, insert/merge" idea. It is not guaranteed exact — single-linkage
    (which VAT/iVAT encodes) is sensitive to cross-group bridges — so we measure
    how much cluster quality it costs and how much parallel speedup it buys.

  Approach B — blocked minimax (bottleneck) closure (EXACT, O(n^3), parallel)
    The iVAT dissimilarity U[i,j] is the minimax path distance (Hu's theorem:
    the largest edge on the MST path), i.e. the single-linkage cophenetic
    distance. That is the transitive closure of D in the (min, max) semiring,
    computable by a Floyd-Warshall/Kleene recurrence that tiles into
    independent b x b block "multiplies" — exactly the 2D-blocked parallel
    structure the idea points at. It is exact but O(n^3); the question is
    whether massive parallelism (e.g. GPU tiles) can beat the O(n^2) serial
    engine for useful n. Here we confirm exactness and characterize the cost.

Run:  python -m experiments.dc_vat
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from tribbleclustering.pcvat import compute_ivat_c


# ---------------------------------------------------------------------------
# Data + exact reference
# ---------------------------------------------------------------------------
def make_blobs(n, d, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-40, 40, size=(k, d))
    lbl = rng.integers(0, k, size=n)
    X = (rng.standard_normal((n, d)) * 2.0 + centers[lbl]).astype(np.float64)
    return np.ascontiguousarray(X), lbl


@njit(cache=True)
def _pdist(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                s += diff * diff
            D[i, j] = D[j, i] = np.sqrt(s)
    return D


# ---------------------------------------------------------------------------
# Ordering-quality metrics (vs ground-truth labels)
# ---------------------------------------------------------------------------
def boundary_count(order, labels):
    """Number of adjacent pairs in the order whose true labels differ. A
    perfect VAT order has exactly (#clusters - 1) boundaries: every cluster is
    one contiguous run. Lower is better."""
    lab = labels[order]
    return int(np.sum(lab[1:] != lab[:-1]))


def contiguity(order, labels):
    """Fraction of adjacent pairs sharing a label (1.0 = perfectly grouped)."""
    lab = labels[order]
    return float(np.mean(lab[1:] == lab[:-1]))


# ---------------------------------------------------------------------------
# Approach A: recursive bisection VAT ("2D merge-sort")
# ---------------------------------------------------------------------------
def _diameter_split(D, idx):
    """Split idx into two groups by the two farthest points (the VAT seed pair),
    assigning every point to its nearer pole. Balanced-ish, cheap, O(|idx|^2)
    for the pair search on the sub-block already in hand."""
    sub = D[np.ix_(idx, idx)]
    flat = int(np.argmax(sub))
    a, b = flat // len(idx), flat % len(idx)
    da = sub[a]
    db = sub[b]
    mask_a = da <= db
    A = idx[mask_a]
    B = idx[~mask_a]
    # guard against a degenerate all-to-one split
    if len(A) == 0 or len(B) == 0:
        half = len(idx) // 2
        A, B = idx[:half], idx[half:]
    return A, B


def _serial_order(D, idx):
    sub = np.ascontiguousarray(D[np.ix_(idx, idx)])
    _, _, p = compute_ivat_c(sub, inplace=True)
    return idx[p]


def _merge_blocks(D, oA, oB):
    """Concatenate two ordered blocks in the orientation whose touching
    endpoints are most similar (the 'insertion' seam). No interleaving."""
    ends = {
        ("AB"): (oA, oB, D[oA[-1], oB[0]]),
        ("ABr"): (oA, oB[::-1], D[oA[-1], oB[-1]]),
        ("rAB"): (oA[::-1], oB, D[oA[0], oB[0]]),
        ("rABr"): (oA[::-1], oB[::-1], D[oA[0], oB[-1]]),
    }
    best = min(ends.values(), key=lambda t: t[2])
    return np.concatenate([best[0], best[1]])


def dc_vat_order(D, base=256, max_workers=None):
    """Recursive bisection VAT. Sub-blocks below `base` are ordered exactly by
    the serial engine; larger ones are split, ordered in parallel, and merged."""
    n = D.shape[0]

    def rec(idx, depth):
        if len(idx) <= base:
            return _serial_order(D, idx)
        A, B = _diameter_split(D, idx)
        if depth < 3 and max_workers != 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fA = ex.submit(rec, A, depth + 1)
                fB = ex.submit(rec, B, depth + 1)
                oA, oB = fA.result(), fB.result()
        else:
            oA, oB = rec(A, depth + 1), rec(B, depth + 1)
        return _merge_blocks(D, oA, oB)

    return rec(np.arange(n), 0)


# ---------------------------------------------------------------------------
# Approach B: exact minimax (bottleneck) closure via (min, max) Floyd-Warshall
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=False)
def minimax_closure(D):
    """Transitive closure of D in the (min, max) semiring = minimax path
    distance = single-linkage cophenetic distance. O(n^3). The k-loop is
    sequential but each k-step is an independent n x n block op that tiles and
    parallelizes (Kleene); this reference keeps it simple."""
    n = D.shape[0]
    U = D.copy()
    for k in range(n):
        for i in range(n):
            uik = U[i, k]
            for j in range(n):
                m = uik if uik > U[k, j] else U[k, j]
                if m < U[i, j]:
                    U[i, j] = m
    return U


def cophenetic_single_linkage(D):
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="single")
    return squareform(cophenet(Z))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def quality_experiment():
    print("\n=== Approach A: recursive-bisection VAT quality vs exact serial ===")
    print("(boundary count: #adjacent label changes; ideal = k-1. "
          "contiguity: fraction of same-label neighbours, 1.0 ideal)\n")
    print(f"{'n':>6} {'k':>3} {'ideal_bnd':>9} "
          f"{'serial_bnd':>10} {'serial_ctg':>10} "
          f"{'dc_bnd':>8} {'dc_ctg':>8}")
    for n, k in [(2000, 10), (4000, 15), (8000, 20), (16000, 30)]:
        X, lbl = make_blobs(n, 12, k, seed=1)
        D = _pdist(X)
        _, _, p_ser = compute_ivat_c(D.copy(), inplace=False)
        o_dc = dc_vat_order(D.copy(), base=max(256, n // 16))
        print(f"{n:>6} {k:>3} {k-1:>9} "
              f"{boundary_count(p_ser, lbl):>10} {contiguity(p_ser, lbl):>10.3f} "
              f"{boundary_count(o_dc, lbl):>8} {contiguity(o_dc, lbl):>8.3f}")


def speed_experiment():
    print("\n=== Approach A: parallel speedup (wall-clock) ===\n")
    print(f"{'n':>6} {'serial_ms':>10} {'dc_par_ms':>10} {'dc_ser_ms':>10} {'speedup':>8}")
    for n in [8000, 16000, 32000]:
        X, lbl = make_blobs(n, 12, 25, seed=2)
        D = _pdist(X)
        t0 = time.perf_counter()
        compute_ivat_c(D.copy(), inplace=False)
        t_ser = (time.perf_counter() - t0) * 1e3
        base = n // 16
        t0 = time.perf_counter()
        dc_vat_order(D.copy(), base=base)
        t_par = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        dc_vat_order(D.copy(), base=base, max_workers=1)
        t_dcser = (time.perf_counter() - t0) * 1e3
        print(f"{n:>6} {t_ser:>10.1f} {t_par:>10.1f} {t_dcser:>10.1f} "
              f"{t_ser / t_par:>7.2f}x")


def exactness_experiment():
    print("\n=== Approach B: (min,max) closure exactness vs single-linkage ===\n")
    for n in [200, 400]:
        X, lbl = make_blobs(n, 8, 6, seed=3)
        D = _pdist(X)
        U = minimax_closure(D)
        C = cophenetic_single_linkage(D)
        err = float(np.max(np.abs(U - C)))
        # also confirm it equals the exact iVAT values (permutation-invariant:
        # compare sorted upper triangles)
        ivat, _, _ = compute_ivat_c(D.copy(), inplace=False)
        iu = np.sort(ivat[np.triu_indices(n, 1)])
        uu = np.sort(U[np.triu_indices(n, 1)])
        err_ivat = float(np.max(np.abs(iu - uu)))
        print(f"  n={n}: max|closure - single_linkage_cophenetic| = {err:.2e}, "
              f"max|sorted(closure) - sorted(iVAT)| = {err_ivat:.2e}")


if __name__ == "__main__":
    exactness_experiment()
    quality_experiment()
    speed_experiment()
