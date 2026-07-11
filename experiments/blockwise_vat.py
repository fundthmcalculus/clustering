"""Spike: block-decomposition ("divide-and-conquer") VAT.

Partition the n points into N groups, extract each group's within-group
dissimilarity sub-matrix D_i, run VAT on each independently, and concatenate the
sub-orderings into one global order. Each sub-VAT is O((n/N)^2), so the total
MST/iVAT work drops ~N x and the N blocks are embarrassingly parallel.

The catch (which the user observed empirically): **pseudo-clusters appear at the
partition boundaries.** Within-group VAT never interleaves points across groups,
so every group becomes a contiguous run in the merged order and shows up as a
dark diagonal block — even when a true cluster straddles a boundary (then it
appears as two half-clusters) or when a group is a random mix of clusters (then
each group repeats the whole cluster set). This makes the method APPROXIMATE:
a speed-for-accuracy trade whose error is concentrated at the seams.

This script quantifies both sides and renders the artifact.

Run:  python -m experiments.blockwise_vat
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c, pairwise_distances_c_64  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"


def make_blobs(n, d, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-40, 40, size=(k, d))
    lbl = rng.integers(0, k, n)
    X = rng.standard_normal((n, d)) * 2.0 + centers[lbl]
    return np.ascontiguousarray(X), lbl


# ---------------------------------------------------------------------------
# Partition strategies (how the n points are split into N groups)
# ---------------------------------------------------------------------------
def partition(n, N, X, how, seed=0):
    rng = np.random.default_rng(seed)
    if how == "random":
        g = rng.integers(0, N, n)
        return [np.where(g == i)[0] for i in range(N)]
    if how == "sequential":
        return [np.array(a) for a in np.array_split(np.arange(n), N)]
    if how == "coordinate":
        # spatially coherent blocks: sort by the first principal coordinate
        proj = X @ (X.mean(axis=0) - X[0]) if X.shape[1] > 1 else X[:, 0]
        order = np.argsort(np.asarray(proj).ravel())
        return [np.array(a) for a in np.array_split(order, N)]
    raise ValueError(how)


# ---------------------------------------------------------------------------
# Block-decomposition VAT
# ---------------------------------------------------------------------------
def _vat_order_of_block(D, idx):
    sub = np.ascontiguousarray(D[np.ix_(idx, idx)])
    _, _, p = compute_ivat_c(sub, inplace=False)  # p = local VAT order
    return idx[p]


def blockwise_vat(D, N, groups, merge="concat"):
    """Return (merged_global_order, boundary_positions, block_times_ms)."""
    orders, times = [], []
    for g in groups:
        t = time.perf_counter()
        orders.append(_vat_order_of_block(D, g))
        times.append((time.perf_counter() - t) * 1e3)
    if merge == "chain":
        # greedily chain blocks so adjacent block-ends are most similar, and
        # orient each block to minimise the seam (a mild mitigation).
        orders = _chain_blocks(D, orders)
    merged = np.concatenate(orders)
    bounds = np.cumsum([len(o) for o in orders])[:-1]
    return merged, bounds, times


def _chain_blocks(D, orders):
    remaining = list(range(len(orders)))
    cur = remaining.pop(0)
    chain = [orders[cur]]
    while remaining:
        tail = chain[-1][-1]
        # pick the remaining block whose nearer endpoint is closest to `tail`
        best, best_d, best_flip = None, np.inf, False
        for j in remaining:
            dh, dt = D[tail, orders[j][0]], D[tail, orders[j][-1]]
            if dh < best_d:
                best, best_d, best_flip = j, dh, False
            if dt < best_d:
                best, best_d, best_flip = j, dt, True
        remaining.remove(best)
        chain.append(orders[best][::-1] if best_flip else orders[best])
    return chain


@njit(cache=True)
def ivat_image_from_order(D, order):
    n = order.shape[0]
    V = np.empty((n, n))
    for i in range(n):
        oi = order[i]
        for j in range(n):
            V[i, j] = D[oi, order[j]]
    for r in range(1, n):
        jj = 0
        mn = V[r, 0]
        for c in range(1, r):
            if V[r, c] < mn:
                mn = V[r, c]
                jj = c
        for c in range(r):
            if c == jj:
                V[r, c] = mn
            else:
                cur = V[jj, c] if jj > c else V[c, jj]
                V[r, c] = mn if mn > cur else cur
    for i in range(1, n):
        for j in range(i):
            V[j, i] = V[i, j]
    return V


# ---------------------------------------------------------------------------
# Quality metrics (vs ground-truth labels)
# ---------------------------------------------------------------------------
def n_label_runs(order, labels):
    """Number of maximal same-label runs in the order. Ideal = #clusters."""
    lab = labels[order]
    return int(1 + np.sum(lab[1:] != lab[:-1]))


def adjusted_rand(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua = np.unique(a, return_inverse=True)[1]
    ub = np.unique(b, return_inverse=True)[1]
    cont = np.zeros((ua.max() + 1, ub.max() + 1), dtype=np.int64)
    for i, j in zip(ua, ub):
        cont[i, j] += 1
    from math import comb
    sum_c = sum(comb(v, 2) for v in cont.flatten())
    sa = sum(comb(v, 2) for v in cont.sum(axis=1))
    sb = sum(comb(v, 2) for v in cont.sum(axis=0))
    n = len(a)
    exp = sa * sb / comb(n, 2)
    mx = 0.5 * (sa + sb)
    return (sum_c - exp) / (mx - exp) if mx != exp else 1.0


def labels_from_order(order, ivat_img, k):
    """Cut the iVAT superdiagonal at the top (k-1) gaps -> contiguous segments."""
    n = len(order)
    diag = np.diag(ivat_img, k=1)
    cut_at = np.sort(np.argsort(diag)[-(k - 1):]) if k > 1 else np.array([], int)
    labels = np.empty(n, dtype=np.int64)
    seg, prev = 0, 0
    for c in list(cut_at) + [n - 1]:
        labels[order[prev:c + 1]] = seg
        seg += 1
        prev = c + 1
    return labels


# ---------------------------------------------------------------------------
# Figures + report
# ---------------------------------------------------------------------------
def quality_figure(n=1600, d=8, k=6, how="random", seed=1):
    X, lbl = make_blobs(n, d, k, seed)
    D = pairwise_distances_c_64(X)
    ivat_ex, _, p_ex = compute_ivat_c(D.copy(), inplace=False)

    panels = [("exact serial VAT", ivat_ex, None)]
    for N in (2, 4, 8):
        groups = partition(n, N, X, how, seed)
        order, bounds, _ = blockwise_vat(D, N, groups, merge="concat")
        panels.append((f"blockwise N={N}", ivat_image_from_order(D, order), bounds))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    vmax = np.percentile(ivat_ex, 99)
    for ax, (title, img, bounds) in zip(axes, panels):
        ax.imshow(img, cmap="viridis", vmax=vmax, aspect="equal")
        if bounds is not None:
            for b in bounds:
                ax.axhline(b - 0.5, color="red", lw=0.8, alpha=0.7)
                ax.axvline(b - 0.5, color="red", lw=0.8, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Block-decomposition VAT — iVAT image (n={n}, {k} true clusters, "
        f"'{how}' partition). Red lines = block boundaries; note the "
        f"pseudo-clusters that appear there.", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"blockwise_vat_quality_{how}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def report():
    print("\n=== quality vs N (n=4000, k=10; ideal runs = k = 10) ===")
    n, d, k = 4000, 10, 10
    X, lbl = make_blobs(n, d, k, seed=2)
    D = pairwise_distances_c_64(X)
    ivat_ex, _, p_ex = compute_ivat_c(D.copy(), inplace=False)
    ari_ex = adjusted_rand(labels_from_order(p_ex, ivat_ex, k), lbl)
    print(f"  exact serial: runs={n_label_runs(p_ex, lbl):3d}  ARI={ari_ex:.3f}")
    for how in ("random", "coordinate"):
        for N in (2, 4, 8):
            groups = partition(n, N, X, how, seed=2)
            order, bounds, _ = blockwise_vat(D, N, groups, merge="concat")
            img = ivat_image_from_order(D, order)
            ari = adjusted_rand(labels_from_order(order, img, k), lbl)
            print(f"  {how:10s} N={N}: runs={n_label_runs(order, lbl):3d}  "
                  f"ARI={ari:.3f}")

    print("\n=== performance vs N (MST+iVAT work; sub-VAT is parallelizable) ===")
    print(f"{'n':>6} {'exact_ms':>10} {'N':>3} {'sum_blocks_ms':>14} "
          f"{'max_block_ms':>13} {'work_speedup':>13}")
    for n in (4000, 8000, 16000):
        X, lbl = make_blobs(n, 10, 20, seed=3)
        D = pairwise_distances_c_64(X)
        compute_ivat_c(D.copy(), inplace=False)  # warm
        t = time.perf_counter()
        compute_ivat_c(D.copy(), inplace=False)
        t_exact = (time.perf_counter() - t) * 1e3
        for N in (2, 4, 8):
            groups = partition(n, N, X, "coordinate", seed=3)
            _, _, times = blockwise_vat(D, N, groups, merge="concat")
            ssum, smax = sum(times), max(times)
            print(f"{n:>6} {t_exact:>10.1f} {N:>3} {ssum:>14.1f} "
                  f"{smax:>13.1f} {t_exact / smax:>12.2f}x")


if __name__ == "__main__":
    print("Block-decomposition (divide-and-conquer) VAT spike")
    print("==================================================")
    for how in ("random", "coordinate"):
        p = quality_figure(how=how)
        print(f"wrote {p}")
    report()
