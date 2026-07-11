"""Full-scale performance sweep of the divide-and-conquer VAT spectrum.

Endpoints:
  * NAIVE block-decomposition (partition into N, VAT each block, concatenate) —
    approximate, ~N^2 parallel.
  * exact serial Prim VAT — the N=1 anchor.
  * BORUVKA MST VAT (CPU Numba, GPU CuPy) — exact, parallel MST.
Middle:
  * STITCHED (structure-aware partition + light cross-block stitch) — near-exact.

We sweep dataset scale n and partition size N and render:
  Fig 1  heatmap: naive-blockwise ideal-parallel speedup vs exact serial (n x N).
  Fig 2  heatmaps: cluster quality (ARI) for naive vs stitched (n x N).
  Fig 3  spectrum line plot: wall-time vs n for all methods (naive & Boruvka ends).

VAT-ORDER production time is the measured quantity (the stage divide-and-conquer
targets; the O(n^2) iVAT recurrence is a common downstream step). naive and exact
use the same C Prim kernel, so their speedups are apples-to-apples; naive's
ideal-parallel time is the largest block (blocks are independent). Boruvka CPU is
Numba, GPU is CuPy on a device-resident matrix.

Run:  python -m experiments.dc_vat_scaling
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from tribbleclustering.pcvat import (  # noqa: E402
    vat_prim_mst_c,
    compute_ivat_c,
    pairwise_distances_c_64,
)
from experiments.blockwise_vat import (  # noqa: E402
    make_blobs,
    partition,
    ivat_image_from_order,
    adjusted_rand,
    labels_from_order,
)
from experiments.stitched_vat import stitched_vat  # noqa: E402
from experiments.boruvka_vat import boruvka_mst_numba, vat_order_from_mst  # noqa: E402

try:
    import cupy as _cp
    from experiments.boruvka_gpu import boruvka_mst_gpu

    _HAS_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _HAS_CUPY = False

FIG_DIR = Path(__file__).parent / "figures"
SIZES = [2000, 4000, 8000, 16000, 32000]
NS = [2, 4, 8, 16, 32]


def _t(fn, *a, warm=False):
    if warm:
        fn(*a)
    t = time.perf_counter()
    r = fn(*a)
    return (time.perf_counter() - t) * 1e3, r


def naive_block_times(D, groups):
    """Per-block VAT-order times (ms) using the C Prim kernel."""
    times = []
    orders = []
    for g in groups:
        sub = np.ascontiguousarray(D[np.ix_(g, g)])
        t = time.perf_counter()
        heap, _ = vat_prim_mst_c(sub)
        times.append((time.perf_counter() - t) * 1e3)
        orders.append(g[heap])
    return times, np.concatenate(orders)


def boruvka_gpu_mst_ms(D):
    """Time the device-resident Boruvka MST build only (matrix already on GPU).
    Excludes the O(n) order traversal to stay comparable to the MST-build timing
    of the other methods (and to avoid a redundant host argmax confound)."""
    Dg = _cp.asarray(D)
    _cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    boruvka_mst_gpu(Dg)
    _cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t) * 1e3
    del Dg
    _cp.get_default_memory_pool().free_all_blocks()
    return dt


def run():
    # results grids
    naive_speedup = np.full((len(NS), len(SIZES)), np.nan)
    naive_ari = np.full((len(NS), len(SIZES)), np.nan)
    stitched_ari = np.full((len(NS), len(SIZES)), np.nan)
    exact_ms, bor_cpu_ms, bor_gpu_ms = [], [], []
    naive8_ms = []
    QUALITY_MAX_N = 8000  # ARI needs the O(n^2) iVAT image; cap it

    print(f"CuPy GPU: {_HAS_CUPY}")
    if _HAS_CUPY:  # warm the RawModule compile + kernels
        Xw, _ = make_blobs(512, 10, 4, seed=0)
        boruvka_gpu_mst_ms(pairwise_distances_c_64(Xw))
    print(f"{'n':>6} {'exact_ms':>9} {'borCPU_ms':>10} {'borGPU_ms':>10}")
    for ci, n in enumerate(SIZES):
        X, lbl = make_blobs(n, 10, 20, seed=7)
        D = pairwise_distances_c_64(X)
        te, (heap_ex, _) = _t(vat_prim_mst_c, D, warm=(ci == 0))
        exact_ms.append(te)
        # MST-build times only (fair cross-method comparison; excludes the
        # shared O(n) order traversal / redundant host argmax).
        tbc, _ = _t(boruvka_mst_numba, D, warm=(ci == 0))
        bor_cpu_ms.append(tbc)
        tbg = boruvka_gpu_mst_ms(D) if _HAS_CUPY else np.nan
        bor_gpu_ms.append(tbg)
        print(f"{n:>6} {te:>9.1f} {tbc:>10.1f} {tbg:>10.1f}")

        # exact ARI reference (moderate n only)
        if n <= QUALITY_MAX_N:
            iv_ex, _, p_ex = compute_ivat_c(D.copy(), inplace=False)
            ari_ex = adjusted_rand(labels_from_order(p_ex, iv_ex, 20), lbl)

        for ri, N in enumerate(NS):
            groups = partition(n, N, X, "coordinate", seed=7)
            btimes, o_naive = naive_block_times(D, groups)
            naive_speedup[ri, ci] = te / max(btimes)  # ideal parallel
            if N == 8:
                naive8_ms.append(max(btimes))
            if n <= QUALITY_MAX_N:
                img = ivat_image_from_order(D, o_naive)
                naive_ari[ri, ci] = adjusted_rand(
                    labels_from_order(o_naive, img, 20), lbl
                )
                o_st = stitched_vat(D, X, N, n_repr=24, seed=7)
                img_st = ivat_image_from_order(D, o_st)
                stitched_ari[ri, ci] = adjusted_rand(
                    labels_from_order(o_st, img_st, 20), lbl
                )

    _plot_speedup_heatmap(naive_speedup)
    _plot_quality_heatmaps(naive_ari, stitched_ari, QUALITY_MAX_N)
    _plot_spectrum(exact_ms, bor_cpu_ms, bor_gpu_ms, naive8_ms)
    return dict(
        naive_speedup=naive_speedup,
        naive_ari=naive_ari,
        stitched_ari=stitched_ari,
        exact_ms=exact_ms,
        bor_cpu_ms=bor_cpu_ms,
        bor_gpu_ms=bor_gpu_ms,
    )


def _annot(ax, M, fmt="{:.1f}"):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    fmt.format(M[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )


def _plot_speedup_heatmap(sp):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sp, cmap="viridis", aspect="auto", norm=LogNorm())
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_yticks(range(len(NS)))
    ax.set_yticklabels(NS)
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("N (partition size)")
    ax.set_title(
        "Naive block-decomposition VAT — ideal-parallel speedup\n"
        "vs exact serial Prim (VAT-order production)"
    )
    _annot(ax, sp, "{:.1f}x")
    fig.colorbar(im, ax=ax, label="speedup (x)")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "dc_vat_speedup_heatmap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")


def _plot_quality_heatmaps(naive_ari, stitched_ari, qmax):
    ncol = sum(1 for n in SIZES if n <= qmax)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, M, title in (
        (axes[0], naive_ari, "naive block-decomposition"),
        (axes[1], stitched_ari, "structure-aware + stitch"),
    ):
        sub = M[:, :ncol]
        im = ax.imshow(sub, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(ncol))
        ax.set_xticklabels(SIZES[:ncol])
        ax.set_yticks(range(len(NS)))
        ax.set_yticklabels(NS)
        ax.set_xlabel("n (samples)")
        ax.set_ylabel("N (partition size)")
        ax.set_title(f"ARI vs truth — {title}")
        _annot(ax, sub, "{:.2f}")
        fig.colorbar(im, ax=ax, label="ARI (1=exact clustering)")
    fig.suptitle(
        "Cluster quality across scale x partition size "
        "(naive degrades with N; stitch stays ~exact)",
        fontsize=12,
    )
    fig.tight_layout()
    p = FIG_DIR / "dc_vat_quality_heatmap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")


def _plot_spectrum(exact, bcpu, bgpu, naive8):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(SIZES, exact, "o-", label="exact serial Prim (N=1)")
    ax.plot(SIZES, naive8, "s-", label="naive block-decomp N=8 (ideal parallel)")
    ax.plot(SIZES, bcpu, "^-", label="Boruvka MST (CPU, Numba)")
    if _HAS_CUPY:
        ax.plot(SIZES, bgpu, "v-", label="Boruvka MST (GPU, resident)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("MST-build time (ms)")
    ax.set_title(
        "Divide-and-conquer VAT spectrum\n"
        "naive (approx, fast) <-> Boruvka (exact, parallel)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "dc_vat_spectrum.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    print("Divide-and-conquer VAT — full-scale performance sweep")
    print("=====================================================")
    run()
