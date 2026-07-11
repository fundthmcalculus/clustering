"""Gap 2: does a PRINCIPLED bounded cross-edge stitch restore robustness
without collapsing to full (O(n^2)) Boruvka?

The light stitch (random representatives, 1 cheapest cross-edge per block pair)
is fragile on non-convex data (ARI swings 0<->1; see HARDENING_FINDINGS.md). We
test whether two bounded improvements fix it:
  * boundary-aware representatives (farthest-point sampling within each block),
  * top-m cross-edges per block pair (redundancy),
against the exact full-cross-edge oracle (== Boruvka merge, O(n^2)).

Same partition-adversarial grid as the fragility test (two-moons, partition
strategy x N). A robust stitch should flatten the checkerboard toward the oracle.

Run:  python -m experiments.principled_stitch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.blockwise_vat import (  # noqa: E402
    ivat_image_from_order, adjusted_rand, labels_from_order,
)
from experiments.stitched_vat import (  # noqa: E402
    stitch_core, maximin_partition, kmeans_partition,
)
from experiments.hardening_eval import (  # noqa: E402
    d_euclidean, adversarial_partition, random_partition, coordinate_partition,
)
from experiments.adversarial_eval import two_moons, circles  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"

# Ablation: which factor fixes the light stitch — boundary-aware reps, or
# cross-edge redundancy (top-m), or both? (exact-VAT on the full MST = 1.0 target,
# partition-independent, shown in text not the grid.)
VARIANTS = {
    "light (random, m=1)": dict(reps="random", m_edges=1),
    "top-m only (random, m=8)": dict(reps="random", m_edges=8),
    "fps only (fps, m=1)": dict(reps="fps", m_edges=1),
    "principled (fps, m=8)": dict(reps="fps", m_edges=8),
}
PTYPES = ["kmeans", "maximin(D)", "coordinate", "random", "adversarial"]
NS = [2, 4, 8, 16, 32]


def _parts(X, y, D, N):
    return {
        "kmeans": kmeans_partition(X, N, seed=3),
        "maximin(D)": maximin_partition(D, N, seed=3),
        "coordinate": coordinate_partition(X, N),
        "random": random_partition(len(X), N, seed=3),
        "adversarial": adversarial_partition(y, N),
    }


def sweep(X, y, k, tag):
    D = d_euclidean(X)
    grids = {v: np.full((len(PTYPES), len(NS)), np.nan) for v in VARIANTS}
    for cj, N in enumerate(NS):
        P = _parts(X, y, D, N)
        for ri, pt in enumerate(PTYPES):
            for vname, kw in VARIANTS.items():
                order = stitch_core(D, P[pt], n_repr=24, seed=3, **kw)
                ari = adjusted_rand(
                    y, labels_from_order(order, ivat_image_from_order(D, order), k))
                grids[vname][ri, cj] = ari
    return grids


def _plot(grids, tag):
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5.2 * len(VARIANTS), 4.4))
    for ax, (vname, G) in zip(axes, grids.items()):
        im = ax.imshow(G, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(NS))); ax.set_xticklabels(NS)
        ax.set_yticks(range(len(PTYPES))); ax.set_yticklabels(PTYPES)
        ax.set_xlabel("N")
        ax.set_title(f"{vname}\nmean={np.nanmean(G):.2f} min={np.nanmin(G):.2f}",
                     fontsize=10)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle(f"Stitch ablation — {tag}: only boundary-aware reps (fps) AND "
                 "top-m cross-edges together are robust (stitched ARI, partition x N)",
                 fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / f"principled_stitch_{tag}.png"
    fig.savefig(p, dpi=115); plt.close(fig)
    print(f"wrote {p}")


def run():
    for tag, gen, k in [("two_moons", lambda: two_moons(1500, 0.07, 2), 2),
                        ("circles", lambda: circles(1500, 0.06, 2), 2)]:
        X, y = gen()
        grids = sweep(X, y, k, tag)
        print(f"\n=== {tag} — mean / min ARI over the partition x N grid ===")
        for vname, G in grids.items():
            print(f"  {vname:24s} mean={np.nanmean(G):.3f}  min={np.nanmin(G):.3f}"
                  f"  frac>=0.9={np.mean(G >= 0.9):.2f}")
        _plot(grids, tag)


if __name__ == "__main__":
    print("Principled bounded cross-edge stitch — robustness")
    print("=================================================")
    run()
