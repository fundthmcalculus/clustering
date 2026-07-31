"""VAT sequence-variation & consensus analysis (pre-ACO/GA hot-start study).

VAT's ordering is Prim's insertion order over the (start-independent) MST — so
*which* city you start from re-linearises the same tree into a different sequence.
This experiment runs the VAT order from many starting points on a small instance
(n in 50-500), then asks:

  * where do the sequences AGREE — the **consistent subsequences** (stable chains
    that recur regardless of start) — and where do they DIVERGE?
  * the divergence links are the natural **2-opt swap points**: the flexible seams
    where the order is free to reconfigure, so a local search should spend its
    effort there and freeze the stable runs.

The by-product is a **co-adjacency matrix** C[a,b] = fraction of starts in which a
and b are consecutive — a consensus edge prior that will seed the ACO/GA pheromone
in the follow-up step. Saved to experiments/figures/vat_tsp_seqvar_coadj_<name>.npz.

Repeatable data only: nearest-size EUC_2D TSPLIB instance for each target n.

Run:  python -m experiments.vat_tsp_seqvar
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.vat_tsp_tsplib import nearest_euc_instance  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"


def pdist(coords):
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d * d).sum(-1))


def vat_order(D, start):
    """Prim insertion order (the VAT ordering) grown from city `start`."""
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = D[start].astype(np.float64).copy()
    key[start] = np.inf
    order = np.empty(n, dtype=np.int64)
    in_tree[start] = True
    order[0] = start
    for i in range(1, n):
        u = int(np.argmin(np.where(in_tree, np.inf, key)))
        in_tree[u] = True
        order[i] = u
        du = D[u]
        upd = (~in_tree) & (du < key)
        key[upd] = du[upd]
    return order


def canonical_start(D):
    """The standard VAT start: an endpoint of the globally largest distance."""
    n = D.shape[0]
    return int(np.argmax(D) // n)


def all_orders(D, max_starts=256):
    """VAT orders from every start (or an evenly-spaced sample if n>max_starts)."""
    n = D.shape[0]
    starts = (
        np.arange(n)
        if n <= max_starts
        else np.linspace(0, n - 1, max_starts, dtype=int)
    )
    return np.stack([vat_order(D, int(s)) for s in starts]), starts


def co_adjacency(orders, n):
    """C[a,b] = fraction of orders in which a,b are consecutive (either dir)."""
    C = np.zeros((n, n))
    for o in orders:
        a, b = o[:-1], o[1:]
        np.add.at(C, (a, b), 1.0)
        np.add.at(C, (b, a), 1.0)
    return C / len(orders)


def segments_and_swaps(ref, C, tau):
    """Along the reference VAT order, split into consistent subsequences (runs of
    consecutive pairs with co-adjacency >= tau) and the swap points between them."""
    runs, swaps = [], []
    cur = [int(ref[0])]
    for i in range(len(ref) - 1):
        if C[ref[i], ref[i + 1]] >= tau:
            cur.append(int(ref[i + 1]))
        else:
            runs.append(cur)
            swaps.append(i)  # low-consensus link sits between positions i, i+1
            cur = [int(ref[i + 1])]
    runs.append(cur)
    return runs, swaps


def analyse(target, tau=0.5, max_starts=256):
    name, coords, dim = nearest_euc_instance(target)
    D = pdist(coords)
    orders, starts = all_orders(D, max_starts)
    C = co_adjacency(orders, dim)
    ref = vat_order(D, canonical_start(D))

    runs, swaps = segments_and_swaps(ref, C, tau)
    seglen = np.array([len(r) for r in runs])
    covered = int(seglen[seglen >= 2].sum())
    coverage = 100.0 * covered / dim

    # position spread: how far each city's index moves across starts
    pos = np.empty((len(orders), dim), dtype=np.float64)
    for k, o in enumerate(orders):
        pos[k, o] = np.arange(dim)
    spread = pos.max(0) - pos.min(0)  # per-city index range across starts

    # geometry check: are swap-point links the long edges of the reference order?
    ref_edge_len = np.array([D[ref[i], ref[i + 1]] for i in range(dim - 1)])
    swap_mask = np.zeros(dim - 1, dtype=bool)
    swap_mask[swaps] = True
    mean_swap_len = ref_edge_len[swap_mask].mean() if swaps else 0.0
    mean_run_len = ref_edge_len[~swap_mask].mean() if (~swap_mask).any() else 0.0
    return dict(
        name=name,
        n=dim,
        coords=coords,
        D=D,
        orders=orders,
        starts=starts,
        C=C,
        ref=ref,
        runs=runs,
        swaps=swaps,
        seglen=seglen,
        coverage=coverage,
        spread=spread,
        ref_edge_len=ref_edge_len,
        swap_mask=swap_mask,
        mean_swap_len=mean_swap_len,
        mean_run_len=mean_run_len,
        tau=tau,
    )


def figure_primary(r):
    name, coords, C, ref = r["name"], r["coords"], r["C"], r["ref"]
    n, runs, swaps = r["n"], r["runs"], r["swaps"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))

    # (a) co-adjacency heatmap, ordered by the reference VAT order
    Cord = C[np.ix_(ref, ref)]
    im = ax[0].imshow(Cord, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax[0].set_title(
        f"co-adjacency C (ordered by VAT ref)\n{name} n={n}, "
        f"{len(r['orders'])} starts"
    )
    ax[0].set_xlabel("position in reference VAT order")
    ax[0].set_ylabel("position in reference VAT order")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="fraction of starts adjacent")

    # (b) point map: consistent subsequences as colour chains, swaps dashed
    cmap = plt.cm.tab20
    for k, run in enumerate(runs):
        if len(run) >= 2:
            pts = coords[run]
            ax[1].plot(pts[:, 0], pts[:, 1], "-", lw=1.6, color=cmap(k % 20), zorder=2)
    for i in swaps:  # swap-point links (low-consensus seams)
        seg = coords[[ref[i], ref[i + 1]]]
        ax[1].plot(seg[:, 0], seg[:, 1], "--", lw=0.8, color="0.5", zorder=1)
    ax[1].plot(coords[:, 0], coords[:, 1], ".", ms=3, color="k", zorder=3)
    ax[1].set_title(
        f"consistent subsequences (tau={r['tau']}): "
        f"{(r['seglen'] >= 2).sum()} runs, {r['coverage']:.0f}% "
        f"of cities\n{len(swaps)} swap points (dashed)"
    )
    ax[1].set_aspect("equal")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # (c) per-city position spread across starts (sorted)
    sp = np.sort(r["spread"])[::-1]
    ax[2].plot(sp, color="tab:blue")
    ax[2].axhline(0, color="0.7", lw=0.5)
    ax[2].set_title(
        "per-city index spread across starts\n"
        f"(max positional wander; median={np.median(r['spread']):.0f})"
    )
    ax[2].set_xlabel("city (sorted by spread)")
    ax[2].set_ylabel("index range (max-min position)")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_seqvar.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def figure_summary(rows, taus):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(ns, [r["coverage"] for r in rows], "o-", color="tab:green")
    ax[0].set_title(f"stable coverage vs n (tau={rows[0]['tau']})")
    ax[0].set_ylabel("% cities in consistent subsequences")
    ax[1].plot(
        ns,
        [len(r["swaps"]) for r in rows],
        "o-",
        color="tab:red",
        label="# swap points",
    )
    ax[1].plot(
        ns,
        [(r["seglen"] >= 2).sum() for r in rows],
        "s-",
        color="tab:blue",
        label="# stable runs",
    )
    ax[1].set_title("swap points & stable runs vs n")
    ax[1].legend(fontsize=8)
    # tau sensitivity on the largest instance
    big = rows[-1]
    cov, nsw = [], []
    for t in taus:
        runs, swaps = segments_and_swaps(big["ref"], big["C"], t)
        sl = np.array([len(x) for x in runs])
        cov.append(100.0 * sl[sl >= 2].sum() / big["n"])
        nsw.append(len(swaps))
    ax[2].plot(taus, cov, "o-", color="tab:green", label="coverage %")
    ax2b = ax[2].twinx()
    ax2b.plot(taus, nsw, "s-", color="tab:red", label="# swap pts")
    ax[2].set_title(f"tau sensitivity ({big['name']} n={big['n']})")
    ax[2].set_xlabel("consensus threshold tau")
    ax[2].set_ylabel("coverage %", color="tab:green")
    ax2b.set_ylabel("# swap points", color="tab:red")
    for a in ax[:2]:
        a.set_xlabel("n (cities)")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "vat_tsp_seqvar_summary.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("VAT sequence-variation & consensus analysis")
    print("=" * 66)
    primary = analyse(200, tau=0.5)
    print(
        f"\nPrimary: {primary['name']} n={primary['n']}  "
        f"({len(primary['orders'])} starts, tau={primary['tau']})"
    )
    print(f"  consistent subsequences (len>=2): {(primary['seglen'] >= 2).sum()}")
    print(f"  cities covered by stable runs:    {primary['coverage']:.1f}%")
    print(f"  swap points (low-consensus links): {len(primary['swaps'])}")
    print(f"  longest stable run:               {primary['seglen'].max()} cities")
    print(
        f"  mean ref-edge length  swap links: {primary['mean_swap_len']:.1f}  "
        f"vs stable links: {primary['mean_run_len']:.1f}  "
        f"(ratio {primary['mean_swap_len']/max(primary['mean_run_len'],1e-9):.1f}x)"
    )
    print(
        f"  median per-city position spread:  {np.median(primary['spread']):.0f}"
        f" of {primary['n']}"
    )

    # save the consensus edge prior for the ACO/GA follow-up
    FIG_DIR.mkdir(exist_ok=True)
    npz = FIG_DIR / f"vat_tsp_seqvar_coadj_{primary['name']}.npz"
    np.savez_compressed(
        npz, C=primary["C"], ref=primary["ref"], coords=primary["coords"]
    )
    print(f"  saved consensus edge prior -> {npz.name}")

    print(f"\nwrote {figure_primary(primary)}")

    print("\nSummary sweep (nearest EUC_2D to each target):")
    rows = []
    for tgt in (52, 100, 200, 500):
        r = analyse(tgt, tau=0.5)
        rows.append(r)
        print(
            f"  {r['name']:>9s} n={r['n']:4d}  coverage {r['coverage']:5.1f}%  "
            f"runs {int((r['seglen'] >= 2).sum()):4d}  swaps {len(r['swaps']):4d}  "
            f"swap/stable edge-len {r['mean_swap_len']/max(r['mean_run_len'],1e-9):4.1f}x"
        )
    print(f"\nwrote {figure_summary(rows, taus=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))}")
