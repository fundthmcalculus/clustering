"""Gap 1: auto-k meta-heuristics for VAT/iVAT on adversarial data.

Can we recover the number of clusters k WITHOUT being told it? Evaluated on the
adversarial datasets (where structure is clean for some, absent for others):

  * max-gap   — the repo's parameter-free rule (get_ivat_levels): the largest
                jump in the SORTED iVAT superdiagonal sets a threshold; entries
                above it are cluster boundaries.
  * silhouette— cut the VAT order at the top (k-1) superdiagonal gaps for
                k=2..Kmax, score each partition by the silhouette on D
                (precomputed, so it works on arbitrary dissimilarity), pick the
                best k.

For each we report the predicted k-hat and the ARI at that k-hat vs ground truth.

Run:  python -m experiments.autok_eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c  # noqa: E402
from tribbleclustering.pvat import get_ivat_levels  # noqa: E402
from experiments.blockwise_vat import adjusted_rand, labels_from_order  # noqa: E402
from experiments.adversarial_eval import DATASETS  # noqa: E402
from experiments.hardening_eval import d_euclidean  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"
KMAX = 8


def superdiagonal(ivat, order):
    return np.array([ivat[order[i], order[i + 1]] for i in range(len(order) - 1)])


def khat_repo(X, ivat, order, y):
    """The SHIPPED parameter-free rule (pvat.get_ivat_levels, n_clusters=-1)."""
    res = get_ivat_levels(X, ivat, np.asarray(order), n_levels=1, n_clusters=-1)
    cids = res.cluster_city_ids
    labels = np.full(len(X), -1)
    for c, ids in enumerate(cids):
        labels[np.asarray(ids)] = c
    mask = y >= 0
    return len(cids), adjusted_rand(y[mask], labels[mask])


def silhouette_precomputed(D, labels):
    labs = np.unique(labels)
    if len(labs) < 2:
        return -1.0
    n = len(labels)
    sil = np.zeros(n)
    masks = {c: labels == c for c in labs}
    for i in range(n):
        ci = labels[i]
        same = masks[ci].copy(); same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, masks[c]].mean() for c in labs if c != ci)
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


def khat_silhouette(D, ivat, order, d):
    best_k, best_s = 2, -2.0
    for k in range(2, KMAX + 1):
        cuts = np.sort(np.argsort(d)[-(k - 1):])
        labels = labels_from_order(order, ivat, k)
        s = silhouette_precomputed(D, labels)
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def run():
    print(f"{'dataset':16s} {'k_true':>6s} | {'repo_k':>6s} {'repo_ARI':>9s}"
          f" | {'sil_k':>6s} {'sil_ARI':>8s}")
    rows = []
    for dname, gen, ktrue in DATASETS:
        X, y = gen()
        D = d_euclidean(X)
        ivat, _, order = compute_ivat_c(D.copy(), inplace=False)
        d = superdiagonal(ivat, order)
        krepo, ari_repo = khat_repo(X, ivat, order, y)
        ksil = khat_silhouette(D, ivat, order, d)
        ari_sil = adjusted_rand(y[y >= 0],
                                labels_from_order(order, ivat, ksil)[y >= 0])
        rows.append((dname, ktrue, krepo, ari_repo, ksil, ari_sil, d))
        print(f"{dname:16s} {ktrue:>6d} | {krepo:>6d} {ari_repo:>9.2f} | "
              f"{ksil:>6d} {ari_sil:>8.2f}")
    _figure(rows)


def _figure(rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (dname, ktrue, krepo, ari_repo, ksil, ari_sil, d) in zip(
            axes.ravel(), rows):
        s = np.sort(d)[::-1]
        ax.plot(range(1, len(s) + 1), s, ".-", ms=3)
        ax.axvline(ktrue - 0.5, color="green", ls=":", lw=1.5,
                   label=f"true k-1={ktrue-1}")
        ax.axvline(krepo - 0.5, color="red", ls="--", lw=1,
                   label=f"repo k̂-1={krepo-1}")
        ax.axvline(ksil - 0.5, color="blue", ls="-.", lw=1,
                   label=f"sil k̂-1={ksil-1}")
        ax.set_xlim(0, min(20, len(s)))
        ax.set_title(f"{dname}: true={ktrue}, repo k̂={krepo} (ARI {ari_repo:.2f}), "
                     f"sil k̂={ksil} (ARI {ari_sil:.2f})", fontsize=9)
        ax.set_xlabel("boundary rank (sorted desc)"); ax.set_ylabel("iVAT superdiag")
        ax.legend(fontsize=7)
    fig.suptitle("Auto-k on the iVAT superdiagonal: a clean knee at true k-1 "
                 "exactly when single-linkage structure is valid (moons/circles/"
                 "blobs); no rule recovers k where VAT itself fails (aniso/bridged)",
                 fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "autok_eval.png"
    fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    print("Auto-k meta-heuristics for VAT/iVAT")
    print("===================================")
    run()
