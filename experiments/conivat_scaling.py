"""ConiVAT scale study — wall-clock vs. sample count n (N = 50 .. 5000).

Measures the runtime of the pure-Python ConiVAT reference (constraint
pre-processing -> MMC metric learning -> impose "similar" constraints ->
path-based minimax / iVAT transform -> VAT ordering) as n grows, and breaks it
against the plain iVAT baseline it builds on.

Data is a set of well-separated 2D Gaussian blobs; constraints are sampled from
the (synthetic) ground-truth labels, matching the paper's protocol. The number
of features and constraints is held fixed so the plot isolates scaling in n.

The dominant cost is the O(n^2) pairwise-distance + iVAT transform, so the
curve should track n^2; metric learning is O(|constraints| * p^2) and n-independent.

Run:  python -m experiments.conivat_scaling
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import (  # noqa: E402
    compute_conivat,
    compute_ivat,
    generate_constraints_from_labels,
    pairwise_distances,
)

FIG_DIR = Path(__file__).parent / "figures"
SIZES = [50, 100, 250, 500, 1000, 2000, 3500, 5000]
N_FEATURES = 2
N_CLUSTERS = 4
N_CONSTRAINTS = 30
SEED = 7


def make_blobs(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """n points split across N_CLUSTERS well-separated 2D Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, size=(N_CLUSTERS, N_FEATURES)) * 30.0
    sizes = [n // N_CLUSTERS] * N_CLUSTERS
    for k in range(n - sum(sizes)):  # scatter the remainder
        sizes[k] += 1
    parts, labels = [], []
    for c, (center, size) in enumerate(zip(centers, sizes)):
        parts.append(rng.standard_normal((size, N_FEATURES)) + center)
        labels.append(np.full(size, c))
    return np.vstack(parts).astype(np.float64), np.concatenate(labels)


def _time(fn, *args, repeats: int = 3) -> float:
    """Best-of-`repeats` wall time in milliseconds (numba already warmed)."""
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def run() -> dict:
    # Warm the numba JIT (compile cost must not land in the first timed point).
    Xw, yw = make_blobs(64)
    compute_ivat(pairwise_distances(np.ascontiguousarray(Xw)), inplace=False)
    compute_conivat(Xw, labels=yw, random_state=SEED)

    conivat_ms, conivat_noml_ms, ivat_ms = [], [], []
    print(f"{'n':>6} {'iVAT_ms':>10} {'ConiVAT_ms':>12} {'ConiVAT(noML)_ms':>18}")
    for n in SIZES:
        X, y = make_blobs(n)
        ml, cl = generate_constraints_from_labels(y, N_CONSTRAINTS, random_state=SEED)

        def _ivat():
            D = pairwise_distances(np.ascontiguousarray(X))
            return compute_ivat(D, inplace=True)

        def _conivat():
            return compute_conivat(
                X, must_link=ml, cannot_link=cl, metric_learning=True, inplace=True
            )

        def _conivat_noml():
            return compute_conivat(
                X, must_link=ml, cannot_link=cl, metric_learning=False, inplace=True
            )

        t_iv = _time(_ivat)
        t_cv = _time(_conivat)
        t_cv0 = _time(_conivat_noml)
        ivat_ms.append(t_iv)
        conivat_ms.append(t_cv)
        conivat_noml_ms.append(t_cv0)
        print(f"{n:>6} {t_iv:>10.2f} {t_cv:>12.2f} {t_cv0:>18.2f}")

    _plot(ivat_ms, conivat_ms, conivat_noml_ms)
    return dict(
        sizes=SIZES,
        ivat_ms=ivat_ms,
        conivat_ms=conivat_ms,
        conivat_noml_ms=conivat_noml_ms,
    )


def _plot(ivat_ms, conivat_ms, conivat_noml_ms) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(SIZES, ivat_ms, "o-", label="iVAT baseline")
    ax.plot(
        SIZES, conivat_noml_ms, "^-", label="ConiVAT (constraints, no metric learn)"
    )
    ax.plot(SIZES, conivat_ms, "s-", label="ConiVAT (full: + MMC metric learning)")

    # O(n^2) reference anchored at the largest iVAT point.
    ref = np.array(SIZES, dtype=float) ** 2
    ref = ref / ref[-1] * ivat_ms[-1]
    ax.plot(SIZES, ref, "k--", alpha=0.5, label=r"$O(n^2)$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("wall time (ms)")
    ax.set_title(
        "ConiVAT scaling, n = 50 .. 5000\n"
        "(2D blobs, 30 constraints; pure-Python reference)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "conivat_scaling.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    print("ConiVAT scale study (N = 50 .. 5000)")
    print("====================================")
    run()
