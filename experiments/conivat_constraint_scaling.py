"""ConiVAT constraint-dial evaluation — wall-clock vs #constraints, N fixed.

The companion to the sample-count sweep: here n is held at 5000 and the *number
of constraints* is swept from 5 to 500. This isolates ConiVAT's second dial.

Constraints do NOT touch the O(n^2) core (pairwise distances + the iVAT
minimax transform is constraint-independent). They drive two n-independent
stages instead:

  * expand_constraints — transitive closure (union-find) + expansion of the
    must-link cliques / cannot-link cross-products. Cost grows with the number
    of *expanded* pairs, which can exceed the raw count.
  * learn_metric (MMC)  — builds the S / D difference-vector sets from the
    expanded pairs and iterates gradient-ascent + projection; cost grows with
    |expanded pairs| x features^2.

So the expectation is: total ConiVAT time is ~flat (the fixed core dominates),
while the constraint-handling overhead rises with the constraint count. This
script measures both and reports the expanded-pair counts that explain it.

Run:  python -m experiments.conivat_constraint_scaling
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
    expand_constraints,
    generate_constraints_from_labels,
    learn_metric,
)
from experiments.conivat_scaling import make_blobs  # noqa: E402

try:
    from tribbleclustering.pcvat import pairwise_distances_c  # noqa: F401,E402

    HAS_COMPILED = True
except ImportError:
    HAS_COMPILED = False

FIG_DIR = Path(__file__).parent / "figures"
N = 5000
CONSTRAINTS = [5, 10, 20, 50, 75, 100, 150, 200, 300, 400, 500]
SEED = 7
BACKEND = "cython"
# The ~380 ms constraint-independent core dominates and its run-to-run jitter
# swamps the few-ms constraint signal, so take many samples and keep the best
# (min removes upward OS/thermal noise). The cheap constraint-only stages get
# far more samples since each costs only a few ms.
REPEATS_HEAVY = 15  # full / core / baseline (each ~380 ms)
REPEATS_LIGHT = 200  # expand_constraints / learn_metric (each a few ms)


def _time(fn, *args, repeats: int = REPEATS_HEAVY) -> float:
    """Best-of-`repeats` wall time in milliseconds (kernels already warmed)."""
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def run() -> dict:
    if not HAS_COMPILED and BACKEND == "cython":
        raise SystemExit("Build the pcvat extension first (setup.py build_ext).")

    X, y = make_blobs(N)

    # Warm the compiled core + numba, and a first metric-learning solve.
    compute_conivat(X, labels=y, random_state=SEED, backend=BACKEND)

    # Constraint-free baseline: the core cost the dial can never remove.
    def _baseline():
        return compute_conivat(
            X,
            must_link=[],
            cannot_link=[],
            metric_learning=False,
            inplace=True,
            backend=BACKEND,
        )

    baseline_ms = _time(_baseline)

    full_ms, core_ms, expand_ms, mmc_ms = [], [], [], []
    ml_counts, cl_counts = [], []
    print(f"N={N}, backend={BACKEND}, constraint-free core = {baseline_ms:.1f} ms\n")
    hdr = (
        f"{'#req':>5} {'|ML*|':>7} {'|CL*|':>7} {'expand_ms':>10} "
        f"{'mmc_ms':>8} {'core_ms':>9} {'full_ms':>9}"
    )
    print(hdr)
    for c in CONSTRAINTS:
        ml, cl = generate_constraints_from_labels(y, c, random_state=SEED)
        ml_exp, cl_exp = expand_constraints(ml, cl, N)
        ml_counts.append(len(ml_exp))
        cl_counts.append(len(cl_exp))

        t_expand = _time(lambda: expand_constraints(ml, cl, N), repeats=REPEATS_LIGHT)
        t_mmc = _time(lambda: learn_metric(X, ml_exp, cl_exp), repeats=REPEATS_LIGHT)

        def _full():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=True,
                inplace=True,
                backend=BACKEND,
            )

        def _core():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=False,
                inplace=True,
                backend=BACKEND,
            )

        t_full = _time(_full)
        t_core = _time(_core)
        expand_ms.append(t_expand)
        mmc_ms.append(t_mmc)
        core_ms.append(t_core)
        full_ms.append(t_full)
        print(
            f"{c:>5} {len(ml_exp):>7} {len(cl_exp):>7} {t_expand:>10.2f} "
            f"{t_mmc:>8.2f} {t_core:>9.1f} {t_full:>9.1f}"
        )

    _plot(baseline_ms, full_ms, core_ms, expand_ms, mmc_ms, ml_counts, cl_counts)
    return dict(
        n=N,
        constraints=CONSTRAINTS,
        baseline_ms=baseline_ms,
        full_ms=full_ms,
        core_ms=core_ms,
        expand_ms=expand_ms,
        mmc_ms=mmc_ms,
        ml_expanded=ml_counts,
        cl_expanded=cl_counts,
    )


def _plot(baseline_ms, full_ms, core_ms, expand_ms, mmc_ms, ml_counts, cl_counts):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: total ConiVAT time vs #constraints (fixed core dominates).
    ax.plot(CONSTRAINTS, full_ms, "v-", color="tab:green", label="ConiVAT full (+MMC)")
    ax.plot(
        CONSTRAINTS,
        core_ms,
        "s-",
        color="tab:blue",
        label="ConiVAT core (constraints, no MMC)",
    )
    ax.axhline(
        baseline_ms,
        color="grey",
        ls=":",
        label=f"constraint-free core ({baseline_ms:.0f} ms)",
    )
    ax.set_xlabel("# constraints requested")
    ax.set_ylabel("total wall time (ms)")
    ax.set_title(f"ConiVAT total time vs constraint count (N={N})")
    # Anchor at 0 so the residual ~few-percent core jitter reads as the flat
    # line it is, rather than being magnified by an auto-zoomed axis.
    ax.set_ylim(0, max(max(full_ms), max(core_ms), baseline_ms) * 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Right: the constraint-handling stages (what actually scales), plus the
    # expanded-pair counts they scale with, on a secondary axis.
    ax2.plot(
        CONSTRAINTS, expand_ms, "o-", color="tab:purple", label="expand_constraints"
    )
    ax2.plot(CONSTRAINTS, mmc_ms, "^-", color="tab:orange", label="learn_metric (MMC)")
    ax2.set_xlabel("# constraints requested")
    ax2.set_ylabel("stage wall time (ms)")
    ax2.set_title("Constraint-handling cost (the part the dial moves)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    axc = ax2.twinx()
    axc.plot(
        CONSTRAINTS, ml_counts, "--", color="tab:red", alpha=0.5, label="|ML*| expanded"
    )
    axc.plot(
        CONSTRAINTS,
        cl_counts,
        "--",
        color="tab:blue",
        alpha=0.5,
        label="|CL*| expanded",
    )
    axc.set_ylabel("expanded pair count")
    axc.legend(loc="lower right")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "conivat_constraint_scaling.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nwrote {p}")
    return p


if __name__ == "__main__":
    print("ConiVAT constraint-dial evaluation (N=5000, constraints 5..500)")
    print("==============================================================")
    run()
