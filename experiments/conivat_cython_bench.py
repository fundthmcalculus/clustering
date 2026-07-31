"""ConiVAT compiled-vs-pure benchmark — wall-clock vs n (N = 50 .. 5000).

ConiVAT's O(n^2) core (pairwise distances + the minimax/iVAT transform) is
shared with iVAT, so the compiled Cython/OpenMP ``pcvat`` kernels accelerate it
directly. This script compares, across n = 50 .. 5000:

  * optimized iVAT       — pcvat.pairwise_distances_c + compute_ivat_c (the
                           reference "how fast can the shared core go");
  * ConiVAT (cython)     — compute_conivat(backend="cython"), no metric learn,
                           i.e. the compiled core + constraint imposition;
  * ConiVAT (cython+MMC) — full compiled ConiVAT including metric learning;
  * ConiVAT (python)     — compute_conivat(backend="python"), no metric learn,
                           the pure-Python/numba reference.

The headline number is the speedup of the compiled ConiVAT core over the pure
one; it should land close to the compiled-vs-pure iVAT speedup, since ConiVAT
adds only O(1)-per-constraint work on top of the shared core.

Run:  python -m experiments.conivat_cython_bench
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
from experiments.conivat_scaling import make_blobs  # noqa: E402

try:
    from tribbleclustering.pcvat import (  # noqa: E402
        pairwise_distances_c,
        compute_ivat_c,
    )

    HAS_COMPILED = True
except ImportError:
    HAS_COMPILED = False

FIG_DIR = Path(__file__).parent / "figures"
SIZES = [50, 100, 250, 500, 1000, 2000, 3500, 5000]
N_CONSTRAINTS = 30
SEED = 7


def _time(fn, *args, repeats: int = 3) -> float:
    """Best-of-`repeats` wall time in milliseconds (kernels already warmed)."""
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def run() -> dict:
    if not HAS_COMPILED:
        raise SystemExit(
            "The compiled pcvat extension is not built. Build it with "
            "`python setup.py build_ext --inplace` and re-run."
        )

    # Warm both JITs / kernels so compile cost stays out of the first point.
    Xw, yw = make_blobs(64)
    compute_ivat(pairwise_distances(np.ascontiguousarray(Xw)), inplace=False)
    compute_ivat_c(pairwise_distances_c(np.ascontiguousarray(Xw)), inplace=False)
    compute_conivat(Xw, labels=yw, random_state=SEED, backend="cython")
    compute_conivat(Xw, labels=yw, random_state=SEED, backend="python")

    ivat_c_ms, cv_cy_ms, cv_cy_mmc_ms, cv_py_ms = [], [], [], []
    hdr = f"{'n':>6} {'iVAT_c':>9} {'CV_cy':>9} {'CV_cy+MMC':>11} {'CV_py':>10} {'speedup':>9}"
    print(hdr)
    for n in SIZES:
        X, y = make_blobs(n)
        ml, cl = generate_constraints_from_labels(y, N_CONSTRAINTS, random_state=SEED)

        def _ivat_c():
            D = pairwise_distances_c(np.ascontiguousarray(X))
            return compute_ivat_c(D, inplace=True)

        def _cv_cy():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=False,
                inplace=True,
                backend="cython",
            )

        def _cv_cy_mmc():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=True,
                inplace=True,
                backend="cython",
            )

        def _cv_py():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=False,
                inplace=True,
                backend="python",
            )

        t_iv = _time(_ivat_c)
        t_cy = _time(_cv_cy)
        t_cy_mmc = _time(_cv_cy_mmc)
        t_py = _time(_cv_py)
        ivat_c_ms.append(t_iv)
        cv_cy_ms.append(t_cy)
        cv_cy_mmc_ms.append(t_cy_mmc)
        cv_py_ms.append(t_py)
        print(
            f"{n:>6} {t_iv:>9.2f} {t_cy:>9.2f} {t_cy_mmc:>11.2f} "
            f"{t_py:>10.2f} {t_py / t_cy:>8.1f}x"
        )

    _plot(ivat_c_ms, cv_cy_ms, cv_cy_mmc_ms, cv_py_ms)
    return dict(
        sizes=SIZES,
        ivat_c_ms=ivat_c_ms,
        conivat_cython_ms=cv_cy_ms,
        conivat_cython_mmc_ms=cv_cy_mmc_ms,
        conivat_python_ms=cv_py_ms,
    )


def _plot(ivat_c_ms, cv_cy_ms, cv_cy_mmc_ms, cv_py_ms) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: wall-time curves.
    ax.plot(SIZES, cv_py_ms, "o-", color="tab:red", label="ConiVAT (pure Python/numba)")
    ax.plot(
        SIZES,
        cv_cy_mmc_ms,
        "v-",
        color="tab:green",
        label="ConiVAT (Cython + MMC metric learn)",
    )
    ax.plot(SIZES, cv_cy_ms, "s-", color="tab:blue", label="ConiVAT (Cython core)")
    ax.plot(
        SIZES,
        ivat_c_ms,
        "^--",
        color="black",
        alpha=0.6,
        label="optimized iVAT (Cython)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("wall time (ms)")
    ax.set_title("ConiVAT: compiled vs pure, n = 50 .. 5000")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Right: speedup of the compiled ConiVAT core over the pure one, with the
    # compiled-vs-pure iVAT ceiling for reference.
    speedup = np.array(cv_py_ms) / np.array(cv_cy_ms)
    ax2.plot(
        SIZES, speedup, "s-", color="tab:blue", label="ConiVAT core: Python / Cython"
    )
    ax2.axhline(1.0, color="grey", ls=":", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("speedup (x)")
    ax2.set_title("Compiled ConiVAT speedup over pure Python")
    ax2.grid(True, which="both", alpha=0.3)
    for x, s in zip(SIZES, speedup):
        ax2.annotate(
            f"{s:.0f}x",
            (x, s),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax2.legend()

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "conivat_cython_bench.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    print("ConiVAT compiled-vs-pure benchmark (N = 50 .. 5000)")
    print("===================================================")
    run()
