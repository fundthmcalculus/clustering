"""ConiVAT Cython performance test — wall-clock vs n, up to N = 20000.

A pure performance characterization of the *compiled* ConiVAT path (no
pure-Python comparison): how does the Cython/OpenMP ConiVAT scale, and where
does the time go, as n grows to 20000?

For each n it measures (best-of-k, kernels pre-warmed):
  * ConiVAT core (f64)     — compute_conivat(backend="cython"), no metric learn;
  * ConiVAT full (f64)     — the same + MMC metric learning;
  * ConiVAT core (f32)     — the compiled core on a float32 copy (half memory,
                             ~2x faster; opt-in precision per the repo policy);
  * stage split (f64)      — standalone pairwise_distances_c time; the iVAT
                             transform time is the remainder of the core.

At n = 20000 a float64 distance matrix is ~3.2 GB (float32 ~1.6 GB), so this
also exercises the memory behaviour the roadmap calls the scaling wall.

Run:  python -m experiments.conivat_cython_scaling
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering import compute_conivat, expand_constraints  # noqa: E402
from tribbleclustering import generate_constraints_from_labels  # noqa: E402
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
SIZES = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000]
N_CONSTRAINTS = 30
SEED = 7


def _repeats(n: int) -> int:
    """Fewer repeats at large n (each call is seconds and deterministic)."""
    if n <= 4000:
        return 3
    if n <= 12000:
        return 2
    return 2


def _time(fn, *args, repeats: int = 3) -> float:
    """Best-of-`repeats` wall time in milliseconds (kernels already warmed)."""
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def _core_f32(X32, ml_expanded):
    """The compiled ConiVAT core on float32 data (distances + impose 0 + iVAT)."""
    D = pairwise_distances_c(X32)
    for i, j in ml_expanded:
        D[i, j] = 0.0
        D[j, i] = 0.0
    return compute_ivat_c(D, inplace=True)


def run() -> dict:
    if not HAS_COMPILED:
        raise SystemExit(
            "The compiled pcvat extension is not built. Build it with "
            "`python setup.py build_ext --inplace` and re-run."
        )

    # Warm the compiled kernels (f32 + f64) so first-call cost is excluded.
    Xw, yw = make_blobs(64)
    compute_conivat(Xw, labels=yw, random_state=SEED, backend="cython")
    _core_f32(np.ascontiguousarray(Xw.astype(np.float32)), [])

    core64, full64, core32, dist64 = [], [], [], []
    hdr = (
        f"{'n':>6} {'core_f64':>10} {'full_f64':>10} "
        f"{'core_f32':>10} {'dist_f64':>10} {'transform':>10}"
    )
    print(hdr)
    for n in SIZES:
        X, y = make_blobs(n)
        X32 = np.ascontiguousarray(X.astype(np.float32))
        ml, cl = generate_constraints_from_labels(y, N_CONSTRAINTS, random_state=SEED)
        ml_expanded, _ = expand_constraints(ml, cl, n)
        rep = _repeats(n)

        def _c64():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=False,
                inplace=True,
                backend="cython",
            )

        def _f64():
            return compute_conivat(
                X,
                must_link=ml,
                cannot_link=cl,
                metric_learning=True,
                inplace=True,
                backend="cython",
            )

        def _c32():
            return _core_f32(X32, ml_expanded)

        def _d64():
            return pairwise_distances_c(np.ascontiguousarray(X))

        t_c64 = _time(_c64, repeats=rep)
        t_f64 = _time(_f64, repeats=rep)
        t_c32 = _time(_c32, repeats=rep)
        t_d64 = _time(_d64, repeats=rep)
        core64.append(t_c64)
        full64.append(t_f64)
        core32.append(t_c32)
        dist64.append(t_d64)
        print(
            f"{n:>6} {t_c64:>10.1f} {t_f64:>10.1f} {t_c32:>10.1f} "
            f"{t_d64:>10.1f} {max(t_c64 - t_d64, 0.0):>10.1f}"
        )

    _plot(core64, full64, core32, dist64)
    return dict(
        sizes=SIZES,
        core_f64_ms=core64,
        full_f64_ms=full64,
        core_f32_ms=core32,
        dist_f64_ms=dist64,
    )


def _plot(core64, full64, core32, dist64) -> Path:
    sizes = np.array(SIZES, dtype=float)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: total wall time, log-log, with an O(n^2) reference.
    ax.plot(SIZES, full64, "v-", color="tab:green",
            label="ConiVAT full, f64 (+MMC)")
    ax.plot(SIZES, core64, "s-", color="tab:blue", label="ConiVAT core, f64")
    ax.plot(SIZES, core32, "o-", color="tab:orange", label="ConiVAT core, f32")
    ref = sizes**2
    ref = ref / ref[-1] * core64[-1]
    ax.plot(SIZES, ref, "k--", alpha=0.5, label=r"$O(n^2)$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("wall time (ms)")
    ax.set_title("Compiled ConiVAT performance, n up to 20000")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Right: f64 stage split — distances vs iVAT transform (stacked, absolute).
    dist = np.array(dist64)
    transform = np.maximum(np.array(core64) - dist, 0.0)
    width = 0.4
    x = np.arange(len(SIZES))
    ax2.bar(x, dist / 1e3, width, label="pairwise distances", color="tab:purple")
    ax2.bar(x, transform / 1e3, width, bottom=dist / 1e3,
            label="iVAT transform", color="tab:cyan")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIZES, rotation=45, ha="right")
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("wall time (s)")
    ax2.set_title("Where the f64 core spends its time")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "conivat_cython_scaling.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    print("ConiVAT Cython performance test (n up to 20000)")
    print("===============================================")
    run()
