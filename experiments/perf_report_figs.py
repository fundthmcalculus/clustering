"""Render the performance figures that were previously only tables, for the
consolidated performance-report.md.

These plot the *documented, previously-measured* numbers (from the per-experiment
findings / PR bodies, taken earlier under clean thermal conditions and with the
correct methodology — e.g. GPU pairwise timings INCLUDE the device->host copy of
the result). They are constants here so the consolidated report stays consistent
with the PRs and white-paper rather than re-measuring under end-of-session
thermal throttling. Source is cited per figure.

    python -m experiments.perf_report_figs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"


# 1. CPU baseline scaling — corrected subprocess-isolated baseline (PR #16), f64.
def cpu_baseline():
    n = [4000, 16000, 32000]
    pw = [32.8, 426.6, 1916.4]
    vat = [59.4, 748.5, 3287.6]
    ivat = [113.0, 1502.4, 6966.1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(n, pw, "o-", label="pairwise distances")
    ax.plot(n, vat, "s-", label="VAT (MST + gather)")
    ax.plot(n, ivat, "^-", label="iVAT (full)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (samples)"); ax.set_ylabel("time (ms)")
    ax.set_title("CPU baseline — exact VAT/iVAT (float64, 32-core Intel)\n"
                 "subprocess-isolated benchmark (PR #16)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "cpu_baseline_scaling.png", dpi=120)
    plt.close(fig); print("wrote cpu_baseline_scaling.png")


# 2. Memory reduction — analytic (exact), 64 GB wall.
def memory_reduction():
    n = np.linspace(2000, 100000, 300)
    def gb(m): return m * n * n * 8 / 1e9
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, lab in [(3, "original iVAT (3 matrices)"),
                   (2, "in-place over VAT (2) — PR #17"),
                   (1, "in-place permutation (1) — PR #18")]:
        ax.plot(n, gb(m), label=lab)
    ax.axhline(64, color="red", ls="--", lw=1.2, label="64 GB RAM")
    for m, col in [(3, "C0"), (2, "C1"), (1, "C2")]:
        nmax = np.sqrt(64e9 / (m * 8))
        ax.axvline(nmax, color=col, ls=":", lw=0.9)
        ax.text(nmax, 3, f"n≈{nmax/1000:.0f}k", color=col, fontsize=8,
                rotation=90, va="bottom", ha="right")
    ax.set_xlabel("n (samples)"); ax.set_ylabel("iVAT peak memory (GB, float64)")
    ax.set_ylim(0, 100)
    ax.set_title("Memory reduction lifts the feasible n\n"
                 "(float64 iVAT peak vs the 64 GB wall — analytic)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "memory_reduction.png", dpi=120)
    plt.close(fig); print("wrote memory_reduction.png")


# 3. GPU FCM — documented (benchmarks/gpu_fcm.md), m=2, k=10, d=20.
def gpu_fcm():
    n = [50000, 200000, 500000]
    cpu = [1480, 4759, 15933]
    gpu = [46, 108, 286]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(n, cpu, "o-", label="CPU FCM (NumPy/BLAS, 32 cores)")
    ax.plot(n, gpu, "v-", label="GPU FCM (CuPy, data-resident)")
    ax.set_xscale("log"); ax.set_yscale("log")
    for x, c, g in zip(n, cpu, gpu):
        ax.annotate(f"{c/g:.0f}x", (x, g), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=10, color="tab:green")
    ax.set_xlabel("n (samples)"); ax.set_ylabel("fit time (ms)")
    ax.set_title("GPU Fuzzy-C-Means — 30–56× (data-resident iteration, k=10, d=20)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "gpu_fcm_speedup.png", dpi=120)
    plt.close(fig); print("wrote gpu_fcm_speedup.png")


# 4. GPU pairwise regime — documented (benchmarks/gpu_pairwise.md), n=16000.
#    Speedup = GPU/CPU wall-clock INCLUDING device->host copy of the n x n result.
def gpu_pairwise_regime():
    d = [10, 50, 200, 784]
    f64 = [0.34, 0.51, 1.03, 0.50]
    f32 = [0.53, 0.82, 1.31, 1.46]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(d, f64, "o-", label="float64")
    ax.plot(d, f32, "s-", label="float32 (double-accum)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="parity (CPU = GPU)")
    ax.annotate("float32 fast-accum reaches 2.47x at d=200 (n=32000)",
                (200, 1.31), textcoords="offset points", xytext=(-30, 24),
                fontsize=8, color="tab:orange",
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.8))
    ax.set_xlabel("feature dimension d"); ax.set_ylabel("GPU speedup vs CPU (x)")
    ax.set_title("GPU pairwise distances — loses at low d / float64,\n"
                 "wins only at higher d / float32 (n=16000, incl. D->H transfer)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "gpu_pairwise_regime.png", dpi=120)
    plt.close(fig); print("wrote gpu_pairwise_regime.png")


if __name__ == "__main__":
    print("Rendering performance-report figures (from documented measurements)")
    FIG_DIR.mkdir(exist_ok=True)
    cpu_baseline()
    memory_reduction()
    gpu_fcm()
    gpu_pairwise_regime()
