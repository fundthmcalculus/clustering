"""Plot ``mst_pareto`` results: time vs n, memory vs n, and the Pareto front.

python -m benchmarks.plot_mst_pareto benchmarks/baselines/mst_pareto_<host>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled), at most 8 line series per chart.
# Remaining methods (1-thread, numba) appear only as grey Pareto points.
COLORS = {
    "prim_c": "#257de0",
    "prim_c_points": "#f47230",
    "dlmst_c": "#00b894",
    "dlmst_c_points": "#f2a600",
    "prim_rust": "#ee7fac",
    "dlmst_rust": "#008a17",
    "dlmst_rust_points": "#6a4ce2",
    "dlmst_py": "#ec4647",
    "ivat_c": "#257de0",
    "ivat_c_inplace": "#f47230",
    "ivat_dlmst_c": "#00b894",
    "ivat_dlmst_c_points": "#f2a600",
    "ivat_numba": "#ee7fac",
    "ivat_dlmst_rust": "#008a17",
    "ivat_dlmst_rust_points": "#6a4ce2",
    "ivat_dlmst_py_points": "#ec4647",
}
LABEL = {
    "prim_c": "Prim (C, matrix)",
    "prim_c_points": "Prim (C, points)",
    "dlmst_c": "Cert+filter (C, matrix)",
    "dlmst_c_points": "Cert+filter (C, points)",
    "dlmst_c_1t": "Cert+filter (C, matrix, 1 thr)",
    "dlmst_c_points_1t": "Cert+filter (C, points, 1 thr)",
    "prim_numba": "Prim (numba, matrix)",
    "dlmst_py": "Cert+filter (numpy, matrix)",
    "dlmst_py_points": "Cert+filter (numpy, points)",
    "prim_rust": "Prim (Rust, matrix)",
    "dlmst_rust": "Cert+filter (Rust, matrix)",
    "dlmst_rust_points": "Cert+filter (Rust, points)",
    "ivat_c": "IVAT (C, 2 buffers)",
    "ivat_c_inplace": "IVAT (C, in place)",
    "ivat_dlmst_c": "IVAT from MST (C, matrix)",
    "ivat_dlmst_c_points": "IVAT from MST (C, points)",
    "ivat_numba": "IVAT (numba)",
    "ivat_dlmst_py_points": "IVAT from MST (numpy, points)",
    "ivat_dlmst_rust": "IVAT from MST (Rust, matrix)",
    "ivat_dlmst_rust_points": "IVAT from MST (Rust, points)",
}
INK, INK2, GRID, GREY = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#b0b0b0"


def _pareto(pts):
    """Indices of non-dominated (memory, time) points, sorted by memory."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    front, best = [], float("inf")
    for i in order:
        if pts[i][1] < best:
            front.append(i)
            best = pts[i][1]
    return front


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main(path: str, dtype: str = "f64") -> None:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data["rows"] if "time_ms" in r and r["dtype"] == dtype]
    out_dir = Path(path).parent
    for task in ("order", "ivat"):
        tr = [r for r in rows if r["task"] == task]
        methods = [m for m in COLORS if any(r["method"] == m for r in tr)]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        for m in methods:
            mr = sorted((r for r in tr if r["method"] == m), key=lambda r: r["n"])
            ns = [r["n"] for r in mr]
            kw = dict(color=COLORS[m], linewidth=2, marker="o", markersize=5)
            axes[0].plot(ns, [r["time_ms"] / 1e3 for r in mr], label=LABEL[m], **kw)
            axes[1].plot(ns, [max(r["delta_gb"], 1e-3) for r in mr], **kw)
        nmax = max(r["n"] for r in tr)
        top = [r for r in tr if r["n"] == nmax]
        pts = [(max(r["delta_gb"], 1e-3), r["time_ms"] / 1e3) for r in top]
        front = _pareto(pts)
        fp = [pts[i] for i in front]
        axes[2].plot(
            [p[0] for p in fp],
            [p[1] for p in fp],
            color=INK2,
            linewidth=1,
            linestyle="--",
            zorder=1,
            label="Pareto front",
        )
        for i, r in enumerate(top):
            on = i in front
            axes[2].scatter(
                *pts[i],
                s=90 if on else 50,
                color=COLORS.get(r["method"], GREY),
                edgecolor="white",
                linewidth=2,
                zorder=3,
            )
            axes[2].annotate(
                LABEL[r["method"]] + (" *" if on else ""),
                pts[i],
                fontsize=7,
                color=INK,
                xytext=(6, 4),
                textcoords="offset points",
            )
        name = "VAT order (MST + Prim order)" if task == "order" else "IVAT image"
        _style(axes[0], f"{name}: runtime", "n", "seconds (log)")
        _style(axes[1], "Peak working memory", "n", "GB above baseline (log)")
        _style(axes[2], f"Pareto front at n={nmax}", "GB (log)", "seconds (log)")
        for ax in axes:
            ax.set_yscale("log")
        for ax in axes[:2]:
            ax.set_xscale("log")
        axes[2].set_xscale("log")
        fig.legend(
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=9,
            labelcolor=INK,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            f"{name} from points, {dtype}, d={data['meta']['d']}, "
            f"{data['meta']['cpus']} CPUs  (* = Pareto-optimal)",
            color=INK,
            x=0.01,
            ha="left",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        f = out_dir / f"mst_pareto_{task}_{dtype}.png"
        fig.savefig(f, dpi=110, facecolor="white")
        print(f"wrote {f}")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
