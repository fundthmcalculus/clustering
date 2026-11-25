"""Runtime-vs-memory Pareto benchmark: dense Prim vs certificate-and-filter MST.

Compares every way the package can produce a VAT order or an IVAT image,
end to end *from points* -- so methods that need the ``n x n`` distance matrix
pay for building and holding it, and methods that work from points do not.

Usage:
    python -m benchmarks.mst_pareto --quick
    python -m benchmarks.mst_pareto --sizes 2000 4000 8000 16000 24000

Each ``(method, n, dtype)`` runs in a fresh subprocess (see README.md: peak
RSS is a process-lifetime high-water mark). Results go to
``benchmarks/baselines/mst_pareto_<host>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# name -> (task, needs_matrix, python_only, env)
# python_only: pure-numpy reference; skipped above --py-max-n and for f32.
METHODS: dict[str, tuple[str, bool, bool, dict]] = {
    "prim_numba": ("order", True, True, {}),
    "prim_c": ("order", True, False, {}),
    "prim_c_points": ("order", False, False, {}),
    "dlmst_py": ("order", True, True, {}),
    "dlmst_py_points": ("order", False, True, {}),
    "dlmst_c": ("order", True, False, {}),
    "dlmst_c_points": ("order", False, False, {}),
    "dlmst_c_1t": ("order", True, False, {"OMP_NUM_THREADS": "1"}),
    "dlmst_c_points_1t": ("order", False, False, {"OMP_NUM_THREADS": "1"}),
    "prim_rust": ("order", True, False, {}),
    "dlmst_rust": ("order", True, False, {}),
    "dlmst_rust_points": ("order", False, False, {}),
    "ivat_numba": ("ivat", True, True, {}),
    "ivat_c": ("ivat", True, False, {}),
    "ivat_c_inplace": ("ivat", True, False, {}),
    "ivat_dlmst_py_points": ("ivat", False, True, {}),
    "ivat_dlmst_c": ("ivat", True, False, {}),
    "ivat_dlmst_c_points": ("ivat", False, False, {}),
    "ivat_dlmst_rust": ("ivat", True, False, {}),
    "ivat_dlmst_rust_points": ("ivat", False, False, {}),
}


def _fn(method: str):
    from benchmarks import rust_dlmst as rs
    from tribbleclustering import dlmst, pvat
    from tribbleclustering.cdlmst import compute_ivat_dlmst_c, vat_order_dlmst_c
    from tribbleclustering.pcvat import (
        compute_ivat_c,
        pairwise_distances_c,
        vat_prim_mst_c,
        vat_prim_mst_seq_c,
    )

    pd = pairwise_distances_c
    return {
        "prim_numba": lambda X: pvat.vat_prim_mst(pd(X)),
        "prim_c": lambda X: vat_prim_mst_c(pd(X)),
        "prim_c_points": lambda X: vat_prim_mst_seq_c(X),
        "dlmst_py": lambda X: dlmst.vat_order_py(pd(X)),
        "dlmst_py_points": lambda X: dlmst.vat_order_py(X, from_points=True),
        "dlmst_c": lambda X: vat_order_dlmst_c(pd(X)),
        "dlmst_c_points": lambda X: vat_order_dlmst_c(X, from_points=True),
        "dlmst_c_1t": lambda X: vat_order_dlmst_c(pd(X)),
        "dlmst_c_points_1t": lambda X: vat_order_dlmst_c(X, from_points=True),
        "ivat_numba": lambda X: pvat.compute_ivat(pd(X)),
        "ivat_c": lambda X: compute_ivat_c(pd(X)),
        "ivat_c_inplace": lambda X: compute_ivat_c(pd(X), inplace=True),
        "ivat_dlmst_py_points": lambda X: dlmst.compute_ivat_dlmst_py(
            X, from_points=True
        ),
        "ivat_dlmst_c": lambda X: compute_ivat_dlmst_c(pd(X)),
        "ivat_dlmst_c_points": lambda X: compute_ivat_dlmst_c(X, from_points=True),
        "prim_rust": lambda X: rs.prim_order(pd(X)),
        "dlmst_rust": lambda X: rs.vat_order(pd(X)),
        "dlmst_rust_points": lambda X: rs.vat_order(X, from_points=True),
        "ivat_dlmst_rust": lambda X: rs.compute_ivat(pd(X)),
        "ivat_dlmst_rust_points": lambda X: rs.compute_ivat(X, from_points=True),
    }[method]


def _worker(method: str, n: int, d: int, dt: str, n_clusters: int) -> None:
    import gc

    from benchmarks.memprobe import measure_peak_rss
    from benchmarks.scale_bench import make_blobs

    dtype = np.float32 if dt == "f32" else np.float64
    fn = _fn(method)
    fn(make_blobs(64, d, 2, dtype, seed=1))  # JIT / thread-pool warm-up
    X = make_blobs(n, d, n_clusters, dtype)
    gc.collect()
    with measure_peak_rss() as m:
        t0 = time.perf_counter()
        fn(X)
        ms = (time.perf_counter() - t0) * 1e3
    print("RESULT " + json.dumps({"time_ms": ms, "delta_gb": m.delta_gb}))


def _run(method, n, d, dt, n_clusters, timeout):
    env = dict(os.environ, **METHODS[method][3])
    cmd = [sys.executable, "-m", "benchmarks.mst_pareto", "--worker", method]
    cmd += [str(n), str(d), dt, str(n_clusters)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    return {"error": (proc.stderr or proc.stdout or "no RESULT").strip()[-300:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=5, metavar=("METHOD", "N", "D", "DT", "K"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--dtypes", nargs="+", default=["f64", "f32"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--py-max-n", type=int, default=16000)
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.worker:
        m, n, d, dt, k = a.worker
        _worker(m, int(n), int(d), dt, int(k))
        return
    if a.quick:
        a.sizes = [500, 1500]

    if any(m.endswith("rust") or "_rust_" in m for m in a.methods):
        from benchmarks import rust_dlmst

        rust_dlmst.build()  # once, before any timed subprocess
    host = socket.gethostname()
    out = Path(a.out or Path(__file__).parent / "baselines" / f"mst_pareto_{host}.json")
    rows = []
    for n in a.sizes:
        for dt in a.dtypes:
            mat_gb = n * n * (4 if dt == "f32" else 8) / 1e9
            for m in a.methods:
                task, needs_mat, py_only, _ = METHODS[m]
                if py_only and (dt == "f32" or n > a.py_max_n):
                    continue
                need = mat_gb * ((2 if needs_mat else 0) + (1 if task == "ivat" else 0))
                if need > a.max_gb:
                    r = {"skipped": f"needs ~{need:.1f} GB"}
                else:
                    r = _run(m, n, a.d, dt, a.clusters, a.timeout)
                row = {"method": m, "task": task, "n": n, "dtype": dt, **r}
                rows.append(row)
                print(json.dumps(row), flush=True)
    meta = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host,
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "d": a.d,
        "clusters": a.clusters,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
