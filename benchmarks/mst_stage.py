"""MST/VAT-order stage alone, given the distance matrix (median of reps).

Complements ``mst_pareto`` (end to end from points), whose matrix-path
timings are dominated by building the distance matrix.

    python -m benchmarks.mst_stage --sizes 8000 16000 --reps 3
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from benchmarks import rust_dlmst as rs
from benchmarks.scale_bench import make_blobs
from tribbleclustering.cdlmst import vat_order_dlmst_c
from tribbleclustering.pcvat import pairwise_distances_c, vat_prim_mst_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 16000])
    ap.add_argument("--dtypes", nargs="+", default=["f64", "f32"])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    fns = {
        "prim_c": vat_prim_mst_c,
        "prim_rust": rs.prim_order,
        "dlmst_c": vat_order_dlmst_c,
        "dlmst_rust": rs.vat_order,
        "dlmst_c_1t": lambda R: vat_order_dlmst_c(R, nthreads=1),
        "dlmst_rust_1t": lambda R: rs.vat_order(R, nthreads=1),
    }
    print("| n | dtype | " + " | ".join(fns) + " |")
    print("|---" * (len(fns) + 2) + "|")
    for n in a.sizes:
        for dt in a.dtypes:
            X = make_blobs(n, 10, 8, np.float32 if dt == "f32" else np.float64)
            R = pairwise_distances_c(X)
            ref = vat_prim_mst_c(R)[0]
            cells = []
            for f in fns.values():
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    p = f(R)[0]
                    ts.append(time.perf_counter() - t0)
                assert (p == ref).all()
                cells.append(f"{statistics.median(ts) * 1e3:.0f} ms")
            print(f"| {n} | {dt} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
