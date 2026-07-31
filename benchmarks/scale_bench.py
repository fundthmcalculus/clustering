"""Scale benchmark for the VAT/IVAT pipeline.

Measures wall-clock and peak RSS for each O(n^2) stage across dataset sizes,
for float32 and float64, and records a JSON baseline plus a markdown report.

Usage:
    python -m benchmarks.scale_bench --sizes 2000 8000 16000 32000
    python -m benchmarks.scale_bench --sizes 50000 --dtypes f64 --max-gb 55
    python -m benchmarks.scale_bench --quick        # tiny sanity sweep

Each stage is measured in a *fresh subprocess*. This is essential for accurate
peak-memory numbers: Windows keeps a process-lifetime working-set high-water
mark (no reset API) and does not promptly return freed pages to the OS, so
sequential large allocations in one process contaminate each other's peak. A
clean process per measurement gives a true, isolated peak.

Every stage estimates its peak memory *before* running and is skipped (not
crashed) if it would exceed --max-gb, so pushing toward the 64 GB envelope is
safe. Correctness is verified against reference implementations at small n.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_BYTES = {"f32": 4, "f64": 8}
_STAGES = ("pairwise", "vat", "ivat", "ivat_inplace")
# Memory cost as a multiple of one n x n matrix, per stage.
#   pairwise      : the output distance matrix                    -> 1
#   vat           : input + permuted output                       -> 2
#   ivat          : input + shared VAT/IVAT buffer (built in place)-> 2
#   ivat_inplace  : input consumed in place (P.M.Pt + IVAT), O(n)  -> 1
#                   scratch only; no second n x n buffer
_STAGE_MATRICES = {"pairwise": 1, "vat": 2, "ivat": 2, "ivat_inplace": 1}
_STAGE_HDR = {
    "pairwise": "pairwise",
    "vat": "VAT(2buf)",
    "ivat": "IVAT(2buf)",
    "ivat_inplace": "IVAT-inplace(1buf)",
}


def make_blobs(n: int, d: int, n_clusters: int, dtype, seed: int = 0) -> np.ndarray:
    """Gaussian blobs — gives the distance matrix real cluster structure so the
    MST/VAT ordering is meaningful (not the degenerate uniform-noise case)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-50.0, 50.0, size=(n_clusters, d))
    sizes = np.full(n_clusters, n // n_clusters)
    sizes[: n - int(sizes.sum())] += 1
    parts = [
        rng.standard_normal((s, d)) * 2.0 + centers[k] for k, s in enumerate(sizes)
    ]
    X = np.vstack(parts).astype(dtype)
    rng.shuffle(X)
    return np.ascontiguousarray(X)


# ---------------------------------------------------------------------------
# Worker: runs ONE stage in a clean process, prints a JSON line.
# ---------------------------------------------------------------------------
def _worker(n: int, d: int, dt: str, n_clusters: int, stage: str) -> None:
    import gc

    from benchmarks.memprobe import measure_peak_rss
    from tribbleclustering.pcvat import (
        compute_ivat_c,
        compute_ivat_c_32,
        compute_ivat_c_64,
        compute_vat_c_32,
        compute_vat_c_64,
        pairwise_distances_c_32,
        pairwise_distances_c_64,
    )

    dtype = np.float32 if dt == "f32" else np.float64
    pd_fn = pairwise_distances_c_32 if dt == "f32" else pairwise_distances_c_64
    vat_fn = compute_vat_c_32 if dt == "f32" else compute_vat_c_64
    ivat_fn = compute_ivat_c_32 if dt == "f32" else compute_ivat_c_64

    X = make_blobs(n, d, n_clusters, dtype)
    if stage == "pairwise":
        fn, args = pd_fn, (X,)
    else:
        D = pd_fn(X)  # untimed prerequisite
        if stage == "vat":
            fn, args = vat_fn, (D,)
        elif stage == "ivat":
            fn, args = ivat_fn, (D,)
        elif stage == "ivat_inplace":
            # consumes D in place; measured once, so destroying it is fine
            fn, args = (lambda M: compute_ivat_c(M, inplace=True)), (D,)
        else:
            raise ValueError(f"unknown stage {stage}")

    # warm the thread pool / branch predictors with a cheap call is unnecessary
    # here: each subprocess measures a single large call where startup is noise.
    gc.collect()
    with measure_peak_rss() as m:
        t0 = time.perf_counter()
        fn(*args)
        dt_ms = (time.perf_counter() - t0) * 1e3
    print(
        "RESULT "
        + json.dumps({"time_ms": dt_ms, "peak_gb": m.peak_gb, "delta_gb": m.delta_gb})
    )


def _run_worker_subprocess(n, d, dt, n_clusters, stage, timeout=1800):
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.scale_bench",
        "--worker",
        str(n),
        str(d),
        dt,
        str(n_clusters),
        stage,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    return {"error": (proc.stderr or proc.stdout or "no RESULT").strip()[-400:]}


# ---------------------------------------------------------------------------
# Correctness (runs in-process; small n).
# ---------------------------------------------------------------------------
def verify_correctness(n: int = 400, d: int = 6, n_clusters: int = 5) -> dict:
    from scipy.spatial.distance import pdist, squareform

    from tribbleclustering.pcvat import compute_ivat_c_64, pairwise_distances_c_64
    from tribbleclustering.pvat import compute_ivat as ref_ivat

    out: dict = {}
    X = make_blobs(n, d, n_clusters, np.float64, seed=7)
    D_c = pairwise_distances_c_64(X)
    D_ref = squareform(pdist(X, metric="euclidean"))
    out["pairwise_max_abs_err"] = float(np.max(np.abs(D_c - D_ref)))

    ivat_c, _, _ = compute_ivat_c_64(D_ref)
    ivat_ref, _, _ = ref_ivat(D_ref, inplace=False)
    out["ivat_sorted_max_abs_err"] = float(
        np.max(np.abs(np.sort(ivat_c.ravel()) - np.sort(ivat_ref.ravel())))
    )
    out["ivat_symmetric"] = bool(np.allclose(ivat_c, ivat_c.T))
    out["passed"] = (
        out["pairwise_max_abs_err"] < 1e-9
        and out["ivat_sorted_max_abs_err"] < 1e-9
        and out["ivat_symmetric"]
    )
    return out


def _fmt(stage_res: dict | None) -> str:
    if stage_res is None:
        return "skip(mem)"
    if "error" in stage_res:
        return f"ERR:{stage_res['error'][:18]}"
    return f"{stage_res['time_ms']:8.1f}ms /{stage_res['peak_gb']:6.2f}GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worker",
        nargs=5,
        default=None,
        metavar=("N", "D", "DT", "NCLUST", "STAGE"),
        help="internal: measure one stage and print JSON",
    )
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 8000, 16000, 32000])
    ap.add_argument(
        "--dtypes", nargs="+", default=["f32", "f64"], choices=["f32", "f64"]
    )
    ap.add_argument("--stages", nargs="+", default=list(_STAGES), choices=list(_STAGES))
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--max-gb", type=float, default=55.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.worker is not None:
        n, d, dt, nclust, stage = args.worker
        _worker(int(n), int(d), dt, int(nclust), stage)
        return

    if args.quick:
        args.sizes = [500, 2000]

    print(f"host={socket.gethostname()}  cpu={platform.processor()}")
    print(
        f"sizes={args.sizes} dtypes={args.dtypes} d={args.d} "
        f"n_clusters={args.n_clusters} max_gb={args.max_gb}"
    )

    verify = None
    if not args.no_verify:
        print("\nverifying correctness vs scipy/numba reference (n=400)...")
        verify = verify_correctness()
        print(f"  {verify}")
        if not verify["passed"]:
            print("  !! correctness check FAILED — results below are suspect")

    results = []
    stages = [s for s in _STAGES if s in args.stages]
    colw = 22
    hdr = "{:>7} {:>4} {:>6}  ".format("n", "dt", "1mat") + "".join(
        f"{_STAGE_HDR[s]:>{colw}}" for s in stages
    )
    print("\n" + hdr)
    for n in args.sizes:
        for dt in args.dtypes:
            b = _BYTES[dt]
            mat_gb = n * n * b / 1e9
            row = {"n": n, "d": args.d, "dtype": dt, "matrix_gb": mat_gb, "stages": {}}
            cells = []
            for stage in stages:
                need = _STAGE_MATRICES[stage] * mat_gb
                if need > args.max_gb:
                    row["stages"][stage] = None
                    cells.append("skip(mem)")
                    continue
                sr = _run_worker_subprocess(n, args.d, dt, args.n_clusters, stage)
                row["stages"][stage] = sr
                cells.append(_fmt(sr))
            results.append(row)
            print(
                "{:>7} {:>4} {:>6.1f}  ".format(n, dt, mat_gb)
                + "".join(f"{c:>{colw}}" for c in cells)
            )

    payload = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "d": args.d,
            "n_clusters": args.n_clusters,
            "max_gb": args.max_gb,
        },
        "correctness": verify,
        "results": results,
    }
    out_dir = Path(__file__).parent / "baselines"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}_{socket.gethostname()}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
