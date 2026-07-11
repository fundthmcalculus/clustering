"""DGX Spark (GB10, Grace-Blackwell unified memory) study of GPU Boruvka VAT.

The device-side GPU Boruvka MST (``experiments/boruvka_gpu.py``) was first
benchmarked on a discrete GPU, where a brutal host->device transfer tax
(~10x the resident kernel time) made it impractical for host-resident data and a
small device VRAM capped the reachable ``n``. The DGX Spark's GB10 changes both:
the CPU and GPU share ~128 GB of coherent LPDDR5X over NVLink-C2C, so (a) the
transfer collapses and (b) an ``n x n`` matrix far larger than any discrete GPU
can hold fits in the shared pool.

This spike measures four things on the GB10:

  1. transfer-mode comparison — the same MST under three input residencies:
       * device-resident (matrix already a cupy array)
       * host numpy       (per-call cp.asarray copy, the discrete-GPU tax)
       * unified/managed  (one shared allocation, no explicit copy)
  2. large-n capability — the distance matrix *born on the device* via a tiled
     GPU pairwise, MST built out to n=100000 (an 80 GB f64 matrix), with the
     discrete-GPU VRAM ceilings marked for contrast.
  3. precision study — MST build time and accuracy (MST weight, VAT order match)
     with the matrix stored at f64 / f32 / f16 (~2x faster per precision step;
     f32 exact, f16 near-identical).
  4. capacity study — equal ~80 GB matrix, n growing as the element narrows
     (f64 100k -> f32 141k -> f16 200k), showing the reachable n scale with dtype.

Correctness (MST weight vs the CPU Boruvka) is checked inline in (1) and (3).

Outputs four figures under experiments/figures/:
  * boruvka_dgx_transfer.png   — MST time by input residency vs n
  * boruvka_dgx_largen.png     — GPU pairwise + MST time vs n, with VRAM walls
  * boruvka_dgx_precision.png  — MST time + accuracy at f64/f32/f16
  * (capacity study prints a table; no figure)

Run:  python -m experiments.boruvka_dgx_spark
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import (  # noqa: E402
    pairwise_distances_c_64,
    vat_prim_mst_c,
    compute_ivat_c,
)
from experiments.boruvka_vat import (  # noqa: E402
    make_blobs,
    boruvka_mst_numba,
    vat_order_from_mst,
)
from experiments.boruvka_gpu import (  # noqa: E402
    boruvka_mst_gpu,
    pairwise_distances_gpu,
    as_unified,
    _HAS_CUPY,
)

if _HAS_CUPY:
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


def _sync():
    cp.cuda.Stream.null.synchronize()


def _time_cpu(fn, *a, rep=2, warm=1):
    for _ in range(warm):
        fn(*a)
    best = np.inf
    for _ in range(rep):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def _time_gpu(fn, arg, rep=3):
    _sync()
    fn(arg)
    _sync()
    best = np.inf
    for _ in range(rep):
        _sync()
        t = time.perf_counter()
        fn(arg)
        _sync()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def _mst_weight(D, mu, mv):
    return float(sum(D[a, b] for a, b in zip(mu.tolist(), mv.tolist())))


# ---------------------------------------------------------------------------
# 1) transfer-mode comparison
# ---------------------------------------------------------------------------
def transfer_figure():
    sizes = [2000, 4000, 8000, 16000, 32000]
    t_prim, t_numba = [], []
    t_resident, t_host, t_unified = [], [], []
    for n in sizes:
        X = make_blobs(n, 10, 25, seed=7)
        D = pairwise_distances_c_64(X)
        t_prim.append(_time_cpu(lambda M: vat_prim_mst_c(M), D))
        t_numba.append(_time_cpu(lambda M: boruvka_mst_numba(M), D))

        Dg = cp.asarray(D)
        _sync()
        Du = as_unified(D)
        t_resident.append(_time_gpu(boruvka_mst_gpu, Dg))
        t_host.append(_time_gpu(boruvka_mst_gpu, D))  # per-call host->device copy
        t_unified.append(_time_gpu(boruvka_mst_gpu, Du))

        # correctness: MST weight parity with CPU Boruvka
        mu, mv = boruvka_mst_gpu(Du)
        nu, nv = boruvka_mst_numba(D)
        ok = np.isclose(_mst_weight(D, mu, mv), _mst_weight(D, nu, nv))
        print(
            f"  n={n:6d}: prim {t_prim[-1]:7.1f}  numba {t_numba[-1]:7.1f}  "
            f"gpu-resident {t_resident[-1]:7.1f}  gpu-host+copy {t_host[-1]:7.1f}  "
            f"gpu-unified {t_unified[-1]:7.1f}  (ms)  MST_ok={bool(ok)}"
        )
        del Dg, Du
        cp.get_default_memory_pool().free_all_blocks()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(sizes, t_prim, "o-", color="0.4", label="serial Prim (C/OpenMP)")
    ax.plot(sizes, t_numba, "s-", color="0.6", label="Boruvka (Numba, CPU)")
    ax.plot(
        sizes,
        t_host,
        "^--",
        color="tab:red",
        alpha=0.6,
        label="GPU Boruvka (host numpy, per-call copy)",
    )
    ax.plot(
        sizes,
        t_unified,
        "D-",
        color="tab:orange",
        label="GPU Boruvka (unified/managed, no copy)",
    )
    ax.plot(
        sizes, t_resident, "^-", color="tab:blue", label="GPU Boruvka (device-resident)"
    )
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("MST build time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "GB10 unified memory: the host->device transfer tax collapses\n"
        "(unified/managed input tracks device-resident; explicit copy is the old wall)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "boruvka_dgx_transfer.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, dict(
        sizes=sizes,
        prim=t_prim,
        numba=t_numba,
        gpu_resident=t_resident,
        gpu_host_copy=t_host,
        gpu_unified=t_unified,
    )


# ---------------------------------------------------------------------------
# 2) large-n capability (matrix born on the device)
# ---------------------------------------------------------------------------
def large_n_figure():
    sizes = [16000, 32000, 65536, 100000]
    t_pw, t_mst, gib = [], [], []
    for n in sizes:
        X = make_blobs(n, 10, 25, seed=7)
        t = time.perf_counter()
        D = pairwise_distances_gpu(X)  # born on device, single unified allocation
        _sync()
        t_pw.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        mu, mv = boruvka_mst_gpu(D)
        _sync()
        t_mst.append((time.perf_counter() - t) * 1e3)
        gib.append(D.nbytes / 1e9)
        edges = len(mu)
        print(
            f"  n={n:7d}: D={gib[-1]:6.1f}GB  pairwise(gpu) {t_pw[-1]:8.0f}ms  "
            f"MST(gpu) {t_mst[-1]:8.0f}ms  edges={edges}/{n - 1}"
        )
        del D
        cp.get_default_memory_pool().free_all_blocks()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(sizes, t_mst, "^-", color="tab:blue", label="GPU Boruvka MST")
    ax.plot(
        sizes,
        t_pw,
        "o--",
        color="tab:green",
        alpha=0.7,
        label="GPU pairwise (tiled, one-time build)",
    )
    # discrete-GPU f64 n x n VRAM ceilings: n_max = sqrt(bytes / 8)
    for vram_gb, name in [(24, "24 GB"), (48, "48 GB"), (80, "80 GB")]:
        n_max = (vram_gb * 1e9 / 8) ** 0.5
        ax.axvline(n_max, color="0.6", ls=":", lw=1)
        ax.text(
            n_max,
            ax.get_ylim()[0] * 1.4,
            f"{name}\nVRAM wall",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.4",
        )
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "GB10 128 GB unified pool: dense VAT MST past every discrete-GPU VRAM wall\n"
        "(n=100000 is an 80 GB f64 distance matrix; discrete GPUs OOM long before)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "boruvka_dgx_largen.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, dict(sizes=sizes, pairwise=t_pw, mst=t_mst, d_gb=gib)


# ---------------------------------------------------------------------------
# 3) precision study — MST time and accuracy at f64 / f32 / f16
# ---------------------------------------------------------------------------
_DTYPES = [("f64", np.float64), ("f32", np.float32), ("f16", np.float16)]


def precision_figure():
    """Same distances stored at f64/f32/f16: MST build time and accuracy vs f64.

    Accuracy is judged on the *reference f64 distances*: reldiff of the MST weight
    (edges chosen by the low-precision MST, re-measured at f64) and the VAT
    order match against serial Prim (the tree traversal itself always reads the
    exact f64 matrix, so this isolates the effect of a lower-precision MST build).
    """
    sizes = [4000, 8000, 16000, 32000]
    times = {tag: [] for tag, _ in _DTYPES}
    reldiff = {tag: [] for tag, _ in _DTYPES}
    ordm = {tag: [] for tag, _ in _DTYPES}
    for n in sizes:
        X = make_blobs(n, 10, 25, seed=7)
        Dc = pairwise_distances_c_64(X)
        nu, nv = boruvka_mst_numba(Dc)
        w_ref = _mst_weight(Dc, nu, nv)
        _, _, p_serial = compute_ivat_c(Dc.copy(), inplace=False)
        line = f"  n={n:6d}:"
        for tag, dt in _DTYPES:
            Du = as_unified(Dc, dtype=dt)  # same distances, downcast storage
            times[tag].append(_time_gpu(boruvka_mst_gpu, Du))
            mu, mv = boruvka_mst_gpu(Du)
            w_true = _mst_weight(Dc, mu, mv)  # re-measure chosen edges at f64
            reldiff[tag].append(abs(w_true - w_ref) / w_ref)
            order = vat_order_from_mst(Dc, mu, mv)
            ordm[tag].append(float(np.mean(order == p_serial)))
            line += (
                f"  {tag} {times[tag][-1]:6.1f}ms"
                f"(rd={reldiff[tag][-1]:.1e},om={ordm[tag][-1]:.4f})"
            )
            del Du
            cp.get_default_memory_pool().free_all_blocks()
        print(line)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"f64": "tab:blue", "f32": "tab:orange", "f16": "tab:green"}
    for tag, _ in _DTYPES:
        ax.plot(sizes, times[tag], "o-", color=colors[tag], label=f"GPU Boruvka {tag}")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("MST build time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MST build time by storage precision")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for tag, _ in _DTYPES:
        ax2.plot(
            sizes,
            reldiff[tag],
            "s-",
            color=colors[tag],
            label=f"{tag} MST-weight reldiff",
        )
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("|MST weight − f64| / f64")
    ax2.set_xscale("log")
    ax2.set_yscale("symlog", linthresh=1e-16)
    ax2.set_title("Accuracy vs f64 (weight of chosen edges, re-measured at f64)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig.suptitle(
        "GB10 VAT MST at f64 / f32 / f16 — f32 is exact; f16 trades ~1e-6 weight "
        "error for half the memory again",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "boruvka_dgx_precision.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, dict(sizes=sizes, times=times, reldiff=reldiff, order_match=ordm)


# ---------------------------------------------------------------------------
# 4) capacity study — largest n each dtype reaches in the shared pool
# ---------------------------------------------------------------------------
def capacity_table():
    """Equal ~80 GB matrix, tripled n as precision narrows (born on device).

    f64 @ 100k, f32 @ 141k, f16 @ 200k are all ~80 GB dense matrices — showing
    that narrowing the element type buys √(bytes-ratio)× more samples at the same
    footprint, i.e. f16 reaches 2× the n of f64 (and past any discrete GPU).
    """
    cases = [
        ("f64", np.float64, 100000),
        ("f32", np.float32, 141000),
        ("f16", np.float16, 200000),
    ]
    rows = []
    for tag, dt, n in cases:
        X = make_blobs(n, 10, 25, seed=7)
        t = time.perf_counter()
        D = pairwise_distances_gpu(X, dtype=dt)  # born on device at target dtype
        _sync()
        t_pw = time.perf_counter() - t
        t = time.perf_counter()
        mu, mv = boruvka_mst_gpu(D)
        _sync()
        t_mst = time.perf_counter() - t
        rows.append((tag, n, D.nbytes / 1e9, t_pw, t_mst, len(mu)))
        print(
            f"  {tag} n={n:7d}  D={D.nbytes / 1e9:5.1f}GB  "
            f"pairwise {t_pw:6.1f}s  MST {t_mst * 1e3:7.0f}ms  edges={len(mu)}/{n - 1}"
        )
        del D
        cp.get_default_memory_pool().free_all_blocks()
    return rows


if __name__ == "__main__":
    print("Boruvka VAT on DGX Spark (GB10)\n" + "=" * 31)
    print(f"CuPy GPU available: {_HAS_CUPY}")
    if not _HAS_CUPY:
        raise SystemExit("no CUDA device — nothing to measure")
    print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"unified pool: {total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free\n")

    print("1) transfer-mode comparison (MST time by input residency)...")
    tpath, tdata = transfer_figure()
    print(f"   wrote {tpath}\n")

    print("2) large-n capability (matrix born on device)...")
    lpath, ldata = large_n_figure()
    print(f"   wrote {lpath}\n")

    print("3) precision study (MST time + accuracy at f64/f32/f16)...")
    ppath, pdata = precision_figure()
    print(f"   wrote {ppath}\n")

    print("4) capacity study (equal ~80 GB matrix, n grows as precision narrows)...")
    capacity_table()
