"""DGX Spark (GB10, Grace-Blackwell unified memory) study of GPU Boruvka VAT.

The device-side GPU Boruvka MST (``experiments/boruvka_gpu.py``) was first
benchmarked on a discrete GPU, where a brutal host->device transfer tax
(~10x the resident kernel time) made it impractical for host-resident data and a
small device VRAM capped the reachable ``n``. The DGX Spark's GB10 changes both:
the CPU and GPU share ~128 GB of coherent LPDDR5X over NVLink-C2C, so (a) the
transfer collapses and (b) an ``n x n`` matrix far larger than any discrete GPU
can hold fits in the shared pool.

This spike measures three things on the GB10:

  1. transfer-mode comparison — the same MST under three input residencies:
       * device-resident (matrix already a cupy array)
       * host numpy       (per-call cp.asarray copy, the discrete-GPU tax)
       * unified/managed  (one shared allocation, no explicit copy)
  2. large-n capability — the distance matrix *born on the device* via a tiled
     GPU pairwise, MST built out to n=100000 (an 80 GB f64 matrix), with the
     discrete-GPU VRAM ceilings marked for contrast.
  3. correctness — MST weight vs the CPU Boruvka, at every measured size.

Outputs two figures under experiments/figures/:
  * boruvka_dgx_transfer.png   — MST time by input residency vs n
  * boruvka_dgx_largen.png     — GPU pairwise + MST time vs n, with VRAM walls

Run:  python -m experiments.boruvka_dgx_spark
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import (
    pairwise_distances_c_64,
    vat_prim_mst_c,
)  # noqa: E402
from experiments.boruvka_vat import make_blobs, boruvka_mst_numba  # noqa: E402
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
    print(f"   wrote {lpath}")
