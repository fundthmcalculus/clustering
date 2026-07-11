"""GPU-accelerated primitives (optional, CuPy).

Currently provides a tiled dense pairwise-distance matrix. The design targets
the scaling regime where the n x n output does not fit in the 12 GB-class VRAM
of a laptop GPU: distances are computed in row-tiles on the device and streamed
into a host (NumPy) matrix, so only one R x n tile is ever resident on the GPU.
The full matrix lives in host RAM (n=64000 float64 = 32 GB), where the CPU
VAT/IVAT engine then consumes it.

Accuracy matches the C/OpenMP kernel exactly: each distance is the direct
sqrt(sum_k (x_ik - x_jk)^2) with the squared sum accumulated in double
precision (for both float32 and float64 inputs). The gram-trick
(|x_i|^2 + |x_j|^2 - 2 x_i.x_j) is deliberately NOT used — it suffers
catastrophic cancellation for nearby points (~1e-7 error even in float64),
which would break the exact-VAT guarantee.

Everything degrades gracefully: if CuPy or a CUDA device is unavailable,
``is_available()`` returns False and callers fall back to the CPU path.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as _cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - environment dependent
    _cp = None
    _HAS_CUPY = False


def is_available() -> bool:
    """True if CuPy is importable and a CUDA device is present and usable."""
    if not _HAS_CUPY:
        return False
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


# One thread per output cell (i, j) of the current row-tile; each thread walks
# the d features once and accumulates the squared difference in double. `row0`
# offsets the tile into the full matrix. Two instantiations (float/double) are
# cached lazily.
# `ACC` is the accumulator type: `double` (default) reproduces the C/OpenMP
# kernel bit-for-bit (float64) or to ~1e-6 (float32, whose CPU path also
# accumulates in double). `float` accumulation is much faster for float32 on
# consumer GPUs whose float64 rate is a fraction of float32, at a small
# accuracy cost — selected via high_precision=False.
_KERNEL_SRC = r"""
extern "C" __global__
void {name}(const {T}* __restrict__ X, int n, int d, int R,
            int row0, long long tile_elems, {T}* __restrict__ out) {{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_elems) return;
    int i = (int)(idx / n);                        // local tile row [0, R)
    int j = (int)(idx % n);                        // column [0, n)
    const {T}* xi = X + (long long)(row0 + i) * d; // full-matrix tile row
    const {T}* xj = X + (long long)(j) * d;        // full-matrix column row
    {ACC} acc = 0;
    for (int k = 0; k < d; ++k) {{
        {ACC} diff = ({ACC})xi[k] - ({ACC})xj[k];
        acc += diff * diff;
    }}
    out[idx] = ({T})sqrt(acc);
}}
"""

_KERNELS: dict = {}


def _kernel(dtype, high_precision: bool):
    key = (np.dtype(dtype).name, high_precision)
    if key not in _KERNELS:
        name_dt = np.dtype(dtype).name
        if name_dt == "float32":
            T = "float"
        elif name_dt == "float64":
            T = "double"
        else:
            raise TypeError(f"Expected float32 or float64, got {dtype}")
        acc = "double" if high_precision else T
        name = f"dist_tile_{name_dt}_{'hp' if high_precision else 'fast'}"
        src = _KERNEL_SRC.format(name=name, T=T, ACC=acc)
        _KERNELS[key] = _cp.RawKernel(src, name)
    return _KERNELS[key]


def _tile_rows_for_budget(n: int, itemsize: int, budget_bytes: int) -> int:
    """Largest row-tile height R such that two R x n device buffers (the tile
    plus headroom for the copy/allocation) fit the budget. At least 1 row."""
    per_row = n * itemsize
    r = max(1, int(budget_bytes // (2 * per_row)))
    return min(r, n)


def pairwise_distances_gpu(
    data: np.ndarray,
    out: np.ndarray | None = None,
    tile_rows: int | None = None,
    vram_fraction: float = 0.35,
    high_precision: bool = True,
) -> np.ndarray:
    """Dense Euclidean pairwise-distance matrix, computed on the GPU in
    row-tiles and streamed into a host matrix.

    Parameters
    ----------
    data : (n, d) float32/float64 array (host or device).
    out : optional (n, n) host array to write into (same dtype). Allocated if
        None.
    tile_rows : rows per GPU tile. Auto-sized to ``vram_fraction`` of free VRAM
        if None.
    vram_fraction : fraction of *free* VRAM to budget for tiles when auto-sizing.
    high_precision : accumulate the squared sum in float64 (default), matching
        the CPU kernel. Set False to accumulate in the input dtype — markedly
        faster for float32 on consumer GPUs (whose float64 rate is a fraction of
        float32) at ~1e-5 accuracy cost; ignored for float64 input.

    Returns
    -------
    (n, n) host ndarray with a zero diagonal, dtype matching the input.
    """
    if not is_available():
        raise RuntimeError("CuPy/CUDA device not available")

    data = np.asarray(data)
    if data.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected float32 or float64, got {data.dtype}")
    n, d = data.shape
    dtype = data.dtype
    itemsize = dtype.itemsize
    hp = high_precision or dtype == np.float64

    if out is None:
        out = np.empty((n, n), dtype=dtype)
    elif out.shape != (n, n) or out.dtype != dtype:
        raise ValueError("out must be (n, n) with the same dtype as data")
    if n == 0:
        return out

    X_dev = _cp.asarray(np.ascontiguousarray(data))  # (n, d), small
    kern = _kernel(dtype, hp)

    if tile_rows is None:
        free_bytes, _ = _cp.cuda.Device().mem_info
        tile_rows = _tile_rows_for_budget(n, itemsize, int(free_bytes * vram_fraction))

    threads = 256
    for a in range(0, n, tile_rows):
        b = min(a + tile_rows, n)
        R = b - a
        tile_elems = R * n
        tile_dev = _cp.empty((R, n), dtype=dtype)
        blocks = (tile_elems + threads - 1) // threads
        kern(
            (blocks,), (threads,),
            (X_dev, np.int32(n), np.int32(d), np.int32(R),
             np.int32(a), np.int64(tile_elems), tile_dev),
        )
        # Zero this tile's diagonal contribution (rows a..b) exactly, matching
        # the CPU kernel's zero diagonal (sqrt of a tiny residual could be ~1e-7
        # for float32).
        out[a:b, :] = _cp.asnumpy(tile_dev)
    np.fill_diagonal(out, 0)
    del X_dev
    _cp.get_default_memory_pool().free_all_blocks()
    return out


# Empirical crossover on a consumer GPU (RTX 4080, weak float64, PCIe D2H of
# the O(n^2) result): the GPU only beats 32 CPU cores at higher feature
# dimension AND for float32. float64 loses at every dimension on this class of
# card (FP64 ~1/64 of FP32); below the crossover d the transfer cost loses too.
_GPU_DIM_CROSSOVER = 64


def gpu_pairwise_beneficial(data: np.ndarray) -> bool:
    """Whether routing this input's pairwise distances to the GPU is expected
    to be a win on consumer-class hardware: a device is available, the data is
    float32, and the feature dimension clears the crossover."""
    data = np.asarray(data)
    if data.ndim != 2:
        return False
    return (is_available()
            and data.dtype == np.float32
            and data.shape[1] >= _GPU_DIM_CROSSOVER)


def pairwise_distances(data: np.ndarray, backend: str = "auto",
                       high_precision: bool = True) -> np.ndarray:
    """Dense Euclidean pairwise-distance matrix with backend selection.

    backend='auto' (default) uses the GPU only where it is expected to win on
    this class of hardware (see ``gpu_pairwise_beneficial`` /
    benchmarks/gpu_pairwise.md) and otherwise the CPU C/OpenMP kernel.
    'gpu'/'cpu' force a backend.
    """
    from .pcvat import pairwise_distances_c

    if backend == "gpu":
        return pairwise_distances_gpu(data, high_precision=high_precision)
    if backend == "cpu":
        return pairwise_distances_c(data)
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'gpu', or 'cpu', got {backend!r}")
    if gpu_pairwise_beneficial(data):
        return pairwise_distances_gpu(data, high_precision=high_precision)
    return pairwise_distances_c(data)


# ---------------------------------------------------------------------------
# GPU Fuzzy C-Means
#
# Unlike pairwise distances (a single pass producing a huge O(n^2) result that
# must transfer back), FCM is iterative: the data stays resident on the device
# across ~100 iterations and only the tiny (k, d) centers move. That amortizes
# the transfer, so this is the regime where the GPU reliably wins for large n.
#
# Distances here feed membership *ratios* in an iterative fixed-point solve, so
# — unlike VAT — the gram formulation ||x_i - c_j||^2 = |x_i|^2 - 2 x_i.c_j +
# |c_j|^2 is appropriate: it turns the per-iteration distance into a single
# (n x d)(d x k) GEMM (cuBLAS), and its cancellation error is immaterial to the
# converged partition.
# ---------------------------------------------------------------------------
def fuzzy_c_means_gpu(
    x: np.ndarray,
    n: int,
    m: float = 2.0,
    *,
    initial_guess: np.ndarray | None = None,
    indices=None,
    max_iter: int = 100,
    tol: float = 1e-5,
):
    """GPU Fuzzy C-Means. Mirrors ``fcm.fuzzy_c_means`` and returns
    ``(centers, membership)`` with shapes ``(n, d)`` and ``(n_samples, n)``.

    Falls back to the CPU implementation if no CUDA device is available.
    """
    if not is_available():
        from .fcm import fuzzy_c_means
        return fuzzy_c_means(x, n, m=m, indices=indices, initial_guess=initial_guess)

    x = np.asarray(x)
    dtype = x.dtype if x.dtype in (np.float32, np.float64) else np.float64
    Xd = _cp.asarray(np.ascontiguousarray(x, dtype=dtype))  # (n_samples, d), resident
    n_samples, d = Xd.shape

    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")
    if indices is not None:
        C = Xd[_cp.asarray(np.asarray(indices))].copy()
    elif initial_guess is not None:
        if initial_guess.shape != (n, d):
            raise ValueError(
                f"initial_guess must have shape ({n}, {d}), got {initial_guess.shape}")
        C = _cp.asarray(np.ascontiguousarray(initial_guess, dtype=dtype))
    else:
        idx = np.random.choice(n_samples, size=n * 2, replace=False)
        C = _cp.asarray(np.ascontiguousarray(x[idx], dtype=dtype)).reshape(
            n, 2, d).mean(axis=1)

    q = 1.0 / (m - 1.0)          # membership exponent on squared distance
    sqx = _cp.sum(Xd * Xd, axis=1)  # (n_samples,), constant across iterations

    def _membership(C):
        # squared distances via the gram identity, clamped to >= 0
        D2 = sqx[:, None] - 2.0 * (Xd @ C.T) + _cp.sum(C * C, axis=1)[None, :]
        _cp.maximum(D2, 0.0, out=D2)
        zero = D2 == 0.0
        any_zero = _cp.any(zero, axis=1)
        # u_ij = D2_ij^{-q} / sum_l D2_il^{-q}   (== the ratio-sum FCM weight)
        with np.errstate(divide="ignore"):
            inv = D2 ** (-q)
        U = inv / _cp.sum(inv, axis=1, keepdims=True)
        # points sitting exactly on centers: hard-assign among the zero cells
        if bool(any_zero.any()):
            hz = zero.astype(dtype)
            hz /= _cp.sum(hz, axis=1, keepdims=True)
            U = _cp.where(any_zero[:, None], hz, U)
        return U

    for _ in range(max_iter):
        U = _membership(C)
        Um = U ** m                              # (n_samples, n)
        C_new = (Um.T @ Xd) / _cp.sum(Um, axis=0)[:, None]
        if bool(_cp.all(_cp.abs(C_new - C) <= (1e-8 + tol * _cp.abs(C)))):
            C = C_new
            break
        C = C_new

    U = _membership(C)
    return _cp.asnumpy(C), _cp.asnumpy(U)
