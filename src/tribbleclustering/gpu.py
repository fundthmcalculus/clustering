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
            (blocks,),
            (threads,),
            (
                X_dev,
                np.int32(n),
                np.int32(d),
                np.int32(R),
                np.int32(a),
                np.int64(tile_elems),
                tile_dev,
            ),
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
# dimension. Below this d the transfer/compute cost loses to the CPU C kernel.
_GPU_DIM_CROSSOVER = 64


def pairwise_distances(
    data: np.ndarray, backend: str = "auto", high_precision: bool = True
) -> np.ndarray:
    """Dense Euclidean pairwise-distance matrix with backend selection.

    backend='auto' (default) uses the GPU only where it is expected to win on
    this class of hardware — a CUDA device is available and the feature
    dimension is at least ~64 (see benchmarks/gpu_pairwise.md) — and otherwise
    the CPU C/OpenMP kernel. 'gpu'/'cpu' force a backend.
    """
    from .pcvat import pairwise_distances_c

    data = np.asarray(data)
    d = data.shape[1] if data.ndim == 2 else 0
    if backend == "gpu":
        return pairwise_distances_gpu(data, high_precision=high_precision)
    if backend == "cpu":
        return pairwise_distances_c(data)
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'gpu', or 'cpu', got {backend!r}")
    if is_available() and d >= _GPU_DIM_CROSSOVER:
        return pairwise_distances_gpu(data, high_precision=high_precision)
    return pairwise_distances_c(data)
