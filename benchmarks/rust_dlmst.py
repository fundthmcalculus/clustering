"""ctypes loader for the dev-only Rust port in ``benchmarks/rust_dlmst``.

Builds the cdylib with ``cargo build --release`` on first use. Exposes the
same call shapes as ``tribbleclustering.cdlmst`` so ``mst_pareto`` can swap
implementations.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent / "rust_dlmst"
_EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
_PREFIX = "" if sys.platform == "win32" else "lib"
_LIB_PATH = _DIR / "target" / "release" / f"{_PREFIX}rust_dlmst.{_EXT}"
_lib = None


def build() -> Path:
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=_DIR, check=True)
    return _LIB_PATH


def _load():
    global _lib
    if _lib is None:
        if not _LIB_PATH.exists():
            build()
        _lib = ctypes.CDLL(str(_LIB_PATH))
    return _lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


def _sfx(a: np.ndarray) -> str:
    if a.dtype == np.float64:
        return "f64"
    if a.dtype == np.float32:
        return "f32"
    raise TypeError(f"Expected float32 or float64, got {a.dtype}")


def vat_order(data, from_points=False, k=4, nthreads=0):
    """``(p, q, w)`` as ``cdlmst.vat_order_dlmst_c``."""
    a = np.ascontiguousarray(data)
    n = a.shape[0]
    d = a.shape[1] if from_points else 0
    p = np.empty(n, np.int32)
    q = np.empty(n, np.int32)
    w = np.zeros(n, a.dtype)
    fn = getattr(_load(), f"dlmst_order_{_sfx(a)}")
    fn(
        _ptr(a),
        ctypes.c_size_t(n),
        ctypes.c_size_t(d),
        ctypes.c_size_t(k),
        ctypes.c_int(nthreads),
        _ptr(p),
        _ptr(q),
        _ptr(w),
    )
    return p, q, w


def compute_ivat(data, from_points=False, k=4, nthreads=0):
    """``(ivat, p)`` -- IVAT built from the MST."""
    a = np.ascontiguousarray(data)
    n = a.shape[0]
    d = a.shape[1] if from_points else 0
    out = np.empty((n, n), a.dtype)
    p = np.empty(n, np.int32)
    fn = getattr(_load(), f"dlmst_ivat_{_sfx(a)}")
    fn(
        _ptr(a),
        ctypes.c_size_t(n),
        ctypes.c_size_t(d),
        ctypes.c_size_t(k),
        ctypes.c_int(nthreads),
        _ptr(out),
        _ptr(p),
    )
    return out, p


def prim_order(R, nthreads=0):
    """``(p, q)`` from the compact dense Prim, as ``vat_prim_mst_c``."""
    a = np.ascontiguousarray(R)
    n = a.shape[0]
    p = np.empty(n, np.int32)
    q = np.empty(n, np.int32)
    fn = getattr(_load(), f"prim_order_{_sfx(a)}")
    fn(_ptr(a), ctypes.c_size_t(n), ctypes.c_int(nthreads), _ptr(p), _ptr(q))
    return p, q
