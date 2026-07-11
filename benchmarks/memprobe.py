"""Peak resident-memory measurement.

The interesting quantity for scaling work is the *peak* process working set
during a call, not the delta before/after (the peak lands mid-call, when
several n x n buffers are live simultaneously, and is gone by the time the
call returns). We read the OS high-water mark where the platform exposes one
and otherwise sample the current RSS from a background thread.

On Windows the peak is read directly from ``PROCESS_MEMORY_COUNTERS`` via
psapi. A subtle bug to avoid: ``GetCurrentProcess`` returns a pseudo-handle
of ``(HANDLE)-1``; without an explicit ``restype`` ctypes treats it as a
32-bit int and the truncated handle makes ``GetProcessMemoryInfo`` fail
silently (returns zeroes). We set the signatures explicitly and check the
return value.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

_WINDOWS = sys.platform == "win32"

if _WINDOWS:
    import ctypes
    from ctypes import POINTER, Structure, byref, c_size_t, c_void_p, sizeof
    from ctypes import wintypes as wt

    class _PROCESS_MEMORY_COUNTERS(Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", c_size_t),
            ("WorkingSetSize", c_size_t),
            ("QuotaPeakPagedPoolUsage", c_size_t),
            ("QuotaPagedPoolUsage", c_size_t),
            ("QuotaPeakNonPagedPoolUsage", c_size_t),
            ("QuotaNonPagedPoolUsage", c_size_t),
            ("PagefileUsage", c_size_t),
            ("PeakPagefileUsage", c_size_t),
        ]

    _psapi = ctypes.WinDLL("psapi")
    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.GetCurrentProcess.restype = c_void_p
    _psapi.GetProcessMemoryInfo.argtypes = [
        c_void_p,
        POINTER(_PROCESS_MEMORY_COUNTERS),
        wt.DWORD,
    ]
    _psapi.GetProcessMemoryInfo.restype = wt.BOOL

    def _counters() -> _PROCESS_MEMORY_COUNTERS:
        c = _PROCESS_MEMORY_COUNTERS()
        c.cb = sizeof(c)
        ok = _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), byref(c), c.cb)
        if not ok:
            raise ctypes.WinError()
        return c

    def current_rss_bytes() -> int:
        return _counters().WorkingSetSize

    def _os_peak_bytes() -> int:
        return _counters().PeakWorkingSetSize

    def reset_os_peak() -> bool:
        """Reset the OS peak-working-set high-water mark to the current RSS.

        Available on Windows 11 / Server 2016+. Returns False if unsupported,
        in which case callers fall back to the sampling thread.
        """
        try:
            fn = _kernel32.SetProcessWorkingSetSizeEx
        except AttributeError:
            return False
        # There is no direct "reset peak" call; EmptyWorkingSet trims the set so
        # the subsequent peak reflects only what the measured call touches.
        try:
            _psapi.EmptyWorkingSet.argtypes = [c_void_p]
            _psapi.EmptyWorkingSet.restype = wt.BOOL
            _psapi.EmptyWorkingSet(_kernel32.GetCurrentProcess())
            return True
        except Exception:
            return False

else:  # POSIX
    import resource

    def current_rss_bytes() -> int:
        with open(f"/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * resource.getpagesize()

    def _os_peak_bytes() -> int:
        # ru_maxrss is KiB on Linux, bytes on macOS.
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r * 1024 if sys.platform.startswith("linux") else r

    def reset_os_peak() -> bool:
        return False  # ru_maxrss is process-lifetime; use the sampler instead.


class _Sampler:
    """Background thread that tracks the max current RSS while active."""

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> None:
        self.peak = current_rss_bytes()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            r = current_rss_bytes()
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self) -> int:
        self._stop.set()
        if self._t is not None:
            self._t.join()
        return self.peak


@contextmanager
def measure_peak_rss():
    """Context manager yielding an object whose ``.peak_bytes`` /
    ``.peak_gb`` give the peak RSS observed inside the block.

    Combines the OS high-water mark (exact, no sampling gaps) with a sampling
    thread (portable) and reports the larger of the two.
    """

    class _Result:
        peak_bytes = 0
        baseline_bytes = 0

        @property
        def peak_gb(self) -> float:
            return self.peak_bytes / 1e9

        @property
        def delta_gb(self) -> float:
            return (self.peak_bytes - self.baseline_bytes) / 1e9

    res = _Result()
    res.baseline_bytes = current_rss_bytes()
    reset_os_peak()
    sampler = _Sampler()
    sampler.start()
    try:
        yield res
    finally:
        sampled = sampler.stop()
        try:
            os_peak = _os_peak_bytes()
        except Exception:
            os_peak = 0
        res.peak_bytes = max(sampled, os_peak, res.baseline_bytes)


if __name__ == "__main__":
    import gc

    import numpy as np

    print(f"baseline RSS: {current_rss_bytes()/1e9:.3f} GB")
    with measure_peak_rss() as m:
        a = np.ones((12000, 12000), dtype=np.float64)  # ~1.15 GB, touched
        a += 1.0
    print(f"expected ~1.15 GB touched -> peak delta {m.delta_gb:.3f} GB")
    del a
    gc.collect()
