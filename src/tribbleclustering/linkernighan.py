"""sklearn-style Lin-Kernighan TSP solver wrapper.

``LinKernighan`` is a thin, sklearn-flavoured wrapper over the Lin-Kernighan
local search. Like ``FuzzyCMeans``/``IVATMeans`` it prefers the compiled
Cython/OpenMP kernel (``clk``) and transparently falls back to the pure-numpy
reference (``lk``) when the extension is not built (see ``CLAUDE.md``).

The solver takes either a point cloud (Euclidean distances are computed for
you) or a precomputed symmetric distance matrix, and exposes the result as
``tour_`` (visit order) and ``tour_length_``.
"""

import os
from typing import Optional

import numpy as np
from numpy import ndarray

from .lk import tour_length
from .util import pairwise_distances as _pairwise_distances

try:
    from .clk import lin_kernighan_c as _lk_compiled

    _has_compiled_lk = True
except ImportError:  # pragma: no cover - exercised only without the extension
    _has_compiled_lk = False

from .lk import lin_kernighan as _lk_pure


class LinKernighan:
    """Lin-Kernighan heuristic solver for the symmetric TSP.

    Parameters
    ----------
    n_starts : int, optional
        Number of independent nearest-neighbour restarts, each optimized to a
        local optimum; the best tour is returned. When ``None`` (default) this
        defaults to the CPU count so the parallel restart loop saturates the
        machine. More starts trade runtime for solution quality.
    max_depth : int, optional
        Maximum depth of each variable-depth LK move chain. Default 5.
    neighbors : int, optional
        Size of the candidate nearest-neighbour list per city. Default 8.
    num_threads : int, optional
        OpenMP threads used for the parallel restart loop (compiled path only;
        ignored by the pure-Python fallback). ``0`` (default) lets OpenMP pick,
        capped at ``n_starts``. ``1`` forces serial optimization.
    dtype : {"float64", "float32"}, optional
        Working precision for the distance matrix. ``float32`` roughly halves
        memory and time at a small accuracy cost. Default ``"float64"``.
    random_state : int, optional
        Seed for start-city selection (only matters when ``n_starts`` exceeds
        the number of cities). Default ``None``.

    Attributes
    ----------
    tour_ : ndarray of shape (n_cities,)
        Best tour found, as a permutation of ``range(n_cities)``.
    tour_length_ : float
        Length of ``tour_`` under the (possibly computed) distance matrix.
    distance_matrix_ : ndarray of shape (n_cities, n_cities)
        The distance matrix that was optimized over.
    """

    def __init__(
        self,
        n_starts: Optional[int] = None,
        max_depth: int = 5,
        neighbors: int = 8,
        num_threads: int = 0,
        dtype: str = "float64",
        random_state: Optional[int] = None,
    ):
        self.n_starts = n_starts
        self.max_depth = max_depth
        self.neighbors = neighbors
        self.num_threads = num_threads
        self.dtype = dtype
        self.random_state = random_state
        self.tour_: Optional[ndarray] = None
        self.tour_length_: Optional[float] = None
        self.distance_matrix_: Optional[ndarray] = None

    @property
    def uses_compiled(self) -> bool:
        """True when the compiled (OpenMP, multi-threaded) kernel is in use."""
        return _has_compiled_lk

    def _resolve_distance_matrix(self, X: ndarray, precomputed: bool) -> ndarray:
        np_dtype = np.float32 if self.dtype == "float32" else np.float64
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if precomputed:
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "precomputed=True requires a square distance matrix, got "
                    f"shape {X.shape}"
                )
            return np.ascontiguousarray(X, dtype=np_dtype)
        # Point cloud -> Euclidean pairwise-distance matrix.
        points = np.ascontiguousarray(X, dtype=np_dtype)
        return np.ascontiguousarray(_pairwise_distances(points), dtype=np_dtype)

    def solve(self, X: ndarray, precomputed: bool = False) -> "LinKernighan":
        """Solve the TSP for ``X``.

        Parameters
        ----------
        X : ndarray
            Either a point cloud of shape ``(n_cities, n_features)`` (Euclidean
            distances are computed), or a precomputed square distance matrix
            when ``precomputed=True``.
        precomputed : bool, optional
            Interpret ``X`` as a distance matrix rather than points.

        Returns
        -------
        self : LinKernighan
            Fitted solver with ``tour_`` and ``tour_length_`` populated.
        """
        distances = self._resolve_distance_matrix(X, precomputed)
        self.distance_matrix_ = distances

        n = distances.shape[0]
        n_starts = self.n_starts if self.n_starts is not None else (os.cpu_count() or 1)
        n_starts = max(1, min(int(n_starts), max(1, n)))

        if _has_compiled_lk:
            tour, length = _lk_compiled(
                distances,
                n_starts=n_starts,
                max_depth=self.max_depth,
                neighbors=self.neighbors,
                num_threads=self.num_threads,
                seed=self.random_state,
            )
        else:
            tour, length = _lk_pure(
                distances,
                n_starts=n_starts,
                max_depth=self.max_depth,
                neighbors=self.neighbors,
                seed=self.random_state,
            )

        self.tour_ = np.asarray(tour, dtype=np.int64)
        self.tour_length_ = float(length)
        return self

    def fit(self, X: ndarray, y: Optional[ndarray] = None) -> "LinKernighan":
        """Alias for :meth:`solve` on a point cloud (sklearn API consistency)."""
        return self.solve(X, precomputed=False)

    def fit_predict(self, X: ndarray, y: Optional[ndarray] = None) -> ndarray:
        """Solve and return the tour order.

        Returns
        -------
        tour : ndarray of shape (n_cities,)
            The best tour found.
        """
        self.solve(X, precomputed=False)
        assert self.tour_ is not None  # set by solve()
        return self.tour_

    def tour_length_for(self, tour: ndarray) -> float:
        """Length of an arbitrary ``tour`` under this solver's distance matrix."""
        if self.distance_matrix_ is None:
            raise ValueError("Solver has not been fitted yet. Call solve() first.")
        return tour_length(tour, self.distance_matrix_)
