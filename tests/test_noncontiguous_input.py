"""Every public compiled entry point must accept a non-C-contiguous array.

The compiled and pure-Python paths sit behind a silent import-time fallback,
so they are meant to be interchangeable. A `[:, ::1]` memoryview in a kernel
signature makes them not interchangeable for strided input -- `data[:, ::2]`,
a transpose, an F-ordered load -- and the caller gets a working call or a bare
`ValueError: ndarray is not C-contiguous` depending on whether the extension
happened to build on their machine. Issue #102.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, squareform

import tribbleclustering as tc

try:
    from tribbleclustering import cfcm, clk, pcvat

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def assert_same_result(got, ref):
    """Every element of the returned tuple must match, not just the first.

    `compute_vat_c` returns `(matrix, ..., order)` and `lin_kernighan_c`
    returns `(tour, length)`; comparing only element 0 would let a
    layout-dependent difference in the ordering or the tour length -- the
    outputs a caller actually consumes -- pass unnoticed.
    """
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    assert len(got) == len(ref)
    for i, (g, r) in enumerate(zip(got, ref)):
        assert_allclose(np.asarray(g), np.asarray(r), err_msg=f"return value {i}")


@pytest.fixture
def strided_points():
    """(n, d) point cloud that is a strided view, plus its contiguous twin."""
    rng = np.random.default_rng(0)
    padded = np.zeros((40, 6))
    padded[:, ::2] = rng.standard_normal((40, 3))
    view = padded[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    return view, np.ascontiguousarray(view)


@pytest.fixture
def strided_distances(strided_points):
    """(n, n) dissimilarity matrix that is a strided view, plus its twin."""
    _, points = strided_points
    d = squareform(pdist(points)).astype(np.float64)
    padded = np.zeros((d.shape[0], 2 * d.shape[1]))
    padded[:, ::2] = d
    view = padded[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    return view, np.ascontiguousarray(view)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
class TestCompiledAcceptsStridedInput:
    """Strided input must give the same answer as its contiguous twin."""

    def test_cfcm_fuzzy_c_means(self, strided_points):
        view, contig = strided_points
        # Deliberately one shared, contiguous initial_guess: `x` is the
        # variable under test, and identical starts make the two runs
        # comparable iteration for iteration.
        guess = contig[:3].copy()

        c_view, w_view, n_view, _ = cfcm.fuzzy_c_means(view, 3, initial_guess=guess)
        c_ref, w_ref, n_ref, _ = cfcm.fuzzy_c_means(contig, 3, initial_guess=guess)

        assert n_view == n_ref
        assert_allclose(c_view, c_ref)
        assert_allclose(w_view, w_ref)

    def test_cfcm_matches_the_pure_path(self, strided_points):
        """The gap that motivated #102: one path accepted it, the other raised."""
        view, _ = strided_points
        # Here the guess comes off `view` rather than the twin, to prove the
        # strided array is genuinely usable end to end and not merely tolerated
        # once a contiguous array has been threaded in beside it.
        guess = np.ascontiguousarray(view[:3])

        c_py, w_py = tc.fuzzy_c_means(view, 3, initial_guess=guess)
        c_cy, w_cy, _, _ = cfcm.fuzzy_c_means(view, 3, initial_guess=guess)

        assert_allclose(c_cy, c_py, rtol=1e-6, atol=1e-8)
        assert_allclose(w_cy, w_py, rtol=1e-6, atol=1e-8)

    def test_cfcm_strided_indices(self, strided_points):
        """`indices` is taken as an int64 `[::1]` view too."""
        view, contig = strided_points
        padded = np.zeros(12, dtype=np.int64)
        padded[::2] = np.arange(6, dtype=np.int64)
        strided_idx = padded[::2]
        assert not strided_idx.flags["C_CONTIGUOUS"]

        c_view, _, _, _ = cfcm.fuzzy_c_means(view, 3, indices=strided_idx)
        c_ref, _, _, _ = cfcm.fuzzy_c_means(
            contig, 3, indices=np.ascontiguousarray(strided_idx)
        )
        assert_allclose(c_view, c_ref)

    def test_pcvat_pairwise_distances(self, strided_points):
        view, contig = strided_points
        assert_same_result(
            pcvat.pairwise_distances_c(view), pcvat.pairwise_distances_c(contig)
        )

    @pytest.mark.parametrize("fn_name", ["compute_vat_c", "compute_ivat_c"])
    def test_pcvat_vat_family(self, strided_distances, fn_name):
        view, contig = strided_distances
        fn = getattr(pcvat, fn_name)
        # inplace=False by default (pcvat.pyx:728, :965), but pass copies anyway
        # so a future in-place default cannot make this compare a matrix
        # against itself.
        assert_same_result(fn(view.copy()), fn(contig.copy()))

    def test_pcvat_prim_mst(self, strided_distances):
        view, contig = strided_distances
        assert_same_result(pcvat.vat_prim_mst_c(view), pcvat.vat_prim_mst_c(contig))

    def test_clk_lin_kernighan(self, strided_distances):
        """Tour and length both, and both are reproducible.

        The multi-start/OpenMP structure looks like it should vary per run; it
        does not. Measured 8 runs on a 60-node instance: one distinct tour, one
        distinct length. So comparing two runs is a fair test, not a flaky one.
        """
        view, contig = strided_distances
        assert_same_result(clk.lin_kernighan_c(view), clk.lin_kernighan_c(contig))


class TestPurePathAcceptsStridedInput:
    """The reference implementations set the bar the compiled ones must meet."""

    def test_fuzzy_c_means(self, strided_points):
        view, _ = strided_points
        c, w = tc.fuzzy_c_means(view, 3, initial_guess=np.ascontiguousarray(view[:3]))
        assert np.all(np.isfinite(c))
        assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)

    @pytest.mark.parametrize("fn_name", ["compute_vat", "compute_ivat"])
    def test_vat_family(self, strided_distances, fn_name):
        view, contig = strided_distances
        fn = getattr(tc, fn_name)
        assert_same_result(fn(view.copy()), fn(contig.copy()))

    def test_pairwise_distances(self, strided_points):
        view, contig = strided_points
        assert_same_result(tc.pairwise_distances(view), tc.pairwise_distances(contig))
