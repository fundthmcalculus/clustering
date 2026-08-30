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
        guess = contig[:3].copy()

        c_view, w_view, n_view, _ = cfcm.fuzzy_c_means(view, 3, initial_guess=guess)
        c_ref, w_ref, n_ref, _ = cfcm.fuzzy_c_means(contig, 3, initial_guess=guess)

        assert n_view == n_ref
        assert_allclose(c_view, c_ref)
        assert_allclose(w_view, w_ref)

    def test_cfcm_matches_the_pure_path(self, strided_points):
        """The gap that motivated #102: one path accepted it, the other raised."""
        view, _ = strided_points
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
        assert_allclose(
            pcvat.pairwise_distances_c(view), pcvat.pairwise_distances_c(contig)
        )

    @pytest.mark.parametrize("fn_name", ["compute_vat_c", "compute_ivat_c"])
    def test_pcvat_vat_family(self, strided_distances, fn_name):
        view, contig = strided_distances
        fn = getattr(pcvat, fn_name)
        # inplace=False by default, but pass copies anyway so a future in-place
        # default cannot make this test compare a matrix against itself.
        got = fn(view.copy())
        ref = fn(contig.copy())
        assert_allclose(np.asarray(got[0]), np.asarray(ref[0]))

    def test_pcvat_prim_mst(self, strided_distances):
        view, contig = strided_distances
        assert_allclose(
            np.asarray(pcvat.vat_prim_mst_c(view)[0]),
            np.asarray(pcvat.vat_prim_mst_c(contig)[0]),
        )

    def test_clk_lin_kernighan(self, strided_distances):
        view, contig = strided_distances
        assert_allclose(
            np.asarray(clk.lin_kernighan_c(view)[0]),
            np.asarray(clk.lin_kernighan_c(contig)[0]),
        )


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
        assert_allclose(
            np.asarray(fn(view.copy())[0]), np.asarray(fn(contig.copy())[0])
        )

    def test_pairwise_distances(self, strided_points):
        view, contig = strided_points
        assert_allclose(tc.pairwise_distances(view), tc.pairwise_distances(contig))
