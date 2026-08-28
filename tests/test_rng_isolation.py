"""RNG threading and process-global isolation (issue #92).

Two properties are asserted for every seedable estimator/kernel here:

1. **Repeatability** -- the same ``random_state`` gives the same fit. This is a
   pre-existing public-API contract for ``FuzzyCMeans.random_state``; the switch
   from ``np.random.seed(...)`` to a threaded ``np.random.Generator`` changed the
   *mechanism*, so these tests pin the contract that must survive it.

2. **Isolation** -- fitting must not read from or write to the process-global
   ``np.random`` stream. Before the fix, ``FuzzyCMeans.fit`` called
   ``np.random.seed(self.random_state)``, which rewound the global MT19937 for
   every other consumer in the process. ``ivatmeans.py`` already documents that
   position; these tests enforce it.

The global-stream tests deliberately compare against draws taken with *no* fit in
between, so they fail both if the estimator reseeds the stream (the old bug) and
if it merely consumes from it.
"""

import numpy as np
import pytest

from tribbleclustering import KMeans, fuzzy_c_means, gpu
from tribbleclustering.fuzzycmeans import FuzzyCMeans
from tribbleclustering.kmeans import kmeans

try:
    from tribbleclustering.cfcm import fuzzy_c_means as fuzzy_c_means_cython

    CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on whether the ext is built
    fuzzy_c_means_cython = None
    CYTHON_AVAILABLE = False


@pytest.fixture
def blobs():
    """Three well-separated gaussian blobs, built without the global RNG."""
    rng = np.random.default_rng(7)
    return np.vstack(
        [
            rng.normal(loc=(0.0, 0.0), scale=0.35, size=(60, 2)),
            rng.normal(loc=(5.0, 5.0), scale=0.35, size=(60, 2)),
            rng.normal(loc=(0.0, 6.0), scale=0.35, size=(60, 2)),
        ]
    )


def _global_draws_around(fn, n=5, seed=1234):
    """Return (without_fn, with_fn): global draws taken from an identical
    starting state, with and without ``fn`` running in between."""
    np.random.seed(seed)
    without = np.random.random(n)
    np.random.seed(seed)
    fn()
    with_fn = np.random.random(n)
    return without, with_fn


class TestFuzzyCMeansRepeatability:
    """``FuzzyCMeans.random_state`` is public API -- keep its meaning."""

    def test_same_random_state_gives_identical_labels(self, blobs):
        a = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        b = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        assert np.array_equal(a, b)

    def test_same_random_state_gives_identical_centers(self, blobs):
        a = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).cluster_centers_
        b = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).cluster_centers_
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_repeatable_across_an_intervening_global_reseed(self, blobs):
        """The fit must not depend on the state of the global stream."""
        a = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        np.random.seed(999)
        np.random.random(17)
        b = FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        assert np.array_equal(a, b)

    def test_random_state_none_still_fits(self, blobs):
        model = FuzzyCMeans(n_clusters=3).fit(blobs)
        assert model.cluster_centers_ is not None
        assert np.asarray(model.cluster_centers_).shape == (3, 2)


class TestFuzzyCMeansGlobalRngIsolation:
    """``fit`` must leave ``np.random`` exactly as it found it."""

    def test_fit_does_not_disturb_global_stream(self, blobs):
        without, with_fit = _global_draws_around(
            lambda: FuzzyCMeans(n_clusters=3, random_state=42).fit(blobs)
        )
        assert np.array_equal(without, with_fit), (
            "FuzzyCMeans.fit(random_state=42) perturbed the process-global "
            f"np.random stream: {without} -> {with_fit}"
        )

    def test_unseeded_fit_does_not_disturb_global_stream(self, blobs):
        without, with_fit = _global_draws_around(
            lambda: FuzzyCMeans(n_clusters=3).fit(blobs)
        )
        assert np.array_equal(without, with_fit)

    def test_free_function_does_not_disturb_global_stream(self, blobs):
        without, with_call = _global_draws_around(
            lambda: fuzzy_c_means(blobs, 3, random_state=42)
        )
        assert np.array_equal(without, with_call)

    def test_unseeded_free_function_does_not_disturb_global_stream(self, blobs):
        without, with_call = _global_draws_around(lambda: fuzzy_c_means(blobs, 3))
        assert np.array_equal(without, with_call)


class TestFuzzyCMeansFreeFunctionSeeding:
    """The free function is exported in ``__all__``; it must be seedable."""

    def test_accepts_random_state(self, blobs):
        a = fuzzy_c_means(blobs, 3, random_state=11).cluster_centers_
        b = fuzzy_c_means(blobs, 3, random_state=11).cluster_centers_
        assert np.array_equal(a, b)

    def test_accepts_a_generator(self, blobs):
        a = fuzzy_c_means(
            blobs, 3, random_state=np.random.default_rng(5)
        ).cluster_centers_
        b = fuzzy_c_means(
            blobs, 3, random_state=np.random.default_rng(5)
        ).cluster_centers_
        assert np.array_equal(a, b)

    def test_distinct_seeds_reach_distinct_initializations(self, blobs):
        """max_iter=1 keeps the output close to the random init, so the seed
        must be visible in the result. (At convergence these blobs reach the
        same partition from any start, which is why this is measured at
        iteration 1 rather than at convergence.)"""
        seen = {
            fuzzy_c_means(
                blobs, 3, max_iter=1, random_state=s
            ).cluster_centers_.tobytes()
            for s in range(6)
        }
        assert len(seen) > 1

    def test_seed_is_ignored_when_centers_are_given(self, blobs):
        """indices / initial_guess are already deterministic; random_state must
        not change them."""
        guess = blobs[:3].copy()
        a = fuzzy_c_means(blobs, 3, initial_guess=guess, random_state=1)
        b = fuzzy_c_means(blobs, 3, initial_guess=guess, random_state=2)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)

        idx = [0, 61, 121]
        c = fuzzy_c_means(blobs, 3, indices=idx, random_state=1)
        d = fuzzy_c_means(blobs, 3, indices=idx, random_state=2)
        assert np.array_equal(c.cluster_centers_, d.cluster_centers_)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
class TestCompiledKernelSeeding:
    """The compiled kernel is the one ``FuzzyCMeans`` actually calls when it is
    built, so it must honour the same contract as the pure path."""

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_same_random_state_gives_identical_centers(self, blobs, dtype):
        x = blobs.astype(dtype)
        a = fuzzy_c_means_cython(x, 3, random_state=11)[0]
        b = fuzzy_c_means_cython(x, 3, random_state=11)[0]
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_accepts_a_generator(self, blobs, dtype):
        x = blobs.astype(dtype)
        a = fuzzy_c_means_cython(x, 3, random_state=np.random.default_rng(5))[0]
        b = fuzzy_c_means_cython(x, 3, random_state=np.random.default_rng(5))[0]
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_does_not_disturb_global_stream(self, blobs, dtype):
        x = blobs.astype(dtype)
        without, with_call = _global_draws_around(
            lambda: fuzzy_c_means_cython(x, 3, random_state=11)
        )
        assert np.array_equal(without, with_call)

    def test_distinct_seeds_reach_distinct_initializations(self, blobs):
        seen = {
            fuzzy_c_means_cython(blobs, 3, max_iter=1, random_state=s)[0].tobytes()
            for s in range(6)
        }
        assert len(seen) > 1


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")
class TestPureAndCompiledAgree:
    """CLAUDE.md requires the compiled and pure kernels to stay behaviourally
    equivalent, and that rule is why ``FuzzyCMeans.fit`` was NOT given a silent
    fallback to the pure path when the extension is stale. Nothing pinned it,
    so an edit that changed only one kernel's draw (``rng.choice`` ->
    ``rng.integers``, say) would have sailed through the suite.

    Measured at ``max_iter=1``, one update off the initialization, because
    these blobs converge to the same partition from any start -- so at
    convergence a changed draw would be invisible. At iteration 1 the two
    kernels agree to 6.2e-15 while two *different* seeds are ~2-3.7 apart, a
    separation of ~15 orders of magnitude."""

    @pytest.mark.parametrize("n_clusters", [2, 3, 5])
    @pytest.mark.parametrize("seed", [0, 1, 11])
    def test_same_random_state_reaches_the_same_initialization(
        self, blobs, n_clusters, seed
    ):
        pure = np.asarray(
            fuzzy_c_means(
                blobs, n_clusters, max_iter=1, random_state=seed
            ).cluster_centers_
        )
        compiled = np.asarray(
            fuzzy_c_means_cython(
                np.ascontiguousarray(blobs),
                n_clusters,
                2.0,
                max_iter=1,
                random_state=seed,
            )[0]
        )
        assert np.allclose(
            np.sort(pure, axis=0), np.sort(compiled, axis=0), rtol=0, atol=1e-9
        )

    def test_the_comparison_is_not_vacuous(self, blobs):
        """Guard the guard: if two different seeds produced the same centers at
        max_iter=1, the test above would pass no matter what either kernel drew."""
        a = np.asarray(
            fuzzy_c_means(blobs, 3, max_iter=1, random_state=0).cluster_centers_
        )
        b = np.asarray(
            fuzzy_c_means(blobs, 3, max_iter=1, random_state=1).cluster_centers_
        )
        assert np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max() > 1e-3


class TestKMeansRepeatability:
    """``KMeans.random_state`` carries the same public-API contract."""

    def test_same_random_state_gives_identical_labels(self, blobs):
        a = KMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        b = KMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("init", ["k-means++", "random"])
    def test_same_random_state_gives_identical_centers(self, blobs, init):
        a = KMeans(n_clusters=3, init=init, random_state=42).fit(blobs)
        b = KMeans(n_clusters=3, init=init, random_state=42).fit(blobs)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)
        assert a.inertia_ == b.inertia_

    def test_repeatable_across_an_intervening_global_reseed(self, blobs):
        a = KMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        np.random.seed(999)
        np.random.random(17)
        b = KMeans(n_clusters=3, random_state=42).fit(blobs).labels_
        assert np.array_equal(a, b)

    def test_random_state_none_still_fits(self, blobs):
        model = KMeans(n_clusters=3).fit(blobs)
        assert model.cluster_centers_ is not None
        assert model.cluster_centers_.shape == (3, 2)


class TestKMeansGlobalRngIsolation:
    @pytest.mark.parametrize("init", ["k-means++", "random"])
    def test_fit_does_not_disturb_global_stream(self, blobs, init):
        without, with_fit = _global_draws_around(
            lambda: KMeans(n_clusters=3, init=init, random_state=42).fit(blobs)
        )
        assert np.array_equal(without, with_fit), (
            f"KMeans(init={init!r}).fit perturbed the process-global np.random "
            f"stream: {without} -> {with_fit}"
        )

    def test_unseeded_fit_does_not_disturb_global_stream(self, blobs):
        without, with_fit = _global_draws_around(
            lambda: KMeans(n_clusters=3).fit(blobs)
        )
        assert np.array_equal(without, with_fit)

    def test_free_function_does_not_disturb_global_stream(self, blobs):
        without, with_call = _global_draws_around(
            lambda: kmeans(blobs, 3, random_state=42)
        )
        assert np.array_equal(without, with_call)


class TestKMeansFreeFunctionSeeding:
    @pytest.mark.parametrize("init", ["k-means++", "random"])
    def test_accepts_random_state(self, blobs, init):
        a = kmeans(blobs, 3, init=init, random_state=11)
        b = kmeans(blobs, 3, init=init, random_state=11)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)

    def test_accepts_a_generator(self, blobs):
        a = kmeans(blobs, 3, random_state=np.random.default_rng(5))
        b = kmeans(blobs, 3, random_state=np.random.default_rng(5))
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)

    def test_distinct_seeds_reach_distinct_initializations(self, blobs):
        """k-means++ picks the first center uniformly at random, so different
        seeds must produce different draws. Measured at max_iter=1 for the
        same reason as the FCM case: at convergence these blobs are
        seed-independent."""
        seen = {
            kmeans(blobs, 3, max_iter=1, random_state=s).cluster_centers_.tobytes()
            for s in range(6)
        }
        assert len(seen) > 1

    def test_seed_is_ignored_when_centers_are_given(self, blobs):
        guess = blobs[:3].copy()
        a = kmeans(blobs, 3, initial_guess=guess, random_state=1)
        b = kmeans(blobs, 3, initial_guess=guess, random_state=2)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)


class TestGpuFallbackSeeding:
    """``gpu.py`` mirrors the CPU kernels and must expose the same knob. Only
    the CPU-fallback branch runs here -- ``gpu.is_available()`` is False
    without CUDA, so the device branches are NOT covered by these tests."""

    def test_fcm_gpu_fallback_accepts_random_state(self, blobs):
        a = gpu.fuzzy_c_means_gpu(blobs, 3, random_state=11)
        b = gpu.fuzzy_c_means_gpu(blobs, 3, random_state=11)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)

    def test_fcm_gpu_fallback_does_not_disturb_global_stream(self, blobs):
        without, with_call = _global_draws_around(
            lambda: gpu.fuzzy_c_means_gpu(blobs, 3, random_state=11)
        )
        assert np.array_equal(without, with_call)

    def test_fcm_gpu_survives_more_clusters_than_half_the_rows(self):
        """n_samples < 2*n_clusters. The device branch drew with
        ``replace=False`` and would have raised ValueError on this shape --
        the same defect ``test_fcm_more_clusters_than_half_the_rows`` pins for
        the CPU kernels. Only the fallback executes here (no CUDA), so this
        pins the entry point's contract, NOT the device code."""
        x = np.random.default_rng(0).normal(size=(10, 2))  # 10 rows, 2n = 12
        res = gpu.fuzzy_c_means_gpu(x, 6, random_state=0)
        assert np.asarray(res.cluster_centers_).shape == (6, 2)

    def test_kmeans_gpu_fallback_accepts_random_state(self, blobs):
        a = gpu.kmeans_gpu(blobs, 3, random_state=11)
        b = gpu.kmeans_gpu(blobs, 3, random_state=11)
        assert np.array_equal(a.cluster_centers_, b.cluster_centers_)

    def test_kmeans_gpu_fallback_does_not_disturb_global_stream(self, blobs):
        without, with_call = _global_draws_around(
            lambda: gpu.kmeans_gpu(blobs, 3, random_state=11)
        )
        assert np.array_equal(without, with_call)
