from typing import Callable, Optional, Union

import numpy as np
from numpy import ndarray

from .pvat import get_ivat_levels, get_ivat_hierarchy, IvatMeansResult, ClusterNode
from .nerfcm import relational_fuzzy_c_means, relational_out_of_sample_membership
from . import gpu as _gpu
from . import gpu_vat as _gpu_vat

try:
    from .pcvat import pairwise_distances_c as _pairwise_distances
    from .pcvat import compute_ivat_c as _compute_ivat

    _has_compiled_distances = True
except ImportError:
    from .util import pairwise_distances as _pairwise_distances
    from .pvat import compute_ivat as _compute_ivat

    _has_compiled_distances = False

MetricLike = Union[None, str, Callable[[ndarray, ndarray], ndarray]]

_METRICS: dict[str, Callable[[ndarray, ndarray], ndarray]] = {
    "euclidean": lambda a, b: np.linalg.norm(
        a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=2
    ),
    "manhattan": lambda a, b: np.sum(
        np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]), axis=2
    ),
    "cosine": lambda a, b: 1.0
    - (
        (a @ b.T)
        / (
            np.linalg.norm(a, axis=1)[:, np.newaxis]
            * np.linalg.norm(b, axis=1)[np.newaxis, :]
            + 1e-12
        )
    ),
}


def _resolve_metric(metric: MetricLike) -> Callable[[ndarray, ndarray], ndarray]:
    """Resolve the ``metric`` constructor argument to a pairwise-distance callable.

    This is the "advanced feature" hook from issue #54: it controls how new
    points are related to the discovered prototypes (nearest-medoid distance,
    or the single-linkage nearest-neighbor step of the relational
    out-of-sample extension). It does *not* change the VAT/iVAT pairwise
    stage itself, which stays L2 (that stage feeds the compiled MST kernels
    and is orthogonal to this issue).
    """
    if metric is None:
        return _METRICS["euclidean"]
    if callable(metric):
        return metric
    if metric in _METRICS:
        return _METRICS[metric]
    raise ValueError(
        f"metric must be a callable, one of {sorted(_METRICS)}, or None, got {metric!r}"
    )


def _minimax_medoid(d_sub: ndarray) -> int:
    """Index (within ``d_sub``) minimizing the maximum in-cluster distance.

    ``d_sub`` is the (k, k) minimax/iVAT sub-matrix restricted to one
    cluster's members -- the Bien & Tibshirani (2011) minimax-linkage
    prototype: always a real member of the cluster, never off-data.
    """
    return int(np.argmin(np.max(d_sub, axis=1)))


class IVATMeans:
    """
    IVAT-based clustering algorithm with scikit-learn compatible interface.

    ``refine`` controls how the clusters iVAT finds are represented and how
    new points are assigned to them (see GitHub issue #54 /
    ``docs/novel-niche.md``): iVAT's minimax recurrence recovers non-convex,
    elongated and chained structure, but a Euclidean-mean prototype and
    nearest-centroid assignment discard that advantage (the mean of a ring is
    in the hole). The options are:

    - ``"medoid"`` (default) -- each cluster's prototype is its
      minimax-linkage medoid (the member minimizing the maximum in-cluster
      minimax distance), so it is always a real point inside the cluster.
      Assignment is crisp nearest-prototype under ``metric``.
    - ``"relational"`` -- fits Non-Euclidean Relational FCM (NERFCM,
      Hathaway & Bezdek 1994) directly on the iVAT minimax matrix, producing
      a soft partition without ever taking a Euclidean mean. New points are
      scored via the relational out-of-sample extension, using ``metric``
      only to find each new point's single nearest training neighbor (the
      single-linkage insertion step). Sets ``membership_``.
    - ``"euclidean"`` -- the original behavior: Euclidean-mean prototypes,
      Euclidean nearest-centroid assignment. Kept for backward compatibility
      but is no longer the default since it reintroduces the geometry
      mismatch described in issue #54.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        n_levels: int = 1,
        random_state: Optional[int] = None,
        distance_backend: str = "auto",
        on_device: bool = False,
        dtype: str = "float32",
        refine: str = "medoid",
        m: float = 2.0,
        metric: MetricLike = None,
    ):
        self.n_clusters = n_clusters
        self.n_levels = n_levels
        self.random_state = random_state
        if refine not in ("medoid", "relational", "euclidean"):
            raise ValueError(
                f"refine must be 'medoid', 'relational', or 'euclidean', got {refine!r}"
            )
        self.refine = refine
        self.m = m
        self.metric = metric
        # distance_backend controls the pairwise-distance stage of fit():
        #   "auto" — GPU only when it is expected to win (float32, high feature
        #            dimension, CUDA present; see gpu.gpu_pairwise_beneficial),
        #            else the CPU C/OpenMP kernel;
        #   "gpu"  — force GPU (errors if no device);
        #   "cpu"  — force the CPU kernel.
        self.distance_backend = distance_backend
        # on_device: run the WHOLE VAT pipeline (distances + exact Boruvka MST +
        # ordering + iVAT recurrence) on the GPU with the dissimilarity matrix
        # kept resident — nothing but the O(n) order and the final image return
        # to the host (see gpu_vat.ivat_gpu). Opt-in; requires the matrix to fit
        # VRAM. At dtype="float32" the result matches the CPU path.
        self.on_device = on_device
        # dtype: storage precision of the resident matrix on the on_device path.
        #   "float32" (default) — matches the CPU result, half the memory;
        #   "float16"           — max scale, near-exact (a few near-tie flips);
        #   "float64"           — downgraded to float32 with a warning (use the
        #                         CPU path for exact float64). Ignored when the
        #                         on_device path is not taken.
        self.dtype = dtype
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self.hierarchy_: Optional[ClusterNode] = None
        # Soft membership matrix (n_samples, n_clusters_found); only set when
        # refine="relational".
        self.membership_: Optional[ndarray] = None
        self._ivat_result: Optional[IvatMeansResult] = None
        # Out-of-sample state for refine="relational" (see _fit_relational /
        # _predict_relational): kept in VAT-order (position) space so it lines
        # up with the iVAT matrix without an extra n x n permutation copy.
        self._relational_X_train: Optional[ndarray] = None
        self._relational_R_train: Optional[ndarray] = None
        self._relational_u_train: Optional[ndarray] = None
        self._relational_beta: float = 0.0

    def _use_on_device(self, X: ndarray) -> bool:
        if not self.on_device or not _gpu.is_available():
            return False
        # resident n x n matrix must fit VRAM (leave headroom for the reorder)
        n = X.shape[0]
        itemsize = 8 if np.asarray(X).dtype != np.float32 else 4
        try:
            free_bytes, _ = _gpu._cp.cuda.Device().mem_info
        except Exception:
            return False
        return n * n * itemsize < 0.6 * free_bytes

    def _compute_distances(self, X: ndarray) -> ndarray:
        backend = self.distance_backend
        if backend == "gpu" or (backend == "auto" and _gpu.gpu_pairwise_beneficial(X)):
            return _gpu.pairwise_distances_gpu(X)
        if backend not in ("auto", "cpu", "gpu"):
            raise ValueError(
                f"distance_backend must be 'auto', 'gpu', or 'cpu', got {backend!r}"
            )
        return _pairwise_distances(X)

    def fit(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> "IVATMeans":
        """
        Fit the IVAT clustering model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : IVATMeans
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self._use_on_device(X):
            # Whole VAT pipeline on the GPU (distances + exact Boruvka MST +
            # ordering + iVAT recurrence), matrix resident; only the O(n) order
            # and the final image return to the host. Apply the GPU-VAT dtype
            # policy here (f32 default, f16 opt-in, f64 -> f32 with a warning).
            store_dtype = _gpu_vat._resolve_vat_dtype(self.dtype)
            ivat_matrix, vat_order = _gpu_vat.ivat_gpu(X, dtype=store_dtype)
        else:
            distances = self._compute_distances(X)
            # `distances` is a throwaway intermediate, so let IVAT consume it in
            # place: the VAT/IVAT transform reorders it into the result rather
            # than allocating additional n x n buffers. This roughly halves peak
            # memory on large inputs (the dominant cost of fitting).
            ivat_matrix, _, vat_order = _compute_ivat(distances, inplace=True)

        ivat_result = get_ivat_levels(
            X, ivat_matrix, vat_order, n_levels=1, n_clusters=self.n_clusters
        )
        # n_levels=1 always yields a single result, never a list.
        assert isinstance(ivat_result, IvatMeansResult)
        self._ivat_result = ivat_result

        # Compute the hierarchy with the requested number of levels
        self.hierarchy_ = get_ivat_hierarchy(
            X, ivat_matrix, vat_order, n_levels=self.n_levels
        )

        if self.refine == "euclidean":
            self.cluster_centers_ = ivat_result.initial_centroids
            self.membership_ = None
            self.labels_ = self._assign_clusters(X)
        elif self.refine == "medoid":
            self.cluster_centers_ = self._compute_medoids(
                X, ivat_matrix, vat_order, ivat_result
            )
            self.membership_ = None
            self.labels_ = self._assign_clusters(X)
        else:  # "relational"
            self.cluster_centers_, self.membership_ = self._fit_relational(
                X, ivat_matrix, vat_order, ivat_result
            )
            self.labels_ = np.argmax(self.membership_, axis=1).astype(np.int32)

        return self

    def _compute_medoids(
        self,
        X: ndarray,
        ivat_matrix: ndarray,
        vat_order: ndarray,
        ivat_result: IvatMeansResult,
    ) -> ndarray:
        """Minimax-linkage medoid per cluster (Bien & Tibshirani 2011).

        The prototype is the cluster member minimizing the maximum in-cluster
        minimax (iVAT) distance -- always a real data point, unlike the
        Euclidean mean (see issue #54).
        """
        pos = np.argsort(vat_order)
        n_clusters = len(ivat_result.cluster_city_ids)
        medoids = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)
        for k, cluster_ids in enumerate(ivat_result.cluster_city_ids):
            positions = pos[cluster_ids]
            d_sub = ivat_matrix[np.ix_(positions, positions)]
            local_medoid = _minimax_medoid(d_sub)
            medoids[k] = X[cluster_ids[local_medoid]]
        return medoids

    def _fit_relational(
        self,
        X: ndarray,
        ivat_matrix: ndarray,
        vat_order: ndarray,
        ivat_result: IvatMeansResult,
    ) -> tuple[ndarray, ndarray]:
        """Fit NERFCM directly on the iVAT minimax matrix -- no Euclidean mean."""
        n = X.shape[0]
        cluster_city_ids = ivat_result.cluster_city_ids
        n_clusters = len(cluster_city_ids)
        pos = np.argsort(vat_order)

        # Hard initial partition from the iVAT cut, in VAT-order (position)
        # space (ivat_matrix's own index space).
        u_init = np.zeros((n, n_clusters), dtype=np.float64)
        for k, cluster_ids in enumerate(cluster_city_ids):
            u_init[pos[cluster_ids], k] = 1.0

        u_pos, beta = relational_fuzzy_c_means(
            ivat_matrix, n_clusters, self.m, u_init=u_init
        )

        # Scatter back to original sample order for the public membership_.
        membership = np.empty_like(u_pos)
        membership[vat_order] = u_pos

        self._relational_X_train = X[vat_order]
        self._relational_R_train = ivat_matrix
        self._relational_u_train = u_pos
        self._relational_beta = beta

        # Representative point per cluster (highest membership) -- for
        # plotting only, never used for assignment.
        centers = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)
        for k, cluster_ids in enumerate(cluster_city_ids):
            best_local = np.argmax(membership[cluster_ids, k])
            centers[k] = X[cluster_ids[best_local]]

        return centers, membership

    def predict(self, X: ndarray, batch_size: int = 10000) -> ndarray:
        """
        Predict cluster labels for samples in X.

        For large n_samples, prediction is done in batches to avoid
        allocating huge temporary arrays. Batch size can be tuned
        based on available memory.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.
        batch_size : int, optional
            Number of samples to process at once. Default 10000.
            Reduce if you encounter memory errors, increase if you have
            plenty of RAM and want faster prediction.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.refine == "relational":
            if self._relational_R_train is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")
        elif self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if self.refine == "relational":
            return self._predict_relational(X, batch_size)

        assert self.cluster_centers_ is not None  # checked above
        metric = _resolve_metric(self.metric)
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)

        # For small datasets, use direct computation (faster)
        if n_samples <= batch_size:
            distances = metric(X, self.cluster_centers_)
            return np.argmin(distances, axis=1)

        # For large datasets, process in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            distances = metric(X[start:end], self.cluster_centers_)
            labels[start:end] = np.argmin(distances, axis=1)

        return labels

    def _predict_relational(self, X: ndarray, batch_size: int) -> ndarray:
        """Out-of-sample NERFCM assignment via the single-linkage nearest-
        neighbor extension: a new point's minimax distance to any training
        point j is bounded by max(distance to its own nearest neighbor,
        that neighbor's minimax distance to j) -- exactly how Prim's MST
        would attach a new leaf connected by a single edge.
        """
        # Guaranteed by predict(): reachable only after a refine="relational" fit.
        assert self._relational_R_train is not None
        assert self._relational_X_train is not None
        assert self._relational_u_train is not None
        metric = _resolve_metric(self.metric)
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)
        x_train = self._relational_X_train
        r_train = self._relational_R_train
        u_train = self._relational_u_train
        beta = self._relational_beta

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X[start:end]
            dists = metric(x_batch, x_train)
            nn_idx = np.argmin(dists, axis=1)
            nn_dist = dists[np.arange(len(x_batch)), nn_idx]
            r_new = np.maximum(nn_dist[:, np.newaxis], r_train[nn_idx, :])
            membership = relational_out_of_sample_membership(
                r_new, r_train, u_train, self.m, beta=beta
            )
            labels[start:end] = np.argmax(membership, axis=1)

        return labels

    def fit_predict(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Fit the model and predict cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample in X.
        """
        self.fit(X, y, sample_weight)
        assert self.labels_ is not None  # set by fit()
        return self.labels_

    def _assign_clusters(self, X: ndarray) -> ndarray:
        """Assign cluster labels to samples based on nearest center."""
        return self.predict(X)

    def get_soft_labels(self) -> ndarray:
        """
        Get the soft membership values from a ``refine="relational"`` fit.

        Returns
        -------
        membership : ndarray of shape (n_samples, n_clusters)
            Soft membership matrix where membership[i, j] represents the
            partial membership of sample i to cluster j.
        """
        if self.membership_ is None:
            raise ValueError(
                "Soft labels are only available after fitting with "
                "refine='relational'."
            )
        return self.membership_
