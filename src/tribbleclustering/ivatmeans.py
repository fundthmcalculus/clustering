from typing import Optional

import numpy as np
from numpy import ndarray

from .pvat import get_ivat_levels, IvatMeansResult
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


class IVATMeans:
    """
    IVAT-based clustering algorithm with scikit-learn compatible interface.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: Optional[int] = None,
        distance_backend: str = "auto",
        on_device: bool = False,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # distance_backend controls the pairwise-distance stage of fit():
        #   "auto" — GPU only when it is expected to win (float32, high feature
        #            dimension, CUDA present; see gpu.gpu_pairwise_beneficial),
        #            else the CPU C/OpenMP kernel;
        #   "gpu"  — force GPU (errors if no device);
        #   "cpu"  — force the CPU kernel.
        self.distance_backend = distance_backend
        # on_device: run the whole VAT front-end (distances + exact Boruvka MST +
        # ordering) on the GPU with the dissimilarity matrix kept resident, then
        # finish the serial iVAT recurrence on the host (see gpu_vat.ivat_gpu).
        # Output is bit-identical to the CPU path. Opt-in: the front-end is
        # ~5-6x faster, but the iVAT recurrence still needs the matrix on the
        # host, so the one-time device->host copy of the n x n matrix can offset
        # the gain for the *full* fit — benchmark for your n. Requires the matrix
        # to fit VRAM.
        self.on_device = on_device
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self._ivat_result: Optional[IvatMeansResult] = None

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
            # Distances + exact Boruvka MST + ordering on the GPU (matrix
            # resident); serial iVAT recurrence finished on the host.
            ivat_matrix, vat_order = _gpu_vat.ivat_gpu(X)
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

        self.cluster_centers_ = ivat_result.initial_centroids
        self.labels_ = self._assign_clusters(X)

        return self

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
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)

        # For small datasets, use direct computation (faster)
        if n_samples <= batch_size:
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2
            )
            return np.argmin(distances, axis=1)

        # For large datasets, process in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]

            distances = np.linalg.norm(
                X_batch[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
                axis=2,
            )
            labels[start:end] = np.argmin(distances, axis=1)

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
