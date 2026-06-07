from typing import Optional

import numpy as np
from numpy import ndarray

from .pvat import compute_ivat, get_ivat_levels, IvatMeansResult

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

    def __init__(self, n_levels: int = 1, random_state: Optional[int] = None):
        self.n_levels = n_levels
        self.random_state = random_state
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self._ivat_result = None

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

        distances = _pairwise_distances(X)
        ivat_matrix, vat_matrix, _, vat_order = compute_ivat(distances, inplace=False)

        self._ivat_result = get_ivat_levels(
            X, ivat_matrix, vat_order, n_levels=self.n_levels
        )

        if self.n_levels == 1:
            result: IvatMeansResult = self._ivat_result
            self.cluster_centers_ = result.initial_centroids
            self.labels_ = self._assign_clusters(X)
        else:
            result = self._ivat_result[0]
            self.cluster_centers_ = result.initial_centroids
            self.labels_ = self._assign_clusters(X)

        return self

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict cluster labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.

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

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2
        )
        return np.argmin(distances, axis=1)

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
        return self.labels_

    def _assign_clusters(self, X: ndarray) -> ndarray:
        """Assign cluster labels to samples based on nearest center."""
        return self.predict(X)
