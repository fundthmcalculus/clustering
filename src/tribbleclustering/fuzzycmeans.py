from typing import Optional

import numpy as np
from numpy import ndarray

try:
    from .cfcm import fuzzy_c_means as fcm_algorithm

    _has_compiled_fcm = True
except ImportError:
    from .fcm import fuzzy_c_means as fcm_algorithm

    _has_compiled_fcm = False


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering algorithm with scikit-learn compatible interface.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.random_state = random_state
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self.membership_matrix_: Optional[ndarray] = None

    def fit(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> "FuzzyCMeans":
        """
        Fit the Fuzzy C-Means clustering model.

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
        self : FuzzyCMeans
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.cluster_centers_, self.membership_matrix_ = fcm_algorithm(
            X, self.n_clusters, m=self.m
        )

        self.labels_ = self._get_hard_labels()

        return self

    def predict(self, X: ndarray, batch_size: int = 10000) -> ndarray:
        """
        Predict cluster labels for samples in X using hard assignment.

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
                axis=2
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
        return self.labels_

    def get_soft_labels(self) -> ndarray:
        """
        Get the soft membership values for all samples.

        Returns
        -------
        membership : ndarray of shape (n_samples, n_clusters)
            Soft membership matrix where membership[i, j] represents the
            partial membership of sample i to cluster j.
        """
        if self.membership_matrix_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.membership_matrix_

    def _get_hard_labels(self) -> ndarray:
        """Convert soft membership matrix to hard cluster assignments."""
        return np.argmax(self.membership_matrix_, axis=1)
