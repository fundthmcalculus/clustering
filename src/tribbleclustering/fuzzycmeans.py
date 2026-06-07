from typing import Optional, Literal

import numpy as np
from numpy import ndarray

from .fcm import fuzzy_c_means as fcm_algorithm


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering algorithm with scikit-learn compatible interface.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        method: Literal["gd", "iter"] = "iter",
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.method = method
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
            X, self.n_clusters, m=self.m, method=self.method
        )

        self.labels_ = self._get_hard_labels()

        return self

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict cluster labels for samples in X using hard assignment.

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
