"""Base classes and interfaces for clustering methods.

Provides a common interface for all clustering algorithms (KMeans, FuzzyCMeans,
IVATMeans) to enable clean method selection and swapping.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy import ndarray


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms.

    All clustering methods should inherit from this to maintain a consistent
    interface for easy swapping (e.g., from KMeans to FuzzyCMeans).
    """

    n_clusters: int
    random_state: Optional[int]
    cluster_centers_: Optional[ndarray]
    labels_: Optional[ndarray]

    @abstractmethod
    def fit(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> "BaseClusterer":
        """Fit the clustering model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : BaseClusterer
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """Predict cluster labels for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        pass

    @abstractmethod
    def fit_predict(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> ndarray:
        """Fit the model and predict cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
        sample_weight : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        pass
