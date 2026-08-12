"""K-Means clustering with optional GPU acceleration."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray

from .clustering_base import BaseClusterer

try:
    from .cfcm import kmeans as kmeans_algorithm

    _has_compiled_kmeans = True
except ImportError:
    kmeans_algorithm = None
    _has_compiled_kmeans = False


@dataclass
class KMeansResult:
    """Result of K-Means clustering."""

    cluster_centers_: ndarray
    labels_: ndarray
    inertia_: float
    n_iter_: int
    converged: bool


def _kmeans_plusplus(
    X: ndarray, n_clusters: int, random_state: Optional[int] = None
) -> ndarray:
    """Initialize cluster centers using k-means++ algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.
    n_clusters : int
        Number of clusters.
    random_state : int, optional
        Random seed.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Initial cluster centers.
    """
    n_samples = X.shape[0]
    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})"
        )

    if random_state is not None:
        np.random.seed(random_state)

    # Choose first center randomly
    center_idx = np.random.randint(n_samples)
    centers = [X[center_idx]]

    # Choose remaining centers
    for _ in range(1, n_clusters):
        # Compute distances to nearest center
        centers_arr = np.array(centers)
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - centers_arr[np.newaxis, :, :], axis=2
        )
        min_distances = np.min(distances, axis=1)

        # Choose next center with probability proportional to distance squared
        probabilities = min_distances**2
        probabilities /= probabilities.sum()
        cumsum = np.cumsum(probabilities)
        r = np.random.rand()
        next_idx = np.searchsorted(cumsum, r)
        centers.append(X[next_idx])

    return np.array(centers)


def _compute_distances_gram(X: ndarray, centers: ndarray) -> ndarray:
    """Compute squared Euclidean distances using gram identity.

    Computes ||x - c||^2 = ||x||^2 - 2*x*c^T + ||c||^2 using GEMM.
    This is more efficient than the direct ||x - c||^2 computation
    and avoids allocating large intermediate arrays.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.
    centers : ndarray of shape (n_clusters, n_features)
        Cluster centers.

    Returns
    -------
    distances_sq : ndarray of shape (n_samples, n_clusters)
        Squared distances from each point to each center.
    """
    x_norm2 = np.sum(X**2, axis=1, keepdims=True)  # (n_samples, 1)
    centers_norm2 = np.sum(centers**2, axis=1)  # (n_clusters,)
    xc = 2 * np.dot(X, centers.T)  # (n_samples, n_clusters) via GEMM
    distances_sq = x_norm2 + centers_norm2 - xc
    return np.maximum(distances_sq, 0)


def kmeans(
    X: ndarray,
    n_clusters: int,
    *,
    max_iter: int = 100,
    init: str = "k-means++",
    tol: float = 1e-4,
    indices: Optional[ndarray] = None,
    initial_guess: Optional[ndarray] = None,
) -> KMeansResult:
    """
    Compute K-Means clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data points.
    n_clusters : int
        Number of clusters.
    max_iter : int, optional
        Maximum number of iterations. Default 100.
    init : str, optional
        Initialization method: 'k-means++' (default) or 'random'.
    tol : float, optional
        Relative tolerance for convergence. Default 1e-4.
    indices : ndarray of shape (n_clusters,), optional
        Indices of initial cluster centers.
    initial_guess : ndarray of shape (n_clusters, n_features), optional
        Initial cluster centers.

    Returns
    -------
    result : KMeansResult
        Clustering result with cluster_centers_, labels_, inertia_, n_iter_, converged.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})"
        )

    if initial_guess is not None and indices is not None:
        raise ValueError("initial_guess and indices cannot both be provided")

    # Initialize centers
    if indices is not None:
        centers = X[indices].copy()
    elif initial_guess is not None:
        if initial_guess.shape != (n_clusters, X.shape[1]):
            raise ValueError(
                f"initial_guess must have shape ({n_clusters}, {X.shape[1]}), "
                f"got {initial_guess.shape}"
            )
        centers = initial_guess.copy()
    elif init == "k-means++":
        centers = _kmeans_plusplus(X, n_clusters)
    elif init == "random":
        indices = np.random.choice(n_samples, size=n_clusters, replace=False)
        centers = X[indices].copy()
    else:
        raise ValueError(f"init must be 'k-means++' or 'random', got {init!r}")

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # Compute squared distances using gram identity
        distances_sq = _compute_distances_gram(X, centers)
        distances = np.sqrt(distances_sq)

        # Assign points to nearest cluster
        labels = np.argmin(distances, axis=1).astype(np.int32)

        # Update centers
        centers_new = np.empty_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers_new[k] = X[mask].mean(axis=0)
            else:
                # Keep the old center if no points assigned (rare in practice)
                centers_new[k] = centers[k]

        n_iter = iteration + 1

        # Check convergence
        if np.allclose(centers_new, centers, rtol=tol, atol=1e-8):
            converged = True
            centers = centers_new
            break

        centers = centers_new

    # Compute final assignments and inertia
    distances_sq = _compute_distances_gram(X, centers)
    distances = np.sqrt(distances_sq)
    labels = np.argmin(distances, axis=1).astype(np.int32)
    inertia = np.sum(np.min(distances_sq, axis=1))

    return KMeansResult(
        cluster_centers_=centers,
        labels_=labels,
        inertia_=inertia,
        n_iter_=n_iter,
        converged=converged,
    )


_GPU_KMEANS_MIN_SAMPLES = 5000


class KMeans(BaseClusterer):
    """
    K-Means clustering with scikit-learn compatible interface.

    Can be used interchangeably with FuzzyCMeans and IVATMeans via the
    BaseClusterer interface.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        init: str = "k-means++",
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize K-Means clustering.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        init : str, optional
            Initialization method: 'k-means++' (default) or 'random'.
        max_iter : int, optional
            Maximum number of iterations. Default 100.
        tol : float, optional
            Relative tolerance for convergence. Default 1e-4.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.converged: Optional[bool] = None


    def fit(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> "KMeans":
        """
        Fit the K-Means clustering model.

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
        self : KMeans
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        result = kmeans(
            X,
            self.n_clusters,
            max_iter=self.max_iter,
            init=self.init,
            tol=self.tol,
        )

        self.cluster_centers_ = result.cluster_centers_
        self.labels_ = result.labels_
        self.inertia_ = result.inertia_
        self.n_iter_ = result.n_iter_
        self.converged = result.converged

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

        # For small datasets, use direct computation
        if n_samples <= batch_size:
            distances_sq = _compute_distances_gram(X, self.cluster_centers_)
            return np.argmin(distances_sq, axis=1).astype(np.int32)

        # For large datasets, process in batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]
            distances_sq = _compute_distances_gram(X_batch, self.cluster_centers_)
            labels[start:end] = np.argmin(distances_sq, axis=1).astype(np.int32)

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
