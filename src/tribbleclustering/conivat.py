"""ConiVAT — constraint-based iVAT (Rathore, Bezdek, Santi & Ratti, 2020).

Reference
---------
P. Rathore, J. C. Bezdek, P. Santi, and C. Ratti, "ConiVAT: Cluster Tendency
Assessment and Clustering with Partial Background Knowledge," arXiv:2008.09570,
2020. A committed copy lives at ``docs/papers/Rathore_2020_ConiVAT.pdf``.

ConiVAT is a *semi-supervised* extension of iVAT. Given partial background
knowledge as pairwise constraints — a "similar" (must-link) set ``S`` and a
"dissimilar" (cannot-link) set ``D`` — it improves the reordered dissimilarity
image (RDI) for noisy data and data with "bridge" points between clusters, the
known failure mode of plain VAT/iVAT single-linkage chaining.

The algorithm has three constraint-aware stages layered on top of ordinary
iVAT (paper §4):

1. **Constraint pre-processing** (§4.1): expand the given constraints with the
   transitivity property (must-link is an equivalence relation; must-link then
   cannot-link implies cannot-link) and drop constraints that are mutually
   inconsistent.
2. **Metric learning** (§4.2): learn a Mahalanobis metric ``A`` (Xing et al.'s
   MMC) that pulls "similar" points together and pushes "dissimilar" points
   apart, then transform the data into that metric's space.
3. **Minimum transitive dissimilarity** (§4.3): force the "similar" pair
   distances to zero, then apply the path-based minimax (transitive) distance
   transform. The paper notes this transform is *exactly* the non-recursive
   iVAT transform, so we reuse this repository's :func:`compute_ivat` for it.

Finally VAT ordering of that matrix yields the RDI; cutting the ``k - 1``
longest MST edges (what :func:`get_ivat_levels` does off the reordered
diagonal) gives ``k`` single-linkage clusters.

This is a **pure-Python/numpy reference implementation** — it deliberately
mirrors the pure-Python VAT/iVAT path and does not (yet) have a compiled
Cython twin.
"""

from typing import Optional, Sequence

import numpy as np
from numpy import ndarray

from .pvat import compute_ivat, get_ivat_levels, IvatMeansResult
from .util import pairwise_distances

ConstraintPair = tuple[int, int]
ConstraintList = Sequence[ConstraintPair]


# --------------------------------------------------------------------------- #
# 4.1 Constraint generation and pre-processing
# --------------------------------------------------------------------------- #
def generate_constraints_from_labels(
    labels: ndarray,
    n_constraints: int = 30,
    random_state: Optional[int] = None,
) -> tuple[list[ConstraintPair], list[ConstraintPair]]:
    """Sample must-link / cannot-link constraints from ground-truth labels.

    Mirrors the paper's protocol (§5.3): draw ``n_constraints`` random pairs of
    distinct objects; a pair whose labels match becomes a "similar"
    (must-link) constraint, otherwise a "dissimilar" (cannot-link) constraint.

    Parameters
    ----------
    labels : ndarray of shape (n_samples,)
        Class label per object. Used only to generate constraints.
    n_constraints : int
        Number of random pairs to draw.
    random_state : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    must_link, cannot_link : list of (int, int)
        The sampled "similar" and "dissimilar" pairs.
    """
    labels = np.asarray(labels).ravel()
    n = labels.shape[0]
    if n < 2:
        return [], []
    rng = np.random.default_rng(random_state)

    must_link: list[ConstraintPair] = []
    cannot_link: list[ConstraintPair] = []
    seen: set[ConstraintPair] = set()
    # Bound the attempts so we cannot loop forever on tiny / degenerate inputs.
    max_attempts = max(n_constraints * 20, 100)
    attempts = 0
    while len(must_link) + len(cannot_link) < n_constraints and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        key = (int(min(i, j)), int(max(i, j)))
        if key in seen:
            continue
        seen.add(key)
        if labels[i] == labels[j]:
            must_link.append(key)
        else:
            cannot_link.append(key)
    return must_link, cannot_link


class _UnionFind:
    """Minimal union-find for building must-link equivalence classes."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def expand_constraints(
    must_link: ConstraintList,
    cannot_link: ConstraintList,
    n_samples: int,
    max_pairs: int = 100_000,
) -> tuple[list[ConstraintPair], list[ConstraintPair]]:
    """Apply the transitive closure to the constraint sets and drop conflicts.

    Must-link is treated as an equivalence relation, so its transitive closure
    is the set of connected components (built with union-find). Cannot-link
    then propagates across whole components: if any member of component ``A``
    cannot-link any member of component ``B``, every cross pair ``(a, b)`` with
    ``a in A, b in B`` is a cannot-link.

    Inconsistent constraints — a cannot-link pair whose two endpoints landed in
    the *same* must-link component — are discarded (paper §4.1).

    Parameters
    ----------
    must_link, cannot_link : sequence of (int, int)
        Input constraint pairs (indices into the data).
    n_samples : int
        Number of objects; indices must be in ``[0, n_samples)``.
    max_pairs : int
        Safety cap on the number of expanded pairs of each type. Expansion is
        quadratic in component size, so this guards against pathological inputs.

    Returns
    -------
    ml_expanded, cl_expanded : list of (int, int)
        Transitively-closed, conflict-free "similar" and "dissimilar" pairs
        (each stored once as ``(i, j)`` with ``i < j``).
    """
    uf = _UnionFind(n_samples)
    for i, j in must_link:
        uf.union(int(i), int(j))

    # Group objects by must-link component.
    components: dict[int, list[int]] = {}
    for idx in range(n_samples):
        components.setdefault(uf.find(idx), []).append(idx)

    def _pair(i: int, j: int) -> ConstraintPair:
        return (i, j) if i < j else (j, i)

    # Expand must-link to all within-component pairs.
    ml_expanded: set[ConstraintPair] = set()
    for members in components.values():
        if len(members) < 2:
            continue
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                ml_expanded.add(_pair(members[a_i], members[b_i]))
                if len(ml_expanded) >= max_pairs:
                    break
            if len(ml_expanded) >= max_pairs:
                break
        if len(ml_expanded) >= max_pairs:
            break

    # Expand cannot-link across components, skipping inconsistent (same-root)
    # pairs. De-duplicate at the component-root level first so we only expand
    # each pair of components once.
    root_pairs: set[ConstraintPair] = set()
    for i, j in cannot_link:
        ri, rj = uf.find(int(i)), uf.find(int(j))
        if ri == rj:
            # Endpoints are must-linked yet declared dissimilar -> inconsistent.
            continue
        root_pairs.add(_pair(ri, rj))

    cl_expanded: set[ConstraintPair] = set()
    for ri, rj in root_pairs:
        for a in components[ri]:
            for b in components[rj]:
                cl_expanded.add(_pair(a, b))
                if len(cl_expanded) >= max_pairs:
                    break
            if len(cl_expanded) >= max_pairs:
                break
        if len(cl_expanded) >= max_pairs:
            break

    return sorted(ml_expanded), sorted(cl_expanded)


# --------------------------------------------------------------------------- #
# 4.2 Metric learning (Xing et al. 2002 MMC, used by ConiVAT)
# --------------------------------------------------------------------------- #
def _project_halfspace(A: ndarray, G: ndarray, g_norm_sq: float) -> ndarray:
    """Project ``A`` onto the half-space ``{A : <G, A>_F <= 1}``.

    ``<G, A>_F = sum_(i,j)-in-S ||x_i - x_j||_A^2`` is linear in ``A``, so the
    "similar" constraint is a single half-space with closed-form projection.
    """
    c = float(np.sum(G * A))
    if c <= 1.0:
        return A
    return A - ((c - 1.0) / g_norm_sq) * G


def _project_psd(A: ndarray) -> ndarray:
    """Project ``A`` onto the PSD cone (symmetrize, clip negative eigenvalues)."""
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals) @ eigvecs.T


def learn_metric(
    X: ndarray,
    must_link: ConstraintList,
    cannot_link: ConstraintList,
    *,
    max_iters: int = 100,
    learning_rate: float = 0.1,
    tol: float = 1e-4,
    projection_iters: int = 20,
) -> ndarray:
    """Learn a Mahalanobis metric matrix ``A`` from pairwise constraints.

    Implements Xing et al.'s metric-learning problem as used by ConiVAT
    (paper §4.2, Eqs. 2-4): maximize the summed distance over "dissimilar"
    pairs subject to the summed squared distance over "similar" pairs being
    ``<= 1`` and ``A`` positive semi-definite, via projected gradient ascent.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data (in the original space).
    must_link, cannot_link : sequence of (int, int)
        The (already expanded) "similar" and "dissimilar" pairs.
    max_iters : int
        Maximum gradient-ascent iterations.
    learning_rate : float
        Gradient-ascent step size.
    tol : float
        Relative objective-change convergence threshold.
    projection_iters : int
        Alternating projections onto the two convex sets per gradient step.

    Returns
    -------
    A : ndarray of shape (n_features, n_features)
        The learned PSD metric matrix. Falls back to the identity (Euclidean)
        when either constraint set is empty.
    """
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[1]
    A = np.eye(p, dtype=np.float64)
    if len(must_link) == 0 or len(cannot_link) == 0:
        return A

    S = np.array([X[i] - X[j] for i, j in must_link], dtype=np.float64)
    Dd = np.array([X[i] - X[j] for i, j in cannot_link], dtype=np.float64)

    # G = sum_(i,j)-in-S (x_i - x_j)(x_i - x_j)^T. The "similar" constraint is
    # <G, A>_F <= 1.
    G = S.T @ S
    g_norm_sq = float(np.sum(G * G))
    if g_norm_sq <= 0.0:
        return A

    A = _project_halfspace(A, G, g_norm_sq)
    prev_obj = -np.inf
    for _ in range(max_iters):
        # grad of sum_d ||d||_A w.r.t. A  =  sum_d (d d^T) / (2 ||d||_A).
        quad = np.einsum("ij,jk,ik->i", Dd, A, Dd)  # d^T A d per dissimilar pair
        quad = np.maximum(quad, 1e-12)
        w = 1.0 / (2.0 * np.sqrt(quad))
        grad = (Dd * w[:, None]).T @ Dd
        A = A + learning_rate * grad

        # Alternate the two convex projections; end on the PSD projection so
        # the returned A is a valid metric.
        for _ in range(projection_iters):
            A = _project_halfspace(A, G, g_norm_sq)
            A = _project_psd(A)

        obj = float(
            np.sum(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Dd, A, Dd), 0.0)))
        )
        if abs(obj - prev_obj) < tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj

    return A


def transform_with_metric(X: ndarray, A: ndarray) -> ndarray:
    """Map ``X`` into the space where Euclidean distance equals ``d_A``.

    Factor ``A = L L^T`` via its eigendecomposition; then Euclidean distances
    of ``X @ L`` equal the Mahalanobis distances ``d_A`` in the original space.
    """
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0.0, None)
    L = eigvecs * np.sqrt(eigvals)
    return np.asarray(X, dtype=np.float64) @ L


# --------------------------------------------------------------------------- #
# 4.3 ConiVAT: minimum transitive dissimilarity + VAT ordering
# --------------------------------------------------------------------------- #
def compute_conivat(
    X: ndarray,
    must_link: Optional[ConstraintList] = None,
    cannot_link: Optional[ConstraintList] = None,
    labels: Optional[ndarray] = None,
    n_constraints: int = 30,
    metric_learning: bool = True,
    random_state: Optional[int] = None,
    inplace: bool = False,
) -> tuple[ndarray, list, ndarray]:
    """Compute the ConiVAT reordered dissimilarity image for ``X``.

    Runs the three constraint-aware stages of ConiVAT and returns the same
    triple as :func:`compute_ivat`, so downstream helpers such as
    :func:`get_ivat_levels` work unchanged.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature vectors. ConiVAT needs the vectors (not just a distance
        matrix) because metric learning operates in feature space.
    must_link, cannot_link : sequence of (int, int), optional
        "Similar" and "dissimilar" constraints. When both are ``None`` and
        ``labels`` is given, constraints are sampled from ``labels``.
    labels : ndarray, optional
        Ground-truth labels used only to generate constraints when explicit
        constraint sets are not supplied.
    n_constraints : int
        Number of constraints to sample when generating from ``labels``.
    metric_learning : bool
        When ``True`` (and both constraint sets are non-empty), learn a
        Mahalanobis metric and transform the data before building distances.
    random_state : int, optional
        Seed for constraint sampling.
    inplace : bool
        Passed through to the iVAT transform (the distance matrix built here
        is a throwaway intermediate, so ``True`` roughly halves peak memory).

    Returns
    -------
    conivat_matrix : ndarray of shape (n_samples, n_samples)
        The ConiVAT (minimum-transitive) reordered dissimilarity matrix.
    argmin_seq : list
        The iVAT argmin sequence.
    vat_order : ndarray
        The VAT permutation (ordering) sequence.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    n = X.shape[0]

    # Stage 4.1 — resolve and pre-process constraints.
    if must_link is None and cannot_link is None and labels is not None:
        must_link, cannot_link = generate_constraints_from_labels(
            labels, n_constraints=n_constraints, random_state=random_state
        )
    must_link = list(must_link) if must_link is not None else []
    cannot_link = list(cannot_link) if cannot_link is not None else []
    ml_expanded, cl_expanded = expand_constraints(must_link, cannot_link, n)

    # Stage 4.2 — metric learning + space transform.
    if metric_learning and ml_expanded and cl_expanded:
        A = learn_metric(X, ml_expanded, cl_expanded)
        X_t = transform_with_metric(X, A)
    else:
        X_t = np.asarray(X, dtype=np.float64)

    # Stage 4.3 — build distances, impose "similar" constraints (distance 0),
    # then apply the path-based minimax (transitive) transform == iVAT.
    distances = pairwise_distances(np.ascontiguousarray(X_t))
    for i, j in ml_expanded:
        distances[i, j] = 0.0
        distances[j, i] = 0.0

    return compute_ivat(distances, inplace=inplace)


class ConiVAT:
    """ConiVAT semi-supervised clustering with a scikit-learn-style interface.

    A constraint-guided sibling of :class:`IVATMeans`: it accepts must-link /
    cannot-link constraints (or samples them from labels passed to
    :meth:`fit`), builds the ConiVAT reordered dissimilarity image, and then
    extracts clusters exactly as IVATMeans does (cut the longest MST edges off
    the reordered diagonal, take per-cluster centroids as centers).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        must_link: Optional[ConstraintList] = None,
        cannot_link: Optional[ConstraintList] = None,
        n_constraints: int = 30,
        metric_learning: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.must_link = must_link
        self.cannot_link = cannot_link
        self.n_constraints = n_constraints
        self.metric_learning = metric_learning
        self.random_state = random_state
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self._ivat_result: Optional[IvatMeansResult] = None

    def fit(
        self,
        X: ndarray,
        y: Optional[ndarray] = None,
        sample_weight: Optional[ndarray] = None,
    ) -> "ConiVAT":
        """Fit the ConiVAT clustering model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray, optional
            Ground-truth labels used only to sample constraints when explicit
            constraint sets were not provided at construction time.
        sample_weight : Ignored
            Present for API consistency by convention.

        Returns
        -------
        self : ConiVAT
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        conivat_matrix, _, vat_order = compute_conivat(
            X,
            must_link=self.must_link,
            cannot_link=self.cannot_link,
            labels=y,
            n_constraints=self.n_constraints,
            metric_learning=self.metric_learning,
            random_state=self.random_state,
            inplace=True,
        )

        ivat_result = get_ivat_levels(
            X, conivat_matrix, vat_order, n_levels=1, n_clusters=self.n_clusters
        )
        # n_levels=1 always yields a single result, never a list.
        assert isinstance(ivat_result, IvatMeansResult)
        self._ivat_result = ivat_result

        self.cluster_centers_ = ivat_result.initial_centroids
        self.labels_ = self._assign_clusters(X)
        return self

    def predict(self, X: ndarray, batch_size: int = 10000) -> ndarray:
        """Predict cluster labels for ``X`` by nearest cluster center.

        For large ``n_samples`` prediction is batched to avoid allocating huge
        temporaries. Reduce ``batch_size`` under memory pressure.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        n_samples = X.shape[0]
        if n_samples <= batch_size:
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2
            )
            return np.argmin(distances, axis=1)

        labels = np.empty(n_samples, dtype=np.int32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            distances = np.linalg.norm(
                X[start:end, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
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
        """Fit the model and return cluster labels for ``X``."""
        self.fit(X, y, sample_weight)
        assert self.labels_ is not None  # set by fit()
        return self.labels_

    def _assign_clusters(self, X: ndarray) -> ndarray:
        """Assign cluster labels to samples based on the nearest center."""
        return self.predict(X)
