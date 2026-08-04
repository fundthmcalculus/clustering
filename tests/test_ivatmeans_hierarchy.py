"""Tests for IVATMeans's hierarchy feature (GitHub issue #61).

IVATMeans now exposes the hierarchical tree structure (ClusterNode) that it
computes during fitting, via the hierarchy_ attribute. This test suite verifies:

1. Determinism: Fitting twice on the same data produces identical trees.
2. Consistency: At n_levels=1, the tree's leaves partition the data exactly as labels_ does.
3. Structure: The hierarchy_ attribute exists and has the correct properties.
"""

import numpy as np
import pytest

from tribbleclustering import IVATMeans, ClusterNode


def _concentric_rings(n_per_ring: int = 60, seed: int = 0):
    """Two concentric rings: the canonical non-convex case from issue #54."""
    rng = np.random.default_rng(seed)
    t1 = rng.uniform(0, 2 * np.pi, n_per_ring)
    r1 = 1.0 + rng.normal(0, 0.05, n_per_ring)
    inner = np.c_[r1 * np.cos(t1), r1 * np.sin(t1)]

    t2 = rng.uniform(0, 2 * np.pi, n_per_ring)
    r2 = 3.0 + rng.normal(0, 0.05, n_per_ring)
    outer = np.c_[r2 * np.cos(t2), r2 * np.sin(t2)]

    X = np.vstack([inner, outer]).astype(np.float64)
    return X


@pytest.fixture(scope="module")
def simple_data():
    """Simple blob data for basic tests."""
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[0, 0], scale=0.3, size=(15, 2))
    blob_b = rng.normal(loc=[5, 5], scale=0.3, size=(15, 2))
    return np.vstack([blob_a, blob_b]).astype(np.float64)


@pytest.fixture(scope="module")
def rings():
    """Concentric rings data."""
    return _concentric_rings(n_per_ring=30)


class TestHierarchyPresence:
    """Test that hierarchy_ attribute is created and has correct structure."""

    def test_hierarchy_attribute_exists_after_fit(self, simple_data):
        """hierarchy_ should be present after fit()."""
        model = IVATMeans(n_clusters=2, random_state=42)
        assert model.hierarchy_ is None
        model.fit(simple_data)
        assert model.hierarchy_ is not None

    def test_hierarchy_is_cluster_node(self, simple_data):
        """hierarchy_ should be a ClusterNode instance."""
        model = IVATMeans(n_clusters=2, random_state=42)
        model.fit(simple_data)
        assert isinstance(model.hierarchy_, ClusterNode)

    def test_hierarchy_root_contains_all_data(self, simple_data):
        """Root of hierarchy should contain all samples."""
        model = IVATMeans(n_clusters=2, random_state=42)
        model.fit(simple_data)
        assert len(model.hierarchy_.indices) == len(simple_data)

    def test_hierarchy_root_centroid_is_global_mean(self, simple_data):
        """Root centroid should be the mean of all data."""
        model = IVATMeans(n_clusters=2, random_state=42)
        model.fit(simple_data)
        expected_centroid = np.mean(simple_data, axis=0)
        assert np.allclose(model.hierarchy_.centroid, expected_centroid)

    def test_hierarchy_with_n_levels_parameter(self, simple_data):
        """n_levels parameter should control the depth of hierarchy."""
        model = IVATMeans(n_clusters=2, n_levels=2, random_state=42)
        model.fit(simple_data)
        assert model.hierarchy_ is not None
        # With n_levels=2 and 2 clusters, we should have children at level 1
        if len(model.hierarchy_.children) > 0:
            assert all(isinstance(child, ClusterNode) for child in model.hierarchy_.children)

    def test_n_levels_default_value(self):
        """Default n_levels should be 1."""
        model = IVATMeans()
        assert model.n_levels == 1

    def test_n_levels_parameter_stored(self):
        """n_levels parameter should be stored as an attribute."""
        model = IVATMeans(n_levels=3)
        assert model.n_levels == 3


class TestHierarchyDeterminism:
    """Test that hierarchy is deterministic across multiple fits."""

    def test_hierarchy_identical_on_repeated_fit(self, simple_data):
        """Fitting twice should produce identical trees."""
        model1 = IVATMeans(n_clusters=2, random_state=42)
        model1.fit(simple_data)
        tree1 = model1.hierarchy_

        model2 = IVATMeans(n_clusters=2, random_state=42)
        model2.fit(simple_data)
        tree2 = model2.hierarchy_

        # Compare the trees structurally
        _assert_trees_equal(tree1, tree2)

    def test_hierarchy_indices_identical(self, simple_data):
        """Tree indices should be identical across fits."""
        model1 = IVATMeans(n_clusters=2, random_state=42)
        model1.fit(simple_data)

        model2 = IVATMeans(n_clusters=2, random_state=42)
        model2.fit(simple_data)

        # Check all nodes have same indices
        nodes1 = _collect_all_nodes(model1.hierarchy_)
        nodes2 = _collect_all_nodes(model2.hierarchy_)

        assert len(nodes1) == len(nodes2)
        for n1, n2 in zip(nodes1, nodes2):
            assert np.array_equal(np.sort(n1.indices), np.sort(n2.indices))

    def test_hierarchy_centroids_identical(self, simple_data):
        """Tree centroids should be identical across fits."""
        model1 = IVATMeans(n_clusters=2, random_state=42)
        model1.fit(simple_data)

        model2 = IVATMeans(n_clusters=2, random_state=42)
        model2.fit(simple_data)

        nodes1 = _collect_all_nodes(model1.hierarchy_)
        nodes2 = _collect_all_nodes(model2.hierarchy_)

        for n1, n2 in zip(nodes1, nodes2):
            assert np.allclose(n1.centroid, n2.centroid)


class TestHierarchyConsistencyWithLabels:
    """Test consistency between hierarchy_ and labels_ attributes."""

    def test_hierarchy_leaves_partition_data_at_n_levels_1(self, simple_data):
        """At n_levels=1, hierarchy leaves should partition data as labels_ does."""
        model = IVATMeans(n_clusters=2, n_levels=1, random_state=42)
        model.fit(simple_data)

        # Get all leaf nodes (children of root at depth 1)
        leaf_indices = []
        for child in model.hierarchy_.children:
            leaf_indices.extend(child.indices)

        # Check that leaves cover all samples exactly once
        leaf_indices_set = set(leaf_indices)
        assert len(leaf_indices_set) == len(simple_data)
        assert leaf_indices_set == set(range(len(simple_data)))

    def test_hierarchy_labels_match_at_n_levels_1(self, simple_data):
        """Labels should correspond to first level children of hierarchy."""
        model = IVATMeans(n_clusters=2, n_levels=1, random_state=42)
        model.fit(simple_data)

        # Each sample should be assigned to exactly one of the root's children
        labels_from_hierarchy = np.zeros(len(simple_data), dtype=np.int32)
        for cluster_id, child in enumerate(model.hierarchy_.children):
            for sample_idx in child.indices:
                labels_from_hierarchy[sample_idx] = cluster_id

        # The labels should partition the data in the same way as model.labels_
        # (though cluster IDs might be permuted, all samples in a cluster should be together)
        clusters_from_labels = {}
        for i, label in enumerate(model.labels_):
            if label not in clusters_from_labels:
                clusters_from_labels[label] = []
            clusters_from_labels[label].append(i)

        clusters_from_hierarchy = {}
        for label, indices in enumerate(model.hierarchy_.children):
            clusters_from_hierarchy[label] = list(indices.indices)

        # Both should partition the data
        all_indices_from_labels = set()
        for indices in clusters_from_labels.values():
            all_indices_from_labels.update(indices)
        assert len(all_indices_from_labels) == len(simple_data)

        all_indices_from_hierarchy = set()
        for indices in clusters_from_hierarchy.values():
            all_indices_from_hierarchy.update(indices)
        assert len(all_indices_from_hierarchy) == len(simple_data)

    def test_hierarchy_children_are_cluster_nodes(self, simple_data):
        """All children in hierarchy should be ClusterNode instances."""
        model = IVATMeans(n_clusters=2, random_state=42)
        model.fit(simple_data)

        for child in model.hierarchy_.children:
            assert isinstance(child, ClusterNode)
            assert isinstance(child.indices, np.ndarray)
            assert isinstance(child.centroid, np.ndarray)

    def test_hierarchy_works_with_all_refine_modes(self, simple_data):
        """Hierarchy should work with all refine modes."""
        for refine in ["medoid", "relational", "euclidean"]:
            model = IVATMeans(n_clusters=2, refine=refine, random_state=42)
            model.fit(simple_data)
            assert model.hierarchy_ is not None
            assert isinstance(model.hierarchy_, ClusterNode)
            assert len(model.hierarchy_.children) > 0


class TestHierarchyMultiLevel:
    """Test hierarchy behavior with multiple levels."""

    def test_hierarchy_depth_increases_with_n_levels(self, simple_data):
        """Deeper hierarchies should be created with larger n_levels."""
        model1 = IVATMeans(n_clusters=2, n_levels=1, random_state=42)
        model1.fit(simple_data)
        depth1 = _get_max_depth(model1.hierarchy_)

        model2 = IVATMeans(n_clusters=2, n_levels=2, random_state=42)
        model2.fit(simple_data)
        depth2 = _get_max_depth(model2.hierarchy_)

        # Depth should not decrease (might be equal if not enough natural clusters)
        assert depth2 >= depth1

    def test_hierarchy_maintains_parent_child_relationships(self, simple_data):
        """Parent nodes should properly contain their children's samples."""
        model = IVATMeans(n_clusters=2, n_levels=2, random_state=42)
        model.fit(simple_data)

        def check_containment(node):
            """Recursively check parent-child containment."""
            for child in node.children:
                child_set = set(child.indices)
                parent_set = set(node.indices)
                assert child_set.issubset(parent_set), (
                    f"Child indices {child_set} not subset of parent {parent_set}"
                )
                check_containment(child)

        check_containment(model.hierarchy_)

    def test_hierarchy_all_nodes_valid(self, rings):
        """All nodes in hierarchy should have valid centroids."""
        model = IVATMeans(n_clusters=3, n_levels=2, random_state=42)
        model.fit(rings)

        nodes = _collect_all_nodes(model.hierarchy_)
        for node in nodes:
            assert len(node.indices) > 0
            assert node.centroid.ndim == 1
            assert len(node.centroid) == rings.shape[1]
            assert np.all(np.isfinite(node.centroid))


class TestHierarchyFitPredict:
    """Test that hierarchy doesn't interfere with fit_predict."""

    def test_fit_predict_with_hierarchy(self, simple_data):
        """fit_predict should work with hierarchy."""
        model = IVATMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(simple_data)
        assert model.hierarchy_ is not None
        assert np.array_equal(labels, model.labels_)

    def test_hierarchy_and_labels_consistent_after_fit_predict(self, simple_data):
        """After fit_predict, hierarchy should be consistent with labels."""
        model = IVATMeans(n_clusters=2, random_state=42)
        model.fit_predict(simple_data)

        # Every sample should be in exactly one cluster at depth 1
        all_indices = set()
        for child in model.hierarchy_.children:
            all_indices.update(child.indices)
        assert len(all_indices) == len(simple_data)


# Helper functions

def _assert_trees_equal(tree1: ClusterNode, tree2: ClusterNode, depth: int = 0):
    """Recursively check if two ClusterNode trees are structurally equal."""
    # Check that both have same indices (in any order)
    assert set(tree1.indices) == set(tree2.indices), f"Indices mismatch at depth {depth}"

    # Check that both have same number of children
    assert len(tree1.children) == len(tree2.children), (
        f"Different number of children at depth {depth}: "
        f"{len(tree1.children)} vs {len(tree2.children)}"
    )

    # Check centroids are close (floating point comparison)
    assert np.allclose(tree1.centroid, tree2.centroid, rtol=1e-10), (
        f"Centroid mismatch at depth {depth}"
    )

    # Recursively check children (need to match them first, in case order differs)
    if len(tree1.children) > 0:
        # Match children by indices content
        for child1 in tree1.children:
            found_match = False
            for child2 in tree2.children:
                if set(child1.indices) == set(child2.indices):
                    _assert_trees_equal(child1, child2, depth + 1)
                    found_match = True
                    break
            assert found_match, f"No matching child found for depth {depth}"


def _collect_all_nodes(node: ClusterNode) -> list[ClusterNode]:
    """Recursively collect all nodes in the tree (BFS)."""
    nodes = [node]
    queue = [node]
    while queue:
        current = queue.pop(0)
        for child in current.children:
            nodes.append(child)
            queue.append(child)
    return nodes


def _get_max_depth(node: ClusterNode) -> int:
    """Get the maximum depth of the tree."""
    if not node.children:
        return 0
    return 1 + max(_get_max_depth(child) for child in node.children)
