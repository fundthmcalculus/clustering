"""Tests for vat_prim_mst_seq bounded behavior fix.

Regression tests to verify that vat_prim_mst_seq correctly computes the VAT
ordering by comparing it with the reference compute_vat implementation.
"""

import numpy as np
import pytest

import tribbleclustering as tc
from tribbleclustering.util import pairwise_distances
from tribbleclustering.pvat import vat_prim_mst_seq


def agreement_ratio(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute the fraction of elements that agree between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")
    return np.mean(seq1 == seq2)


@pytest.fixture
def random_samples():
    """Generate a random sample dataset."""
    np.random.seed(42)
    return np.random.randn(50, 10).astype(np.float64)


@pytest.fixture
def clustered_samples():
    """Generate a dataset with clear cluster structure."""
    np.random.seed(42)
    # Create 3 clusters
    cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 5) + np.array([5, 5, 5, 5, 5])
    cluster3 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
    return np.vstack([cluster1, cluster2, cluster3]).astype(np.float64)


class TestVatPrimMstSeq:
    """Test suite for vat_prim_mst_seq bounded behavior."""

    def test_basic_random_float64(self, random_samples):
        """Test that vat_prim_mst_seq matches compute_vat on random data (float64)."""
        # Compute reference VAT ordering from distance matrix
        dist_matrix = pairwise_distances(random_samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        # Compute VAT ordering directly from samples
        seq_ordering = vat_prim_mst_seq(random_samples)

        # Verify they match (should be identical)
        assert len(seq_ordering) == len(
            vat_ordering
        ), f"Sequence length mismatch: {len(seq_ordering)} vs {len(vat_ordering)}"
        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement with reference, got {agreement:.3f}"

    def test_basic_random_float32(self):
        """Test that vat_prim_mst_seq works correctly with float32 precision."""
        np.random.seed(42)
        samples = np.random.randn(30, 8).astype(np.float32)

        dist_matrix = pairwise_distances(samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        seq_ordering = vat_prim_mst_seq(samples)

        assert len(seq_ordering) == len(vat_ordering)
        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement with reference (float32), got {agreement:.3f}"

    def test_clustered_data(self, clustered_samples):
        """Test that vat_prim_mst_seq handles clustered data correctly."""
        dist_matrix = pairwise_distances(clustered_samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        seq_ordering = vat_prim_mst_seq(clustered_samples)

        assert len(seq_ordering) == len(vat_ordering)
        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement on clustered data, got {agreement:.3f}"

    def test_small_dataset(self):
        """Test on small dataset (n=5)."""
        np.random.seed(42)
        samples = np.random.randn(5, 3).astype(np.float64)

        dist_matrix = pairwise_distances(samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        seq_ordering = vat_prim_mst_seq(samples)

        assert len(seq_ordering) == 5
        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement on small dataset, got {agreement:.3f}"

    def test_medium_dataset(self):
        """Test on medium dataset (n=100)."""
        np.random.seed(42)
        samples = np.random.randn(100, 15).astype(np.float64)

        dist_matrix = pairwise_distances(samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        seq_ordering = vat_prim_mst_seq(samples)

        assert len(seq_ordering) == 100
        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement on medium dataset (n=100), got {agreement:.3f}"

    def test_ordering_is_valid_permutation(self, random_samples):
        """Test that the returned sequence is a valid permutation."""
        seq_ordering = vat_prim_mst_seq(random_samples)

        # All indices should be unique and within bounds
        assert len(seq_ordering) == len(random_samples)
        assert len(np.unique(seq_ordering)) == len(random_samples)
        assert np.min(seq_ordering) == 0
        assert np.max(seq_ordering) == len(random_samples) - 1

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_agreement_across_sizes(self, n):
        """Test that vat_prim_mst_seq has high agreement across different dataset sizes."""
        np.random.seed(42)
        samples = np.random.randn(n, 8).astype(np.float64)

        dist_matrix = pairwise_distances(samples)
        _, vat_ordering = tc.compute_vat(dist_matrix)

        seq_ordering = vat_prim_mst_seq(samples)

        agreement = agreement_ratio(seq_ordering, vat_ordering)
        assert (
            agreement == 1.0
        ), f"Expected 100% agreement for n={n}, got {agreement:.3f}"

    def test_deterministic_result(self, random_samples):
        """Test that vat_prim_mst_seq returns the same result on repeated calls."""
        result1 = vat_prim_mst_seq(random_samples)
        result2 = vat_prim_mst_seq(random_samples)

        np.testing.assert_array_equal(result1, result2)
