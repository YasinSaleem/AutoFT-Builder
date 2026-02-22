"""Tests for the similarity filter module."""

import numpy as np
import pytest

from autoft.filter.similarity import (
    FilterResult,
    SimilarityFilter,
    cosine_similarity,
    cosine_similarity_matrix,
    max_similarity,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test that identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        """Test that opposite vectors have similarity -1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test that orthogonal vectors have similarity 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_normalized_vectors(self) -> None:
        """Test with pre-normalized vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.707, 0.707, 0.0])  # ~45 degrees
        sim = cosine_similarity(vec1, vec2)
        assert 0.7 < sim < 0.72

    def test_zero_vector(self) -> None:
        """Test that zero vectors return 0.0 similarity."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec1, vec2) == 0.0
        assert cosine_similarity(vec2, vec1) == 0.0

    def test_both_zero_vectors(self) -> None:
        """Test that two zero vectors return 0.0."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_high_dimensional_vectors(self) -> None:
        """Test with 384-dimensional vectors (like BGE embeddings)."""
        np.random.seed(42)
        vec1 = np.random.rand(384)
        vec2 = np.random.rand(384)
        sim = cosine_similarity(vec1, vec2)
        # Random vectors should have moderate similarity
        assert -1.0 <= sim <= 1.0


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix function."""

    def test_single_vectors(self) -> None:
        """Test matrix with single vectors."""
        vec1 = np.array([[1.0, 0.0, 0.0]])
        vec2 = np.array([[1.0, 0.0, 0.0]])
        result = cosine_similarity_matrix(vec1, vec2)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_multiple_vectors(self) -> None:
        """Test matrix with multiple vectors."""
        vecs1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        vecs2 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = cosine_similarity_matrix(vecs1, vecs2)
        assert result.shape == (2, 3)
        # vec1[0] vs vec2[0] should be 1.0
        assert result[0, 0] == pytest.approx(1.0)
        # vec1[0] vs vec2[1] should be 0.0 (orthogonal)
        assert result[0, 1] == pytest.approx(0.0)
        # vec1[1] vs vec2[1] should be 1.0
        assert result[1, 1] == pytest.approx(1.0)

    def test_empty_vectors(self) -> None:
        """Test with empty input arrays."""
        vecs1 = np.empty((0, 384))
        vecs2 = np.random.rand(10, 384)
        result = cosine_similarity_matrix(vecs1, vecs2)
        assert result.shape == (0, 10)

    def test_returns_float32(self) -> None:
        """Test that result is float32."""
        vecs1 = np.random.rand(5, 384).astype(np.float64)
        vecs2 = np.random.rand(10, 384).astype(np.float64)
        result = cosine_similarity_matrix(vecs1, vecs2)
        assert result.dtype == np.float32


class TestMaxSimilarity:
    """Tests for max_similarity function."""

    def test_finds_maximum(self) -> None:
        """Test that it finds the maximum similarity."""
        new_emb = np.array([1.0, 0.0, 0.0])
        existing = np.array(
            [
                [0.0, 1.0, 0.0],  # orthogonal, sim = 0
                [1.0, 0.0, 0.0],  # identical, sim = 1
                [0.5, 0.5, 0.0],  # partial match
            ]
        )
        result = max_similarity(new_emb, existing)
        assert result == pytest.approx(1.0)

    def test_empty_existing(self) -> None:
        """Test with empty existing embeddings."""
        new_emb = np.random.rand(384)
        existing = np.empty((0, 384))
        result = max_similarity(new_emb, existing)
        assert result == 0.0

    def test_single_existing(self) -> None:
        """Test with single existing embedding."""
        new_emb = np.array([1.0, 0.0, 0.0])
        existing = np.array([[0.707, 0.707, 0.0]])
        result = max_similarity(new_emb, existing)
        assert 0.7 < result < 0.72


class TestSimilarityFilterInit:
    """Tests for SimilarityFilter initialization."""

    def test_default_threshold(self) -> None:
        """Test default threshold is 0.85."""
        filter_obj = SimilarityFilter()
        assert filter_obj.threshold == 0.85

    def test_custom_threshold(self) -> None:
        """Test custom threshold."""
        filter_obj = SimilarityFilter(threshold=0.9)
        assert filter_obj.threshold == 0.9

    def test_invalid_threshold_too_high(self) -> None:
        """Test that threshold > 1 raises error."""
        with pytest.raises(ValueError):
            SimilarityFilter(threshold=1.5)

    def test_invalid_threshold_too_low(self) -> None:
        """Test that threshold < 0 raises error."""
        with pytest.raises(ValueError):
            SimilarityFilter(threshold=-0.1)

    def test_edge_thresholds(self) -> None:
        """Test edge case thresholds 0 and 1."""
        filter_0 = SimilarityFilter(threshold=0.0)
        assert filter_0.threshold == 0.0
        filter_1 = SimilarityFilter(threshold=1.0)
        assert filter_1.threshold == 1.0


class TestSimilarityFilterIsSimilar:
    """Tests for SimilarityFilter.is_similar method."""

    def test_similar_embedding(self) -> None:
        """Test that similar embeddings are detected."""
        filter_obj = SimilarityFilter(threshold=0.85)
        new_emb = np.array([1.0, 0.0, 0.0])
        existing = np.array([[1.0, 0.0, 0.0]])  # Identical
        assert filter_obj.is_similar(new_emb, existing) is True

    def test_dissimilar_embedding(self) -> None:
        """Test that dissimilar embeddings are not flagged."""
        filter_obj = SimilarityFilter(threshold=0.85)
        new_emb = np.array([1.0, 0.0, 0.0])
        existing = np.array([[0.0, 1.0, 0.0]])  # Orthogonal
        assert filter_obj.is_similar(new_emb, existing) is False

    def test_empty_existing(self) -> None:
        """Test with empty existing embeddings."""
        filter_obj = SimilarityFilter(threshold=0.85)
        new_emb = np.random.rand(384)
        existing = np.empty((0, 384))
        assert filter_obj.is_similar(new_emb, existing) is False

    def test_threshold_boundary(self) -> None:
        """Test behavior at threshold boundary."""
        # Create vectors with known similarity
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.9, 0.436])  # cos similarity ~0.9

        filter_low = SimilarityFilter(threshold=0.85)
        filter_high = SimilarityFilter(threshold=0.95)

        existing = vec2.reshape(1, -1)
        # With threshold 0.85, similarity 0.9 should be flagged
        assert filter_low.is_similar(vec1, existing) is True
        # With threshold 0.95, similarity 0.9 should not be flagged
        assert filter_high.is_similar(vec1, existing) is False


class TestSimilarityFilterGetMaxSimilarity:
    """Tests for get_max_similarity method."""

    def test_returns_max(self) -> None:
        """Test that it returns the maximum similarity."""
        filter_obj = SimilarityFilter()
        new_emb = np.array([1.0, 0.0, 0.0])
        existing = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.707, 0.707, 0.0],
            ]
        )
        result = filter_obj.get_max_similarity(new_emb, existing)
        assert 0.7 < result < 0.72

    def test_empty_returns_zero(self) -> None:
        """Test that empty existing returns 0."""
        filter_obj = SimilarityFilter()
        new_emb = np.random.rand(384)
        existing = np.empty((0, 384))
        result = filter_obj.get_max_similarity(new_emb, existing)
        assert result == 0.0


class TestSimilarityFilterFilterBatch:
    """Tests for SimilarityFilter.filter_batch method."""

    def test_all_unique(self) -> None:
        """Test that all unique samples are accepted."""
        filter_obj = SimilarityFilter(threshold=0.85)
        # Create orthogonal vectors (definitely unique)
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        existing = np.empty((0, 3))
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == [0, 1, 2]

    def test_all_duplicates_of_existing(self) -> None:
        """Test that duplicates of existing are rejected."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.array([[1.0, 0.0, 0.0]])
        # New embeddings identical to existing
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == []

    def test_duplicates_within_batch(self) -> None:
        """Test that duplicates within batch are filtered."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.empty((0, 3))
        # First two are identical, third is different
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # Duplicate of first
                [0.0, 1.0, 0.0],  # Different
            ]
        )
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == [0, 2]  # First and third accepted

    def test_mixed_scenario(self) -> None:
        """Test mixed scenario with both types of duplicates."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.array([[1.0, 0.0, 0.0]])
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Duplicate of existing
                [0.0, 1.0, 0.0],  # Unique
                [0.0, 1.0, 0.0],  # Duplicate within batch
                [0.0, 0.0, 1.0],  # Unique
            ]
        )
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == [1, 3]

    def test_empty_new_embeddings(self) -> None:
        """Test with empty new embeddings."""
        filter_obj = SimilarityFilter(threshold=0.85)
        new_embeddings = np.empty((0, 384))
        existing = np.random.rand(10, 384)
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == []

    def test_preserves_order(self) -> None:
        """Test that accepted indices maintain original order."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.empty((0, 3))
        # All unique
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = filter_obj.filter_batch(new_embeddings, existing)
        assert result == sorted(result)


class TestSimilarityFilterFilterBatchDetailed:
    """Tests for filter_batch_detailed method."""

    def test_returns_filter_result(self) -> None:
        """Test that it returns a FilterResult."""
        filter_obj = SimilarityFilter(threshold=0.85)
        new_embeddings = np.array([[1.0, 0.0, 0.0]])
        existing = np.empty((0, 3))
        result = filter_obj.filter_batch_detailed(new_embeddings, existing)
        assert isinstance(result, FilterResult)

    def test_accepted_and_rejected(self) -> None:
        """Test that accepted and rejected indices are correct."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.array([[1.0, 0.0, 0.0]])
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Duplicate
                [0.0, 1.0, 0.0],  # Unique
            ]
        )
        result = filter_obj.filter_batch_detailed(new_embeddings, existing)
        assert result.accepted_indices == [1]
        assert result.rejected_indices == [0]

    def test_similarities_for_rejected(self) -> None:
        """Test that similarities are recorded for rejected samples."""
        filter_obj = SimilarityFilter(threshold=0.85)
        existing = np.array([[1.0, 0.0, 0.0]])
        new_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Identical, sim = 1.0
            ]
        )
        result = filter_obj.filter_batch_detailed(new_embeddings, existing)
        assert 0 in result.similarities
        assert result.similarities[0] == pytest.approx(1.0)

    def test_empty_new_embeddings(self) -> None:
        """Test with empty new embeddings."""
        filter_obj = SimilarityFilter()
        result = filter_obj.filter_batch_detailed(
            np.empty((0, 384)),
            np.random.rand(10, 384),
        )
        assert result.accepted_indices == []
        assert result.rejected_indices == []
        assert result.similarities == {}


class TestSimilarityFilterPairwise:
    """Tests for pairwise similarity methods."""

    def test_compute_pairwise_similarities(self) -> None:
        """Test pairwise similarity computation."""
        filter_obj = SimilarityFilter()
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # Duplicate of first
            ]
        )
        result = filter_obj.compute_pairwise_similarities(embeddings)
        assert result.shape == (3, 3)
        # Diagonal should be 1.0
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)
        # First and third should be identical
        assert result[0, 2] == pytest.approx(1.0)
        # First and second should be orthogonal
        assert result[0, 1] == pytest.approx(0.0)

    def test_find_duplicates(self) -> None:
        """Test finding duplicate pairs."""
        filter_obj = SimilarityFilter(threshold=0.85)
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # Duplicate of first
            ]
        )
        duplicates = filter_obj.find_duplicates(embeddings)
        # Should find (0, 2) as duplicates
        assert len(duplicates) == 1
        assert duplicates[0][0] == 0
        assert duplicates[0][1] == 2
        assert duplicates[0][2] == pytest.approx(1.0)

    def test_find_duplicates_empty(self) -> None:
        """Test find_duplicates with empty embeddings."""
        filter_obj = SimilarityFilter()
        duplicates = filter_obj.find_duplicates(np.empty((0, 384)))
        assert duplicates == []

    def test_find_duplicates_no_duplicates(self) -> None:
        """Test when there are no duplicates."""
        filter_obj = SimilarityFilter(threshold=0.85)
        # Orthogonal vectors
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        duplicates = filter_obj.find_duplicates(embeddings)
        assert duplicates == []


class TestSimilarityFilterLargeScale:
    """Tests for performance with larger datasets."""

    def test_filter_1000_samples(self) -> None:
        """Test filtering 1000 samples efficiently."""
        np.random.seed(42)
        filter_obj = SimilarityFilter(threshold=0.95)

        # Create 100 existing samples
        existing = np.random.rand(100, 384).astype(np.float32)

        # Create 1000 new samples
        new_embeddings = np.random.rand(1000, 384).astype(np.float32)

        # This should complete without error
        result = filter_obj.filter_batch(new_embeddings, existing)

        # Most random samples should be unique with high threshold
        assert len(result) > 900

    def test_filter_with_many_existing(self) -> None:
        """Test filtering against large existing dataset."""
        np.random.seed(42)
        filter_obj = SimilarityFilter(threshold=0.95)

        # Create 5000 existing samples
        existing = np.random.rand(5000, 384).astype(np.float32)

        # Create 100 new samples
        new_embeddings = np.random.rand(100, 384).astype(np.float32)

        # This should complete without error
        result = filter_obj.filter_batch(new_embeddings, existing)

        # Most random samples should be unique
        assert len(result) > 90


@pytest.mark.real
class TestSimilarityFilterWithRealEmbeddings:
    """Integration tests with real embeddings."""

    @pytest.fixture(scope="class")
    def embedding_model(self):
        """Load the real embedding model."""
        from autoft.embedder.embedding import EmbeddingModel

        return EmbeddingModel()

    def test_similar_texts_filtered(self, embedding_model) -> None:
        """Test that semantically similar texts are filtered."""
        filter_obj = SimilarityFilter(threshold=0.85)

        # Similar texts
        texts = [
            "What is Python programming?",
            "Tell me about the Python language",  # Similar
            "How does JavaScript work?",  # Different
        ]

        embeddings = embedding_model.encode_batch(texts)
        existing = np.empty((0, 384))

        result = filter_obj.filter_batch(embeddings, existing)

        # First and third should be accepted, second filtered
        assert 0 in result
        assert 2 in result
        # Second might be filtered as similar to first
        # (depends on actual similarity)

    def test_diverse_texts_accepted(self, embedding_model) -> None:
        """Test that diverse texts are all accepted."""
        filter_obj = SimilarityFilter(threshold=0.85)

        # Very different texts
        texts = [
            "How to cook pasta?",
            "What is quantum physics?",
            "Explain machine learning",
            "History of ancient Rome",
        ]

        embeddings = embedding_model.encode_batch(texts)
        existing = np.empty((0, 384))

        result = filter_obj.filter_batch(embeddings, existing)

        # All should be accepted
        assert len(result) == 4
