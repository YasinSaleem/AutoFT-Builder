"""Cosine similarity filtering for semantic deduplication."""

from dataclasses import dataclass

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    For normalized vectors (L2 norm = 1), this is equivalent to the dot product.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_similarity_matrix(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between all pairs of vectors.

    This is optimized for batch comparisons using matrix multiplication.

    Args:
        vectors1: Array of shape (n, dim) with first set of vectors.
        vectors2: Array of shape (m, dim) with second set of vectors.

    Returns:
        Array of shape (n, m) with pairwise cosine similarities.
    """
    if vectors1.size == 0 or vectors2.size == 0:
        return np.empty((vectors1.shape[0], vectors2.shape[0]), dtype=np.float32)

    # Normalize vectors
    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)

    # Avoid division by zero
    norms1 = np.where(norms1 == 0, 1, norms1)
    norms2 = np.where(norms2 == 0, 1, norms2)

    normalized1 = vectors1 / norms1
    normalized2 = vectors2 / norms2

    # Compute similarity matrix via dot product
    result: np.ndarray = np.dot(normalized1, normalized2.T).astype(np.float32)
    return result


def max_similarity(
    new_embedding: np.ndarray,
    existing_embeddings: np.ndarray,
) -> float:
    """Find the maximum similarity between a new embedding and existing ones.

    Args:
        new_embedding: Single embedding vector of shape (dim,).
        existing_embeddings: Array of shape (n, dim) with existing embeddings.

    Returns:
        Maximum cosine similarity, or 0.0 if no existing embeddings.
    """
    if existing_embeddings.size == 0:
        return 0.0

    # Reshape for matrix operation
    new_reshaped = new_embedding.reshape(1, -1)
    similarities = cosine_similarity_matrix(new_reshaped, existing_embeddings)
    return float(np.max(similarities))


@dataclass
class FilterResult:
    """Result of filtering a batch of samples."""

    accepted_indices: list[int]
    rejected_indices: list[int]
    similarities: dict[int, float]  # Maps rejected index to max similarity


class SimilarityFilter:
    """Filter samples based on cosine similarity threshold.

    This filter removes samples that are too similar to existing samples
    or to other samples in the same batch, ensuring semantic diversity.

    Example:
        >>> filter = SimilarityFilter(threshold=0.85)
        >>> is_dup = filter.is_similar(new_embedding, existing_embeddings)
        >>> accepted = filter.filter_batch(new_embeddings, existing_embeddings)
    """

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize the similarity filter.

        Args:
            threshold: Similarity threshold (0.0 to 1.0). Samples with similarity
                      above this value are considered duplicates and filtered out.
                      Default is 0.85, recommended range is 0.80-0.90.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold

    def is_similar(
        self,
        new_embedding: np.ndarray,
        existing_embeddings: np.ndarray,
    ) -> bool:
        """Check if a new embedding is similar to any existing embeddings.

        Args:
            new_embedding: Embedding of the new sample, shape (dim,).
            existing_embeddings: Array of existing embeddings, shape (n, dim).

        Returns:
            True if the new embedding has similarity > threshold with any existing one.
        """
        if existing_embeddings.size == 0:
            return False

        max_sim = max_similarity(new_embedding, existing_embeddings)
        return max_sim > self.threshold

    def get_max_similarity(
        self,
        new_embedding: np.ndarray,
        existing_embeddings: np.ndarray,
    ) -> float:
        """Get the maximum similarity between a new embedding and existing ones.

        Args:
            new_embedding: Embedding of the new sample, shape (dim,).
            existing_embeddings: Array of existing embeddings, shape (n, dim).

        Returns:
            Maximum cosine similarity, or 0.0 if no existing embeddings.
        """
        return max_similarity(new_embedding, existing_embeddings)

    def filter_batch(
        self,
        new_embeddings: np.ndarray,
        existing_embeddings: np.ndarray,
    ) -> list[int]:
        """Filter a batch of embeddings, returning indices of unique samples.

        This method filters samples that are:
        1. Too similar to existing embeddings in the database
        2. Too similar to other samples within the same batch (accepted earlier)

        Args:
            new_embeddings: Embeddings of new samples, shape (n, dim).
            existing_embeddings: Existing embeddings to compare against, shape (m, dim).

        Returns:
            List of indices of samples that pass the similarity filter.
        """
        if new_embeddings.size == 0:
            return []

        n_samples = new_embeddings.shape[0]
        accepted_indices: list[int] = []
        accepted_embeddings: list[np.ndarray] = []

        for i in range(n_samples):
            new_emb = new_embeddings[i]

            # Check against existing database embeddings
            if existing_embeddings.size > 0 and self.is_similar(
                new_emb, existing_embeddings
            ):
                continue

            # Check against already accepted samples in this batch
            if accepted_embeddings:
                accepted_array = np.array(accepted_embeddings)
                if self.is_similar(new_emb, accepted_array):
                    continue

            # Sample passes both checks
            accepted_indices.append(i)
            accepted_embeddings.append(new_emb)

        return accepted_indices

    def filter_batch_detailed(
        self,
        new_embeddings: np.ndarray,
        existing_embeddings: np.ndarray,
    ) -> FilterResult:
        """Filter a batch with detailed results including rejection reasons.

        Args:
            new_embeddings: Embeddings of new samples, shape (n, dim).
            existing_embeddings: Existing embeddings to compare against, shape (m, dim).

        Returns:
            FilterResult with accepted indices, rejected indices, and similarities.
        """
        if new_embeddings.size == 0:
            return FilterResult(
                accepted_indices=[],
                rejected_indices=[],
                similarities={},
            )

        n_samples = new_embeddings.shape[0]
        accepted_indices: list[int] = []
        rejected_indices: list[int] = []
        similarities: dict[int, float] = {}
        accepted_embeddings: list[np.ndarray] = []

        for i in range(n_samples):
            new_emb = new_embeddings[i]
            max_sim = 0.0

            # Check against existing database embeddings
            if existing_embeddings.size > 0:
                db_sim = max_similarity(new_emb, existing_embeddings)
                max_sim = max(max_sim, db_sim)

            # Check against already accepted samples in this batch
            if accepted_embeddings:
                accepted_array = np.array(accepted_embeddings)
                batch_sim = max_similarity(new_emb, accepted_array)
                max_sim = max(max_sim, batch_sim)

            if max_sim > self.threshold:
                rejected_indices.append(i)
                similarities[i] = max_sim
            else:
                accepted_indices.append(i)
                accepted_embeddings.append(new_emb)

        return FilterResult(
            accepted_indices=accepted_indices,
            rejected_indices=rejected_indices,
            similarities=similarities,
        )

    def compute_pairwise_similarities(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise similarities within a set of embeddings.

        Useful for analyzing diversity of a dataset.

        Args:
            embeddings: Array of shape (n, dim) with embeddings.

        Returns:
            Symmetric matrix of shape (n, n) with pairwise similarities.
        """
        return cosine_similarity_matrix(embeddings, embeddings)

    def find_duplicates(
        self,
        embeddings: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        """Find all pairs of embeddings that exceed the similarity threshold.

        Args:
            embeddings: Array of shape (n, dim) with embeddings.

        Returns:
            List of (i, j, similarity) tuples for pairs exceeding threshold.
        """
        if embeddings.size == 0:
            return []

        sim_matrix = self.compute_pairwise_similarities(embeddings)
        n = embeddings.shape[0]
        duplicates = []

        # Only check upper triangle (avoid duplicates and self-comparison)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > self.threshold:
                    duplicates.append((i, j, float(sim_matrix[i, j])))

        return duplicates
