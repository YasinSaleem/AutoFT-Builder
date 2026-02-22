"""Similarity filtering module."""

from autoft.filter.similarity import (
    FilterResult,
    SimilarityFilter,
    cosine_similarity,
    cosine_similarity_matrix,
    max_similarity,
)

__all__ = [
    "SimilarityFilter",
    "FilterResult",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "max_similarity",
]
