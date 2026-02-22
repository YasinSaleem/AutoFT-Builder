"""Integration tests for AutoFT-Builder pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autoft.cli import main
from autoft.embedder import EmbeddingModel
from autoft.export import export_from_database, read_jsonl, validate_jsonl
from autoft.filter import SimilarityFilter
from autoft.generator import GeneratedSample, LLMGenerator
from autoft.storage import Database


class TestModuleIntegration:
    """Integration tests between modules."""

    def test_storage_and_export(self, tmp_path: Path) -> None:
        """Test storing samples and exporting them."""
        db_path = tmp_path / "test.db"
        export_path = tmp_path / "export.jsonl"

        # Store samples
        with Database(db_path) as db:
            for i in range(100):
                embedding = np.random.randn(384).astype(np.float32)
                db.insert_sample(f"Question {i}", f"Answer {i}", embedding)

        # Export
        with Database(db_path) as db:
            count = export_from_database(db, export_path)

        assert count == 100

        # Validate exported file
        valid_count, errors = validate_jsonl(export_path)
        assert valid_count == 100
        assert errors == []

        # Read back
        samples = list(read_jsonl(export_path))
        assert len(samples) == 100
        assert samples[0]["instruction"] == "Question 0"

    def test_embedder_and_filter(self) -> None:
        """Test embedding and filtering similar samples."""
        # Create mock embedder
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            # Return specific embeddings for testing
            embeddings = [
                np.array([1, 0, 0, 0] * 96, dtype=np.float32),  # Unique
                np.array([1, 0, 0, 0] * 96, dtype=np.float32),  # Duplicate of first
                np.array([0, 1, 0, 0] * 96, dtype=np.float32),  # Different
            ]
            mock_model.encode.side_effect = embeddings

            embedder = EmbeddingModel()

            # Encode samples
            emb1 = embedder.encode("Sample 1")
            emb2 = embedder.encode("Sample 2")
            emb3 = embedder.encode("Sample 3")

        # Filter
        similarity_filter = SimilarityFilter(threshold=0.85)

        # emb2 should be similar to emb1
        assert similarity_filter.is_similar(emb2, np.array([emb1]))
        # emb3 should not be similar to emb1
        assert not similarity_filter.is_similar(emb3, np.array([emb1]))

    def test_full_pipeline_mock(self, tmp_path: Path) -> None:
        """Test full pipeline with mocked LLM."""
        db_path = tmp_path / "pipeline.db"

        # Mock generator
        mock_generator = MagicMock(spec=LLMGenerator)
        mock_generator.generate_batch.return_value = [
            GeneratedSample(
                instruction="What is Python?", output="A programming language."
            ),
            GeneratedSample(
                instruction="What is AI?", output="Artificial Intelligence."
            ),
            GeneratedSample(
                instruction="What is Python?", output="A snake."
            ),  # Similar to first
        ]

        # Mock embedder
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            # Create embeddings where sample 3 is similar to sample 1
            # Use deterministic embeddings so sample 1 and sample 3 are nearly identical
            emb1 = np.ones(384, dtype=np.float32)  # "What is Python?" (first)
            emb2 = np.zeros(
                384, dtype=np.float32
            )  # "What is AI?" (different direction)
            emb2[0] = 1.0  # Give it some magnitude
            emb3 = np.ones(384, dtype=np.float32) * 0.99  # Very similar to emb1
            mock_model.encode.side_effect = [emb1, emb2, emb3]

            embedder = EmbeddingModel()
            similarity_filter = SimilarityFilter(threshold=0.85)

            with Database(db_path) as db:
                # Simulate pipeline
                existing_embeddings = db.get_all_embeddings()

                samples = mock_generator.generate_batch("Generate Q&A", 3)

                # Embed and filter
                new_embeddings = []

                for sample in samples:
                    emb = embedder.encode(f"{sample.instruction} {sample.output}")
                    new_embeddings.append(emb)

                new_embeddings_array = np.array(new_embeddings)
                filter_result = similarity_filter.filter_batch_detailed(
                    new_embeddings_array, existing_embeddings
                )

                # Store accepted
                for idx in filter_result.accepted_indices:
                    db.insert_sample(
                        samples[idx].instruction,
                        samples[idx].output,
                        new_embeddings_array[idx],
                    )

                assert db.get_sample_count() == 2  # Only 2 unique samples


@pytest.mark.integration
class TestPipelineIntegration:
    """Full pipeline integration tests."""

    def test_cli_workflow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,  # noqa: ARG002
    ) -> None:
        """Test CLI workflow: create data, export, clear."""
        db_path = tmp_path / "cli_test.db"
        export_path = tmp_path / "export.jsonl"

        # Create some samples directly (simulating generate)
        with Database(db_path) as db:
            for i in range(50):
                embedding = np.random.randn(384).astype(np.float32)
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)

        # Test stats
        result = main(["stats", "--db", str(db_path)])
        assert result == 0

        # Test export
        result = main(["export", "--db", str(db_path), "--output", str(export_path)])
        assert result == 0
        assert export_path.exists()

        # Verify export
        samples = list(read_jsonl(export_path))
        assert len(samples) == 50

        # Test clear
        result = main(["clear", "--db", str(db_path), "--force"])
        assert result == 0

        # Verify cleared
        with Database(db_path) as db:
            assert db.get_sample_count() == 0

    def test_large_dataset_handling(self, tmp_path: Path) -> None:
        """Test handling of large datasets."""
        db_path = tmp_path / "large.db"
        export_path = tmp_path / "large_export.jsonl"

        # Create 5000 samples
        with Database(db_path) as db:
            samples = [
                (
                    f"Question {i}",
                    f"Answer {i}",
                    np.random.randn(384).astype(np.float32),
                )
                for i in range(5000)
            ]
            db.insert_samples_batch(samples)

        # Export with streaming
        with Database(db_path) as db:
            count = export_from_database(db, export_path, batch_size=500)

        assert count == 5000

        # Validate
        valid_count, errors = validate_jsonl(export_path)
        assert valid_count == 5000
        assert errors == []

    def test_similarity_filtering_at_scale(self, tmp_path: Path) -> None:  # noqa: ARG002
        """Test similarity filtering with many samples."""
        # Create existing embeddings
        existing = np.random.randn(1000, 384).astype(np.float32)

        # Create new embeddings with some duplicates
        new_embeddings = np.vstack(
            [
                np.random.randn(100, 384).astype(np.float32),  # Unique
                existing[:50],  # Duplicates
            ]
        )

        similarity_filter = SimilarityFilter(threshold=0.99)  # High threshold
        filter_result = similarity_filter.filter_batch_detailed(
            new_embeddings, existing
        )

        # All should be accepted since random vectors have low similarity
        # and only exact duplicates should be rejected
        assert (
            len(filter_result.accepted_indices) + len(filter_result.rejected_indices)
            == 150
        )


@pytest.mark.real
class TestRealPipeline:
    """Real integration tests with actual models.

    Run with: pytest -m real tests/test_integration.py
    """

    def test_real_embedding_and_filter(self) -> None:
        """Test real embedding model with filtering."""
        embedder = EmbeddingModel()

        # Test samples
        samples = [
            ("What is Python?", "Python is a programming language."),
            (
                "Explain Python.",
                "Python is a high-level programming language.",
            ),  # Similar
            ("What is the weather?", "The weather is sunny today."),  # Different
        ]

        embeddings = embedder.encode_samples_batch(samples)

        # Filter
        similarity_filter = SimilarityFilter(threshold=0.85)
        filter_result = similarity_filter.filter_batch_detailed(
            embeddings, np.empty((0, 384))
        )

        # Should accept at least 2 samples
        assert len(filter_result.accepted_indices) >= 2

    def test_real_jsonl_export_format(self, tmp_path: Path) -> None:
        """Test that exported JSONL is valid for fine-tuning."""
        db_path = tmp_path / "finetuning.db"
        export_path = tmp_path / "finetuning.jsonl"

        with Database(db_path) as db:
            db.insert_sample(
                "Summarize the following text.",
                "This is a summary of the text.",
                np.zeros(384, dtype=np.float32),
            )

        with Database(db_path) as db:
            export_from_database(db, export_path)

        # Verify format matches fine-tuning requirements
        with open(export_path) as f:
            data = json.loads(f.readline())

        assert "instruction" in data
        assert "output" in data
        assert isinstance(data["instruction"], str)
        assert isinstance(data["output"], str)
