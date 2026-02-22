"""Tests for the storage module."""

import numpy as np

from autoft.storage.database import (
    Database,
    _deserialize_embedding,
    _serialize_embedding,
)


class TestEmbeddingSerialization:
    """Tests for embedding serialization functions."""

    def test_serialize_embedding(self) -> None:
        """Test that embedding is serialized to bytes."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blob = _serialize_embedding(embedding)
        assert isinstance(blob, bytes)
        assert len(blob) == 3 * 4  # 3 floats * 4 bytes each

    def test_deserialize_embedding(self) -> None:
        """Test that bytes are deserialized back to numpy array."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blob = _serialize_embedding(original)
        restored = _deserialize_embedding(blob, dim=3)
        np.testing.assert_array_almost_equal(original, restored)

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test full roundtrip of serialization."""
        original = np.random.rand(384).astype(np.float32)
        blob = _serialize_embedding(original)
        restored = _deserialize_embedding(blob, dim=384)
        np.testing.assert_array_almost_equal(original, restored)

    def test_serialize_converts_to_float32(self) -> None:
        """Test that serialization converts to float32."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        blob = _serialize_embedding(embedding)
        # float32 is 4 bytes per element
        assert len(blob) == 3 * 4


class TestDatabase:
    """Tests for Database class."""

    def test_database_init(self) -> None:
        """Test that database initializes correctly."""
        db = Database(":memory:")
        assert db.db_path == ":memory:"
        assert db._connection is None
        assert db.embedding_dim == 384

    def test_database_init_custom_dim(self) -> None:
        """Test database with custom embedding dimension."""
        db = Database(":memory:", embedding_dim=768)
        assert db.embedding_dim == 768

    def test_initialize_creates_table(self) -> None:
        """Test that initialize creates the samples table."""
        db = Database(":memory:")
        db.initialize()

        # Check table exists
        conn = db._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='samples'"
        )
        assert cursor.fetchone() is not None
        db.close()

    def test_initialize_is_idempotent(self) -> None:
        """Test that calling initialize multiple times is safe."""
        db = Database(":memory:")
        db.initialize()
        db.initialize()  # Should not raise
        db.close()

    def test_context_manager(self) -> None:
        """Test database as context manager."""
        with Database(":memory:") as db:
            # Table should be initialized
            assert db.get_sample_count() == 0

    def test_insert_sample(self) -> None:
        """Test inserting a single sample."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            sample_id = db.insert_sample(
                instruction="Test instruction",
                output="Test output",
                embedding=embedding,
            )
            assert sample_id == 1
            assert db.get_sample_count() == 1

    def test_insert_multiple_samples(self) -> None:
        """Test inserting multiple samples."""
        with Database(":memory:") as db:
            for i in range(5):
                embedding = np.random.rand(384).astype(np.float32)
                sample_id = db.insert_sample(
                    instruction=f"Instruction {i}",
                    output=f"Output {i}",
                    embedding=embedding,
                )
                assert sample_id == i + 1
            assert db.get_sample_count() == 5

    def test_insert_samples_batch(self) -> None:
        """Test batch insertion of samples."""
        with Database(":memory:") as db:
            samples = [
                (
                    f"Instruction {i}",
                    f"Output {i}",
                    np.random.rand(384).astype(np.float32),
                )
                for i in range(10)
            ]
            ids = db.insert_samples_batch(samples)
            assert len(ids) == 10
            assert ids == list(range(1, 11))
            assert db.get_sample_count() == 10

    def test_get_all_embeddings_empty(self) -> None:
        """Test getting embeddings from empty database."""
        with Database(":memory:") as db:
            embeddings = db.get_all_embeddings()
            assert embeddings.shape == (0, 384)

    def test_get_all_embeddings(self) -> None:
        """Test retrieving all embeddings."""
        with Database(":memory:") as db:
            original_embeddings = []
            for i in range(5):
                embedding = np.random.rand(384).astype(np.float32)
                original_embeddings.append(embedding)
                db.insert_sample(
                    instruction=f"Instruction {i}",
                    output=f"Output {i}",
                    embedding=embedding,
                )

            retrieved = db.get_all_embeddings()
            assert retrieved.shape == (5, 384)
            for i, orig in enumerate(original_embeddings):
                np.testing.assert_array_almost_equal(retrieved[i], orig)

    def test_iter_embeddings(self) -> None:
        """Test iterating over embeddings in batches."""
        with Database(":memory:") as db:
            # Insert 25 samples
            for i in range(25):
                embedding = np.full(384, float(i), dtype=np.float32)
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)

            # Iterate in batches of 10
            batches = list(db.iter_embeddings(batch_size=10))
            assert len(batches) == 3  # 10 + 10 + 5
            assert batches[0].shape == (10, 384)
            assert batches[1].shape == (10, 384)
            assert batches[2].shape == (5, 384)

    def test_get_sample_count_empty(self) -> None:
        """Test sample count on empty database."""
        with Database(":memory:") as db:
            assert db.get_sample_count() == 0

    def test_get_sample_count(self) -> None:
        """Test sample count after inserts."""
        with Database(":memory:") as db:
            for i in range(10):
                embedding = np.random.rand(384).astype(np.float32)
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)
            assert db.get_sample_count() == 10

    def test_get_all_samples_empty(self) -> None:
        """Test getting samples from empty database."""
        with Database(":memory:") as db:
            samples = db.get_all_samples()
            assert samples == []

    def test_get_all_samples(self) -> None:
        """Test retrieving all samples."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            db.insert_sample("What is Python?", "A programming language.", embedding)
            db.insert_sample("What is Java?", "Another language.", embedding)

            samples = db.get_all_samples()
            assert len(samples) == 2
            assert samples[0] == {
                "instruction": "What is Python?",
                "output": "A programming language.",
            }
            assert samples[1] == {
                "instruction": "What is Java?",
                "output": "Another language.",
            }

    def test_iter_samples(self) -> None:
        """Test iterating over samples in batches."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            for i in range(25):
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)

            batches = list(db.iter_samples(batch_size=10))
            assert len(batches) == 3
            assert len(batches[0]) == 10
            assert len(batches[1]) == 10
            assert len(batches[2]) == 5

    def test_get_sample_by_id(self) -> None:
        """Test retrieving a sample by ID."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            db.insert_sample("Test instruction", "Test output", embedding)

            sample = db.get_sample_by_id(1)
            assert sample is not None
            assert sample["id"] == 1
            assert sample["instruction"] == "Test instruction"
            assert sample["output"] == "Test output"
            assert sample["created_at"] is not None

    def test_get_sample_by_id_not_found(self) -> None:
        """Test retrieving a non-existent sample."""
        with Database(":memory:") as db:
            sample = db.get_sample_by_id(999)
            assert sample is None

    def test_clear(self) -> None:
        """Test clearing all samples."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            for i in range(5):
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)

            assert db.get_sample_count() == 5
            deleted = db.clear()
            assert deleted == 5
            assert db.get_sample_count() == 0

    def test_clear_empty_database(self) -> None:
        """Test clearing an empty database."""
        with Database(":memory:") as db:
            deleted = db.clear()
            assert deleted == 0

    def test_clear_resets_autoincrement(self) -> None:
        """Test that clear resets the autoincrement counter."""
        with Database(":memory:") as db:
            embedding = np.random.rand(384).astype(np.float32)
            db.insert_sample("Test", "Test", embedding)
            assert db.get_sample_count() == 1

            db.clear()

            # Insert new sample - should get ID 1 again
            new_id = db.insert_sample("New", "New", embedding)
            assert new_id == 1

    def test_close(self) -> None:
        """Test closing the database connection."""
        db = Database(":memory:")
        db.initialize()
        assert db._connection is not None
        db.close()
        assert db._connection is None

    def test_close_idempotent(self) -> None:
        """Test that closing multiple times is safe."""
        db = Database(":memory:")
        db.initialize()
        db.close()
        db.close()  # Should not raise


class TestDatabaseWithFileStorage:
    """Tests for database with file-based storage."""

    def test_file_persistence(self, tmp_path) -> None:
        """Test that data persists to file."""
        db_path = tmp_path / "test.db"

        # Create and populate database
        with Database(db_path) as db:
            embedding = np.random.rand(384).astype(np.float32)
            db.insert_sample("Test instruction", "Test output", embedding)

        # Reopen and verify
        with Database(db_path) as db:
            assert db.get_sample_count() == 1
            samples = db.get_all_samples()
            assert samples[0]["instruction"] == "Test instruction"

    def test_path_object(self, tmp_path) -> None:
        """Test that Path objects work correctly."""
        from pathlib import Path

        db_path = Path(tmp_path) / "test.db"
        with Database(db_path) as db:
            embedding = np.random.rand(384).astype(np.float32)
            db.insert_sample("Test", "Test", embedding)
            assert db.get_sample_count() == 1


class TestDatabaseLargeScale:
    """Tests for database with larger datasets."""

    def test_handle_10000_samples(self) -> None:
        """Test that database handles 10,000 samples efficiently."""
        with Database(":memory:") as db:
            # Batch insert for efficiency
            batch_size = 1000
            for batch_num in range(10):
                samples = [
                    (
                        f"Instruction {batch_num * batch_size + i}",
                        f"Output {batch_num * batch_size + i}",
                        np.random.rand(384).astype(np.float32),
                    )
                    for i in range(batch_size)
                ]
                db.insert_samples_batch(samples)

            assert db.get_sample_count() == 10000

            # Test retrieval
            embeddings = db.get_all_embeddings()
            assert embeddings.shape == (10000, 384)

    def test_embedding_retrieval_accuracy(self) -> None:
        """Test that embeddings are retrieved with high accuracy."""
        with Database(":memory:") as db:
            # Create deterministic embeddings
            original_embeddings = []
            for i in range(100):
                embedding = np.full(384, float(i) / 100, dtype=np.float32)
                original_embeddings.append(embedding)
                db.insert_sample(f"Instruction {i}", f"Output {i}", embedding)

            retrieved = db.get_all_embeddings()

            for i, orig in enumerate(original_embeddings):
                np.testing.assert_array_almost_equal(retrieved[i], orig, decimal=6)
