"""SQLite database operations for dataset storage."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes for SQLite storage.

    Args:
        embedding: Numpy array to serialize.

    Returns:
        Bytes representation of the array.
    """
    return embedding.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
    """Deserialize bytes back to a numpy array.

    Args:
        blob: Bytes to deserialize.
        dim: Expected dimension of the embedding.

    Returns:
        Numpy array representation.
    """
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


class Database:
    """SQLite database for storing samples with embeddings."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instruction TEXT NOT NULL,
        output TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db_path: Path | str, embedding_dim: int = 384) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory DB.
            embedding_dim: Dimension of embedding vectors (default: 384 for BGE-small).
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._connection: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection.

        Returns:
            Active SQLite connection.
        """
        if self._connection is None:
            # Convert Path to string for sqlite3
            db_path_str = (
                str(self.db_path) if isinstance(self.db_path, Path) else self.db_path
            )
            self._connection = sqlite3.connect(db_path_str)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        conn.execute(self.SCHEMA)
        conn.commit()

    def insert_sample(
        self,
        instruction: str,
        output: str,
        embedding: np.ndarray,
    ) -> int:
        """Insert a new sample into the database.

        Args:
            instruction: The instruction text.
            output: The output text.
            embedding: The embedding vector.

        Returns:
            The ID of the inserted sample.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO samples (instruction, output, embedding)
            VALUES (?, ?, ?)
            """,
            (instruction, output, _serialize_embedding(embedding)),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def insert_samples_batch(
        self,
        samples: list[tuple[str, str, np.ndarray]],
    ) -> list[int]:
        """Insert multiple samples in a single transaction.

        Args:
            samples: List of (instruction, output, embedding) tuples.

        Returns:
            List of inserted sample IDs.
        """
        conn = self._get_connection()
        ids = []
        for instruction, output, embedding in samples:
            cursor = conn.execute(
                """
                INSERT INTO samples (instruction, output, embedding)
                VALUES (?, ?, ?)
                """,
                (instruction, output, _serialize_embedding(embedding)),
            )
            ids.append(cursor.lastrowid or 0)
        conn.commit()
        return ids

    def get_all_embeddings(self) -> np.ndarray:
        """Retrieve all embeddings from the database.

        Returns:
            Array of shape (n_samples, embedding_dim) with all embeddings.
            Returns empty array with shape (0, embedding_dim) if no samples.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT embedding FROM samples ORDER BY id")
        rows = cursor.fetchall()

        if not rows:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = [
            _deserialize_embedding(row["embedding"], self.embedding_dim) for row in rows
        ]
        return np.array(embeddings, dtype=np.float32)

    def iter_embeddings(self, batch_size: int = 1000) -> Iterator[np.ndarray]:
        """Iterate over embeddings in batches for memory efficiency.

        Args:
            batch_size: Number of embeddings per batch.

        Yields:
            Arrays of shape (batch_size, embedding_dim).
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT embedding FROM samples ORDER BY id")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            embeddings = [
                _deserialize_embedding(row["embedding"], self.embedding_dim)
                for row in rows
            ]
            yield np.array(embeddings, dtype=np.float32)

    def get_sample_count(self) -> int:
        """Get the total number of samples in the database.

        Returns:
            Number of samples.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM samples")
        row = cursor.fetchone()
        return row["count"] if row else 0

    def get_all_samples(self) -> list[dict[str, str]]:
        """Retrieve all samples from the database.

        Returns:
            List of dicts with 'instruction' and 'output' keys.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT instruction, output FROM samples ORDER BY id")
        return [
            {"instruction": row["instruction"], "output": row["output"]}
            for row in cursor.fetchall()
        ]

    def iter_samples(self, batch_size: int = 1000) -> Iterator[list[dict[str, str]]]:
        """Iterate over samples in batches for memory efficiency.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            Lists of sample dicts.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT instruction, output FROM samples ORDER BY id")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [
                {"instruction": row["instruction"], "output": row["output"]}
                for row in rows
            ]

    def get_sample_by_id(self, sample_id: int) -> dict[str, str] | None:
        """Retrieve a single sample by ID.

        Args:
            sample_id: The ID of the sample to retrieve.

        Returns:
            Dict with 'id', 'instruction', 'output', and 'created_at' keys,
            or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, instruction, output, created_at
            FROM samples WHERE id = ?
            """,
            (sample_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": row["output"],
            "created_at": row["created_at"],
        }

    def clear(self) -> int:
        """Remove all samples from the database.

        Returns:
            Number of samples deleted.
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM samples")
        count = cursor.rowcount
        conn.commit()
        # Reset autoincrement counter
        conn.execute("DELETE FROM sqlite_sequence WHERE name='samples'")
        conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "Database":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
