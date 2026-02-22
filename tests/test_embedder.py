"""Tests for the embedding module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autoft.embedder.embedding import EmbeddingModel


class TestEmbeddingModelInit:
    """Tests for EmbeddingModel initialization."""

    def test_model_constants(self) -> None:
        """Test that model constants are defined correctly."""
        assert EmbeddingModel.MODEL_NAME == "BAAI/bge-small-en-v1.5"
        assert EmbeddingModel.EMBEDDING_DIM == 384
        assert "Represent this sentence" in EmbeddingModel.PREFIX

    def test_model_init_defaults(self) -> None:
        """Test default initialization."""
        model = EmbeddingModel()
        assert model._model_name == "BAAI/bge-small-en-v1.5"
        assert model._use_prefix is True
        assert model._device == "cpu"
        assert model._model is None

    def test_model_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        model = EmbeddingModel(
            model_name="custom/model",
            use_prefix=False,
            device="cuda",
        )
        assert model._model_name == "custom/model"
        assert model._use_prefix is False
        assert model._device == "cuda"

    def test_is_loaded_initially_false(self) -> None:
        """Test that model is not loaded initially."""
        model = EmbeddingModel()
        assert model.is_loaded is False

    def test_embedding_dim_property(self) -> None:
        """Test embedding_dim property."""
        model = EmbeddingModel()
        assert model.embedding_dim == 384


class TestPrepareText:
    """Tests for text preparation methods."""

    def test_prepare_text_with_prefix(self) -> None:
        """Test that prefix is added when use_prefix is True."""
        model = EmbeddingModel(use_prefix=True)
        result = model._prepare_text("test text")
        assert result.startswith(EmbeddingModel.PREFIX)
        assert result.endswith("test text")

    def test_prepare_text_without_prefix(self) -> None:
        """Test that no prefix is added when use_prefix is False."""
        model = EmbeddingModel(use_prefix=False)
        result = model._prepare_text("test text")
        assert result == "test text"

    def test_prepare_texts_with_prefix(self) -> None:
        """Test batch text preparation with prefix."""
        model = EmbeddingModel(use_prefix=True)
        texts = ["text 1", "text 2", "text 3"]
        result = model._prepare_texts(texts)
        assert len(result) == 3
        for prepared in result:
            assert prepared.startswith(EmbeddingModel.PREFIX)

    def test_prepare_texts_without_prefix(self) -> None:
        """Test batch text preparation without prefix."""
        model = EmbeddingModel(use_prefix=False)
        texts = ["text 1", "text 2", "text 3"]
        result = model._prepare_texts(texts)
        assert result == texts


class TestEmbeddingModelWithMock:
    """Tests for EmbeddingModel using mocked SentenceTransformer."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        mock_model = MagicMock()
        # Return a 384-dimensional embedding
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        return mock_model

    @pytest.fixture
    def mock_sentence_transformer_batch(self):
        """Create a mock SentenceTransformer for batch encoding."""
        mock_model = MagicMock()

        # Return embeddings for batch
        def encode_side_effect(texts, **kwargs):  # noqa: ARG001
            if isinstance(texts, str):
                return np.random.rand(384).astype(np.float32)
            return np.random.rand(len(texts), 384).astype(np.float32)

        mock_model.encode.side_effect = encode_side_effect
        return mock_model

    def test_load_model_creates_sentence_transformer(self) -> None:
        """Test that _load_model creates a SentenceTransformer instance."""
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_instance = MagicMock()
            MockST.return_value = mock_instance

            model = EmbeddingModel()
            result = model._load_model()

            MockST.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cpu")
            assert result == mock_instance
            assert model.is_loaded is True

    def test_load_model_caches_instance(self) -> None:
        """Test that _load_model only creates the model once."""
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_instance = MagicMock()
            MockST.return_value = mock_instance

            model = EmbeddingModel()
            model._load_model()
            model._load_model()  # Second call

            # Should only be called once
            MockST.assert_called_once()

    def test_encode_single_text(self, mock_sentence_transformer) -> None:
        """Test encoding a single text."""
        model = EmbeddingModel()
        model._model = mock_sentence_transformer

        result = model.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32
        mock_sentence_transformer.encode.assert_called_once()

    def test_encode_passes_correct_args(self, mock_sentence_transformer) -> None:
        """Test that encode passes correct arguments to the model."""
        model = EmbeddingModel(use_prefix=True)
        model._model = mock_sentence_transformer

        model.encode("test text", normalize=True)

        call_args = mock_sentence_transformer.encode.call_args
        # First positional arg should be the prepared text
        assert EmbeddingModel.PREFIX in call_args[0][0]
        # Check kwargs
        assert call_args[1]["normalize_embeddings"] is True
        assert call_args[1]["convert_to_numpy"] is True

    def test_encode_batch_multiple_texts(self, mock_sentence_transformer_batch) -> None:
        """Test batch encoding multiple texts."""
        model = EmbeddingModel()
        model._model = mock_sentence_transformer_batch

        texts = ["text 1", "text 2", "text 3"]
        result = model.encode_batch(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_encode_batch_empty_list(self) -> None:
        """Test batch encoding with empty list."""
        model = EmbeddingModel()
        result = model.encode_batch([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)
        assert result.dtype == np.float32

    def test_encode_batch_passes_correct_args(
        self, mock_sentence_transformer_batch
    ) -> None:
        """Test that encode_batch passes correct arguments."""
        model = EmbeddingModel()
        model._model = mock_sentence_transformer_batch

        model.encode_batch(
            ["text 1", "text 2"],
            normalize=True,
            batch_size=16,
            show_progress=True,
        )

        call_args = mock_sentence_transformer_batch.encode.call_args
        assert call_args[1]["normalize_embeddings"] is True
        assert call_args[1]["batch_size"] == 16
        assert call_args[1]["show_progress_bar"] is True

    def test_encode_sample(self, mock_sentence_transformer) -> None:
        """Test encoding an instruction-output pair."""
        model = EmbeddingModel()
        model._model = mock_sentence_transformer

        result = model.encode_sample(
            instruction="What is Python?",
            output="A programming language.",
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

        # Check that the combined text was used
        call_args = mock_sentence_transformer.encode.call_args
        encoded_text = call_args[0][0]
        assert "What is Python?" in encoded_text
        assert "A programming language." in encoded_text

    def test_encode_samples_batch(self, mock_sentence_transformer_batch) -> None:
        """Test batch encoding of instruction-output pairs."""
        model = EmbeddingModel()
        model._model = mock_sentence_transformer_batch

        samples = [
            ("Instruction 1", "Output 1"),
            ("Instruction 2", "Output 2"),
        ]
        result = model.encode_samples_batch(samples)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)

    def test_encode_samples_batch_empty(self) -> None:
        """Test batch encoding with empty samples list."""
        model = EmbeddingModel()
        result = model.encode_samples_batch([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)


@pytest.mark.real
class TestEmbeddingModelReal:
    """Integration tests using the real embedding model.

    These tests are skipped by default. Run with: pytest -m real
    """

    @pytest.fixture(scope="class")
    def real_model(self):
        """Load the real model once for all tests in this class."""
        model = EmbeddingModel()
        # Force model loading
        model._load_model()
        return model

    def test_real_encode_single(self, real_model) -> None:
        """Test real encoding of a single text."""
        embedding = real_model.encode("What is Python?")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        # Normalized embeddings should have L2 norm close to 1
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01

    def test_real_encode_batch(self, real_model) -> None:
        """Test real batch encoding."""
        texts = [
            "What is Python?",
            "How do you define a function?",
            "Explain list comprehensions.",
        ]
        embeddings = real_model.encode_batch(texts)

        assert embeddings.shape == (3, 384)
        # Each embedding should be normalized
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01

    def test_real_similar_texts_have_high_similarity(self, real_model) -> None:
        """Test that similar texts have high cosine similarity."""
        emb1 = real_model.encode("What is Python programming language?")
        emb2 = real_model.encode("Tell me about the Python language")

        # Cosine similarity (since they're normalized, it's just dot product)
        similarity = np.dot(emb1, emb2)
        assert similarity > 0.7  # Similar texts should have high similarity

    def test_real_different_texts_have_lower_similarity(self, real_model) -> None:
        """Test that different texts have lower cosine similarity."""
        emb1 = real_model.encode("What is Python programming language?")
        emb2 = real_model.encode("The weather is sunny today")

        similarity = np.dot(emb1, emb2)
        assert similarity < 0.5  # Different topics should have lower similarity

    def test_real_encode_sample(self, real_model) -> None:
        """Test encoding an instruction-output pair."""
        embedding = real_model.encode_sample(
            instruction="What is a Python list?",
            output="A list is a mutable ordered collection of items.",
        )

        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01
