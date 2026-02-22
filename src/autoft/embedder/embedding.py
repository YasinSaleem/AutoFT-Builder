"""Local embedding model using BAAI/bge-small-en-v1.5."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Embedding model using BAAI/bge-small-en-v1.5.

    This model generates 384-dimensional embeddings optimized for semantic
    similarity tasks. It uses lazy loading to avoid loading the model
    until it's actually needed.

    The BGE model requires a specific prefix for optimal performance when
    encoding queries/documents for retrieval tasks.

    Example:
        >>> model = EmbeddingModel()
        >>> embedding = model.encode("What is Python?")
        >>> embedding.shape
        (384,)
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384
    PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str | None = None,
        use_prefix: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialize the embedding model (lazy loading).

        Args:
            model_name: Model name to use. Defaults to BAAI/bge-small-en-v1.5.
            use_prefix: Whether to prepend the BGE prefix to texts.
            device: Device to run the model on ('cpu', 'cuda', etc.).
                   Defaults to 'cpu' for portability.
        """
        self._model_name = model_name or self.MODEL_NAME
        self._use_prefix = use_prefix
        self._device = device or "cpu"
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> "SentenceTransformer":
        """Load the model if not already loaded.

        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        return self._model

    def _prepare_text(self, text: str) -> str:
        """Prepare text for encoding by optionally adding prefix.

        Args:
            text: Raw text to prepare.

        Returns:
            Text with prefix if use_prefix is True.
        """
        if self._use_prefix:
            return f"{self.PREFIX}{text}"
        return text

    def _prepare_texts(self, texts: list[str]) -> list[str]:
        """Prepare multiple texts for encoding.

        Args:
            texts: List of raw texts to prepare.

        Returns:
            List of texts with prefix if use_prefix is True.
        """
        if self._use_prefix:
            return [f"{self.PREFIX}{text}" for text in texts]
        return texts

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Text to encode.
            normalize: Whether to L2-normalize the embedding. Defaults to True
                      for cosine similarity comparisons.

        Returns:
            384-dimensional embedding vector as float32 numpy array.
        """
        model = self._load_model()
        prepared_text = self._prepare_text(text)

        embedding = model.encode(
            prepared_text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple texts into embedding vectors.

        Args:
            texts: List of texts to encode.
            normalize: Whether to L2-normalize embeddings. Defaults to True.
            batch_size: Batch size for encoding. Defaults to 32.
            show_progress: Whether to show a progress bar. Defaults to False.

        Returns:
            Array of shape (n_texts, 384) with embedding vectors as float32.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        model = self._load_model()
        prepared_texts = self._prepare_texts(texts)

        embeddings = model.encode(
            prepared_texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_sample(
        self,
        instruction: str,
        output: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode an instruction-output pair into a single embedding.

        This concatenates the instruction and output with a space separator
        before encoding, as specified in the PRD.

        Args:
            instruction: The instruction text.
            output: The output text.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            384-dimensional embedding vector.
        """
        combined_text = f"{instruction} {output}"
        return self.encode(combined_text, normalize=normalize)

    def encode_samples_batch(
        self,
        samples: list[tuple[str, str]],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple instruction-output pairs into embeddings.

        Args:
            samples: List of (instruction, output) tuples.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            Array of shape (n_samples, 384) with embedding vectors.
        """
        if not samples:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        combined_texts = [f"{instr} {out}" for instr, out in samples]
        return self.encode_batch(
            combined_texts,
            normalize=normalize,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM
