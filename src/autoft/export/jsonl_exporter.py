"""JSONL export functionality for fine-tuning datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class SampleDict(Protocol):
    """Protocol for sample dictionaries."""

    def __getitem__(self, key: str) -> str:
        """Get item by key."""
        ...


class SampleProvider(Protocol):
    """Protocol for objects that provide samples."""

    def iter_samples(self, batch_size: int = 1000) -> Iterator[list[dict[str, str]]]:
        """Iterate over samples in batches."""
        ...

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        ...


def export_to_jsonl(
    samples: Iterable[dict[str, str]],
    output_path: Path | str,
    ensure_keys: tuple[str, ...] = ("instruction", "output"),
) -> int:
    """Export samples to a JSONL file.

    Args:
        samples: Iterable of sample dicts with 'instruction' and 'output' keys.
        output_path: Path to output JSONL file.
        ensure_keys: Keys that must be present in each sample.

    Returns:
        Number of samples exported.

    Raises:
        ValueError: If a sample is missing required keys.
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Validate required keys
            missing_keys = [key for key in ensure_keys if key not in sample]
            if missing_keys:
                raise ValueError(
                    f"Sample missing required keys: {missing_keys}. "
                    f"Sample has keys: {list(sample.keys())}"
                )

            # Export only the required keys (not embedding, created_at, etc.)
            export_sample = {key: sample[key] for key in ensure_keys}
            f.write(json.dumps(export_sample, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Exported {count} samples to {output_path}")
    return count


def export_from_database(
    provider: SampleProvider,
    output_path: Path | str,
    batch_size: int = 1000,
    ensure_keys: tuple[str, ...] = ("instruction", "output"),
) -> int:
    """Export samples from a database provider to JSONL using streaming.

    This is memory-efficient for large datasets as it processes samples
    in batches without loading all samples into memory at once.

    Args:
        provider: Object implementing SampleProvider protocol (e.g., Database).
        output_path: Path to output JSONL file.
        batch_size: Number of samples to process per batch.
        ensure_keys: Keys that must be present in each sample.

    Returns:
        Number of samples exported.
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in provider.iter_samples(batch_size=batch_size):
            for sample in batch:
                # Export only the required keys
                export_sample = {key: sample[key] for key in ensure_keys}
                f.write(json.dumps(export_sample, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Exported {count} samples to {output_path}")
    return count


def validate_jsonl(
    file_path: Path | str, required_keys: tuple[str, ...] = ("instruction", "output")
) -> tuple[int, list[str]]:
    """Validate a JSONL file for fine-tuning compatibility.

    Args:
        file_path: Path to JSONL file to validate.
        required_keys: Keys required in each line.

    Returns:
        Tuple of (valid_count, list of error messages).
    """
    file_path = Path(file_path)
    errors: list[str] = []
    valid_count = 0

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            if not isinstance(data, dict):
                errors.append(
                    f"Line {line_num}: Expected object, got {type(data).__name__}"
                )
                continue

            missing = [key for key in required_keys if key not in data]
            if missing:
                errors.append(f"Line {line_num}: Missing keys {missing}")
                continue

            valid_count += 1

    return valid_count, errors


def read_jsonl(file_path: Path | str) -> Iterator[dict[str, str]]:
    """Read samples from a JSONL file.

    Args:
        file_path: Path to JSONL file.

    Yields:
        Sample dictionaries.

    Raises:
        ValueError: If a line contains invalid JSON.
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e


def count_jsonl(file_path: Path | str) -> int:
    """Count the number of samples in a JSONL file.

    Args:
        file_path: Path to JSONL file.

    Returns:
        Number of non-empty lines.
    """
    file_path = Path(file_path)
    count = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count
