"""Tests for the export module."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoft.export.jsonl_exporter import (
    count_jsonl,
    export_from_database,
    export_to_jsonl,
    read_jsonl,
    validate_jsonl,
)


class TestExportToJsonl:
    """Tests for export_to_jsonl function."""

    def test_export_single_sample(self, tmp_path: Path) -> None:
        """Test exporting a single sample."""
        samples = [{"instruction": "What is 2+2?", "output": "4"}]
        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl(samples, output_path)

        assert count == 1
        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data == {"instruction": "What is 2+2?", "output": "4"}

    def test_export_multiple_samples(self, tmp_path: Path) -> None:
        """Test exporting multiple samples."""
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
            {"instruction": "Q3", "output": "A3"},
        ]
        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl(samples, output_path)

        assert count == 3
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["instruction"] == f"Q{i + 1}"
                assert data["output"] == f"A{i + 1}"

    def test_export_empty_samples(self, tmp_path: Path) -> None:
        """Test exporting empty list."""
        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl([], output_path)

        assert count == 0
        assert output_path.exists()
        with open(output_path) as f:
            assert f.read() == ""

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        output_path = tmp_path / "nested" / "dir" / "test.jsonl"
        samples = [{"instruction": "test", "output": "test"}]

        export_to_jsonl(samples, output_path)

        assert output_path.exists()

    def test_export_with_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work."""
        samples = [{"instruction": "test", "output": "test"}]
        output_path = str(tmp_path / "test.jsonl")

        count = export_to_jsonl(samples, output_path)

        assert count == 1
        assert Path(output_path).exists()

    def test_export_missing_key_raises_error(self, tmp_path: Path) -> None:
        """Test that missing keys raise ValueError."""
        samples = [{"instruction": "test"}]  # Missing 'output'
        output_path = tmp_path / "test.jsonl"

        with pytest.raises(ValueError, match="missing required keys"):
            export_to_jsonl(samples, output_path)

    def test_export_only_includes_required_keys(self, tmp_path: Path) -> None:
        """Test that only required keys are exported."""
        samples = [
            {
                "instruction": "test",
                "output": "response",
                "extra_key": "should not appear",
                "another": 123,
            }
        ]
        output_path = tmp_path / "test.jsonl"

        export_to_jsonl(samples, output_path)

        with open(output_path) as f:
            data = json.loads(f.read().strip())
            assert data == {"instruction": "test", "output": "response"}
            assert "extra_key" not in data

    def test_export_custom_keys(self, tmp_path: Path) -> None:
        """Test exporting with custom required keys."""
        samples = [{"question": "What?", "answer": "This."}]
        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl(
            samples, output_path, ensure_keys=("question", "answer")
        )

        assert count == 1
        with open(output_path) as f:
            data = json.loads(f.read().strip())
            assert data == {"question": "What?", "answer": "This."}

    def test_export_unicode_content(self, tmp_path: Path) -> None:
        """Test exporting unicode content."""
        samples = [
            {"instruction": "Translate to Japanese", "output": "こんにちは"},
            {"instruction": "Chinese greeting", "output": "你好"},
        ]
        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl(samples, output_path)

        assert count == 2
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            assert data1["output"] == "こんにちは"
            assert data2["output"] == "你好"

    def test_export_with_generator(self, tmp_path: Path) -> None:
        """Test exporting from a generator (iterable)."""

        def sample_generator():
            for i in range(5):
                yield {"instruction": f"Q{i}", "output": f"A{i}"}

        output_path = tmp_path / "test.jsonl"

        count = export_to_jsonl(sample_generator(), output_path)

        assert count == 5


class TestExportFromDatabase:
    """Tests for export_from_database function."""

    def test_export_from_mock_provider(self, tmp_path: Path) -> None:
        """Test exporting from a mock database provider."""
        # Create a mock that behaves like Database
        mock_provider = MagicMock()
        mock_provider.iter_samples.return_value = iter(
            [
                [
                    {"instruction": "Q1", "output": "A1"},
                    {"instruction": "Q2", "output": "A2"},
                ],
                [
                    {"instruction": "Q3", "output": "A3"},
                ],
            ]
        )

        output_path = tmp_path / "test.jsonl"

        count = export_from_database(mock_provider, output_path)

        assert count == 3
        mock_provider.iter_samples.assert_called_once_with(batch_size=1000)

    def test_export_from_database_custom_batch_size(self, tmp_path: Path) -> None:
        """Test custom batch size is passed to provider."""
        mock_provider = MagicMock()
        mock_provider.iter_samples.return_value = iter([])

        output_path = tmp_path / "test.jsonl"

        export_from_database(mock_provider, output_path, batch_size=500)

        mock_provider.iter_samples.assert_called_once_with(batch_size=500)

    def test_export_from_empty_database(self, tmp_path: Path) -> None:
        """Test exporting from empty database."""
        mock_provider = MagicMock()
        mock_provider.iter_samples.return_value = iter([])

        output_path = tmp_path / "test.jsonl"

        count = export_from_database(mock_provider, output_path)

        assert count == 0
        assert output_path.exists()


class TestValidateJsonl:
    """Tests for validate_jsonl function."""

    def test_validate_valid_file(self, tmp_path: Path) -> None:
        """Test validating a valid JSONL file."""
        file_path = tmp_path / "valid.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "Q1", "output": "A1"}\n')
            f.write('{"instruction": "Q2", "output": "A2"}\n')

        valid_count, errors = validate_jsonl(file_path)

        assert valid_count == 2
        assert errors == []

    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        """Test validating file with invalid JSON."""
        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "valid", "output": "ok"}\n')
            f.write("not valid json\n")
            f.write('{"instruction": "valid2", "output": "ok2"}\n')

        valid_count, errors = validate_jsonl(file_path)

        assert valid_count == 2
        assert len(errors) == 1
        assert "Line 2" in errors[0]
        assert "Invalid JSON" in errors[0]

    def test_validate_missing_keys(self, tmp_path: Path) -> None:
        """Test validating file with missing keys."""
        file_path = tmp_path / "missing.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "ok", "output": "ok"}\n')
            f.write('{"instruction": "missing output"}\n')
            f.write('{"output": "missing instruction"}\n')

        valid_count, errors = validate_jsonl(file_path)

        assert valid_count == 1
        assert len(errors) == 2
        assert "Line 2" in errors[0]
        assert "Missing keys" in errors[0]

    def test_validate_non_object_line(self, tmp_path: Path) -> None:
        """Test validating file with non-object line."""
        file_path = tmp_path / "array.jsonl"
        with open(file_path, "w") as f:
            f.write('["array", "not", "object"]\n')
            f.write('{"instruction": "ok", "output": "ok"}\n')

        valid_count, errors = validate_jsonl(file_path)

        assert valid_count == 1
        assert len(errors) == 1
        assert "Expected object" in errors[0]

    def test_validate_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        file_path = tmp_path / "empty.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "Q1", "output": "A1"}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"instruction": "Q2", "output": "A2"}\n')

        valid_count, errors = validate_jsonl(file_path)

        assert valid_count == 2
        assert errors == []

    def test_validate_custom_keys(self, tmp_path: Path) -> None:
        """Test validation with custom required keys."""
        file_path = tmp_path / "custom.jsonl"
        with open(file_path, "w") as f:
            f.write('{"question": "Q", "answer": "A"}\n')

        valid_count, errors = validate_jsonl(
            file_path, required_keys=("question", "answer")
        )

        assert valid_count == 1
        assert errors == []


class TestReadJsonl:
    """Tests for read_jsonl function."""

    def test_read_valid_file(self, tmp_path: Path) -> None:
        """Test reading a valid JSONL file."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "Q1", "output": "A1"}\n')
            f.write('{"instruction": "Q2", "output": "A2"}\n')

        samples = list(read_jsonl(file_path))

        assert len(samples) == 2
        assert samples[0] == {"instruction": "Q1", "output": "A1"}
        assert samples[1] == {"instruction": "Q2", "output": "A2"}

    def test_read_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"instruction": "Q1", "output": "A1"}\n')
            f.write("\n")
            f.write('{"instruction": "Q2", "output": "A2"}\n')

        samples = list(read_jsonl(file_path))

        assert len(samples) == 2

    def test_read_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid JSON raises ValueError."""
        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("not valid\n")

        with pytest.raises(ValueError, match="Invalid JSON at line 2"):
            list(read_jsonl(file_path))

    def test_read_is_iterator(self, tmp_path: Path) -> None:
        """Test that read_jsonl returns an iterator."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"a": 1}\n')

        result = read_jsonl(file_path)

        assert isinstance(result, Iterator)


class TestCountJsonl:
    """Tests for count_jsonl function."""

    def test_count_samples(self, tmp_path: Path) -> None:
        """Test counting samples in a file."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"a": 1}\n')
            f.write('{"b": 2}\n')
            f.write('{"c": 3}\n')

        count = count_jsonl(file_path)

        assert count == 3

    def test_count_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are not counted."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"b": 2}\n')

        count = count_jsonl(file_path)

        assert count == 2

    def test_count_empty_file(self, tmp_path: Path) -> None:
        """Test counting empty file."""
        file_path = tmp_path / "empty.jsonl"
        file_path.touch()

        count = count_jsonl(file_path)

        assert count == 0


class TestRoundTrip:
    """Tests for round-trip export and read."""

    def test_export_then_read(self, tmp_path: Path) -> None:
        """Test exporting and reading back."""
        original_samples = [
            {"instruction": "What is Python?", "output": "A programming language."},
            {"instruction": "Explain ML", "output": "Machine learning is..."},
        ]
        file_path = tmp_path / "roundtrip.jsonl"

        export_to_jsonl(original_samples, file_path)
        read_samples = list(read_jsonl(file_path))

        assert read_samples == original_samples

    def test_export_validate_read(self, tmp_path: Path) -> None:
        """Test full workflow: export, validate, read."""
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
            {"instruction": "Q3", "output": "A3"},
        ]
        file_path = tmp_path / "workflow.jsonl"

        # Export
        export_count = export_to_jsonl(samples, file_path)
        assert export_count == 3

        # Validate
        valid_count, errors = validate_jsonl(file_path)
        assert valid_count == 3
        assert errors == []

        # Count
        count = count_jsonl(file_path)
        assert count == 3

        # Read
        read_samples = list(read_jsonl(file_path))
        assert read_samples == samples


class TestIntegrationWithDatabase:
    """Integration tests with actual Database class."""

    def test_export_from_real_database(self, tmp_path: Path) -> None:
        """Test exporting from an actual Database instance."""
        import numpy as np

        from autoft.storage import Database

        db_path = tmp_path / "test.db"
        output_path = tmp_path / "export.jsonl"

        # Create database and insert samples
        with Database(db_path) as db:
            db.insert_sample("Q1", "A1", np.zeros(384, dtype=np.float32))
            db.insert_sample("Q2", "A2", np.zeros(384, dtype=np.float32))
            db.insert_sample("Q3", "A3", np.zeros(384, dtype=np.float32))

            # Export using the database
            count = export_from_database(db, output_path)

        assert count == 3

        # Verify exported content
        samples = list(read_jsonl(output_path))
        assert len(samples) == 3
        assert samples[0] == {"instruction": "Q1", "output": "A1"}
        assert samples[1] == {"instruction": "Q2", "output": "A2"}
        assert samples[2] == {"instruction": "Q3", "output": "A3"}

    def test_large_export_streaming(self, tmp_path: Path) -> None:
        """Test that large exports use streaming correctly."""
        import numpy as np

        from autoft.storage import Database

        db_path = tmp_path / "large.db"
        output_path = tmp_path / "large_export.jsonl"

        # Create 1000 samples
        with Database(db_path) as db:
            samples = [
                (
                    f"Instruction {i}",
                    f"Output {i}",
                    np.random.randn(384).astype(np.float32),
                )
                for i in range(1000)
            ]
            db.insert_samples_batch(samples)

            # Export with small batch size to test streaming
            count = export_from_database(db, output_path, batch_size=100)

        assert count == 1000

        # Verify
        valid_count, errors = validate_jsonl(output_path)
        assert valid_count == 1000
        assert errors == []
