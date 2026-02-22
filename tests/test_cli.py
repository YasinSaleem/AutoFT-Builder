"""Tests for the CLI module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autoft.cli import (
    cmd_clear,
    cmd_export,
    cmd_generate,
    cmd_stats,
    create_parser,
    main,
    setup_logging,
)
from autoft.config import Config
from autoft.storage import Database


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    """Reset logging configuration before each test.

    This ensures logging.basicConfig() works for each test.
    """
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Reset the autoft logger
    autoft_logger = logging.getLogger("autoft")
    for handler in autoft_logger.handlers[:]:
        autoft_logger.removeHandler(handler)

    # Reset basicConfig flag
    logging.root.handlers = []


@pytest.fixture
def setup_test_logging(capfd: pytest.CaptureFixture) -> None:  # noqa: ARG001
    """Set up logging for test capture.

    This configures logging to output to stdout so capsys/capfd can capture it.
    """
    setup_logging(verbose=False, quiet=False)


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_exists(self) -> None:
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None

    def test_version_flag(self) -> None:
        """Test that --version flag works."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_generate_command_requires_task(self) -> None:
        """Test that generate command requires --task."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate"])

    def test_generate_command_with_task(self) -> None:
        """Test that generate command accepts --task."""
        parser = create_parser()
        args = parser.parse_args(["generate", "--task", "Test task"])
        assert args.command == "generate"
        assert args.task == "Test task"
        assert args.batch_size == 10  # default

    def test_generate_command_with_all_options(self) -> None:
        """Test generate command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "generate",
                "--task",
                "Test task",
                "--batch-size",
                "20",
                "--threshold",
                "0.9",
                "--batches",
                "3",
                "--format",
                "Keep it short",
            ]
        )
        assert args.task == "Test task"
        assert args.batch_size == 20
        assert args.threshold == 0.9
        assert args.batches == 3
        assert args.format == "Keep it short"

    def test_export_command(self) -> None:
        """Test export command parsing."""
        parser = create_parser()
        args = parser.parse_args(["export", "--output", "test.jsonl"])
        assert args.command == "export"
        assert args.output == "test.jsonl"

    def test_export_default_output(self) -> None:
        """Test export command default output."""
        parser = create_parser()
        args = parser.parse_args(["export"])
        assert args.output == "dataset.jsonl"

    def test_stats_command(self) -> None:
        """Test stats command parsing."""
        parser = create_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_clear_command(self) -> None:
        """Test clear command parsing."""
        parser = create_parser()
        args = parser.parse_args(["clear"])
        assert args.command == "clear"
        assert args.force is False

    def test_clear_command_with_force(self) -> None:
        """Test clear command with --force flag."""
        parser = create_parser()
        args = parser.parse_args(["clear", "--force"])
        assert args.command == "clear"
        assert args.force is True

    def test_verbose_flag(self) -> None:
        """Test --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "stats"])
        assert args.verbose is True

    def test_db_option(self) -> None:
        """Test --db option for all commands."""
        parser = create_parser()

        for cmd in ["stats", "export", "clear"]:
            args = parser.parse_args([cmd, "--db", "/custom/path.db"])
            assert args.db == "/custom/path.db"


class TestMain:
    """Tests for main CLI entry point."""

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """Test that no command shows help."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()

    @patch("autoft.config.load_dotenv")
    def test_generate_requires_api_key(
        self,
        mock_load_dotenv: MagicMock,  # noqa: ARG002
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that generate command requires API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with caplog.at_level(logging.ERROR, logger="autoft"):
            result = main(["generate", "--task", "Test"])

        assert result == 1
        assert "OPENROUTER_API_KEY" in caplog.text

    @patch("autoft.config.load_dotenv")
    def test_stats_works_without_api_key(
        self,
        mock_load_dotenv: MagicMock,  # noqa: ARG002
        capsys: pytest.CaptureFixture,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that stats command works without API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        db_path = tmp_path / "test.db"

        result = main(["stats", "--db", str(db_path)])
        assert result == 0


class TestCmdStats:
    """Tests for stats command."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create test config."""
        return Config(
            openrouter_api_key="test",
            openrouter_base_url="https://test.api",
            llm_model="test/model",
            similarity_threshold=0.85,
            database_path=tmp_path / "test.db",
        )

    def test_stats_empty_database(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test stats with empty database."""
        # Create empty database
        with Database(config.database_path):
            pass

        args = MagicMock()
        args.db = None

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_stats(args, config)

        assert result == 0
        assert "Total samples:" in caplog.text

    def test_stats_with_samples(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test stats with samples in database."""
        with Database(config.database_path) as db:
            for i in range(5):
                db.insert_sample(f"Q{i}", f"A{i}", np.zeros(384, dtype=np.float32))

        args = MagicMock()
        args.db = None

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_stats(args, config)

        assert result == 0
        assert "Total samples:" in caplog.text
        assert "Database size:" in caplog.text

    def test_stats_database_not_found(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test stats when database doesn't exist."""
        args = MagicMock()
        args.db = None

        with caplog.at_level(logging.WARNING, logger="autoft"):
            result = cmd_stats(args, config)

        assert result == 0
        assert "Database not found" in caplog.text


class TestCmdExport:
    """Tests for export command."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create test config."""
        return Config(
            openrouter_api_key="test",
            openrouter_base_url="https://test.api",
            llm_model="test/model",
            similarity_threshold=0.85,
            database_path=tmp_path / "test.db",
        )

    def test_export_success(
        self, config: Config, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test successful export."""
        # Create database with samples
        with Database(config.database_path) as db:
            db.insert_sample("Q1", "A1", np.zeros(384, dtype=np.float32))
            db.insert_sample("Q2", "A2", np.zeros(384, dtype=np.float32))

        output_path = tmp_path / "export.jsonl"
        args = MagicMock()
        args.db = None
        args.output = str(output_path)

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_export(args, config)

        assert result == 0
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

        assert "Successfully exported 2 samples" in caplog.text

    def test_export_empty_database(
        self, config: Config, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test export with empty database."""
        with Database(config.database_path):
            pass

        args = MagicMock()
        args.db = None
        args.output = str(tmp_path / "export.jsonl")

        with caplog.at_level(logging.WARNING, logger="autoft"):
            result = cmd_export(args, config)

        assert result == 0
        assert "No samples in database" in caplog.text

    def test_export_database_not_found(
        self, config: Config, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test export when database doesn't exist."""
        args = MagicMock()
        args.db = None
        args.output = str(tmp_path / "export.jsonl")

        with caplog.at_level(logging.ERROR, logger="autoft"):
            result = cmd_export(args, config)

        assert result == 1
        assert "Database not found" in caplog.text


class TestCmdClear:
    """Tests for clear command."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create test config."""
        return Config(
            openrouter_api_key="test",
            openrouter_base_url="https://test.api",
            llm_model="test/model",
            similarity_threshold=0.85,
            database_path=tmp_path / "test.db",
        )

    def test_clear_with_force(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test clear with --force flag."""
        with Database(config.database_path) as db:
            db.insert_sample("Q1", "A1", np.zeros(384, dtype=np.float32))
            db.insert_sample("Q2", "A2", np.zeros(384, dtype=np.float32))

        args = MagicMock()
        args.db = None
        args.force = True

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_clear(args, config)

        assert result == 0
        assert "Deleted 2 samples" in caplog.text

        # Verify database is empty
        with Database(config.database_path) as db:
            assert db.get_sample_count() == 0

    def test_clear_aborted(
        self,
        config: Config,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test clear aborted by user."""
        with Database(config.database_path) as db:
            db.insert_sample("Q1", "A1", np.zeros(384, dtype=np.float32))

        args = MagicMock()
        args.db = None
        args.force = False

        # Simulate user typing 'n'
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_clear(args, config)

        assert result == 0
        assert "Aborted" in caplog.text

        # Verify database still has data
        with Database(config.database_path) as db:
            assert db.get_sample_count() == 1

    def test_clear_empty_database(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test clear on empty database."""
        with Database(config.database_path):
            pass

        args = MagicMock()
        args.db = None
        args.force = True

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_clear(args, config)

        assert result == 0
        assert "already empty" in caplog.text

    def test_clear_database_not_found(
        self, config: Config, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test clear when database doesn't exist."""
        args = MagicMock()
        args.db = None
        args.force = True

        with caplog.at_level(logging.WARNING, logger="autoft"):
            result = cmd_clear(args, config)

        assert result == 0
        assert "Database not found" in caplog.text


class TestCmdGenerate:
    """Tests for generate command with mocked components."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create test config."""
        return Config(
            openrouter_api_key="test_key",
            openrouter_base_url="https://test.api",
            llm_model="test/model",
            similarity_threshold=0.85,
            database_path=tmp_path / "test.db",
        )

    @patch("autoft.cli.LLMGenerator")
    @patch("autoft.cli.EmbeddingModel")
    def test_generate_basic_flow(
        self,
        mock_embedder_class: MagicMock,
        mock_generator_class: MagicMock,
        config: Config,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test basic generate flow with mocks."""
        from autoft.generator import GeneratedSample

        # Set up mock generator
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_batch.return_value = [
            GeneratedSample(instruction="Q1", output="A1"),
            GeneratedSample(instruction="Q2", output="A2"),
        ]

        # Set up mock embedder
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        # Return distinct embeddings so they're not filtered
        mock_embedder.encode_samples_batch.return_value = np.array(
            [
                np.random.randn(384).astype(np.float32),
                np.random.randn(384).astype(np.float32),
            ]
        )

        args = MagicMock()
        args.task = "Generate test data"
        args.batch_size = 2
        args.batches = 1
        args.format = None
        args.threshold = None
        args.db = None

        with caplog.at_level(logging.INFO, logger="autoft"):
            result = cmd_generate(args, config)

        assert result == 0
        assert "Generation Complete" in caplog.text
        assert "Total samples generated: 2" in caplog.text

        # Verify samples were stored
        with Database(config.database_path) as db:
            count = db.get_sample_count()
            assert count == 2

    @patch("autoft.cli.LLMGenerator")
    @patch("autoft.cli.EmbeddingModel")
    def test_generate_filters_similar(
        self,
        mock_embedder_class: MagicMock,
        mock_generator_class: MagicMock,
        config: Config,
        caplog: pytest.LogCaptureFixture,  # noqa: ARG002
    ) -> None:
        """Test that similar samples are filtered."""
        from autoft.generator import GeneratedSample

        # Set up mock generator
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_batch.return_value = [
            GeneratedSample(instruction="Q1", output="A1"),
            GeneratedSample(instruction="Q2", output="A2"),
        ]

        # Set up mock embedder - return identical embeddings
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        identical_embedding = np.ones(384, dtype=np.float32)
        mock_embedder.encode_samples_batch.return_value = np.array(
            [
                identical_embedding,
                identical_embedding,
            ]
        )

        args = MagicMock()
        args.task = "Generate test data"
        args.batch_size = 2
        args.batches = 1
        args.format = None
        args.threshold = 0.85
        args.db = None

        result = cmd_generate(args, config)

        assert result == 0

        # Only first sample should be accepted (second is duplicate)
        with Database(config.database_path) as db:
            count = db.get_sample_count()
            assert count == 1

    @patch("autoft.cli.LLMGenerator")
    @patch("autoft.cli.EmbeddingModel")
    def test_generate_handles_error(
        self,
        mock_embedder_class: MagicMock,
        mock_generator_class: MagicMock,
        config: Config,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that generation errors are handled gracefully."""
        from autoft.generator import GenerationError

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_batch.side_effect = GenerationError("API error")

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        args = MagicMock()
        args.task = "Generate test data"
        args.batch_size = 2
        args.batches = 1
        args.format = None
        args.threshold = None
        args.db = None

        with caplog.at_level(logging.ERROR, logger="autoft"):
            result = cmd_generate(args, config)

        assert result == 0  # Should still succeed overall
        # Check for generation failure message
        assert "Generation failed" in caplog.text or "Batch 1/1" in caplog.text


class TestIntegration:
    """Integration tests for CLI commands."""

    def test_full_workflow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,  # noqa: ARG002
    ) -> None:
        """Test full workflow: stats -> clear -> stats."""
        db_path = tmp_path / "test.db"

        # Create database with samples
        with Database(db_path) as db:
            for i in range(10):
                db.insert_sample(
                    f"Q{i}", f"A{i}", np.random.randn(384).astype(np.float32)
                )

        # Check stats
        result = main(["stats", "--db", str(db_path)])
        assert result == 0

        # Export
        export_path = tmp_path / "export.jsonl"
        result = main(["export", "--db", str(db_path), "--output", str(export_path)])
        assert result == 0
        assert export_path.exists()

        # Clear with force
        result = main(["clear", "--db", str(db_path), "--force"])
        assert result == 0

        # Verify empty
        with Database(db_path) as db:
            assert db.get_sample_count() == 0
