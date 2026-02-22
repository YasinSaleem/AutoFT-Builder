"""Tests for the configuration module."""

from unittest.mock import patch

import pytest

from autoft.config import Config, get_config


class TestConfig:
    """Tests for Config class."""

    @patch("autoft.config.load_dotenv")
    def test_config_from_env_with_api_key(
        self,
        mock_load_dotenv: patch,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that config loads correctly with API key set."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")
        # Clear other env vars to test defaults
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
        monkeypatch.delenv("DATABASE_PATH", raising=False)

        config = Config.from_env()

        assert config.openrouter_api_key == "test_api_key"
        assert config.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert config.llm_model == "anthropic/claude-3-haiku"
        assert config.similarity_threshold == 0.85
        assert str(config.database_path) == "data/dataset.db"

    @patch("autoft.config.load_dotenv")
    def test_config_from_env_missing_api_key(
        self,
        mock_load_dotenv: patch,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that config raises error when API key is missing."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            Config.from_env()

        assert "OPENROUTER_API_KEY is required" in str(exc_info.value)

    @patch("autoft.config.load_dotenv")
    def test_config_custom_values(
        self,
        mock_load_dotenv: patch,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that config respects custom environment values."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "custom_key")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.url/api")
        monkeypatch.setenv("LLM_MODEL", "custom/model")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
        monkeypatch.setenv("DATABASE_PATH", "custom/path.db")

        config = Config.from_env()

        assert config.openrouter_api_key == "custom_key"
        assert config.openrouter_base_url == "https://custom.url/api"
        assert config.llm_model == "custom/model"
        assert config.similarity_threshold == 0.75
        assert str(config.database_path) == "custom/path.db"


class TestGetConfig:
    """Tests for get_config function."""

    @patch("autoft.config.load_dotenv")
    def test_get_config_returns_config(
        self,
        mock_load_dotenv: patch,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that get_config returns a Config instance."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        config = get_config()

        assert isinstance(config, Config)
        assert config.openrouter_api_key == "test_key"
