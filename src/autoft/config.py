"""Configuration management for AutoFT-Builder."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""

    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    similarity_threshold: float
    database_path: Path

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            env_path: Optional path to .env file. If not provided,
                     looks for .env in current directory.

        Returns:
            Config instance with loaded values.

        Raises:
            ValueError: If required configuration is missing.
        """
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. "
                "Set it in .env file or as environment variable."
            )

        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            llm_model=os.getenv("LLM_MODEL", "anthropic/claude-3-haiku"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
            database_path=Path(os.getenv("DATABASE_PATH", "data/dataset.db")),
        )


def get_config(env_path: str | None = None) -> Config:
    """Get application configuration.

    Args:
        env_path: Optional path to .env file.

    Returns:
        Config instance.
    """
    return Config.from_env(env_path)
