"""Pytest fixtures and configuration for AutoFT-Builder tests."""

import pytest


@pytest.fixture
def sample_instruction() -> str:
    """Return a sample instruction for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_output() -> str:
    """Return a sample output for testing."""
    return "The capital of France is Paris."


@pytest.fixture
def sample_data() -> dict:
    """Return a sample instruction-output pair for testing."""
    return {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    }
