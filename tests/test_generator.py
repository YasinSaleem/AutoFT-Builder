"""Tests for the LLM generator module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autoft.generator.llm_generator import (
    GeneratedSample,
    GenerationError,
    LLMGenerator,
)


class TestGeneratedSample:
    """Tests for GeneratedSample dataclass."""

    def test_create_sample(self) -> None:
        """Test that GeneratedSample is created correctly."""
        sample = GeneratedSample(
            instruction="What is Python?",
            output="Python is a programming language.",
        )
        assert sample.instruction == "What is Python?"
        assert sample.output == "Python is a programming language."

    def test_sample_equality(self) -> None:
        """Test dataclass equality."""
        s1 = GeneratedSample(instruction="test", output="response")
        s2 = GeneratedSample(instruction="test", output="response")
        assert s1 == s2

    def test_sample_inequality(self) -> None:
        """Test dataclass inequality."""
        s1 = GeneratedSample(instruction="test1", output="response")
        s2 = GeneratedSample(instruction="test2", output="response")
        assert s1 != s2


class TestLLMGeneratorInit:
    """Tests for LLMGenerator initialization."""

    def test_init_with_required_args(self) -> None:
        """Test initialization with required arguments."""
        generator = LLMGenerator(api_key="test_key")
        assert generator.api_key == "test_key"
        assert generator.base_url == "https://openrouter.ai/api/v1"
        assert generator.model == "anthropic/claude-3-haiku"

    def test_init_with_all_args(self) -> None:
        """Test initialization with all arguments."""
        generator = LLMGenerator(
            api_key="test_key",
            base_url="https://custom.api.com/v1",
            model="custom/model",
            max_retries=5,
            retry_delay=2.0,
            retry_multiplier=3.0,
        )
        assert generator.api_key == "test_key"
        assert generator.base_url == "https://custom.api.com/v1"
        assert generator.model == "custom/model"
        assert generator.max_retries == 5
        assert generator.retry_delay == 2.0
        assert generator.retry_multiplier == 3.0

    def test_client_not_initialized(self) -> None:
        """Test that client is not initialized on construction."""
        generator = LLMGenerator(api_key="test_key")
        assert generator._client is None


class TestLLMGeneratorClient:
    """Tests for lazy client initialization."""

    @patch("autoft.generator.llm_generator.OpenAI")
    def test_client_lazy_loading(self, mock_openai: MagicMock) -> None:
        """Test that client is lazily initialized."""
        generator = LLMGenerator(
            api_key="test_key",
            base_url="https://api.test.com/v1",
        )
        assert generator._client is None

        # Access client property
        _ = generator.client

        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.test.com/v1",
        )
        assert generator._client is not None

    @patch("autoft.generator.llm_generator.OpenAI")
    def test_client_cached(self, mock_openai: MagicMock) -> None:
        """Test that client is cached after first access."""
        generator = LLMGenerator(api_key="test_key")

        # Access multiple times
        _ = generator.client
        _ = generator.client
        _ = generator.client

        # Should only create once
        mock_openai.assert_called_once()


class TestBuildPrompts:
    """Tests for prompt building methods."""

    def test_build_system_prompt_basic(self) -> None:
        """Test system prompt without format spec."""
        generator = LLMGenerator(api_key="test_key")
        prompt = generator._build_system_prompt("Generate Q&A pairs")

        assert "synthetic data generator" in prompt
        assert "JSON array" in prompt
        assert "instruction" in prompt
        assert "output" in prompt

    def test_build_system_prompt_with_format_spec(self) -> None:
        """Test system prompt with format spec."""
        generator = LLMGenerator(api_key="test_key")
        prompt = generator._build_system_prompt(
            "Generate Q&A pairs", format_spec="Keep answers under 100 words"
        )

        assert "ADDITIONAL FORMAT REQUIREMENTS" in prompt
        assert "Keep answers under 100 words" in prompt

    def test_build_user_prompt(self) -> None:
        """Test user prompt generation."""
        generator = LLMGenerator(api_key="test_key")
        prompt = generator._build_user_prompt("Generate math problems", 5)

        assert "5" in prompt
        assert "Generate math problems" in prompt
        assert "TASK:" in prompt


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_valid_json_array(self) -> None:
        """Test parsing a valid JSON array."""
        generator = LLMGenerator(api_key="test_key")
        content = json.dumps(
            [
                {"instruction": "What is 2+2?", "output": "4"},
                {"instruction": "What is 3+3?", "output": "6"},
            ]
        )

        samples = generator._parse_response(content, expected_count=2)

        assert len(samples) == 2
        assert samples[0].instruction == "What is 2+2?"
        assert samples[0].output == "4"
        assert samples[1].instruction == "What is 3+3?"
        assert samples[1].output == "6"

    def test_parse_json_with_surrounding_text(self) -> None:
        """Test parsing JSON with surrounding text."""
        generator = LLMGenerator(api_key="test_key")
        content = """Here are the examples:
        [{"instruction": "test", "output": "response"}]
        Hope this helps!"""

        samples = generator._parse_response(content, expected_count=1)

        assert len(samples) == 1
        assert samples[0].instruction == "test"

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        generator = LLMGenerator(api_key="test_key")

        with pytest.raises(GenerationError, match="Failed to parse JSON"):
            generator._parse_response("not valid json", expected_count=1)

    def test_parse_non_array(self) -> None:
        """Test parsing non-array JSON raises error."""
        generator = LLMGenerator(api_key="test_key")

        with pytest.raises(GenerationError, match="Expected JSON array"):
            generator._parse_response('{"key": "value"}', expected_count=1)

    def test_parse_skips_invalid_items(self) -> None:
        """Test that invalid items are skipped."""
        generator = LLMGenerator(api_key="test_key")
        content = json.dumps(
            [
                {"instruction": "valid", "output": "valid"},
                {"instruction": "missing output"},
                "not a dict",
                {"instruction": "", "output": "empty instruction"},
                {"instruction": "valid2", "output": "valid2"},
            ]
        )

        samples = generator._parse_response(content, expected_count=5)

        assert len(samples) == 2
        assert samples[0].instruction == "valid"
        assert samples[1].instruction == "valid2"

    def test_parse_empty_result_raises_error(self) -> None:
        """Test that empty result raises error."""
        generator = LLMGenerator(api_key="test_key")
        content = json.dumps(
            [
                {"instruction": "", "output": ""},
                {"missing": "keys"},
            ]
        )

        with pytest.raises(GenerationError, match="No valid samples"):
            generator._parse_response(content, expected_count=2)

    def test_parse_converts_to_string(self) -> None:
        """Test that non-string values are converted."""
        generator = LLMGenerator(api_key="test_key")
        content = json.dumps(
            [
                {"instruction": 123, "output": True},
            ]
        )

        samples = generator._parse_response(content, expected_count=1)

        assert samples[0].instruction == "123"
        assert samples[0].output == "True"

    def test_parse_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        generator = LLMGenerator(api_key="test_key")
        content = json.dumps(
            [
                {"instruction": "  test  ", "output": "\n\nresponse\n\n"},
            ]
        )

        samples = generator._parse_response(content, expected_count=1)

        assert samples[0].instruction == "test"
        assert samples[0].output == "response"


class TestGenerateBatch:
    """Tests for generate_batch method."""

    def test_invalid_batch_size_zero(self) -> None:
        """Test that batch_size of 0 raises ValueError."""
        generator = LLMGenerator(api_key="test_key")

        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            generator.generate_batch("task", batch_size=0)

    def test_invalid_batch_size_negative(self) -> None:
        """Test that negative batch_size raises ValueError."""
        generator = LLMGenerator(api_key="test_key")

        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            generator.generate_batch("task", batch_size=-5)

    @patch.object(LLMGenerator, "_call_api_with_retry")
    def test_generate_batch_success(self, mock_call: MagicMock) -> None:
        """Test successful batch generation."""
        mock_call.return_value = json.dumps(
            [
                {"instruction": "Q1", "output": "A1"},
                {"instruction": "Q2", "output": "A2"},
            ]
        )

        generator = LLMGenerator(api_key="test_key")
        samples = generator.generate_batch("Generate Q&A", batch_size=2)

        assert len(samples) == 2
        assert samples[0].instruction == "Q1"
        assert samples[1].instruction == "Q2"
        mock_call.assert_called_once()

    @patch.object(LLMGenerator, "_call_api_with_retry")
    def test_generate_batch_with_format_spec(self, mock_call: MagicMock) -> None:
        """Test generation with format specification."""
        mock_call.return_value = json.dumps(
            [
                {"instruction": "Q1", "output": "A1"},
            ]
        )

        generator = LLMGenerator(api_key="test_key")
        generator.generate_batch(
            "Generate Q&A", batch_size=1, format_spec="Keep it short"
        )

        # Check that format_spec was included in the prompt
        call_args = mock_call.call_args[0][0]
        assert any("Keep it short" in msg.get("content", "") for msg in call_args)

    @patch.object(LLMGenerator, "_call_api_with_retry")
    def test_generate_batch_large_warns(self, mock_call: MagicMock) -> None:
        """Test that large batch size logs warning."""
        mock_call.return_value = json.dumps(
            [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(51)]
        )

        generator = LLMGenerator(api_key="test_key")

        with patch("autoft.generator.llm_generator.logger") as mock_logger:
            generator.generate_batch("task", batch_size=51)
            mock_logger.warning.assert_called()


class TestGenerateSingle:
    """Tests for generate_single method."""

    @patch.object(LLMGenerator, "generate_batch")
    def test_generate_single_calls_batch(self, mock_batch: MagicMock) -> None:
        """Test that generate_single calls generate_batch with size 1."""
        mock_batch.return_value = [GeneratedSample(instruction="Q", output="A")]

        generator = LLMGenerator(api_key="test_key")
        sample = generator.generate_single("task", format_spec="test")

        mock_batch.assert_called_once_with("task", batch_size=1, format_spec="test")
        assert sample.instruction == "Q"


class TestApiRetry:
    """Tests for API retry logic."""

    @patch("autoft.generator.llm_generator.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep: MagicMock) -> None:
        """Test retry on rate limit error."""
        from openai import RateLimitError

        generator = LLMGenerator(
            api_key="test_key",
            max_retries=3,
            retry_delay=1.0,
            retry_multiplier=2.0,
        )

        # Create mock client
        mock_client = MagicMock()
        generator._client = mock_client

        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '[{"instruction": "Q", "output": "A"}]'

        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                "rate limit", response=MagicMock(status_code=429), body=None
            ),
            RateLimitError(
                "rate limit", response=MagicMock(status_code=429), body=None
            ),
            mock_response,
        ]

        messages = [{"role": "user", "content": "test"}]
        result = generator._call_api_with_retry(messages)

        assert result == '[{"instruction": "Q", "output": "A"}]'
        assert mock_sleep.call_count == 2
        # Check exponential backoff
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("autoft.generator.llm_generator.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep: MagicMock) -> None:  # noqa: ARG002
        """Test retry on connection error."""
        from openai import APIConnectionError

        generator = LLMGenerator(api_key="test_key", max_retries=2)
        mock_client = MagicMock()
        generator._client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '[{"instruction": "Q", "output": "A"}]'

        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_response,
        ]

        result = generator._call_api_with_retry([])
        assert '[{"instruction": "Q", "output": "A"}]' in result

    @patch("autoft.generator.llm_generator.time.sleep")
    def test_no_retry_on_client_error(self, mock_sleep: MagicMock) -> None:
        """Test no retry on 4xx client errors (except 429)."""
        from openai import APIStatusError

        generator = LLMGenerator(api_key="test_key", max_retries=3)
        mock_client = MagicMock()
        generator._client = mock_client

        error_response = MagicMock()
        error_response.status_code = 400

        mock_client.chat.completions.create.side_effect = APIStatusError(
            "bad request",
            response=error_response,
            body=None,
        )

        with pytest.raises(GenerationError, match="API error"):
            generator._call_api_with_retry([])

        # Should not retry on 400
        mock_sleep.assert_not_called()

    @patch("autoft.generator.llm_generator.time.sleep")
    def test_retry_exhausted(self, mock_sleep: MagicMock) -> None:  # noqa: ARG002
        """Test error when retries are exhausted."""
        from openai import RateLimitError

        generator = LLMGenerator(api_key="test_key", max_retries=2)
        mock_client = MagicMock()
        generator._client = mock_client

        mock_client.chat.completions.create.side_effect = RateLimitError(
            "rate limit",
            response=MagicMock(status_code=429),
            body=None,
        )

        with pytest.raises(GenerationError, match="Failed after 2 attempts"):
            generator._call_api_with_retry([])

    def test_empty_response_raises_error(self) -> None:
        """Test that empty response raises GenerationError."""
        generator = LLMGenerator(api_key="test_key")
        mock_client = MagicMock()
        generator._client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(GenerationError, match="Empty response"):
            generator._call_api_with_retry([])

    @patch("autoft.generator.llm_generator.time.sleep")
    def test_retry_on_server_error(self, mock_sleep: MagicMock) -> None:  # noqa: ARG002
        """Test retry on 5xx server errors."""
        from openai import APIStatusError

        generator = LLMGenerator(api_key="test_key", max_retries=2)
        mock_client = MagicMock()
        generator._client = mock_client

        error_response = MagicMock()
        error_response.status_code = 500

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '[{"instruction": "Q", "output": "A"}]'

        mock_client.chat.completions.create.side_effect = [
            APIStatusError("server error", response=error_response, body=None),
            mock_response,
        ]

        result = generator._call_api_with_retry([])
        assert "instruction" in result


class TestIntegrationWithMock:
    """Integration tests with fully mocked API."""

    def test_full_generation_flow(self) -> None:
        """Test complete generation flow."""
        generator = LLMGenerator(
            api_key="test_key",
            model="test/model",
        )

        # Mock the client
        mock_client = MagicMock()
        generator._client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "instruction": "What is AI?",
                    "output": "AI is artificial intelligence.",
                },
                {"instruction": "Explain ML", "output": "ML is machine learning."},
                {
                    "instruction": "Define NLP",
                    "output": "NLP is natural language processing.",
                },
            ]
        )

        mock_client.chat.completions.create.return_value = mock_response

        samples = generator.generate_batch(
            task_description="Generate AI/ML Q&A pairs",
            batch_size=3,
            format_spec="Technical explanations",
        )

        assert len(samples) == 3
        assert all(isinstance(s, GeneratedSample) for s in samples)
        assert samples[0].instruction == "What is AI?"
        assert "artificial intelligence" in samples[0].output


@pytest.mark.real
class TestRealAPI:
    """Integration tests with real OpenRouter API.

    These tests are skipped by default. Run with:
        pytest -m real tests/test_generator.py

    Requires OPENROUTER_API_KEY environment variable.
    """

    @pytest.fixture
    def generator(self) -> LLMGenerator:
        """Create generator with real API key."""
        import os

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        return LLMGenerator(
            api_key=api_key,
            model="anthropic/claude-3-haiku",  # Use a fast, cheap model
        )

    def test_real_single_generation(self, generator: LLMGenerator) -> None:
        """Test real API single generation."""
        sample = generator.generate_single(
            task_description="Generate a simple math question and answer"
        )

        assert isinstance(sample, GeneratedSample)
        assert len(sample.instruction) > 0
        assert len(sample.output) > 0

    def test_real_batch_generation(self, generator: LLMGenerator) -> None:
        """Test real API batch generation."""
        samples = generator.generate_batch(
            task_description="Generate simple trivia questions about science",
            batch_size=3,
        )

        assert len(samples) >= 1  # May get fewer due to parsing
        assert all(isinstance(s, GeneratedSample) for s in samples)
