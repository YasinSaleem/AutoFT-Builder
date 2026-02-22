"""LLM-based synthetic data generator using OpenRouter API."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class GeneratedSample:
    """A generated instruction-output pair."""

    instruction: str
    output: str


class GenerationError(Exception):
    """Error during sample generation."""

    pass


class LLMGenerator:
    """Generate synthetic training data using LLMs via OpenRouter."""

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_RETRY_MULTIPLIER = 2.0

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-3-haiku",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        retry_multiplier: float = DEFAULT_RETRY_MULTIPLIER,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter API base URL.
            model: Model identifier to use for generation.
            max_retries: Maximum number of retry attempts for API calls.
            retry_delay: Initial delay between retries in seconds.
            retry_multiplier: Multiplier for exponential backoff.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_multiplier = retry_multiplier

        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get or create the OpenAI client (lazy loading).

        Returns:
            OpenAI client configured for OpenRouter.
        """
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _build_system_prompt(
        self, task_description: str, format_spec: str | None = None
    ) -> str:
        """Build the system prompt for generation.

        Args:
            task_description: Description of the task for data generation.
            format_spec: Optional format specification for outputs.

        Returns:
            System prompt string.
        """
        prompt = f"""You are a synthetic data generator creating high-quality training examples.
Your task is to generate diverse instruction-output pairs for fine-tuning language models.

TASK: {task_description}

IMPORTANT RULES:
1. Generate DIVERSE examples - vary structure, length, style, and complexity
2. Each example must be UNIQUE and semantically distinct
3. Output must be valid JSON array format
4. Follow the task description carefully

OUTPUT FORMAT:
You must output a JSON array of objects, where each object has exactly two keys:
- "instruction": The input/question/prompt
- "output": The expected response/answer

Example format:
[
  {{"instruction": "...", "output": "..."}},
  {{"instruction": "...", "output": "..."}}
]"""

        if format_spec:
            prompt += f"\n\nADDITIONAL FORMAT REQUIREMENTS:\n{format_spec}"

        return prompt

    def _build_user_prompt(self, task_description: str, batch_size: int) -> str:
        """Build the user prompt for generation.

        Args:
            task_description: Description of the task.
            batch_size: Number of samples to generate.

        Returns:
            User prompt string.
        """
        return f"""Generate exactly {batch_size} diverse training examples for the following task:

TASK: {task_description}

Remember:
- Make each example unique and different from others
- Vary complexity, length, and style
- Output ONLY the JSON array, no additional text

Generate {batch_size} examples now:"""

    def _parse_response(
        self, content: str, expected_count: int
    ) -> list[GeneratedSample]:
        """Parse the LLM response into GeneratedSample objects.

        Args:
            content: Raw response content from LLM.
            expected_count: Expected number of samples.

        Returns:
            List of GeneratedSample objects.

        Raises:
            GenerationError: If parsing fails or format is invalid.
        """
        # Try to extract JSON array from the response
        content = content.strip()

        # Try to find JSON array in the response
        json_match = re.search(r"\[[\s\S]*\]", content)
        if json_match:
            content = json_match.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise GenerationError(f"Failed to parse JSON response: {e}") from e

        if not isinstance(data, list):
            raise GenerationError(f"Expected JSON array, got {type(data).__name__}")

        samples = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(
                    f"Skipping non-dict item at index {i}: {type(item).__name__}"
                )
                continue

            instruction = item.get("instruction")
            output = item.get("output")

            if instruction is None or output is None:
                logger.warning(
                    f"Skipping item at index {i}: missing instruction or output"
                )
                continue

            # Convert to string if needed
            instruction = str(instruction).strip()
            output = str(output).strip()

            if not instruction or not output:
                logger.warning(
                    f"Skipping item at index {i}: empty instruction or output"
                )
                continue

            samples.append(GeneratedSample(instruction=instruction, output=output))

        if not samples:
            raise GenerationError("No valid samples found in response")

        if len(samples) < expected_count:
            logger.warning(
                f"Generated {len(samples)} samples, expected {expected_count}"
            )

        return samples

    def _call_api_with_retry(self, messages: list[dict[str, str]]) -> str:
        """Call the API with retry logic.

        Args:
            messages: List of message dicts for the API.

        Returns:
            Response content string.

        Raises:
            GenerationError: If all retries fail.
        """
        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.8,  # Higher for diversity
                    max_tokens=4096,
                )

                content = response.choices[0].message.content
                if content is None:
                    raise GenerationError("Empty response from API")

                return content

            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                delay *= self.retry_multiplier

            except APIConnectionError as e:
                last_error = e
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                delay *= self.retry_multiplier

            except APIStatusError as e:
                # Don't retry on client errors (4xx except rate limit)
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    raise GenerationError(f"API error: {e.message}") from e

                last_error = e
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                delay *= self.retry_multiplier

        raise GenerationError(f"Failed after {self.max_retries} attempts: {last_error}")

    def generate_batch(
        self,
        task_description: str,
        batch_size: int = 10,
        format_spec: str | None = None,
    ) -> list[GeneratedSample]:
        """Generate a batch of synthetic training samples.

        Args:
            task_description: Description of the task for data generation.
            batch_size: Number of samples to generate.
            format_spec: Optional format specification for outputs.

        Returns:
            List of generated samples.

        Raises:
            GenerationError: If generation fails.
            ValueError: If batch_size is invalid.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if batch_size > 50:
            logger.warning(
                f"Large batch size ({batch_size}) may result in lower quality. "
                "Consider using smaller batches."
            )

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(task_description, format_spec),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(task_description, batch_size),
            },
        ]

        content = self._call_api_with_retry(messages)
        return self._parse_response(content, batch_size)

    def generate_single(
        self,
        task_description: str,
        format_spec: str | None = None,
    ) -> GeneratedSample:
        """Generate a single training sample.

        Args:
            task_description: Description of the task for data generation.
            format_spec: Optional format specification for outputs.

        Returns:
            Single generated sample.

        Raises:
            GenerationError: If generation fails.
        """
        samples = self.generate_batch(
            task_description, batch_size=1, format_spec=format_spec
        )
        return samples[0]
