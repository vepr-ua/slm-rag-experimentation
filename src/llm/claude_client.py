"""
Claude API client for LLM operations throughout the project.

Can be used for:
- Synthetic data generation
- Evaluation (LLM-as-judge)
- Testing and benchmarking
- Any other Claude API interactions
"""

import time
from typing import Optional

from anthropic import Anthropic
from loguru import logger

from ..data_collection.config import CollectorConfig


class ClaudeClient:
    """
    Wrapper for Anthropic Claude API with rate limiting and error handling.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        Initialize Claude client.

        Args:
            config: Configuration with API key and settings
        """
        self.config = config or CollectorConfig()

        if not self.config.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env. "
                "Get your key at https://console.anthropic.com/"
            )

        self.client = Anthropic(api_key=self.config.anthropic_api_key)
        self.model = self.config.anthropic_model
        self.max_tokens = self.config.anthropic_max_tokens

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

        logger.info(f"Initialized Claude client with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_retries: int = 3,
    ) -> str:
        """
        Generate text using Claude API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_retries: Number of retries on failure

        Returns:
            Generated text

        Raises:
            Exception: If generation fails after retries
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()

                # Make API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract text from response
                generated_text = response.content[0].text

                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Generation failed after {max_retries} attempts")
                    raise

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Generate text for multiple prompts with progress tracking.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            List of generated texts
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(prompts, desc="Generating") if show_progress else prompts

        for prompt in iterator:
            try:
                result = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                results.append("")  # Placeholder for failed generation

        logger.info(
            f"Generated {len([r for r in results if r])} / {len(prompts)} successfully"
        )

        return results

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def estimate_cost(self, num_requests: int, avg_tokens_per_request: int = 2000) -> float:
        """
        Estimate API cost for a batch of requests.

        Args:
            num_requests: Number of API calls
            avg_tokens_per_request: Average tokens per request (input + output)

        Returns:
            Estimated cost in USD

        Note:
            Pricing as of 2024 for Claude 3.5 Sonnet:
            - Input: $3 per million tokens
            - Output: $15 per million tokens
            Assuming 50/50 split for estimation
        """
        # Rough estimate: $9 per million tokens (average of input/output)
        total_tokens = num_requests * avg_tokens_per_request
        estimated_cost = (total_tokens / 1_000_000) * 9.0

        logger.info(
            f"Estimated cost for {num_requests} requests "
            f"({total_tokens:,} tokens): ${estimated_cost:.2f}"
        )

        return estimated_cost

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate(
                prompt="Reply with just 'OK' if you can read this.",
                temperature=0.0,
            )
            logger.info("✅ Claude API connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Claude API connection failed: {e}")
            return False
