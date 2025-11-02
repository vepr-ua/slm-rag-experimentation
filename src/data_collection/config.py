"""
Configuration for data collection.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataCollectionConfig(BaseSettings):
    """Configuration for data collection from various sources."""

    # No prefix - allows overriding via env vars without DATA_COLLECTION_ prefix
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Output paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Cross Validated (StackExchange) settings
    # API Limits: 300 requests/day without key, 10,000/day with key
    stackexchange_tags: list[str] = [
        "experimental-design",
        "hypothesis-testing",
        "p-value",
        "confidence-interval",
        "sample-size",
        "ab-test",
        "statistical-power",
        "multiple-comparisons",
        "statistical-significance",
        "effect-size",
        "randomization",
        "causal-inference",
    ]
    stackexchange_min_score: int = 5  # Minimum question score
    stackexchange_min_answers: int = 1  # Minimum number of answers
    stackexchange_max_questions: int = 500  # Conservative default (use --max-questions to override)
    stackexchange_accepted_only: bool = False  # Only collect accepted answers

    # ArXiv settings
    # API Guidelines: Max 1 request every 3 seconds, use exponential backoff on errors
    arxiv_search_queries: list[str] = [
        "A/B testing",
        "online controlled experiments",
        "randomized controlled trial",
        "experimental design statistics",
        "causal inference experiments",
        "sequential testing",
        "multiple testing correction",
    ]
    arxiv_max_results_per_query: int = 50  # Conservative to respect API (20 requests/min max)
    arxiv_categories: list[str] = ["stat.ME", "stat.AP", "cs.LG"]

    # Quality filters
    min_question_length: int = 20  # characters
    min_answer_length: int = 50  # characters
    max_answer_length: int = 5000  # characters (very long answers might be low quality)

    # Rate limiting - Conservative defaults for API respect
    # Without StackExchange API key: ~10 req/min (safe for 300/day limit)
    # With API key: can increase to ~30 req/min
    requests_per_minute: int = 10
    concurrent_requests: int = 2  # Reduced for safety


class CollectorConfig(BaseSettings):
    """Configuration for individual data collectors."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # StackExchange API
    stackexchange_api_key: str = ""  # Optional, increases rate limit
    stackexchange_site: str = "stats"  # Cross Validated

    # ArXiv doesn't require API key but has rate limits
    arxiv_delay_seconds: float = 3.0  # Delay between requests

    # Anthropic Claude API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_max_tokens: int = 4096
