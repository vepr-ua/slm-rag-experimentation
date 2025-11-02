"""
Data models for training data collection and storage.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Source of the training data."""

    CROSS_VALIDATED = "cross_validated"
    ARXIV = "arxiv"
    TEXTBOOK = "textbook"
    INDUSTRY_BLOG = "industry_blog"
    WIKIPEDIA = "wikipedia"
    SYNTHETIC = "synthetic"


class QuestionType(str, Enum):
    """Type of question being asked."""

    DEFINITION = "definition"  # What is X?
    EXPLANATION = "explanation"  # How does X work?
    COMPARISON = "comparison"  # What's the difference between X and Y?
    METHODOLOGY = "methodology"  # How do I do X?
    CALCULATION = "calculation"  # How do I calculate X?
    TROUBLESHOOTING = "troubleshooting"  # Why is X happening?
    BEST_PRACTICE = "best_practice"  # What's the best way to do X?


class ExperimentationTopic(str, Enum):
    """Experimentation topic taxonomy for organizing training data."""

    # Core Statistics
    HYPOTHESIS_TESTING = "hypothesis_testing"
    P_VALUES = "p_values"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    STATISTICAL_POWER = "statistical_power"
    EFFECT_SIZE = "effect_size"
    SIGNIFICANCE = "significance"

    # Experiment Design
    SAMPLE_SIZE = "sample_size"
    RANDOMIZATION = "randomization"
    STRATIFICATION = "stratification"
    TEST_DURATION = "test_duration"
    METRIC_SELECTION = "metric_selection"

    # Analysis Methods
    VARIANCE_REDUCTION = "variance_reduction"
    MULTIPLE_TESTING = "multiple_testing"
    SEQUENTIAL_TESTING = "sequential_testing"
    BAYESIAN_METHODS = "bayesian_methods"

    # Advanced Topics
    CAUSAL_INFERENCE = "causal_inference"
    HETEROGENEOUS_TREATMENT_EFFECTS = "heterogeneous_treatment_effects"
    NETWORK_EFFECTS = "network_effects"

    # General
    GENERAL = "general"
    OTHER = "other"


class QAPair(BaseModel):
    """A question-answer pair for training."""

    id: str = Field(..., description="Unique identifier")
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")

    # Metadata
    source: DataSource = Field(..., description="Source of this Q&A pair")
    source_url: Optional[str] = Field(None, description="Original URL if applicable")
    source_id: Optional[str] = Field(None, description="ID in source system")

    # Classification
    question_type: Optional[QuestionType] = Field(None, description="Type of question")
    topics: list[ExperimentationTopic] = Field(
        default_factory=list, description="Topics covered"
    )

    # Quality metrics
    score: Optional[float] = Field(None, description="Quality score if available")
    upvotes: Optional[int] = Field(None, description="Upvotes/likes if applicable")
    view_count: Optional[int] = Field(None, description="View count if available")

    # Context
    reasoning: Optional[str] = Field(
        None, description="Step-by-step reasoning (for reasoning-focused training)"
    )
    citations: list[str] = Field(
        default_factory=list, description="Sources cited in answer"
    )
    multi_turn: bool = Field(False, description="Part of multi-turn conversation")
    parent_id: Optional[str] = Field(None, description="Parent Q&A if multi-turn")

    # Timestamps
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    source_created_at: Optional[datetime] = Field(None)

    class Config:
        use_enum_values = True


class CollectionStats(BaseModel):
    """Statistics about collected data."""

    total_examples: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_topic: dict[str, int] = Field(default_factory=dict)
    by_question_type: dict[str, int] = Field(default_factory=dict)
    date_range: Optional[tuple[datetime, datetime]] = None


class DatasetSplit(BaseModel):
    """Train/validation/test split configuration."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    stratify_by: Optional[str] = "topics"  # Stratify split by topic
