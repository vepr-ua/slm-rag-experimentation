"""
Data quality validation utilities.
"""

import re
from typing import Optional

from loguru import logger

from ..models import QAPair


class QualityValidator:
    """Validate quality of collected Q&A pairs."""

    def __init__(
        self,
        min_question_length: int = 20,
        min_answer_length: int = 50,
        max_answer_length: int = 5000,
        max_question_length: int = 1000,
    ):
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.max_question_length = max_question_length

    def validate(self, qa_pair: QAPair) -> tuple[bool, Optional[str]]:
        """
        Validate a Q&A pair.

        Args:
            qa_pair: QAPair to validate

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Length checks
        if len(qa_pair.question) < self.min_question_length:
            return False, f"Question too short ({len(qa_pair.question)} chars)"

        if len(qa_pair.question) > self.max_question_length:
            return False, f"Question too long ({len(qa_pair.question)} chars)"

        if len(qa_pair.answer) < self.min_answer_length:
            return False, f"Answer too short ({len(qa_pair.answer)} chars)"

        if len(qa_pair.answer) > self.max_answer_length:
            return False, f"Answer too long ({len(qa_pair.answer)} chars)"

        # Content quality checks
        if not self._is_meaningful_text(qa_pair.question):
            return False, "Question appears to be low quality"

        if not self._is_meaningful_text(qa_pair.answer):
            return False, "Answer appears to be low quality"

        # Check for excessive URLs (might be spam)
        if self._has_excessive_urls(qa_pair.answer):
            return False, "Answer has too many URLs"

        # Check for code blocks (we want minimal code)
        if self._has_excessive_code(qa_pair.answer):
            return False, "Answer has too much code"

        return True, None

    def _is_meaningful_text(self, text: str) -> bool:
        """Check if text contains meaningful content."""
        # Remove whitespace
        cleaned = text.strip()

        # Check for minimum word count
        words = cleaned.split()
        if len(words) < 5:
            return False

        # Check for excessive repetition
        if self._has_excessive_repetition(cleaned):
            return False

        # Check for reasonable character diversity
        unique_chars = len(set(cleaned.lower()))
        if unique_chars < 10:
            return False

        return True

    def _has_excessive_repetition(self, text: str, max_ratio: float = 0.7) -> bool:
        """Check if text has excessive repetition."""
        words = text.lower().split()
        if len(words) < 10:
            return False

        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))

        return repetition_ratio > max_ratio

    def _has_excessive_urls(self, text: str, max_urls: int = 3) -> bool:
        """Check if text has too many URLs."""
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        urls = re.findall(url_pattern, text)
        return len(urls) > max_urls

    def _has_excessive_code(self, text: str, max_code_ratio: float = 0.4) -> bool:
        """Check if text has too much code."""
        # Simple heuristic: count code-like patterns
        code_indicators = [
            r"```",  # Markdown code blocks
            r"def\s+\w+\(",  # Python functions
            r"function\s+\w+\(",  # JavaScript functions
            r"class\s+\w+",  # Class definitions
            r"import\s+",  # Import statements
            r"<-\s*",  # R assignment
        ]

        code_matches = 0
        for pattern in code_indicators:
            code_matches += len(re.findall(pattern, text))

        # Rough estimate: if we see many code patterns, it's probably code-heavy
        lines = text.split("\n")
        if lines and code_matches > len(lines) * max_code_ratio:
            return True

        return False


class DataCleaner:
    """Clean and normalize collected data."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix common issues
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text.strip()

    @staticmethod
    def remove_code_blocks(text: str) -> str:
        """Remove code blocks from text (for conceptual answers)."""
        # Remove markdown code blocks
        text = re.sub(r"```[^`]*```", "[code example removed]", text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)

        return text

    @staticmethod
    def extract_formulas(text: str) -> tuple[str, list[str]]:
        """
        Extract mathematical formulas from text.

        Returns:
            Tuple of (text_without_formulas, list_of_formulas)
        """
        formulas = []

        # Extract LaTeX formulas
        latex_patterns = [
            r"\$\$[^\$]+\$\$",  # Display math
            r"\$[^\$]+\$",  # Inline math
            r"\\[[^\]]+\\]",  # Alternative display math
            r"\\([^\)]+\\)",  # Alternative inline math
        ]

        text_cleaned = text
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            formulas.extend(matches)
            # Optionally keep formulas or replace with placeholder
            # text_cleaned = re.sub(pattern, "[formula]", text_cleaned)

        return text_cleaned, formulas


def filter_qa_pairs(
    qa_pairs: list[QAPair], validator: Optional[QualityValidator] = None
) -> tuple[list[QAPair], list[tuple[QAPair, str]]]:
    """
    Filter Q&A pairs based on quality criteria.

    Args:
        qa_pairs: List of QAPair objects to filter
        validator: QualityValidator instance (creates default if None)

    Returns:
        Tuple of (valid_pairs, invalid_pairs_with_reasons)
    """
    validator = validator or QualityValidator()

    valid = []
    invalid = []

    for qa_pair in qa_pairs:
        is_valid, reason = validator.validate(qa_pair)
        if is_valid:
            valid.append(qa_pair)
        else:
            invalid.append((qa_pair, reason))
            logger.debug(f"Filtered out {qa_pair.id}: {reason}")

    logger.info(
        f"Filtered {len(qa_pairs)} pairs: {len(valid)} valid, {len(invalid)} invalid"
    )

    return valid, invalid
