"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures for all tests
- Test configuration
- Common test utilities
"""

import pytest


@pytest.fixture
def sample_concept():
    """Sample concept data for testing."""
    return {
        "name": "p-value",
        "definition": "The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true.",
        "category": "statistical_test",
        "related_concepts": ["significance_level", "hypothesis_testing", "null_hypothesis"],
    }


@pytest.fixture
def sample_query():
    """Sample user query for testing."""
    return {
        "question": "What is statistical power?",
        "expected_entities": ["statistical_power"],
        "expected_intent": "definition",
    }


@pytest.fixture
async def mock_surrealdb():
    """Mock SurrealDB client for testing."""
    # TODO: Implement mock database client
    pass


@pytest.fixture
async def mock_llm_client():
    """Mock LLM client for testing."""
    # TODO: Implement mock LLM client
    pass
