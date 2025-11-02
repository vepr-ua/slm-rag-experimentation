"""
Example test file to verify testing setup.

This demonstrates the testing structure and can be used as a template.
"""


def test_example():
    """Basic test to verify pytest is working."""
    assert 1 + 1 == 2


def test_with_fixture(sample_concept):
    """Test using a fixture from conftest.py."""
    assert sample_concept["name"] == "p-value"
    assert "definition" in sample_concept
    assert len(sample_concept["related_concepts"]) > 0


def test_string_operations():
    """Example test for string operations."""
    text = "GraphRAG"
    assert text.lower() == "graphrag"
    assert len(text) == 8


# Async test example
# Uncomment when you have async code to test
# async def test_async_example():
#     """Example async test."""
#     import asyncio
#     await asyncio.sleep(0.001)
#     assert True
