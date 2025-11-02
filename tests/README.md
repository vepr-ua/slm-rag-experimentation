# Tests

Unit and integration tests for the SLM-RAG experimentation project.

## Structure

The test directory mirrors the `src/` structure:

```
tests/
├── conftest.py         # Shared fixtures and pytest configuration
├── test_example.py     # Example test (can be deleted later)
├── api/                # API endpoint tests
├── graph/              # Graph database and traversal tests
├── rag/                # RAG pipeline tests
├── llm/                # LLM integration tests
├── knowledge/          # Knowledge base construction tests
└── utils/              # Utility function tests
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_example.py

# Run specific test
pytest tests/test_example.py::test_example

# Run tests matching a pattern
pytest -k "test_graph"

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s
```

## Writing Tests

### Basic Test

```python
def test_something():
    """Test description."""
    result = my_function()
    assert result == expected_value
```

### Async Test

```python
async def test_async_operation():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

### Using Fixtures

```python
def test_with_fixture(sample_concept):
    """Fixtures are defined in conftest.py."""
    assert sample_concept["name"] is not None
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("p-value", "statistical_test"),
    ("confidence_interval", "statistical_measure"),
])
def test_categorize(input, expected):
    """Test multiple cases."""
    assert categorize_concept(input) == expected
```

## Test Coverage

Coverage reports are generated in `htmlcov/` directory.

```bash
# Generate coverage report
make test-cov

# Open HTML report
open htmlcov/index.html
```

## Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** that describe what's being tested
3. **Use fixtures** for common setup
4. **Mock external dependencies** (database, LLM API, etc.)
5. **Test edge cases** not just happy paths
6. **Keep tests fast** - mock slow operations
7. **Independent tests** - no test should depend on another

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_concept` - Example concept data
- `sample_query` - Example user query
- `mock_surrealdb` - Mock database client
- `mock_llm_client` - Mock LLM client

Add new fixtures to `conftest.py` when they're needed across multiple test files.
