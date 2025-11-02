# Configuration Guide

Complete guide to configuring the data collection system via environment variables.

## Environment Variable Mapping

### CollectorConfig (API Keys & Technical Settings)

Located in: `src/data_collection/config.py`

| Config Field | .env Variable | Default | Description |
|-------------|---------------|---------|-------------|
| `stackexchange_api_key` | `STACKEXCHANGE_API_KEY` | `""` | Optional API key for higher rate limits |
| `stackexchange_site` | `STACKEXCHANGE_SITE` | `"stats"` | StackExchange site (stats = Cross Validated) |
| `arxiv_delay_seconds` | `ARXIV_DELAY_SECONDS` | `3.0` | Delay between ArXiv requests (seconds) |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` | Required for synthetic Q&A generation |
| `anthropic_model` | `ANTHROPIC_MODEL` | `"claude-3-5-sonnet-20241022"` | Claude model to use |
| `anthropic_max_tokens` | `ANTHROPIC_MAX_TOKENS` | `4096` | Max tokens per Claude response |

### DataCollectionConfig (Collection Settings)

Located in: `src/data_collection/config.py`

| Config Field | .env Variable | Default | Description |
|-------------|---------------|---------|-------------|
| `raw_data_dir` | `RAW_DATA_DIR` | `"data/raw"` | Directory for raw collected data |
| `processed_data_dir` | `PROCESSED_DATA_DIR` | `"data/processed"` | Directory for processed data |
| `stackexchange_min_score` | `STACKEXCHANGE_MIN_SCORE` | `5` | Minimum question score to collect |
| `stackexchange_min_answers` | `STACKEXCHANGE_MIN_ANSWERS` | `1` | Minimum number of answers required |
| `stackexchange_max_questions` | `STACKEXCHANGE_MAX_QUESTIONS` | `500` | Max questions to collect (default) |
| `stackexchange_accepted_only` | `STACKEXCHANGE_ACCEPTED_ONLY` | `false` | Only collect accepted answers |
| `arxiv_max_results_per_query` | `ARXIV_MAX_RESULTS_PER_QUERY` | `50` | Max results per ArXiv search query |
| `min_question_length` | `MIN_QUESTION_LENGTH` | `20` | Minimum question length (chars) |
| `min_answer_length` | `MIN_ANSWER_LENGTH` | `50` | Minimum answer length (chars) |
| `max_answer_length` | `MAX_ANSWER_LENGTH` | `5000` | Maximum answer length (chars) |
| `requests_per_minute` | `REQUESTS_PER_MINUTE` | `10` | Rate limit for API requests |
| `concurrent_requests` | `CONCURRENT_REQUESTS` | `2` | Number of concurrent requests |

## Example .env Configuration

### Minimal (No API Keys)

```bash
# Works with defaults and public API access
# StackExchange: 300 requests/day
# ArXiv: Public access with 3s delays

ARXIV_DELAY_SECONDS=3.0
```

### Recommended (With API Keys)

```bash
# StackExchange API (10,000 requests/day)
STACKEXCHANGE_API_KEY=your_stackexchange_key_here
STACKEXCHANGE_SITE=stats

# ArXiv settings
ARXIV_DELAY_SECONDS=3.0
ARXIV_MAX_RESULTS_PER_QUERY=50

# Anthropic Claude API (for synthetic generation)
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MAX_TOKENS=4096

# Optional: Override collection limits
STACKEXCHANGE_MAX_QUESTIONS=1000
REQUESTS_PER_MINUTE=30
```

### Custom Quality Filters

```bash
# Stricter quality requirements
MIN_QUESTION_LENGTH=50
MIN_ANSWER_LENGTH=100
MAX_ANSWER_LENGTH=3000
STACKEXCHANGE_MIN_SCORE=10
STACKEXCHANGE_ACCEPTED_ONLY=true
```

## Getting API Keys

### StackExchange API Key

1. Visit [StackApps](https://stackapps.com/apps/oauth/register)
2. Register a new application
3. Copy your API key
4. Add to `.env`:
   ```bash
   STACKEXCHANGE_API_KEY=your_key_here
   ```

**Benefits:**
- Rate limit: 300 → 10,000 requests/day
- Faster data collection
- No daily quota concerns

### Anthropic Claude API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account (if needed)
3. Generate API key
4. Add to `.env`:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-your_key_here
   ```

**Benefits:**
- Required for synthetic Q&A generation
- Generate training data from textbooks and papers
- Higher quality, domain-specific examples

## Testing Configuration

Verify your configuration is loading correctly:

```bash
python scripts/test_config.py
```

This will display:
- All configuration values
- Which API keys are set
- Warnings for missing keys
- Validation errors (if any)

## Programmatic Access

```python
from src.data_collection.config import DataCollectionConfig, CollectorConfig

# Load from .env
data_config = DataCollectionConfig()
collector_config = CollectorConfig()

# Access settings
print(f"Max questions: {data_config.stackexchange_max_questions}")
print(f"Has API key: {bool(collector_config.stackexchange_api_key)}")

# Override programmatically
custom_config = DataCollectionConfig(
    stackexchange_max_questions=100,
    requests_per_minute=5
)
```

## Common Issues

### Issue: "Configuration not loading from .env"

**Solution:**
1. Ensure `.env` file exists in project root
2. Check variable names match exactly (case-sensitive)
3. No spaces around `=` in `.env` file
4. Run from project root directory

### Issue: "API key not recognized"

**Solution:**
1. Check for extra spaces in .env file
2. Restart application after changing .env
3. Verify key is valid (test with API provider)

### Issue: "Using defaults instead of .env values"

**Solution:**
1. Check that field names in config.py match env var names
2. Use uppercase for env vars in .env file
3. No `DATA_COLLECTION_` prefix needed (we removed it)

## Environment Variable Priority

Settings are loaded in this order (later overrides earlier):

1. **Default values** in config classes
2. **.env file** values
3. **System environment variables**
4. **Programmatic overrides** (when instantiating config)

Example:
```bash
# In .env
STACKEXCHANGE_MAX_QUESTIONS=500

# In shell
export STACKEXCHANGE_MAX_QUESTIONS=1000

# System env var (1000) overrides .env file (500)
```

## Best Practices

1. **Use .env for local development**
   - Don't commit `.env` to git
   - Use `.env.example` as template

2. **Use system env vars for production**
   - More secure
   - Better for containers/CI/CD
   - Override .env values

3. **Override programmatically for scripts**
   - Pass custom values for specific runs
   - Don't modify .env for one-off changes

4. **Validate before collection**
   - Run `python scripts/test_config.py`
   - Check API key warnings
   - Verify rate limits

## Troubleshooting

Enable debug logging to see config loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.data_collection.config import DataCollectionConfig
config = DataCollectionConfig()
```

This will show which env vars were loaded and from where.
