## Data Collection Infrastructure

Collect and prepare training data for the experimentation SLM from multiple sources.

## Overview

This module handles collecting Q&A pairs and source documents from:
- **Cross Validated** (StackExchange for Statistics) - Direct Q&A pairs
- **ArXiv** - Academic papers for synthetic generation
- Future: Textbooks, industry blogs, Wikipedia

## Structure

```
data_collection/
├── models.py              # Data models (QAPair, topics, etc.)
├── config.py              # Configuration settings
├── collectors/            # Data source collectors
│   ├── stackexchange.py   # Cross Validated collector
│   └── arxiv.py           # ArXiv papers collector
├── validators/            # Data quality validation
│   └── quality.py         # Quality filters and validation
└── formatters/            # Output formatters
    └── chatml.py          # ChatML format for Llama 3.2
```

## Quick Start

### 1. Configure API Keys

Add to your `.env` file:

```bash
# Optional: StackExchange API key (increases rate limits)
STACKEXCHANGE_API_KEY=your_key_here

# Required for synthetic generation (coming soon)
ANTHROPIC_API_KEY=your_claude_api_key
```

### 2. Collect Data

**Collect from Cross Validated:**
```bash
python scripts/collect_data.py --source cross-validated --max-questions 1000
```

**Collect from ArXiv:**
```bash
python scripts/collect_data.py --source arxiv --download-pdfs
```

**Collect from all sources:**
```bash
python scripts/collect_data.py --source all
```

### 3. Output

Data is saved to `data/raw/`:
- `cross_validated_raw.json` - Raw Q&A pairs
- `cross_validated_chatml.jsonl` - ChatML formatted for training
- `arxiv_metadata.json` - ArXiv paper metadata
- `arxiv/pdfs/` - Downloaded PDFs (if requested)

## Configuration

Edit `src/data_collection/config.py` or set environment variables:

### Cross Validated Settings

```python
# Tags to search for
stackexchange_tags = [
    "experimental-design",
    "hypothesis-testing",
    "p-value",
    "confidence-interval",
    "sample-size",
    "ab-test",
    ...
]

# Quality filters
stackexchange_min_score = 5        # Minimum question score
stackexchange_min_answers = 1      # Minimum number of answers
stackexchange_max_questions = 5000 # Maximum questions to collect
```

### ArXiv Settings

```python
# Search queries
arxiv_search_queries = [
    "A/B testing",
    "online controlled experiments",
    "randomized controlled trial",
    ...
]

# Categories to filter
arxiv_categories = ["stat.ME", "stat.AP", "cs.LG"]
arxiv_max_results_per_query = 200
```

### Quality Filters

```python
min_question_length = 20    # characters
min_answer_length = 50      # characters
max_answer_length = 5000    # characters
```

## Data Models

### QAPair

Core data structure for training examples:

```python
QAPair(
    id="cv_12345_67890",
    question="What is a p-value?",
    answer="A p-value is...",
    source=DataSource.CROSS_VALIDATED,
    source_url="https://stats.stackexchange.com/q/12345",
    topics=[ExperimentationTopic.P_VALUES],
    score=25.0,
    upvotes=30,
    reasoning="Step-by-step explanation...",  # Optional
    citations=["Source 1", "Source 2"],        # Optional
)
```

### Topics Taxonomy

Training data is organized by topic:

**Core Statistics:**
- `HYPOTHESIS_TESTING`
- `P_VALUES`
- `CONFIDENCE_INTERVALS`
- `STATISTICAL_POWER`
- `EFFECT_SIZE`

**Experiment Design:**
- `SAMPLE_SIZE`
- `RANDOMIZATION`
- `TEST_DURATION`
- `METRIC_SELECTION`

**Analysis Methods:**
- `VARIANCE_REDUCTION`
- `MULTIPLE_TESTING`
- `SEQUENTIAL_TESTING`
- `BAYESIAN_METHODS`

**Advanced:**
- `CAUSAL_INFERENCE`
- `HETEROGENEOUS_TREATMENT_EFFECTS`
- `NETWORK_EFFECTS`

## Data Quality

### Validation Pipeline

All collected data passes through quality checks:

1. **Length filters** - Min/max for questions and answers
2. **Content quality** - Meaningful text, reasonable word diversity
3. **Code filtering** - Limited code blocks (we want conceptual answers)
4. **URL filtering** - Excessive URLs indicate spam
5. **Repetition detection** - Catch low-quality repetitive text

### ChatML Formatting

Data is formatted for Llama 3.2 training:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in experimentation, statistics, and A/B testing...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is a p-value?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
A p-value is the probability of obtaining test results...<|eot_id|>
```

## Usage Examples

### Programmatic Collection

```python
from src.data_collection.collectors.stackexchange import CrossValidatedCollector
from src.data_collection.formatters.chatml import ChatMLFormatter
from src.data_collection.validators.quality import filter_qa_pairs

# Initialize collector
collector = CrossValidatedCollector()

# Collect data
qa_pairs = collector.collect_by_tags(max_questions=1000)

# Filter for quality
valid_pairs, invalid_pairs = filter_qa_pairs(qa_pairs)

# Format to ChatML
formatter = ChatMLFormatter()
formatter.to_jsonl(valid_pairs, "output.jsonl")
```

### Custom Quality Filters

```python
from src.data_collection.validators.quality import QualityValidator

validator = QualityValidator(
    min_question_length=50,
    min_answer_length=100,
    max_answer_length=3000,
)

valid_pairs, invalid_pairs = filter_qa_pairs(qa_pairs, validator)
```

## Next Steps

1. **Synthetic Generation** - Generate Q&A from textbooks and papers using Claude
2. **Topic Classification** - Auto-classify Q&A pairs by topic
3. **Multi-turn Conversations** - Extract and format multi-turn dialogues
4. **Industry Blogs** - Scrape experimentation blogs from Netflix, Booking.com, etc.

## Troubleshooting

**Rate limiting errors:**
- Add `STACKEXCHANGE_API_KEY` to `.env`
- Reduce `requests_per_minute` in config

**Low quality data:**
- Increase `stackexchange_min_score`
- Adjust quality validator thresholds

**ArXiv timeouts:**
- Increase `arxiv_delay_seconds`
- Reduce `arxiv_max_results_per_query`
