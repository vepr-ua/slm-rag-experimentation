# Synthetic Q&A Generators

Generate synthetic training data using Claude API from various source materials.

## Overview

This module creates high-quality Q&A pairs from source materials like ArXiv papers, textbooks, and blog posts. It's the **force multiplier** for our training data:

- 1 ArXiv paper → 5-10 Q&A pairs
- 172 papers → ~860-1,720 synthetic examples
- Combined with real data → 2,000-3,500 total training examples

## Components

### `synthetic_qa.py`
Main synthetic Q&A generator that:
- Processes ArXiv papers (abstracts and conclusions)
- Generates diverse Q&A pairs using Claude
- Parses and structures responses
- Saves in QAPair format

### Claude Client (in `src/llm/`)
- `claude_client.py` - API wrapper with rate limiting
- `prompts.py` - Generation prompts and templates

## Usage

### Quick Test (10 papers, ~$0.10)

```bash
make generate-qa-test
```

### Full Generation (all papers, ~$1-2)

```bash
make generate-qa
```

### Programmatic Usage

```python
from src.data_collection.generators.synthetic_qa import SyntheticQAGenerator
from src.llm.claude_client import ClaudeClient

# Initialize
claude = ClaudeClient()
generator = SyntheticQAGenerator(claude_client=claude)

# Generate from a single paper
qa_pairs = generator.generate_from_arxiv_paper(
    arxiv_id="2402.03231v1",
    title="Improved prediction of future user activity...",
    abstract="In this paper we...",
    num_questions=5
)

# Generate from all ArXiv papers
all_qa = generator.generate_from_arxiv_metadata(
    metadata_file="data/raw/arxiv_metadata.json",
    max_papers=10  # Or None for all
)

# Save results
generator.save_generated_qa(all_qa, "data/processed/synthetic.json")
```

## Generation Strategy

### From ArXiv Papers

For each paper, we extract:
1. **Abstract** → 5 Q&A pairs
   - What problem does it address?
   - What methodology is proposed?
   - What are the key findings?
   - When would practitioners use this?
   - How does it relate to A/B testing?

2. **Conclusions** (future) → 3 Q&A pairs
   - Practical implications
   - When to apply findings
   - Limitations and considerations

### Question Types Generated

- **Definition**: "What is X?"
- **Methodology**: "How do I X?"
- **Comparison**: "What's the difference between X and Y?"
- **Troubleshooting**: "Why is X happening?"
- **Best Practice**: "What's the best way to do X?"

## Quality Control

Generated Q&A pairs are:
1. Parsed from structured format
2. Validated for length (50-5000 chars)
3. Checked for quality (meaningful content, no excessive code)
4. Formatted to ChatML for training

## Output

Generated data is saved in two formats:

**1. Raw JSON** (`data/processed/synthetic_arxiv_raw.json`)
```json
[
  {
    "id": "synthetic_arxiv_2402.03231v1_0",
    "question": "What is the main challenge...",
    "answer": "The main challenge is...",
    "source": "synthetic",
    "citations": ["ArXiv: 2402.03231v1 - ..."],
    ...
  }
]
```

**2. ChatML** (`data/processed/synthetic_arxiv_chatml.jsonl`)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in experimentation...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the main challenge...<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The main challenge is...<|eot_id|>
```

## Cost Estimation

Based on Claude 3.5 Sonnet pricing:
- Input: $3 per million tokens
- Output: $15 per million tokens
- Average: ~$9 per million tokens (50/50 split)

**Estimated costs:**
- 10 papers: ~$0.10
- 50 papers: ~$0.50
- 172 papers: ~$1.50
- 500 papers: ~$4.00

## Prerequisites

1. **Claude API Key**
   ```bash
   # Add to .env
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

2. **ArXiv Metadata**
   ```bash
   # Collect papers first
   make collect-arxiv
   ```

## Tips

1. **Start small**: Test with 10 papers first
2. **Review output**: Check quality before scaling up
3. **Cost tracking**: Monitor your Claude API usage
4. **Rate limiting**: Built-in 1s delay between requests

## Future Enhancements

- [ ] Generate from PDF full text (not just abstracts)
- [ ] Generate from textbooks (OpenStax, etc.)
- [ ] Generate from blog posts
- [ ] Multi-turn conversation generation
- [ ] Topic-specific generation (focus on specific concepts)
- [ ] Quality scoring with LLM-as-judge
