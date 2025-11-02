# API Rate Limits & Best Practices

This document outlines the API rate limits for our data sources and how we respect them.

## StackExchange API (Cross Validated)

### Rate Limits

| Scenario | Limit | Recommendation |
|----------|-------|----------------|
| **Without API Key** | 300 requests/day | Start with `make collect-cv` (100 questions) |
| **With API Key** | 10,000 requests/day | Can use `make collect-cv-full` (1000+ questions) |

### Getting an API Key

1. Register at [StackApps](https://stackapps.com/)
2. Create an application
3. Get your API key
4. Add to `.env`:
   ```bash
   STACKEXCHANGE_API_KEY=your_key_here
   ```

### Our Implementation

- **Default rate**: 10 requests/minute (safe for no API key)
- **With API key**: Can increase to 30 requests/minute in config
- **Automatic warnings**: Script warns if collection may hit limits
- **Conservative defaults**: 500 questions max (overridable via CLI)

### Safe Collection Examples

```bash
# Safe without API key (uses ~100-200 requests)
make collect-cv

# Requires API key (uses ~1000-2000 requests)
make collect-cv-full

# Custom amount
python scripts/collect_data.py --source cross-validated --max-questions 50
```

## ArXiv API

### Rate Limits

| Guideline | Limit | Our Implementation |
|-----------|-------|-------------------|
| **Delay between requests** | 3 seconds minimum | 3.0 seconds (configurable) |
| **Error handling** | Exponential backoff | Implemented |
| **Bulk requests** | Avoid | Limited to 50 results/query |

### Our Implementation

- **Delay**: 3 seconds between all requests
- **Default**: 50 papers per query (7 queries = 350 papers)
- **Estimated time**: ~20 minutes for default collection
- **PDF downloads**: 3 seconds between downloads (even slower)

### Time Estimates

```bash
# Metadata only (~350 papers)
make collect-arxiv          # ~20 minutes

# With PDF downloads (~350 papers)
make collect-arxiv-pdfs     # ~20 minutes + download time (~1-2 hours)
```

## Anthropic API (Claude)

### Rate Limits

Varies by tier. See [Anthropic docs](https://docs.anthropic.com/en/api/rate-limits).

| Tier | Requests/min | Tokens/min |
|------|--------------|------------|
| **Free** | 5 | 40,000 |
| **Build Tier 1** | 50 | 40,000 |
| **Build Tier 2** | 50 | 80,000 |

### Our Implementation (Coming Soon)

- Synthetic Q&A generation respects tier limits
- Exponential backoff on rate limit errors
- Configurable batch sizes

## Configuration

All rate limiting is configurable in `.env`:

```bash
# StackExchange
DATA_COLLECTION_STACKEXCHANGE_MAX_QUESTIONS=500
DATA_COLLECTION_REQUESTS_PER_MINUTE=10
DATA_COLLECTION_CONCURRENT_REQUESTS=2

# ArXiv
DATA_COLLECTION_ARXIV_MAX_RESULTS_PER_QUERY=50
ARXIV_DELAY_SECONDS=3.0

# Anthropic (for synthetic generation)
ANTHROPIC_API_KEY=your_key_here
```

## Monitoring

All collectors log:
- API warnings when limits may be reached
- Estimated time for long operations
- Rate limit errors (if encountered)

Check logs:
```bash
tail -f logs/data_collection.log
```

## Best Practices

### For Development/Testing

1. **Start small**: Use `make collect-cv` (100 questions)
2. **Test incrementally**: Verify quality before collecting more
3. **Use API keys**: Get free keys for higher limits

### For Production Collection

1. **Get API keys**: Both StackExchange and Anthropic
2. **Run during off-hours**: Large collections take time
3. **Monitor logs**: Watch for rate limit warnings
4. **Split collections**: Collect in batches over multiple days if needed

### Respectful Usage

✅ **Do:**
- Use the provided delay configurations
- Add API keys for higher limits
- Monitor and respect warnings
- Report issues if you hit limits

❌ **Don't:**
- Remove or reduce delay times
- Run multiple collectors in parallel
- Ignore rate limit warnings
- Use aggressive retry logic

## Troubleshooting

### "Rate limit exceeded" from StackExchange

**Without API key:**
- You've used 300 requests today
- Wait 24 hours or add API key

**With API key:**
- Rare, but you've used 10,000 requests today
- Wait 24 hours

### ArXiv connection errors

- Normal occasional timeouts
- Script retries automatically
- If persistent, increase `ARXIV_DELAY_SECONDS` to 5.0

### Slow collection times

This is expected! Respecting API limits means:
- **Cross Validated**: ~5-10 minutes for 100 questions
- **ArXiv**: ~20 minutes for 350 papers
- **PDF downloads**: 1-2 hours for 350 papers

## Future Optimizations

While maintaining API respect, we can:
- Implement smarter caching (avoid re-fetching)
- Better deduplication (fewer wasted requests)
- Resume capability (restart after interruption)
- Progress checkpointing (save partial results)

**But we will not:**
- Reduce delays below API recommendations
- Implement aggressive parallel requests
- Circumvent rate limits
