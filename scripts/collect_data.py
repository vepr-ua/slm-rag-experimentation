#!/usr/bin/env python
"""
Main script for collecting training data from various sources.

Usage:
    python scripts/collect_data.py --source cross-validated --output data/raw/cv_data.jsonl
    python scripts/collect_data.py --source arxiv --output data/raw/arxiv_papers.json
    python scripts/collect_data.py --source all
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from src.data_collection.collectors.arxiv import ArXivCollector
from src.data_collection.collectors.stackexchange import CrossValidatedCollector
from src.data_collection.config import CollectorConfig, DataCollectionConfig
from src.data_collection.formatters.chatml import ChatMLFormatter
from src.data_collection.validators.quality import filter_qa_pairs


def collect_cross_validated(output_dir: Path, max_questions: int = None):
    """Collect Q&A pairs from Cross Validated."""
    logger.info("Starting Cross Validated collection...")

    config = DataCollectionConfig()
    collector_config = CollectorConfig()

    collector = CrossValidatedCollector(config, collector_config)

    # Collect Q&A pairs
    qa_pairs = collector.collect_by_tags(max_questions=max_questions)

    # Filter for quality
    valid_pairs, invalid_pairs = filter_qa_pairs(qa_pairs)

    logger.info(
        f"Collected {len(valid_pairs)} valid Q&A pairs "
        f"({len(invalid_pairs)} filtered out)"
    )

    # Save raw data
    raw_output = output_dir / "cross_validated_raw.json"
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_output, "w") as f:
        json.dump(
            [qa.model_dump() for qa in valid_pairs],
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Saved raw data to {raw_output}")

    # Format to ChatML and save
    formatter = ChatMLFormatter()
    chatml_output = output_dir / "cross_validated_chatml.jsonl"
    formatter.to_jsonl(valid_pairs, str(chatml_output))

    logger.info(f"Saved ChatML formatted data to {chatml_output}")

    return valid_pairs


def collect_arxiv(output_dir: Path, download_pdfs: bool = False):
    """Collect papers from ArXiv."""
    logger.info("Starting ArXiv collection...")

    config = DataCollectionConfig()
    collector_config = CollectorConfig()

    collector = ArXivCollector(config, collector_config)

    # Collect papers
    papers = collector.collect_papers()

    # Save metadata
    metadata_file = collector.save_metadata(papers, output_dir / "arxiv_metadata.json")

    logger.info(f"Collected metadata for {len(papers)} papers")

    # Optionally download PDFs
    if download_pdfs:
        logger.info("Downloading PDFs...")
        downloaded = collector.download_pdfs(papers, output_dir / "arxiv" / "pdfs")
        logger.info(f"Downloaded {len(downloaded)} PDFs")

    return papers


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect training data")
    parser.add_argument(
        "--source",
        choices=["cross-validated", "arxiv", "all"],
        default="all",
        help="Data source to collect from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions to collect (Cross Validated only)",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs for ArXiv papers",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/data_collection.log", rotation="10 MB")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data based on source
    if args.source in ["cross-validated", "all"]:
        collect_cross_validated(args.output_dir, args.max_questions)

    if args.source in ["arxiv", "all"]:
        collect_arxiv(args.output_dir, args.download_pdfs)

    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
