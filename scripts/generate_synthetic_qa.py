#!/usr/bin/env python
"""
Generate synthetic Q&A pairs from collected data sources.

Usage:
    python scripts/generate_synthetic_qa.py --source arxiv --max-papers 10
    python scripts/generate_synthetic_qa.py --source arxiv --max-papers 172  # All papers
"""

import argparse
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.data_collection.formatters.chatml import ChatMLFormatter
from src.data_collection.generators.synthetic_qa import SyntheticQAGenerator
from src.data_collection.validators.quality import filter_qa_pairs
from src.llm.claude_client import ClaudeClient

console = Console()


def generate_from_arxiv(max_papers: int = None, output_dir: Path = Path("data/processed")):
    """Generate Q&A pairs from ArXiv papers."""
    console.print("\n[bold cyan]Generating Synthetic Q&A from ArXiv Papers[/bold cyan]\n")

    # Check for ArXiv metadata
    metadata_file = Path("data/raw/arxiv_metadata.json")
    if not metadata_file.exists():
        console.print(
            "[red]❌ ArXiv metadata not found![/red]\n"
            "Run: make collect-arxiv first\n"
        )
        return

    # Initialize components
    console.print("[yellow]Initializing Claude client...[/yellow]")
    try:
        claude = ClaudeClient()
        if not claude.test_connection():
            console.print("[red]❌ Claude API connection failed![/red]")
            console.print("Check your ANTHROPIC_API_KEY in .env")
            return
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize Claude client: {e}[/red]")
        console.print("Make sure ANTHROPIC_API_KEY is set in .env")
        return

    generator = SyntheticQAGenerator(claude_client=claude)

    # Generate Q&A pairs
    console.print(f"\n[yellow]Generating Q&A from ArXiv papers...[/yellow]")
    if max_papers:
        console.print(f"Processing up to {max_papers} papers")

    qa_pairs = generator.generate_from_arxiv_metadata(
        metadata_file=metadata_file,
        max_papers=max_papers,
    )

    if not qa_pairs:
        console.print("[red]❌ No Q&A pairs generated![/red]")
        return

    console.print(f"\n[green]✅ Generated {len(qa_pairs)} Q&A pairs[/green]")

    # Quality filtering
    console.print("\n[yellow]Applying quality filters...[/yellow]")
    valid_pairs, invalid_pairs = filter_qa_pairs(qa_pairs)

    console.print(f"[green]✅ {len(valid_pairs)} valid pairs[/green]")
    console.print(f"[yellow]⚠️  {len(invalid_pairs)} filtered out[/yellow]")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    raw_output = output_dir / "synthetic_arxiv_raw.json"
    generator.save_generated_qa(valid_pairs, raw_output)
    console.print(f"[green]✅ Saved raw data to {raw_output}[/green]")

    # Format to ChatML
    console.print("\n[yellow]Formatting to ChatML for training...[/yellow]")
    formatter = ChatMLFormatter()
    chatml_output = output_dir / "synthetic_arxiv_chatml.jsonl"
    formatter.to_jsonl(valid_pairs, str(chatml_output))
    console.print(f"[green]✅ Saved ChatML data to {chatml_output}[/green]")

    # Summary
    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            f"[bold green]Synthetic Generation Complete![/bold green]\n\n"
            f"Papers processed: {max_papers or 'all'}\n"
            f"Q&A pairs generated: {len(qa_pairs)}\n"
            f"Valid pairs: {len(valid_pairs)}\n"
            f"Filtered out: {len(invalid_pairs)}\n\n"
            f"Output:\n"
            f"  - Raw: {raw_output}\n"
            f"  - Training: {chatml_output}",
            title="Summary",
            border_style="green",
        )
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A pairs from collected data"
    )
    parser.add_argument(
        "--source",
        choices=["arxiv"],
        default="arxiv",
        help="Data source to generate from",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for generated data",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/synthetic_generation.log", rotation="10 MB")

    # Generate based on source
    if args.source == "arxiv":
        generate_from_arxiv(
            max_papers=args.max_papers,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
