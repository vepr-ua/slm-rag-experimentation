#!/usr/bin/env python
"""
Test script to verify configuration loading from .env file.

Usage:
    python scripts/test_config.py
"""

from rich.console import Console
from rich.table import Table

from src.data_collection.config import CollectorConfig, DataCollectionConfig

console = Console()


def test_config():
    """Test configuration loading."""
    console.print("\n[bold cyan]Testing Configuration Loading[/bold cyan]\n")

    # Load configs
    data_config = DataCollectionConfig()
    collector_config = CollectorConfig()

    # Display DataCollectionConfig
    table1 = Table(title="DataCollectionConfig", show_header=True)
    table1.add_column("Setting", style="cyan")
    table1.add_column("Value", style="green")

    table1.add_row("Raw Data Dir", data_config.raw_data_dir)
    table1.add_row("Processed Data Dir", data_config.processed_data_dir)
    table1.add_row("StackExchange Max Questions", str(data_config.stackexchange_max_questions))
    table1.add_row("StackExchange Min Score", str(data_config.stackexchange_min_score))
    table1.add_row("ArXiv Max Results/Query", str(data_config.arxiv_max_results_per_query))
    table1.add_row("Requests Per Minute", str(data_config.requests_per_minute))
    table1.add_row("Concurrent Requests", str(data_config.concurrent_requests))

    console.print(table1)
    console.print()

    # Display CollectorConfig
    table2 = Table(title="CollectorConfig (API Keys)", show_header=True)
    table2.add_column("Setting", style="cyan")
    table2.add_column("Value", style="green")

    # Mask API keys
    se_key = collector_config.stackexchange_api_key
    se_key_display = (
        f"{se_key[:8]}...{se_key[-4:]}" if se_key and len(se_key) > 12 else "Not set"
    )

    anthropic_key = collector_config.anthropic_api_key
    anthropic_key_display = (
        f"{anthropic_key[:8]}...{anthropic_key[-4:]}"
        if anthropic_key and len(anthropic_key) > 12
        else "Not set"
    )

    table2.add_row("StackExchange API Key", se_key_display)
    table2.add_row("StackExchange Site", collector_config.stackexchange_site)
    table2.add_row("ArXiv Delay (seconds)", str(collector_config.arxiv_delay_seconds))
    table2.add_row("Anthropic API Key", anthropic_key_display)
    table2.add_row("Anthropic Model", collector_config.anthropic_model)
    table2.add_row("Anthropic Max Tokens", str(collector_config.anthropic_max_tokens))

    console.print(table2)
    console.print()

    # Warnings
    if not collector_config.stackexchange_api_key:
        console.print(
            "[yellow]⚠️  Warning: STACKEXCHANGE_API_KEY not set in .env[/yellow]"
        )
        console.print(
            "[yellow]   Rate limited to 300 requests/day. Get a key at:[/yellow]"
        )
        console.print("[yellow]   https://stackapps.com/apps/oauth/register[/yellow]\n")

    if not collector_config.anthropic_api_key:
        console.print("[yellow]⚠️  Warning: ANTHROPIC_API_KEY not set in .env[/yellow]")
        console.print(
            "[yellow]   Required for synthetic Q&A generation. Get a key at:[/yellow]"
        )
        console.print("[yellow]   https://console.anthropic.com/[/yellow]\n")

    # Success messages
    if collector_config.stackexchange_api_key:
        console.print("[green]✓ StackExchange API key configured[/green]")

    if collector_config.anthropic_api_key:
        console.print("[green]✓ Anthropic API key configured[/green]")

    console.print("\n[bold green]Configuration test complete![/bold green]\n")


if __name__ == "__main__":
    test_config()
