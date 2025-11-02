#!/usr/bin/env python
"""
Debug script to test StackExchange API and see why we're not getting results.
"""

from stackapi import StackAPI
from rich.console import Console
from rich.table import Table

console = Console()

def test_stackexchange():
    """Test StackExchange API with different queries."""

    console.print("\n[bold cyan]Testing StackExchange API[/bold cyan]\n")

    # Initialize client
    client = StackAPI("stats")  # Cross Validated
    client.max_pages = 1
    client.page_size = 10

    # Test 1: Try with all tags (current approach)
    console.print("[yellow]Test 1: Query with ALL tags (AND logic)[/yellow]")
    all_tags = [
        "experimental-design",
        "hypothesis-testing",
        "p-value",
        "confidence-interval",
        "sample-size",
        "ab-test",
        "statistical-power",
        "multiple-comparisons",
        "statistical-significance",
        "effect-size",
        "randomization",
        "causal-inference",
    ]

    try:
        response = client.fetch(
            "questions",
            tagged=";".join(all_tags),
            sort="votes",
            order="desc",
        )
        console.print(f"  Results: {len(response.get('items', []))} questions\n")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]\n")

    # Test 2: Try with individual tags
    console.print("[yellow]Test 2: Query individual tags[/yellow]")
    test_tags = ["ab-test", "experimental-design", "hypothesis-testing"]

    table = Table(show_header=True)
    table.add_column("Tag", style="cyan")
    table.add_column("Results", style="green")
    table.add_column("Sample Question", style="white")

    for tag in test_tags:
        try:
            response = client.fetch(
                "questions",
                tagged=tag,
                sort="votes",
                order="desc",
            )
            items = response.get("items", [])
            count = len(items)
            sample = items[0]["title"] if items else "N/A"
            table.add_row(tag, str(count), sample[:60] + "..." if len(sample) > 60 else sample)
        except Exception as e:
            table.add_row(tag, f"Error: {e}", "")

    console.print(table)
    console.print()

    # Test 3: Try with OR logic (any tag)
    console.print("[yellow]Test 3: Check API quota/limits[/yellow]")
    try:
        response = client.fetch("questions", tagged="ab-test", sort="votes", order="desc")
        if "quota_remaining" in response:
            console.print(f"  Quota remaining: {response['quota_remaining']}")
        if "backoff" in response:
            console.print(f"  [red]Backoff required: {response['backoff']} seconds[/red]")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")

    console.print()

    # Test 4: Check with minimum score filter
    console.print("[yellow]Test 4: With minimum score filter (score >= 5)[/yellow]")
    try:
        response = client.fetch(
            "questions",
            tagged="ab-test",
            sort="votes",
            order="desc",
            min=5,
        )
        items = response.get("items", [])
        console.print(f"  Results: {len(items)} questions")
        if items:
            console.print(f"  Top question: {items[0]['title']}")
            console.print(f"  Score: {items[0]['score']}")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")

    console.print("\n[bold green]Diagnosis complete![/bold green]\n")


if __name__ == "__main__":
    test_stackexchange()
