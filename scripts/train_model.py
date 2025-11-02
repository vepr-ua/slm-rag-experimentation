#!/usr/bin/env python
"""
Train Llama 3.2 3B with QLoRA on experimentation Q&A data.

Usage:
    # Train with default config
    python scripts/train_model.py

    # Train with custom config
    python scripts/train_model.py --config configs/training_config.json

    # Combine datasets first, then train
    python scripts/train_model.py --combine-datasets

    # Just combine datasets (no training)
    python scripts/train_model.py --combine-only
"""

import argparse
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.training.config import TrainingConfig, load_config, save_config
from src.training.dataset import combine_datasets, get_dataset_stats
from src.training.trainer import QLoRATrainer

console = Console()


def combine_training_data(
    cv_path: Path = Path("data/processed/cross_validated_chatml.jsonl"),
    arxiv_path: Path = Path("data/processed/synthetic_arxiv_chatml.jsonl"),
    output_path: Path = Path("data/processed/combined_train.jsonl"),
    cv_weight: float = 1.0,
    arxiv_weight: float = 1.0,
):
    """Combine Cross Validated and ArXiv datasets."""
    console.print("\n[bold cyan]Combining Training Datasets[/bold cyan]\n")

    # Check if files exist
    if not cv_path.exists():
        console.print(f"[yellow]⚠️  Cross Validated data not found at {cv_path}[/yellow]")
        console.print("Run: make collect-cv first\n")
    if not arxiv_path.exists():
        console.print(f"[yellow]⚠️  ArXiv synthetic data not found at {arxiv_path}[/yellow]")
        console.print("Run: make generate-qa first\n")

    if not cv_path.exists() and not arxiv_path.exists():
        console.print("[red]❌ No training data found![/red]")
        return False

    # Combine datasets
    combine_datasets(
        cv_path=cv_path,
        arxiv_path=arxiv_path,
        output_path=output_path,
        cv_weight=cv_weight,
        arxiv_weight=arxiv_weight,
    )

    # Show statistics
    stats = get_dataset_stats(output_path)

    console.print(
        Panel.fit(
            f"[bold green]Dataset Combined![/bold green]\n\n"
            f"Total examples: {stats['num_examples']}\n"
            f"Average length: {stats['avg_length']:.0f} chars\n"
            f"Max length: {stats['max_length']} chars\n"
            f"Min length: {stats['min_length']} chars\n\n"
            f"Output: {output_path}",
            title="Combined Dataset",
            border_style="green",
        )
    )

    return True


def train(config_path: Path = None):
    """Run training with specified config."""
    console.print("\n[bold cyan]Starting QLoRA Fine-Tuning[/bold cyan]\n")

    # Load config
    if config_path:
        console.print(f"[yellow]Loading config from {config_path}...[/yellow]")
        config = load_config(config_path)
    else:
        console.print("[yellow]Using default configuration...[/yellow]")
        config = TrainingConfig()

    # Check if training data exists
    train_path = config.get_train_data_path()
    if not train_path.exists():
        console.print(
            f"[red]❌ Training data not found at {train_path}![/red]\n"
            "Run with --combine-datasets to combine your datasets first.\n"
        )
        return

    # Display training configuration
    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  Model: {config.model_name}")
    console.print(f"  Training data: {config.train_data_path}")
    console.print(f"  Epochs: {config.num_train_epochs}")
    console.print(f"  Batch size: {config.per_device_train_batch_size}")
    console.print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    console.print(
        f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )
    console.print(f"  Learning rate: {config.learning_rate}")
    console.print(f"  LoRA rank: {config.lora.r}")
    console.print(f"  Output: {config.output_dir}\n")

    # Save config to output dir
    config_output = config.get_output_dir() / "training_config.json"
    save_config(config, config_output)
    console.print(f"[green]✅ Config saved to {config_output}[/green]\n")

    # Create trainer and train
    try:
        trainer = QLoRATrainer(config)
        trainer.train()

        # Save final model
        final_output = config.get_output_dir() / "final"
        trainer.save_model(str(final_output))

        # Evaluate if we have test data
        if config.get_eval_data_path():
            metrics = trainer.evaluate()

        console.print(
            Panel.fit(
                f"[bold green]Training Complete![/bold green]\n\n"
                f"Model saved to: {final_output}\n\n"
                f"Next steps:\n"
                f"  1. Test the model: python scripts/test_model.py\n"
                f"  2. Run evaluation: make evaluate\n"
                f"  3. Deploy: python -m src.api.main",
                title="Success",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        logger.exception("Training error")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Llama 3.2 3B with QLoRA on experimentation Q&A"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config JSON (uses defaults if not provided)",
    )
    parser.add_argument(
        "--combine-datasets",
        action="store_true",
        help="Combine Cross Validated and ArXiv datasets before training",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine datasets (don't train)",
    )
    parser.add_argument(
        "--cv-weight",
        type=float,
        default=1.0,
        help="Weight for Cross Validated examples (default: 1.0)",
    )
    parser.add_argument(
        "--arxiv-weight",
        type=float,
        default=1.0,
        help="Weight for ArXiv synthetic examples (default: 1.0)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/training.log", rotation="100 MB")

    # Combine datasets if requested
    if args.combine_datasets or args.combine_only:
        success = combine_training_data(
            cv_weight=args.cv_weight,
            arxiv_weight=args.arxiv_weight,
        )
        if not success:
            return

    # Train (unless combine-only)
    if not args.combine_only:
        train(config_path=args.config)


if __name__ == "__main__":
    main()
