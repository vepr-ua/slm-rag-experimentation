#!/usr/bin/env python
"""
Test Apple Silicon training setup.

This script verifies that all dependencies are installed and the trainer
can initialize on Apple Silicon with MPS backend.
"""

import torch
from pathlib import Path
from rich.console import Console

console = Console()

def test_pytorch():
    """Test PyTorch and MPS availability."""
    console.print("\n[bold cyan]Testing PyTorch Setup[/bold cyan]\n")

    console.print(f"PyTorch version: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    console.print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        console.print("[green]✅ Apple Silicon MPS backend detected![/green]")
        return True
    else:
        console.print("[red]❌ MPS not available. Are you on Apple Silicon?[/red]")
        return False

def test_dependencies():
    """Test that all required packages are installed."""
    console.print("\n[bold cyan]Testing Dependencies[/bold cyan]\n")

    packages = [
        ("transformers", "Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("trl", "TRL (SFT Trainer)"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
    ]

    all_installed = True
    for package, name in packages:
        try:
            __import__(package)
            console.print(f"[green]✅ {name}[/green]")
        except ImportError:
            console.print(f"[red]❌ {name} not installed[/red]")
            all_installed = False

    return all_installed

def test_config():
    """Test Apple Silicon config exists."""
    console.print("\n[bold cyan]Testing Configuration[/bold cyan]\n")

    config_path = Path("configs/apple_silicon_config.json")
    if config_path.exists():
        console.print(f"[green]✅ Apple Silicon config found: {config_path}[/green]")
        return True
    else:
        console.print(f"[red]❌ Config not found: {config_path}[/red]")
        return False

def test_trainer_init():
    """Test that trainer can initialize."""
    console.print("\n[bold cyan]Testing Trainer Initialization[/bold cyan]\n")

    try:
        from src.training.config import load_config
        from src.training.trainer import QLoRATrainer

        # Load Apple Silicon config
        config = load_config(Path("configs/apple_silicon_config.json"))
        console.print("[green]✅ Config loaded[/green]")

        # Create trainer
        trainer = QLoRATrainer(config)
        console.print("[green]✅ Trainer created[/green]")

        # Detect device
        device = trainer._detect_device()
        console.print(f"[green]✅ Device detected: {device}[/green]")

        if device != "mps":
            console.print(f"[yellow]⚠️  Expected MPS but got {device}[/yellow]")
            return False

        return True

    except Exception as e:
        console.print(f"[red]❌ Trainer initialization failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test that training dataset exists."""
    console.print("\n[bold cyan]Testing Dataset[/bold cyan]\n")

    dataset_path = Path("data/processed/combined_train.jsonl")
    if dataset_path.exists():
        # Count lines
        with open(dataset_path) as f:
            num_examples = sum(1 for _ in f)
        console.print(f"[green]✅ Training dataset found: {num_examples} examples[/green]")
        return True
    else:
        console.print(f"[red]❌ Dataset not found: {dataset_path}[/red]")
        console.print("Run: make combine-datasets")
        return False

def main():
    """Run all tests."""
    console.print("\n" + "="*60)
    console.print("[bold]Apple Silicon Training Setup Test[/bold]")
    console.print("="*60)

    results = {
        "PyTorch & MPS": test_pytorch(),
        "Dependencies": test_dependencies(),
        "Configuration": test_config(),
        "Dataset": test_dataset(),
        "Trainer Init": test_trainer_init(),
    }

    console.print("\n" + "="*60)
    console.print("[bold]Test Results[/bold]")
    console.print("="*60 + "\n")

    for test_name, passed in results.items():
        status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
        console.print(f"{test_name:20} {status}")

    all_passed = all(results.values())

    if all_passed:
        console.print("\n[bold green]🎉 All tests passed! Ready to train on Apple Silicon.[/bold green]")
        console.print("\nRun: [bold]make train-apple[/bold]\n")
    else:
        console.print("\n[bold red]❌ Some tests failed. Fix issues before training.[/bold red]\n")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
