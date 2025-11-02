"""
Dataset loaders for training.

Handles loading and preprocessing ChatML formatted data for fine-tuning.
"""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def load_chatml_dataset(
    train_path: Path,
    eval_path: Optional[Path] = None,
    test_split: float = 0.1,
) -> DatasetDict:
    """
    Load ChatML formatted JSONL data for training.

    Args:
        train_path: Path to training JSONL file
        eval_path: Optional path to evaluation JSONL file
        test_split: Train/test split ratio if eval_path not provided

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    logger.info(f"Loading training data from {train_path}")

    # Load training data
    dataset = load_dataset("json", data_files=str(train_path), split="train")

    logger.info(f"Loaded {len(dataset)} examples from training data")

    # Load or create evaluation split
    if eval_path and eval_path.exists():
        logger.info(f"Loading evaluation data from {eval_path}")
        eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")
        logger.info(f"Loaded {len(eval_dataset)} examples from evaluation data")

        dataset_dict = DatasetDict({"train": dataset, "test": eval_dataset})
    else:
        logger.info(f"Creating {test_split:.0%} test split from training data")
        dataset_dict = dataset.train_test_split(test_size=test_split, seed=42)

    logger.info(
        f"Dataset splits: train={len(dataset_dict['train'])}, test={len(dataset_dict['test'])}"
    )

    return dataset_dict


def preprocess_chatml(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> dict:
    """
    Preprocess ChatML formatted examples for training.

    The input examples should have a 'text' field with ChatML formatted conversations.

    Args:
        examples: Batch of examples with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenized batch with input_ids, attention_mask, and labels
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad dynamically during training
    )

    # For causal LM, labels are the same as input_ids
    # The model will shift them internally for next-token prediction
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def combine_datasets(
    cv_path: Path,
    arxiv_path: Path,
    output_path: Path,
    cv_weight: float = 1.0,
    arxiv_weight: float = 1.0,
) -> None:
    """
    Combine Cross Validated and ArXiv synthetic datasets.

    This creates a balanced training set by optionally oversampling/undersampling.

    Args:
        cv_path: Path to Cross Validated ChatML data
        arxiv_path: Path to ArXiv synthetic ChatML data
        output_path: Output path for combined dataset
        cv_weight: Weight for CV examples (1.0 = no change, 2.0 = double, 0.5 = half)
        arxiv_weight: Weight for ArXiv examples
    """
    logger.info("Combining datasets...")

    # Load datasets
    cv_data = []
    arxiv_data = []

    if cv_path.exists():
        with open(cv_path, "r") as f:
            for line in f:
                cv_data.append(json.loads(line))
        logger.info(f"Loaded {len(cv_data)} Cross Validated examples")
    else:
        logger.warning(f"Cross Validated data not found at {cv_path}")

    if arxiv_path.exists():
        with open(arxiv_path, "r") as f:
            for line in f:
                arxiv_data.append(json.loads(line))
        logger.info(f"Loaded {len(arxiv_data)} ArXiv synthetic examples")
    else:
        logger.warning(f"ArXiv data not found at {arxiv_path}")

    # Apply weights (simple repetition strategy)
    weighted_cv = cv_data * int(cv_weight)
    weighted_arxiv = arxiv_data * int(arxiv_weight)

    # Combine and shuffle
    combined = weighted_cv + weighted_arxiv

    import random

    random.seed(42)
    random.shuffle(combined)

    # Save combined dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    logger.info(
        f"Combined dataset saved to {output_path} with {len(combined)} total examples"
    )
    logger.info(
        f"  - Cross Validated: {len(weighted_cv)} examples (weight={cv_weight})"
    )
    logger.info(f"  - ArXiv Synthetic: {len(weighted_arxiv)} examples (weight={arxiv_weight})")


def get_dataset_stats(dataset_path: Path) -> dict:
    """
    Get statistics about a dataset.

    Args:
        dataset_path: Path to JSONL dataset

    Returns:
        Dictionary with dataset statistics
    """
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    num_examples = len(examples)

    # Calculate length statistics
    lengths = [len(ex.get("text", "")) for ex in examples]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0

    return {
        "num_examples": num_examples,
        "avg_length": avg_length,
        "max_length": max_length,
        "min_length": min_length,
    }
