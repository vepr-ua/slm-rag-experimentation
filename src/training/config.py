"""
Training configuration and hyperparameters.

Defines all settings for QLoRA fine-tuning of Llama 3.2 3B.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    r: int = Field(default=16, description="LoRA rank (higher = more parameters)")
    lora_alpha: int = Field(default=32, description="LoRA scaling factor")
    lora_dropout: float = Field(default=0.05, description="Dropout for LoRA layers")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Transformer modules to apply LoRA to",
    )
    bias: str = Field(default="none", description="Bias training: 'none', 'all', 'lora_only'")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for LoRA")


class QuantizationConfig(BaseModel):
    """4-bit quantization configuration for QLoRA."""

    load_in_4bit: bool = Field(default=True, description="Enable 4-bit quantization")
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16", description="Computation dtype: 'float16' or 'bfloat16'"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4", description="Quantization type: 'nf4' or 'fp4'"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True, description="Enable nested quantization for memory savings"
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters and settings."""

    # Model
    model_name: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description="HuggingFace model identifier",
    )
    max_seq_length: int = Field(
        default=2048, description="Maximum sequence length (context window)"
    )

    # Dataset
    train_data_path: str = Field(
        default="data/processed/combined_train.jsonl",
        description="Path to training data (ChatML format)",
    )
    eval_data_path: Optional[str] = Field(
        default="data/processed/combined_eval.jsonl",
        description="Path to evaluation data (optional)",
    )
    train_test_split: float = Field(
        default=0.1, description="Test split ratio if eval_data not provided"
    )

    # Training hyperparameters
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        default=2, description="Batch size per GPU/CPU for training"
    )
    per_device_eval_batch_size: int = Field(
        default=2, description="Batch size per GPU/CPU for evaluation"
    )
    gradient_accumulation_steps: int = Field(
        default=4, description="Gradient accumulation steps (effective batch = batch_size * steps)"
    )
    learning_rate: float = Field(default=2e-4, description="Peak learning rate")
    warmup_ratio: float = Field(
        default=0.03, description="Warmup ratio (3% of total steps)"
    )
    lr_scheduler_type: str = Field(
        default="cosine", description="Learning rate scheduler: 'linear', 'cosine', 'constant'"
    )

    # Optimization
    optim: str = Field(
        default="paged_adamw_8bit",
        description="Optimizer: 'paged_adamw_8bit', 'adamw_torch', etc.",
    )
    weight_decay: float = Field(default=0.01, description="Weight decay (L2 regularization)")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping threshold")

    # Generation (for evaluation)
    generation_max_length: int = Field(
        default=512, description="Max tokens to generate during evaluation"
    )
    generation_temperature: float = Field(
        default=0.7, description="Temperature for generation"
    )

    # Logging and checkpointing
    output_dir: str = Field(
        default="models/checkpoints", description="Output directory for checkpoints"
    )
    logging_steps: int = Field(default=10, description="Log every N steps")
    eval_steps: int = Field(default=100, description="Evaluate every N steps")
    save_steps: int = Field(default=500, description="Save checkpoint every N steps")
    save_total_limit: int = Field(
        default=3, description="Maximum number of checkpoints to keep"
    )

    # Performance
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=True, description="Use BF16 mixed precision (better for training)")
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing (saves memory)"
    )
    group_by_length: bool = Field(
        default=True, description="Group sequences by length (faster training)"
    )

    # Misc
    seed: int = Field(default=42, description="Random seed for reproducibility")
    report_to: str = Field(
        default="tensorboard", description="Logging backend: 'tensorboard', 'wandb', 'none'"
    )

    # LoRA and Quantization
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    def get_output_dir(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_dir)

    def get_train_data_path(self) -> Path:
        """Get training data path as Path object."""
        return Path(self.train_data_path)

    def get_eval_data_path(self) -> Optional[Path]:
        """Get evaluation data path as Path object."""
        return Path(self.eval_data_path) if self.eval_data_path else None


# Default configuration for Llama 3.2 3B fine-tuning
DEFAULT_CONFIG = TrainingConfig()


def load_config(config_path: Optional[Path] = None) -> TrainingConfig:
    """
    Load training configuration from JSON file.

    Args:
        config_path: Path to config JSON (uses defaults if None)

    Returns:
        TrainingConfig object
    """
    if config_path and config_path.exists():
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return TrainingConfig(**config_dict)
    return DEFAULT_CONFIG


def save_config(config: TrainingConfig, output_path: Path) -> None:
    """
    Save training configuration to JSON file.

    Args:
        config: TrainingConfig object
        output_path: Output file path
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)
