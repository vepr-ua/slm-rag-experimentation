"""
QLoRA fine-tuning trainer using HuggingFace TRL.

Implements efficient fine-tuning of Llama 3.2 3B with:
- 4-bit quantization (QLoRA)
- LoRA adapters
- Supervised Fine-Tuning (SFT)
"""

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from .config import TrainingConfig
from .dataset import load_chatml_dataset, preprocess_chatml


class QLoRATrainer:
    """QLoRA fine-tuning trainer for Llama models."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> None:
        """Set up model, tokenizer, and training components."""
        logger.info("Setting up QLoRA training...")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Llama tokenizers need pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Configure 4-bit quantization
        logger.info("Configuring 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(
                torch, self.config.quantization.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
        )

        # Load model with quantization
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        logger.info("Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Configure LoRA
        logger.info("Configuring LoRA adapters...")
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )

        # Add LoRA adapters to model
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("✅ Model setup complete")

    def load_data(self):
        """Load and preprocess training data."""
        logger.info("Loading training data...")

        train_path = self.config.get_train_data_path()
        eval_path = self.config.get_eval_data_path()

        dataset = load_chatml_dataset(
            train_path=train_path,
            eval_path=eval_path,
            test_split=self.config.train_test_split,
        )

        return dataset

    def create_trainer(self, dataset):
        """
        Create HuggingFace Trainer.

        Args:
            dataset: DatasetDict with train/test splits

        Returns:
            SFTTrainer instance
        """
        logger.info("Creating trainer...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps" if "test" in dataset else "no",
            eval_steps=self.config.eval_steps if "test" in dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            group_by_length=self.config.group_by_length,
            report_to=self.config.report_to,
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # Push to hub disabled by default
            push_to_hub=False,
        )

        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            dataset_text_field="text",  # ChatML formatted text field
            max_seq_length=self.config.max_seq_length,
            args=training_args,
        )

        logger.info("✅ Trainer created")
        return trainer

    def train(self) -> None:
        """Run the full training pipeline."""
        # Setup model and tokenizer
        self.setup()

        # Load data
        dataset = self.load_data()

        # Create trainer
        self.trainer = self.create_trainer(dataset)

        # Train!
        logger.info("🚀 Starting training...")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Training examples: {len(dataset['train'])}")
        if "test" in dataset:
            logger.info(f"  Evaluation examples: {len(dataset['test'])}")
        logger.info(f"  Epochs: {self.config.num_train_epochs}")
        logger.info(
            f"  Batch size: {self.config.per_device_train_batch_size} (per device)"
        )
        logger.info(
            f"  Gradient accumulation: {self.config.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}"
        )
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Output directory: {self.config.output_dir}")

        self.trainer.train()

        logger.info("✅ Training complete!")

    def save_model(self, output_path: str = None) -> None:
        """
        Save the fine-tuned model and tokenizer.

        Args:
            output_path: Output directory (uses config.output_dir if None)
        """
        if output_path is None:
            output_path = self.config.output_dir

        logger.info(f"Saving model to {output_path}...")

        # Save LoRA adapters
        self.model.save_pretrained(output_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"✅ Model saved to {output_path}")

    def evaluate(self) -> dict:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate()

        logger.info("Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return metrics
