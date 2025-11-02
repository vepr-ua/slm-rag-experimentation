# Model Training

QLoRA fine-tuning infrastructure for Llama 3.2 3B on experimentation Q&A data.

## Overview

This module implements **memory-efficient fine-tuning** using:
- **QLoRA** (4-bit quantization + LoRA adapters)
- **HuggingFace TRL** (Supervised Fine-Tuning)
- **Llama 3.2 3B** (strong reasoning, 128K context)

**Training Pipeline:**
1. Combine Cross Validated + ArXiv synthetic datasets
2. Load model with 4-bit quantization
3. Add LoRA adapters (trainable: ~1% of parameters)
4. Fine-tune with SFT
5. Save LoRA adapters + merged model

## Components

### `config.py`
Training configuration and hyperparameters:
- `TrainingConfig`: All training settings
- `LoRAConfig`: LoRA adapter configuration (rank, alpha, target modules)
- `QuantizationConfig`: 4-bit quantization settings

**Default Settings:**
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- LoRA rank: 16 (good balance of quality/speed)
- Batch size: 2 per device, 4 gradient accumulation steps
- Learning rate: 2e-4 with cosine schedule
- Epochs: 3
- Optimizer: `paged_adamw_8bit` (memory efficient)

### `dataset.py`
Dataset loading and preprocessing:
- `load_chatml_dataset()`: Load ChatML formatted JSONL
- `combine_datasets()`: Merge Cross Validated + ArXiv data
- `get_dataset_stats()`: Dataset statistics

### `trainer.py`
Main training logic:
- `QLoRATrainer`: Handles full training pipeline
  - Model setup with 4-bit quantization
  - LoRA adapter configuration
  - HuggingFace Trainer creation
  - Training execution
  - Model saving

## Usage

### Quick Start

```bash
# 1. Combine datasets
make combine-datasets

# 2. Train with default config
make train

# Or do both in one command
make train-full
```

### Detailed Usage

**Combine datasets first:**
```bash
python scripts/train_model.py --combine-datasets
```

**Train with default config:**
```bash
python scripts/train_model.py
```

**Train with custom config:**
```bash
python scripts/train_model.py --config configs/my_config.json
```

**Adjust dataset weights:**
```bash
# Give more weight to real Cross Validated data
python scripts/train_model.py --combine-datasets --cv-weight 2.0 --arxiv-weight 1.0
```

### Programmatic Usage

```python
from src.training.config import TrainingConfig
from src.training.trainer import QLoRATrainer

# Create config (or use defaults)
config = TrainingConfig(
    num_train_epochs=5,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
)

# Create trainer and train
trainer = QLoRATrainer(config)
trainer.train()

# Save model
trainer.save_model("models/my_model")

# Evaluate
metrics = trainer.evaluate()
print(metrics)
```

## Configuration

### Key Hyperparameters

**LoRA Settings:**
```python
lora:
  r: 16                    # Rank (higher = more parameters, better quality)
  lora_alpha: 32           # Scaling factor (typically 2x rank)
  lora_dropout: 0.05       # Dropout rate
  target_modules:          # Which transformer modules to adapt
    - q_proj               # Query projection
    - k_proj               # Key projection
    - v_proj               # Value projection
    - o_proj               # Output projection
    - gate_proj            # MLP gate
    - up_proj              # MLP up
    - down_proj            # MLP down
```

**Training Settings:**
```python
num_train_epochs: 3                    # Number of epochs
per_device_train_batch_size: 2         # Batch per GPU
gradient_accumulation_steps: 4         # Effective batch = 2 * 4 = 8
learning_rate: 2e-4                    # Peak learning rate
warmup_ratio: 0.03                     # 3% warmup
lr_scheduler_type: "cosine"            # Cosine decay
max_seq_length: 2048                   # Context window
```

**Memory Optimization:**
```python
quantization:
  load_in_4bit: true                   # 4-bit quantization
  bnb_4bit_compute_dtype: "bfloat16"   # Computation dtype
  bnb_4bit_quant_type: "nf4"           # NormalFloat 4-bit
  bnb_4bit_use_double_quant: true      # Nested quantization

gradient_checkpointing: true           # Trade compute for memory
bf16: true                             # BF16 mixed precision
```

### Custom Configuration

Create a JSON config file:

```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "num_train_epochs": 5,
  "learning_rate": 1e-4,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "lora": {
    "r": 32,
    "lora_alpha": 64
  },
  "output_dir": "models/my_experiment"
}
```

Load it:
```bash
python scripts/train_model.py --config configs/my_config.json
```

## Output

Training produces:

**Checkpoints** (`models/checkpoints/`):
- Saved every 500 steps (configurable)
- Last 3 checkpoints kept (configurable)
- Contains LoRA adapters + optimizer state

**Final Model** (`models/checkpoints/final/`):
- LoRA adapters
- Tokenizer
- Training configuration

**Logs**:
- TensorBoard logs in `models/checkpoints/runs/`
- Training logs in `logs/training.log`

## Memory Requirements

**With QLoRA (4-bit + LoRA):**
- **Llama 3.2 3B**: ~2-3 GB VRAM
- **Minimum**: 6 GB VRAM (RTX 3060, M1 Max)
- **Recommended**: 12+ GB VRAM (RTX 4070, RTX 3080)

**Without quantization (full fine-tuning):**
- Would require ~40+ GB VRAM (not recommended)

**CPU Training:**
- Possible but very slow (days instead of hours)
- Requires 16+ GB RAM

## Training Time Estimates

Based on 5,000 examples, batch size 8, 3 epochs:

- **RTX 4090**: ~2-3 hours
- **RTX 4070**: ~4-6 hours
- **RTX 3080**: ~5-8 hours
- **M1 Max (32 GB)**: ~8-12 hours
- **CPU only**: ~2-3 days (not recommended)

## Tips

### For Better Quality

1. **Increase LoRA rank**: `r=32` or `r=64` (more parameters)
2. **More epochs**: Try 5-10 epochs with early stopping
3. **Lower learning rate**: `1e-4` for more stable training
4. **Larger batch size**: Increase if you have VRAM

### For Faster Training

1. **Lower LoRA rank**: `r=8` (fewer parameters)
2. **Fewer epochs**: 1-2 epochs
3. **Higher learning rate**: `3e-4` (faster convergence)
4. **Reduce max_seq_length**: `1024` instead of `2048`

### For Less Memory

1. **Reduce batch size**: `per_device_train_batch_size=1`
2. **Enable gradient checkpointing**: Already enabled by default
3. **Reduce sequence length**: `max_seq_length=1024`
4. **Use only essential LoRA modules**: `["q_proj", "v_proj"]`

## Troubleshooting

**Out of Memory (OOM):**
```bash
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Reduce sequence length
max_seq_length: 1024

# Use gradient checkpointing (already default)
gradient_checkpointing: true
```

**Training too slow:**
```bash
# Increase batch size if you have VRAM
per_device_train_batch_size: 4

# Reduce LoRA rank
lora.r: 8

# Reduce sequence length
max_seq_length: 1024
```

**Poor quality results:**
```bash
# Increase LoRA rank
lora.r: 32

# More epochs
num_train_epochs: 5

# Lower learning rate
learning_rate: 1e-4

# More training data
# Run: make collect-cv-full && make generate-qa
```

## Evaluation

After training, evaluate your model:

```bash
# Run evaluation script (to be implemented)
make evaluate
```

This will test on:
- Holdout test set (10% of training data)
- Curated critical questions
- Metrics: accuracy, perplexity, ROUGE, etc.

## Next Steps

1. **Test the model**: `python scripts/test_model.py`
2. **Run evaluation**: `make evaluate`
3. **Deploy API**: Update `src/api/main.py` to use fine-tuned model
4. **Iterate**: Adjust hyperparameters based on results
