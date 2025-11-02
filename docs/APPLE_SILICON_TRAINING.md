# Training on Apple Silicon (M1/M2/M3)

✅ **All tests passed! Ready to train on your Mac.**

## What Changed

We've adapted the training code to work on Apple Silicon without `bitsandbytes` (CUDA-only):

### Architecture Changes

1. **No 4-bit Quantization** - Uses FP32/FP16 instead of 4-bit QLoRA
2. **MPS Backend** - Uses Apple's Metal Performance Shaders instead of CUDA
3. **LoRA Still Active** - Still uses LoRA adapters (trainable: ~1% of parameters)
4. **Smaller Batches** - Batch size 1 with gradient accumulation to fit in memory

### Performance Trade-offs

| Metric | Linux GPU (QLoRA) | Apple Silicon (LoRA) |
|--------|------------------|---------------------|
| **Quantization** | 4-bit (BitsAndBytes) | None (FP32/FP16) |
| **Memory Usage** | ~3-4 GB VRAM | ~8-12 GB RAM |
| **Training Time** | 2-8 hours | 8-16 hours |
| **Quality** | Excellent | Excellent (same LoRA) |
| **Cost** | $1-2 (cloud GPU) | $0 (your Mac) |

## Requirements

- **Mac with Apple Silicon** (M1, M2, M3, M4)
- **16+ GB RAM** (32 GB recommended for comfort)
- **50+ GB free disk space** (for model checkpoints)
- **macOS Monterey or later** (for MPS support)

### Memory Estimate

- **Llama 3.2 3B in FP32**: ~12 GB
- **LoRA adapters**: ~500 MB
- **Optimizer states**: ~2-3 GB
- **Batch processing**: ~2 GB
- **Total**: ~18-20 GB during training

**Recommendation**: Close other apps during training to free up memory.

## Setup

Already done! Run the test to verify:

```bash
source .venv/bin/activate
python scripts/test_apple_silicon.py
```

You should see:
```
🎉 All tests passed! Ready to train on Apple Silicon.
```

## Training

### Option 1: Quick Start (Recommended)

```bash
make train-apple
```

This uses the optimized Apple Silicon config:
- Batch size: 1
- Gradient accumulation: 8 (effective batch = 8)
- No quantization
- FP32 precision (most compatible)
- Gradient checkpointing enabled

### Option 2: Monitor Progress

```bash
# Terminal 1: Start training
source .venv/bin/activate
python scripts/train_model.py --config configs/apple_silicon_config.json

# Terminal 2: Monitor activity
top -pid $(pgrep -f train_model)
```

### Option 3: Background Training

```bash
# Run in background
nohup make train-apple > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Expected Timeline

**Total training time: 8-16 hours** (depending on your Mac)

- **M3 Max/Ultra**: ~8-10 hours
- **M2 Pro/Max**: ~10-12 hours
- **M1 Pro/Max**: ~12-16 hours
- **M1 (base)**: ~16-20 hours

**Progress breakdown:**
- Epoch 1: ~3-6 hours (slowest, includes setup)
- Epoch 2: ~2-5 hours
- Epoch 3: ~2-5 hours
- Saving model: ~10-30 minutes

## What to Expect During Training

### Normal Behavior

1. **Initial loading (5-15 min)**:
   - Downloading model from HuggingFace (~3 GB)
   - Loading into memory
   - Setting up LoRA adapters

2. **Training starts**:
   - You'll see progress bars
   - Loss will decrease gradually
   - Temperature: expect your Mac to get warm (normal!)
   - Fan noise: likely to ramp up

3. **Checkpoints**:
   - Saved every 250 steps (~45-90 minutes)
   - Keeps last 2 checkpoints (saves disk space)

4. **Memory usage**:
   - Should stay around 16-20 GB
   - If it crashes with OOM, see troubleshooting below

### Warning Signs

- **Memory pressure**: Activity Monitor shows yellow/red memory pressure
  - Solution: Close other apps, reduce batch size to 1
- **Thermal throttling**: Mac gets very hot and slows down
  - Solution: Use a laptop stand, ensure good ventilation
- **Crashes**: Python process killed
  - Solution: Reduce `max_seq_length` to 1024 in config

## Configuration

The Apple Silicon config (`configs/apple_silicon_config.json`) has these optimizations:

```json
{
  "per_device_train_batch_size": 1,      // Small batch for memory
  "gradient_accumulation_steps": 8,      // Effective batch = 8
  "fp16": false,                         // FP32 more stable on MPS
  "bf16": false,
  "gradient_checkpointing": true,        // Saves memory
  "optim": "adamw_torch",                // Native PyTorch optimizer
  "save_steps": 250,                     // Frequent checkpoints
  "save_total_limit": 2,                 // Keep last 2 only
  "quantization": {
    "load_in_4bit": false                // No quantization on MPS
  }
}
```

## Troubleshooting

### "RuntimeError: MPS backend out of memory"

**Solution 1: Reduce batch size** (already at 1, so skip to solution 2)

**Solution 2: Reduce sequence length**

Edit `configs/apple_silicon_config.json`:
```json
{
  "max_seq_length": 1024  // Down from 2048
}
```

**Solution 3: Close other apps**

Quit Chrome, Slack, etc. to free up RAM.

**Solution 4: Reduce LoRA rank**

Edit `configs/apple_silicon_config.json`:
```json
{
  "lora": {
    "r": 8  // Down from 16 (fewer trainable parameters)
  }
}
```

### "Training is very slow"

This is normal! Apple Silicon is slower than a dedicated GPU:
- M1 Max: ~10-16 hours for 3 epochs
- RTX 4090: ~2-3 hours for 3 epochs

**Speed improvements:**
1. Use FP16 (less stable but faster):
   ```json
   "fp16": true
   ```
2. Reduce epochs to 2:
   ```json
   "num_train_epochs": 2
   ```
3. Use smaller LoRA rank:
   ```json
   "lora": { "r": 8 }
   ```

### "Process was killed"

macOS killed the process due to memory pressure.

**Solutions:**
1. Reduce `max_seq_length` to 1024
2. Reduce `gradient_accumulation_steps` to 4
3. Disable gradient checkpointing (uses more memory but try it):
   ```json
   "gradient_checkpointing": false
   ```
4. Upgrade to 32 GB RAM Mac if you have 16 GB

### Training crashes with error

**Check the logs:**
```bash
tail -100 logs/training.log
```

Common errors:
- `NotImplementedError: The operator 'X' is not currently implemented for the MPS device`
  - Some PyTorch ops don't support MPS yet
  - Fallback: Use CPU (very slow) by setting `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `bitsandbytes` import errors
  - Should not happen anymore, but if it does: re-install without cuda extras

## Monitoring Training

### Activity Monitor

Watch these metrics during training:

1. **Memory**:
   - Should stay around 16-20 GB used
   - Memory pressure: green (good) or yellow (acceptable)
   - Red = too much pressure, may crash

2. **CPU**:
   - Python process using 400-800% (multi-core)
   - Normal during training

3. **GPU** (if shown):
   - Metal GPU activity should be high
   - Indicates MPS backend is working

### Terminal Output

You'll see:
```
Epoch 1/3: 100%|████████| 124/124 [2:15:32<00:00, 65.42s/it]
Train Loss: 1.234
Eval Loss: 1.456
Saving checkpoint to models/checkpoints/checkpoint-250/
```

### Log Files

```bash
# Training logs
tail -f logs/training.log

# TensorBoard (if enabled)
# Currently disabled in Apple Silicon config (report_to: "none")
```

## After Training

Once training completes:

1. **Find your model**:
   ```bash
   ls -lh models/checkpoints/final/
   ```

2. **Model files**:
   - `adapter_config.json` - LoRA config
   - `adapter_model.safetensors` - LoRA weights (~500 MB)
   - `tokenizer_config.json` - Tokenizer config
   - `training_config.json` - Training hyperparameters

3. **Next steps**:
   - Test the model: `python scripts/test_model.py` (to be implemented)
   - Merge adapters with base model for standalone deployment
   - Run evaluation: `make evaluate` (to be implemented)

## Cost Analysis

**Training on Apple Silicon vs Cloud GPU:**

| Option | Time | Cost | Pros | Cons |
|--------|------|------|------|------|
| **Your Mac M1 Max** | 12 hours | $0 | Free, private | Slow, ties up Mac |
| **Cloud GPU (RTX 4090)** | 3 hours | $1.50 | Fast, frees up Mac | Costs money, upload data |
| **Google Colab Pro** | 4-6 hours | $10/month | Moderate speed | Limited hours |

**Recommendation**: Use your Mac for the first training run (it's free!). If you iterate frequently, consider cloud GPU.

## Tips for Best Results

1. **Start overnight**: Let it run while you sleep (8-16 hours)
2. **Keep Mac plugged in**: Don't run on battery
3. **Good ventilation**: Use a laptop stand or external cooling
4. **Close other apps**: Free up RAM
5. **Don't use your Mac heavily**: Light browsing ok, heavy workloads will slow training
6. **Be patient**: 10+ hours is normal!

## Comparison: QLoRA vs LoRA

**Linux GPU (QLoRA):**
- ✅ Faster (2-8 hours)
- ✅ Less memory (3-4 GB VRAM)
- ✅ 4-bit quantization
- ❌ Requires Linux + GPU
- ❌ Costs money (cloud)

**Apple Silicon (LoRA):**
- ✅ Free (your Mac)
- ✅ Private (data stays local)
- ✅ Same quality (LoRA rank same)
- ❌ Slower (8-16 hours)
- ❌ More memory (16-20 GB RAM)
- ❌ No 4-bit quantization

**Bottom line**: Both produce excellent results. Choose based on your constraints.

## Ready to Train!

Everything is set up and tested. Run:

```bash
make train-apple
```

Then:
1. Go make coffee ☕
2. Check back in 2-3 hours for first checkpoint
3. Let it run overnight for best results
4. Wake up to a trained model! 🎉

---

**Questions?** Check `TRAINING_READY.md` for more general training info.
