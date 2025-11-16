# Memory Optimization Guide for Transfer Quantization Experiment

## Problem: "Killed" or Out of Memory

If the experiment is killed during execution, it's likely due to insufficient memory. Here are solutions:

## Quick Fixes

### 1. Use Smaller Configuration (Recommended for Testing)

```bash
# Very small - good for quick testing (< 8GB RAM)
./run_transfer_experiment.sh transfer_config_tiny.json

# Small - moderate memory (< 16GB RAM)  
./run_transfer_experiment.sh transfer_config_small.json

# Original - requires more memory (32GB+ RAM)
./run_transfer_experiment.sh transfer_config.json
```

### 2. Use CPU Instead of GPU

Edit your config file:
```json
{
  "device_map": "cpu"
}
```

### 3. Reduce Sample Counts

The most effective parameters to reduce:

```json
{
  "n_calibration_samples": 16,      // Default: 128, Minimum: 8
  "calibration_block_size": 256,    // Default: 512, Minimum: 128
  "n_test_samples": 5,              // Default: 20, Minimum: 3
  "test_block_size": 512            // Default: 1024, Minimum: 256
}
```

## Configuration Comparison

| Config | Models | Memory | Time | Samples |
|--------|--------|--------|------|---------|
| **tiny** | GPT2 → GPT2-Medium | ~6GB | ~10 min | 16 calib, 5 test |
| **small** | OPT-350M → OPT-1.3B | ~12GB | ~20 min | 32 calib, 10 test |
| **default** | OPT-350M → OPT-1.3B | ~24GB | ~45 min | 128 calib, 20 test |

## Available Configurations

### 1. `transfer_config_tiny.json` - Minimal Memory (~6GB)
- **Models:** GPT2 (124M) → GPT2-Medium (355M)
- **Dataset:** WikiText-2
- **Use when:** Testing on laptop or limited resources
- **Pros:** Fast, very low memory
- **Cons:** Results may be less accurate due to fewer samples

### 2. `transfer_config_small.json` - Moderate Memory (~12GB)
- **Models:** OPT-350M → OPT-1.3B
- **Dataset:** Pile
- **Use when:** You have moderate RAM/VRAM
- **Pros:** Good balance of speed and accuracy
- **Cons:** Still limited calibration data

### 3. `transfer_config.json` - Full Memory (~24GB)
- **Models:** OPT-350M → OPT-1.3B
- **Dataset:** Pile
- **Use when:** You have sufficient resources
- **Pros:** Most accurate results
- **Cons:** Requires significant memory

## Step-by-Step Memory Troubleshooting

### If killed during dataset loading:
```json
{
  "n_calibration_samples": 8,
  "calibration_block_size": 128
}
```

### If killed during source model evaluation:
```json
{
  "source_model": "gpt2",  // Smaller model
  "device_map": "cpu"
}
```

### If killed during target model evaluation:
```json
{
  "target_model": "gpt2-medium",  // Smaller target
  "n_test_samples": 3
}
```

## Running with Reduced Memory

### Option 1: Use Tiny Config (Fastest)
```bash
chmod +x run_transfer_experiment.sh
./run_transfer_experiment.sh transfer_config_tiny.json
```

### Option 2: Use Small Config
```bash
./run_transfer_experiment.sh transfer_config_small.json
```

### Option 3: Create Custom Config
```bash
# Copy and edit
cp transfer_config_small.json my_config.json
nano my_config.json  # Edit to your needs

# Run
./run_transfer_experiment.sh my_config.json
```

## Monitoring Memory Usage

### During execution:
```bash
# In another terminal
watch -n 1 free -h          # Linux
watch -n 1 vm_stat          # macOS

# Or use htop
htop
```

### Check GPU memory (if using GPU):
```bash
watch -n 1 nvidia-smi
```

## Progressive Testing Strategy

Start small and scale up:

```bash
# 1. First test with tiny config (should work on any system)
./run_transfer_experiment.sh transfer_config_tiny.json

# 2. If successful, try small config
./run_transfer_experiment.sh transfer_config_small.json

# 3. If successful, try full config
./run_transfer_experiment.sh transfer_config.json
```

## Expected Memory Usage

### Tiny Config (GPT2 → GPT2-Medium):
- **Peak RAM:** ~4-6 GB
- **Peak VRAM (if GPU):** ~2-3 GB
- **Runtime:** ~10-15 minutes

### Small Config (OPT-350M → OPT-1.3B):
- **Peak RAM:** ~10-14 GB
- **Peak VRAM (if GPU):** ~6-8 GB
- **Runtime:** ~20-30 minutes

### Full Config (OPT-350M → OPT-1.3B):
- **Peak RAM:** ~20-28 GB
- **Peak VRAM (if GPU):** ~12-16 GB
- **Runtime:** ~45-60 minutes

## Additional Memory-Saving Tips

### 1. Close Other Applications
Free up system memory before running.

### 2. Use Swap Space (Linux)
```bash
# Check current swap
free -h

# Add swap if needed (8GB example)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Run One Phase at a Time
Modify the script to run phases separately if needed.

### 4. Use Gradient Checkpointing
Add to config (if supported):
```json
{
  "use_gradient_checkpointing": true
}
```

## Troubleshooting Specific Errors

### "Killed" (Exit Code 137)
- **Cause:** Out of memory (OOM)
- **Solution:** Use smaller config or add swap

### "CUDA out of memory"
- **Cause:** GPU VRAM insufficient
- **Solution:** Use `"device_map": "cpu"` or smaller models

### "Cannot allocate memory"
- **Cause:** System RAM full
- **Solution:** Close applications, use smaller samples

## Recommended Approach for Your System

Based on the error, try this:

```bash
# Start with tiny config to verify everything works
./run_transfer_experiment.sh transfer_config_tiny.json

# If successful, the experiment framework is working!
# You can then gradually increase sample sizes.
```

## Still Having Issues?

1. **Check available memory:**
   ```bash
   free -h  # Linux
   vm_stat  # macOS
   ```

2. **Use absolute minimal config:**
   ```json
   {
     "n_calibration_samples": 4,
     "n_test_samples": 2,
     "calibration_block_size": 128,
     "test_block_size": 256
   }
   ```

3. **Run on CPU only:**
   ```json
   {
     "device_map": "cpu",
     "torch_dtype": "float32"
   }
   ```

## Success Indicators

You'll know it's working when you see:
```
✓ Source Raw - PPL: X.XX, Size: XXX.XX MB
✓ Source Quantized - PPL: X.XX, Size: XXX.XX MB
✓ Target Raw - PPL: X.XX, Size: XXX.XX MB
...
```

If the experiment completes without being killed, you can gradually increase sample sizes for more accurate results!
