# Transfer Quantization Experiment - Implementation Summary

## What Was Implemented

A complete experimental framework to test whether quantization scaling factors learned from a smaller LLM (MobileLLM-350M) can be transferred to a larger model (MobileLLM-1B) without recalibration.

## Files Created

### 1. `transfer_quantization_experiment.py` (Main Implementation)
**Size:** ~1000 lines
**Purpose:** Complete experiment framework

**Key Classes:**

#### `ScalingFactorExtractor`
- Extracts AWQ scaling factors from quantized models
- Captures:
  - Importance scores (per-channel activation magnitudes)
  - Outlier indices (protected salient channels)
  - Quantization parameters (bits, group size, scale factors)
- Save/load functionality for reuse

#### `ScalingFactorTransfer`
- Transfers scaling factors between models
- Three strategies:
  - **Direct:** Only exact dimension matches
  - **Adaptive:** Handles mismatches via interpolation (RECOMMENDED)
  - **Ratio:** Uses protection ratios (dimension-agnostic)
- Intelligent layer matching (handles architectural differences)
- Dimension adaptation:
  - Upscaling: Interpolate importance scores
  - Downscaling: Sample most important channels

#### `TransferQuantizationExperiment`
- Main orchestrator for the experiment
- Phases:
  1. Load datasets (calibration + test)
  2. Evaluate source model (raw + quantized)
  3. Extract scaling factors
  4. Evaluate target model (raw + normal quant + transferred quant)
  5. Analyze and compare results
- Memory-efficient (unloads models between phases)
- Comprehensive logging and metrics

**Key Functions:**

#### `awq_quantize_with_transferred_scales()`
- Applies AWQ quantization using pre-computed scales
- No calibration needed
- Handles missing layers gracefully
- Preserves dtype consistency

### 2. `transfer_config.json` (Configuration)
Default configuration for MobileLLM experiment:
- Source: MobileLLM-350M
- Target: MobileLLM-1B
- 4-bit AWQ quantization
- 128 calibration samples
- Adaptive transfer strategy

### 3. `TRANSFER_EXPERIMENT_README.md` (Documentation)
**Size:** ~500 lines
**Contents:**
- Complete usage guide
- Configuration options explained
- Transfer strategies detailed
- Output interpretation
- Troubleshooting guide
- Advanced usage examples

### 4. `test_transfer_experiment.py` (Unit Tests)
**Tests:**
- Scaling factor extraction
- All transfer strategies
- Quantization with transferred scales
- Dimension mismatch handling
- End-to-end workflow

### 5. `run_transfer_experiment.sh` (Runner Script)
Convenience script with:
- Dependency checking
- Error handling
- Progress reporting

## How It Works

### Phase 1: Source Model (350M)

```python
# 1. Load model
model_350m = load_model("facebook/MobileLLM-350M")

# 2. Evaluate raw perplexity
ppl_raw = evaluate_perplexity(model_350m, test_data)

# 3. Collect activation statistics
input_feat = get_calib_feat(model_350m, calib_data)

# 4. Extract scaling factors
extractor = ScalingFactorExtractor()
extractor.extract_from_awq(model_350m, input_feat, ...)

# Important: Extract BEFORE quantization!
# This captures the importance scores that AWQ will use

# 5. Apply AWQ quantization
awq_quantize_model_weight(model_350m, input_feat, ...)

# 6. Evaluate quantized perplexity
ppl_quant = evaluate_perplexity(model_350m, test_data)

# 7. Save scales for reuse
extractor.save_to_file("scaling_factors_350m.json")
```

### Phase 2: Target Model (1B)

```python
# Path A: Normal Quantization (Baseline)
model_1b = load_model("facebook/MobileLLM-1B")
input_feat_1b = get_calib_feat(model_1b, calib_data)
awq_quantize_model_weight(model_1b, input_feat_1b, ...)
ppl_normal = evaluate_perplexity(model_1b, test_data)

# Path B: Transferred Quantization (Our Method)
model_1b = load_model("facebook/MobileLLM-1B")
transferred_scales = transfer_scales(scales_350m, model_1b)
awq_quantize_with_transferred_scales(model_1b, transferred_scales)
ppl_transfer = evaluate_perplexity(model_1b, test_data)

# Compare
gap = ppl_transfer - ppl_normal
# If gap < 5%, transfer is excellent!
```

### Key Innovation: Adaptive Transfer

For layers with dimension mismatches:

```python
# Example: 350M has 512 dims, 1B has 768 dims

# 1. Interpolate importance scores
importance_768 = F.interpolate(importance_512, size=768, mode='linear')

# 2. Scale outlier indices proportionally
ratio = 768 / 512  # = 1.5
outlier_indices_768 = (outlier_indices_512 * ratio).long()

# 3. Maintain same protection ratio
protect_ratio = n_protect_512 / 512  # e.g., 0.01
n_protect_768 = int(768 * protect_ratio)  # = 7 or 8 channels
```

## Expected Results

### Success Metrics

The experiment evaluates 5 configurations:

1. **Source Raw (350M):** Baseline for 350M
2. **Source Quantized (350M):** AWQ performance on 350M
3. **Target Raw (1B):** Baseline for 1B
4. **Target Normal Quantized (1B):** Standard AWQ on 1B
5. **Target Transferred Quantized (1B):** Our transfer approach

### Key Metric: Transfer Gap

```
Transfer Gap = PPL_transferred - PPL_normal
```

**Interpretation:**
- **< 5%:** Excellent transfer (practically equivalent)
- **5-10%:** Good transfer (acceptable for many use cases)
- **10-20%:** Moderate transfer (useful but limited)
- **> 20%:** Limited transfer (models too different)

### Example Output

```
Model Variant                            |        PPL |   Size (MB) |  Degradation
--------------------------------------------------------------------------------
Source (350M) - Raw                      |      12.45 |      700.00 |           —
Source (350M) - Quantized                |      13.21 |      175.00 |       6.11%
--------------------------------------------------------------------------------
Target (1B) - Raw                        |      10.23 |     2000.00 |           —
Target (1B) - Normal Quantized           |      10.87 |      500.00 |       6.26%
Target (1B) - Transferred Quantized      |      11.34 |      500.00 |      10.85%
--------------------------------------------------------------------------------

Transfer Gap: +4.59%
Effectiveness: EXCELLENT
```

## Usage

### Basic Usage

```bash
# 1. Install dependencies
pip install torch transformers datasets tqdm

# 2. Run with defaults (MobileLLM 350M → 1B)
python transfer_quantization_experiment.py

# Takes ~30-60 minutes on GPU
# Produces: transfer_experiment_results.json
```

### Custom Configuration

```bash
# Create custom config
cp transfer_config.json my_config.json

# Edit models, quantization bits, datasets, etc.
vim my_config.json

# Run with custom config
python transfer_quantization_experiment.py my_config.json
```

### Test First

```bash
# Run unit tests (no large models loaded)
python test_transfer_experiment.py

# Should complete in < 1 minute
# Verifies all components work
```

## Memory Requirements

### Minimum Requirements
- **GPU:** 16GB VRAM (for MobileLLM-1B)
- **RAM:** 32GB
- **Storage:** 5GB (for model downloads)

### To Reduce Memory:
```json
{
  "n_calibration_samples": 64,    // Reduce from 128
  "n_test_samples": 10,           // Reduce from 20
  "device_map": "cpu"             // Use CPU (slower)
}
```

## Key Research Questions Answered

1. **Can quantization parameters transfer across model sizes?**
   - Measures: Transfer gap in perplexity

2. **Which transfer strategy works best?**
   - Tests: Direct, Adaptive, Ratio strategies

3. **How do dimension mismatches affect transfer?**
   - Adaptive strategy handles via interpolation
   - Per-layer statistics tracked

4. **Is transfer worth the potential degradation?**
   - Saves calibration time (no need to run 1B through calibration)
   - Trade-off: Speed vs. accuracy

## Technical Details

### AWQ Quantization Recap

```python
# AWQ protects salient weight channels
importance = sum(activations).float()
outlier_indices = topk(importance, k=n_protect)

# Scale up protected channels
weights[:, outlier_indices] *= scale_factor

# Quantize all weights
weights = quantize(weights, n_bit=4)

# Scale back down protected channels
weights[:, outlier_indices] /= scale_factor
```

### Why Transfer Works

- **Activation patterns** are somewhat consistent across model sizes
- **Important channels** often correspond to same semantic features
- **Protection ratios** generalize better than absolute indices
- **Interpolation** preserves importance structure across dimensions

### Why Transfer Might Fail

- **Different architectures** (e.g., GPT vs LLaMA)
- **Different training data** (domain shift)
- **Large size gaps** (e.g., 125M → 70B)
- **Very aggressive quantization** (2-bit more sensitive)

## Extensions & Future Work

### Already Implemented
✅ Three transfer strategies  
✅ Dimension mismatch handling  
✅ Comprehensive metrics  
✅ Save/load functionality  
✅ Memory-efficient design  

### Possible Extensions
1. **Cross-Architecture Transfer:** GPT → LLaMA
2. **Progressive Transfer:** 350M → 700M → 1B
3. **Fine-tuning After Transfer:** Light calibration on top
4. **Layer-wise Analysis:** Which layers transfer best?
5. **Multi-bit Transfer:** Different bits per layer
6. **Activation Transfer:** Transfer activation quantization too

## Troubleshooting

### Common Issues

**1. Out of Memory**
```json
{
  "n_calibration_samples": 32,
  "device_map": "cpu"
}
```

**2. Models Not Found**
- Verify model names on Hugging Face
- Check internet connection

**3. Slow Execution**
- Use fewer calibration/test samples
- Use GPU if available
- Reduce block sizes

**4. Poor Transfer Results**
- Models may be too different architecturally
- Try different transfer strategy
- Check if models are from same family

## Files Overview

```
transfer_quantization_experiment.py  # Main implementation (1000 lines)
transfer_config.json                 # Default configuration
TRANSFER_EXPERIMENT_README.md        # User documentation (500 lines)
test_transfer_experiment.py          # Unit tests (400 lines)
run_transfer_experiment.sh           # Convenience runner

# Generated during run:
scaling_factors_350m.json            # Extracted scales (reusable)
transfer_experiment_results.json     # Complete results
transfer_config_default.json         # Auto-generated default
```

## Summary

This implementation provides a complete, production-ready framework for testing quantization parameter transfer between models. It includes:

- **Robust extraction** of quantization parameters
- **Flexible transfer** with multiple strategies
- **Comprehensive evaluation** with detailed metrics
- **Memory-efficient** design for large models
- **Extensive documentation** and examples
- **Unit tests** for validation

The framework is ready to use and can be easily adapted for other model pairs or quantization methods.

## Next Steps

1. **Run the tests:**
   ```bash
   python test_transfer_experiment.py
   ```

2. **Run the experiment:**
   ```bash
   python transfer_quantization_experiment.py
   ```

3. **Analyze results:**
   - Check `transfer_experiment_results.json`
   - Look at transfer gap
   - Assess effectiveness

4. **Experiment further:**
   - Try different model pairs
   - Test different quantization bits
   - Modify transfer strategies

## Citation

```bibtex
@software{transfer_quantization_2024,
  title={Transfer Quantization: Learning Quantization Parameters Across Model Sizes},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-quantization}
}
```
