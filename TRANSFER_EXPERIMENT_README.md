# Transfer Quantization Experiment

## Overview

This experiment tests whether quantization scaling factors learned from a smaller LLM (350M parameters) can be transferred to a larger model (1B parameters) to achieve comparable quantization quality without recalibrating on the larger model.

## Research Question

**Can we amortize the calibration cost across model sizes by learning quantization parameters once and transferring them?**

This has practical implications for:
- Rapid quantization of model families
- Reducing calibration dataset requirements for larger models
- Understanding what makes quantization transferable

## Experiment Design

### Phase 1: Source Model (MobileLLM-350M)
1. Load the 350M model
2. Evaluate raw perplexity
3. Collect activation statistics from calibration data
4. Extract AWQ scaling factors (importance scores, outlier indices)
5. Apply AWQ quantization
6. Evaluate quantized perplexity
7. Save scaling factors for transfer

### Phase 2: Target Model (MobileLLM-1B)
1. Load the 1B model
2. Evaluate raw perplexity
3. **Normal Quantization Path:**
   - Collect own activation statistics
   - Apply normal AWQ quantization
   - Evaluate perplexity
4. **Transfer Quantization Path:**
   - Load scaling factors from 350M
   - Transfer using adaptive strategy (handles dimension mismatches)
   - Apply quantization with transferred factors
   - Evaluate perplexity

### Phase 3: Analysis
Compare perplexities to determine:
- How much does transfer quantization degrade compared to normal?
- Is the gap acceptable for practical use?
- Which layers transfer well vs. poorly?

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers datasets tqdm
```

### 2. Run with Default Configuration

```bash
python transfer_quantization_experiment.py
```

This will:
- Use MobileLLM-350M as source, MobileLLM-1B as target
- Apply 4-bit AWQ quantization
- Use 128 calibration samples from Pile dataset
- Save results to `transfer_experiment_results.json`

### 3. Run with Custom Configuration

```bash
python transfer_quantization_experiment.py transfer_config.json
```

## Configuration Options

### Models
```json
{
  "source_model": "facebook/MobileLLM-350M",
  "target_model": "facebook/MobileLLM-1B"
}
```

### Transfer Strategy
- **`"adaptive"`** (Recommended): Handles dimension mismatches with interpolation
- **`"direct"`**: Only transfers to exactly matching layers
- **`"ratio"`**: Uses protection ratios instead of absolute indices

```json
{
  "transfer_strategy": "adaptive"
}
```

### AWQ Quantization
```json
{
  "awq_config": {
    "w_bit": 4,              # Quantization bits (4 for 4-bit)
    "q_group_size": 128,     # Group size for quantization
    "protect_ratio": 0.01,   # Protect top 1% of channels
    "scale_factor": 2.0      # Scale factor for protected channels
  }
}
```

### Calibration & Test Data
```json
{
  "calibration_dataset": "mit-han-lab/pile-val-backup",
  "n_calibration_samples": 128,
  "calibration_block_size": 512,
  
  "test_dataset": "mit-han-lab/pile-val-backup",
  "n_test_samples": 20,
  "test_block_size": 1024
}
```

## Output

### Console Output

The experiment prints detailed progress and results:

```
================================================================================
TRANSFER QUANTIZATION EXPERIMENT
================================================================================
Source Model: facebook/MobileLLM-350M
Target Model: facebook/MobileLLM-1B
Transfer Strategy: adaptive
================================================================================

PHASE 1: SOURCE MODEL EVALUATION
...

PHASE 2: TARGET MODEL EVALUATION
...

ANALYSIS & COMPARISON
--------------------------------------------------------------------------------
Model Variant                            |        PPL |   Size (MB) |  Degradation
--------------------------------------------------------------------------------
Source (350M) - Raw                      |      12.45 |      700.00 |           —
Source (350M) - Quantized                |      13.21 |      175.00 |       6.11%
--------------------------------------------------------------------------------
Target (1B) - Raw                        |      10.23 |     2000.00 |           —
Target (1B) - Normal Quantized           |      10.87 |      500.00 |       6.26%
Target (1B) - Transferred Quantized      |      11.34 |      500.00 |      10.85%
--------------------------------------------------------------------------------

KEY INSIGHTS
================================================================================

1. Quantization Impact on Source (350M):
   Degradation: 6.11%

2. Quantization Impact on Target (1B):
   Normal Quantization: 6.26%
   Transferred Quantization: 10.85%

3. Transfer Effectiveness:
   Gap between Normal and Transferred: +4.59%
   Effectiveness: EXCELLENT
   → Transferred quantization performs nearly as well as normal quantization!

4. Size Reduction:
   Source: 75.0% reduction
   Target: 75.0% reduction
================================================================================
```

### Saved Files

1. **`scaling_factors_350m.json`**
   - Extracted scaling factors from source model
   - Includes importance scores, outlier indices, dimensions
   - Can be reused for multiple target models

2. **`transfer_experiment_results.json`**
   - Complete results with all perplexity values
   - Includes configuration and timestamp
   - Structured for further analysis

3. **`transfer_config_default.json`**
   - Default configuration for reference
   - Auto-generated if no config provided

## Transfer Strategies Explained

### Adaptive Strategy (Recommended)

Handles dimension mismatches intelligently:

**For layers with larger dimensions (upscaling):**
```python
# Example: 350M has 512 dims, 1B has 768 dims
# Interpolate importance scores: 512 → 768
importance_768 = interpolate(importance_512, size=768)

# Scale outlier indices proportionally
outlier_indices_768 = outlier_indices_512 * (768 / 512)
```

**For layers with smaller dimensions (downscaling):**
```python
# Example: 350M has 2048 dims, 1B has 1536 dims
# Sample importance scores: 2048 → 1536
importance_1536 = importance_2048[sampled_indices]

# Keep top-k most important channels
outlier_indices_1536 = topk(importance_1536, k=n_protect)
```

### Direct Strategy

Only transfers to exactly matching layers:
- Safest approach, no interpolation errors
- May skip many layers if architectures differ
- Best for very similar model architectures

### Ratio Strategy

Uses protection ratios instead of absolute indices:
- Most dimension-agnostic
- Doesn't use activation importance (limitation)
- Good baseline for understanding transfer benefits

## Understanding the Results

### Success Criteria

**Excellent Transfer (Gap < 5%):**
- Transferred quantization nearly matches normal quantization
- Strong evidence that scaling factors are transferable
- Practical for deployment

**Good Transfer (Gap 5-10%):**
- Acceptable degradation for many use cases
- May be worth it to avoid recalibration
- Some model-specific adaptation still beneficial

**Moderate Transfer (Gap 10-20%):**
- Significant but usable degradation
- Consider hybrid approach (transfer + light fine-tuning)
- Useful for understanding transferability limits

**Limited Transfer (Gap > 20%):**
- Models may be too different architecturally
- Activation patterns don't align well
- Normal quantization recommended

### What Affects Transferability?

1. **Architectural Similarity:**
   - Same layer types (attention, MLP)
   - Similar hidden dimensions
   - Same number of layers

2. **Training Data Alignment:**
   - Models trained on similar data have similar activations
   - Domain shift reduces transferability

3. **Model Size Ratio:**
   - Smaller ratios transfer better (350M → 1B easier than 350M → 70B)
   - Closer dimensions = better interpolation

4. **Quantization Aggressiveness:**
   - Lower bits (2-3 bit) are more sensitive to transfer errors
   - Higher bits (8 bit) transfer more reliably

## Advanced Usage

### Experiment with Different Model Pairs

```json
{
  "source_model": "gpt2",
  "target_model": "gpt2-medium"
}
```

```json
{
  "source_model": "facebook/opt-125m",
  "target_model": "facebook/opt-350m"
}
```

### Try Different Quantization Bits

Test 2-bit, 3-bit, 4-bit, 8-bit:

```json
{
  "awq_config": {
    "w_bit": 2
  }
}
```

### Use Different Datasets

```json
{
  "calibration_dataset": "wikitext",
  "calibration_dataset_config": "wikitext-2-raw-v1"
}
```

### Adjust Calibration Samples

Balance between accuracy and speed:

```json
{
  "n_calibration_samples": 64,   // Faster, less accurate
  "n_calibration_samples": 256   // Slower, more accurate
}
```

## Analyzing Results

### Load Results Programmatically

```python
import json

with open('transfer_experiment_results.json', 'r') as f:
    results = json.load(f)

# Extract key metrics
source_deg = results['results']['source_quantized']['degradation_%']
target_normal_deg = results['results']['target_normal_quantized']['degradation_%']
target_transfer_deg = results['results']['target_transferred_quantized']['degradation_%']

transfer_gap = target_transfer_deg - target_normal_deg

print(f"Transfer effectiveness gap: {transfer_gap:.2f}%")
```

### Visualize Results

```python
import matplotlib.pyplot as plt

models = ['Source\nRaw', 'Source\nQuant', 'Target\nRaw', 
          'Target\nNormal', 'Target\nTransfer']
ppls = [
    results['results']['source_raw']['perplexity'],
    results['results']['source_quantized']['perplexity'],
    results['results']['target_raw']['perplexity'],
    results['results']['target_normal_quantized']['perplexity'],
    results['results']['target_transferred_quantized']['perplexity'],
]

plt.bar(models, ppls)
plt.ylabel('Perplexity (lower is better)')
plt.title('Transfer Quantization Experiment Results')
plt.show()
```

## Troubleshooting

### Out of Memory

Reduce memory usage:
```json
{
  "n_calibration_samples": 64,
  "calibration_block_size": 256,
  "n_test_samples": 10,
  "test_block_size": 512
}
```

Or use CPU:
```json
{
  "device_map": "cpu"
}
```

### Models Not Found

Ensure model names are correct:
```bash
# Check available models on Hugging Face
# https://huggingface.co/models
```

For MobileLLM, verify:
- `facebook/MobileLLM-350M`
- `facebook/MobileLLM-1B`

### Slow Execution

To speed up:
1. Reduce calibration samples (64 instead of 128)
2. Reduce test samples (10 instead of 20)
3. Use smaller block sizes
4. Use GPU if available

## Implementation Details

### Key Components

1. **`ScalingFactorExtractor`**
   - Extracts importance scores and outlier indices
   - Saves to JSON for reuse
   - Compatible with any model architecture

2. **`ScalingFactorTransfer`**
   - Handles dimension mismatches
   - Three transfer strategies
   - Layer name matching logic

3. **`awq_quantize_with_transferred_scales`**
   - Applies AWQ quantization using pre-computed scales
   - No calibration needed
   - Falls back gracefully for missing layers

4. **`TransferQuantizationExperiment`**
   - Orchestrates entire experiment
   - Memory-efficient (unloads models between phases)
   - Comprehensive logging and analysis

### Memory Management

The script carefully manages memory:
- Loads models one at a time
- Unloads models after each phase
- Clears CUDA cache between evaluations
- Uses garbage collection

## Citation

If you use this experiment in your research:

```bibtex
@misc{transfer_quantization_experiment,
  title={Transfer Quantization: Learning Quantization Parameters Across Model Sizes},
  author={Your Name},
  year={2024},
  howpublished={GitHub}
}
```

## Related Work

- **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression"
- **GPTQ**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- **MobileLLM**: Meta AI, "MobileLLM: Optimizing Sub-billion Parameter Language Models"

## Future Directions

1. **Cross-Architecture Transfer**: Transfer between different architectures (GPT → LLaMA)
2. **Progressive Transfer**: Transfer in stages (350M → 700M → 1B)
3. **Fine-tuning After Transfer**: Light calibration on top of transferred scales
4. **Layer-wise Analysis**: Identify which layer types transfer best
5. **Multi-bit Transfer**: Transfer different bit-widths simultaneously

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example configurations
3. Examine the console output for specific errors
4. Adjust configuration parameters based on your hardware

## License

This experiment framework is provided as-is for research purposes.
