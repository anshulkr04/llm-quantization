"""
Transfer Quantization Experiment: MobileLLM 350M → 1B

This script tests whether quantization scaling factors learned from a smaller model
(350M parameters) can be transferred to a larger model (1B parameters) to achieve
comparable quantization quality without recalibrating on the larger model.

Experiment Design:
1. Quantize MobileLLM-350M with AWQ and extract scaling factors
2. Transfer these scaling factors to MobileLLM-1B
3. Compare perplexity of:
   - Raw 350M vs Quantized 350M
   - Raw 1B vs Quantized 1B (normal) vs Quantized 1B (transferred)

Key Research Question:
Can we amortize the calibration cost across model sizes by learning
quantization parameters once and transferring them?
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import copy
import gc

from quantization_utils import (
    load_model_and_tokenizer,
    unload_model,
    get_calibration_dataset,
    get_test_dataset,
    get_calib_feat,
    evaluate_perplexity,
    get_model_size,
    MiB,
)

from awq_quantizer import awq_quantize_model_weight
from quantization_utils import pseudo_quantize_tensor


# ==============================================================================
# SCALING FACTOR EXTRACTION
# ==============================================================================

class ScalingFactorExtractor:
    """Extract and store quantization scaling factors from a quantized model."""
    
    def __init__(self):
        self.layer_scales: Dict[str, Dict[str, Any]] = {}
    
    def extract_from_awq(
        self,
        model: nn.Module,
        input_feat: Dict[str, List[torch.Tensor]],
        w_bit: int,
        q_group_size: int,
        protect_ratio: float = 0.01,
        scale_factor: float = 2.0
    ) -> None:
        """
        Extract scaling factors that AWQ would use during quantization.
        
        Args:
            model: Model to analyze
            input_feat: Activation statistics from calibration
            w_bit: Number of bits for quantization
            q_group_size: Group size
            protect_ratio: Fraction of channels to protect
            scale_factor: Scale factor for protected channels
        """
        print("Extracting AWQ scaling factors...")
        
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if name not in input_feat:
                    continue
                
                # Compute importance from activation magnitudes
                importance = sum(input_feat[name]).float()
                
                # Find top protect_ratio% channels
                n_protect = max(1, int(len(importance) * protect_ratio))
                outlier_values, outlier_indices = torch.topk(importance, n_protect)
                
                # Store scaling information
                self.layer_scales[name] = {
                    'importance_scores': importance.cpu(),
                    'outlier_indices': outlier_indices.cpu(),
                    'outlier_values': outlier_values.cpu(),
                    'n_protect': n_protect,
                    'protect_ratio': protect_ratio,
                    'scale_factor': scale_factor,
                    'w_bit': w_bit,
                    'q_group_size': q_group_size,
                    'input_dim': m.weight.shape[1],
                    'output_dim': m.weight.shape[0],
                }
        
        print(f"  → Extracted scales for {len(self.layer_scales)} layers")
    
    def save_to_file(self, filepath: str) -> None:
        """Save scaling factors to JSON file."""
        # Convert tensors to lists for JSON serialization
        serializable_scales = {}
        for name, scales in self.layer_scales.items():
            serializable_scales[name] = {
                'importance_scores': scales['importance_scores'].tolist(),
                'outlier_indices': scales['outlier_indices'].tolist(),
                'outlier_values': scales['outlier_values'].tolist(),
                'n_protect': scales['n_protect'],
                'protect_ratio': scales['protect_ratio'],
                'scale_factor': scales['scale_factor'],
                'w_bit': scales['w_bit'],
                'q_group_size': scales['q_group_size'],
                'input_dim': scales['input_dim'],
                'output_dim': scales['output_dim'],
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_scales, f, indent=2)
        
        print(f"Scaling factors saved to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load scaling factors from JSON file."""
        with open(filepath, 'r') as f:
            serializable_scales = json.load(f)
        
        # Convert lists back to tensors
        self.layer_scales = {}
        for name, scales in serializable_scales.items():
            self.layer_scales[name] = {
                'importance_scores': torch.tensor(scales['importance_scores']),
                'outlier_indices': torch.tensor(scales['outlier_indices'], dtype=torch.long),
                'outlier_values': torch.tensor(scales['outlier_values']),
                'n_protect': scales['n_protect'],
                'protect_ratio': scales['protect_ratio'],
                'scale_factor': scales['scale_factor'],
                'w_bit': scales['w_bit'],
                'q_group_size': scales['q_group_size'],
                'input_dim': scales['input_dim'],
                'output_dim': scales['output_dim'],
            }
        
        print(f"Scaling factors loaded from {filepath}")


# ==============================================================================
# SCALING FACTOR TRANSFER
# ==============================================================================

class ScalingFactorTransfer:
    """Transfer scaling factors from source model to target model."""
    
    def __init__(self, source_scales: Dict[str, Dict[str, Any]]):
        self.source_scales = source_scales
        self.target_scales: Dict[str, Dict[str, Any]] = {}
        self.transfer_stats: Dict[str, str] = {}
    
    def transfer_to_model(
        self,
        target_model: nn.Module,
        strategy: str = 'adaptive'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Transfer scaling factors to target model.
        
        Strategies:
        - 'direct': Only transfer to exactly matching layers
        - 'adaptive': Handle dimension mismatches with interpolation
        - 'ratio': Use protection ratios instead of absolute indices
        
        Args:
            target_model: Model to transfer scales to
            strategy: Transfer strategy
            
        Returns:
            Dictionary of transferred scales for target model
        """
        print(f"\nTransferring scaling factors using '{strategy}' strategy...")
        
        target_layer_info = {}
        for name, m in target_model.named_modules():
            if isinstance(m, nn.Linear):
                target_layer_info[name] = {
                    'input_dim': m.weight.shape[1],
                    'output_dim': m.weight.shape[0],
                }
        
        for target_name, target_info in target_layer_info.items():
            # Try to find matching source layer
            source_scale = self._find_matching_source(target_name, target_info)
            
            if source_scale is None:
                self.transfer_stats[target_name] = 'no_match'
                continue
            
            # Transfer based on strategy
            if strategy == 'direct':
                transferred = self._transfer_direct(source_scale, target_info)
            elif strategy == 'adaptive':
                transferred = self._transfer_adaptive(source_scale, target_info)
            elif strategy == 'ratio':
                transferred = self._transfer_ratio(source_scale, target_info)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if transferred is not None:
                self.target_scales[target_name] = transferred
        
        self._print_transfer_summary()
        return self.target_scales
    
    def _find_matching_source(
        self,
        target_name: str,
        target_info: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching source layer for target layer."""
        # First try exact name match
        if target_name in self.source_scales:
            return self.source_scales[target_name]
        
        # Try partial name matching (e.g., 'model.layers.0.mlp.gate_proj')
        for source_name, source_scale in self.source_scales.items():
            # Extract layer number and component
            if self._is_similar_layer(target_name, source_name):
                return source_scale
        
        return None
    
    def _is_similar_layer(self, name1: str, name2: str) -> bool:
        """Check if two layer names refer to similar architectural components."""
        # Extract key components (e.g., 'mlp', 'attention', 'gate_proj')
        components1 = set(name1.split('.'))
        components2 = set(name2.split('.'))
        
        # Check for common architectural keywords
        key_words = {'mlp', 'attention', 'gate', 'up', 'down', 'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        common_keywords = (components1 & key_words) & (components2 & key_words)
        
        return len(common_keywords) > 0
    
    def _transfer_direct(
        self,
        source_scale: Dict[str, Any],
        target_info: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Direct transfer - only works for exact dimension matches."""
        if (source_scale['input_dim'] == target_info['input_dim'] and
            source_scale['output_dim'] == target_info['output_dim']):
            
            self.transfer_stats[f"layer_{len(self.transfer_stats)}"] = 'direct_match'
            return source_scale.copy()
        
        return None
    
    def _transfer_adaptive(
        self,
        source_scale: Dict[str, Any],
        target_info: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Adaptive transfer - handles dimension mismatches."""
        source_dim = source_scale['input_dim']
        target_dim = target_info['input_dim']
        
        transferred = source_scale.copy()
        transferred['input_dim'] = target_dim
        transferred['output_dim'] = target_info['output_dim']
        
        if source_dim == target_dim:
            # Perfect match
            self.transfer_stats[f"layer_{len(self.transfer_stats)}"] = 'exact_match'
            return transferred
        
        # Handle dimension mismatch
        importance_scores = source_scale['importance_scores']
        
        if target_dim > source_dim:
            # Interpolate/extrapolate
            # Strategy: Repeat the importance pattern proportionally
            scale_ratio = target_dim / source_dim
            indices = torch.linspace(0, source_dim - 1, target_dim)
            transferred['importance_scores'] = torch.nn.functional.interpolate(
                importance_scores.unsqueeze(0).unsqueeze(0),
                size=target_dim,
                mode='linear',
                align_corners=True
            ).squeeze()
            
            # Adjust outlier indices proportionally
            outlier_indices = source_scale['outlier_indices']
            transferred['outlier_indices'] = (outlier_indices.float() * scale_ratio).long()
            transferred['outlier_indices'] = torch.clamp(
                transferred['outlier_indices'], 0, target_dim - 1
            )
            
            # Keep same protection ratio
            n_protect = max(1, int(target_dim * source_scale['protect_ratio']))
            transferred['n_protect'] = n_protect
            
            self.transfer_stats[f"layer_{len(self.transfer_stats)}"] = f'upscaled_{source_dim}_to_{target_dim}'
            
        else:
            # Downscale: Select top-k channels
            # Strategy: Keep the most important channels based on source importance
            n_protect = max(1, int(target_dim * source_scale['protect_ratio']))
            
            # Sample importance scores proportionally
            indices = torch.linspace(0, source_dim - 1, target_dim).long()
            transferred['importance_scores'] = importance_scores[indices]
            
            # Select top-k from downsampled importance
            _, top_indices = torch.topk(transferred['importance_scores'], n_protect)
            transferred['outlier_indices'] = top_indices
            transferred['n_protect'] = n_protect
            
            self.transfer_stats[f"layer_{len(self.transfer_stats)}"] = f'downscaled_{source_dim}_to_{target_dim}'
        
        return transferred
    
    def _transfer_ratio(
        self,
        source_scale: Dict[str, Any],
        target_info: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Transfer using protection ratio - dimension agnostic."""
        target_dim = target_info['input_dim']
        
        transferred = source_scale.copy()
        transferred['input_dim'] = target_dim
        transferred['output_dim'] = target_info['output_dim']
        
        # Apply same protection ratio to target dimension
        protect_ratio = source_scale['protect_ratio']
        n_protect = max(1, int(target_dim * protect_ratio))
        
        # Create uniform importance scores (no activation stats available)
        # This is a limitation of pure transfer without calibration
        transferred['importance_scores'] = torch.ones(target_dim)
        transferred['outlier_indices'] = torch.arange(n_protect)  # Protect first n channels
        transferred['n_protect'] = n_protect
        
        self.transfer_stats[f"layer_{len(self.transfer_stats)}"] = 'ratio_based'
        
        return transferred
    
    def _print_transfer_summary(self):
        """Print summary of transfer operations."""
        print(f"\nTransfer Summary:")
        print(f"  Total target layers: {len(self.transfer_stats)}")
        
        from collections import Counter
        stats_count = Counter(self.transfer_stats.values())
        for stat, count in stats_count.items():
            print(f"  - {stat}: {count} layers")


# ==============================================================================
# AWQ WITH TRANSFERRED SCALES
# ==============================================================================

@torch.no_grad()
def awq_quantize_with_transferred_scales(
    model: nn.Module,
    transferred_scales: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> None:
    """
    Apply AWQ quantization using pre-computed transferred scaling factors.
    
    Args:
        model: Model to quantize
        transferred_scales: Scaling factors transferred from source model
        verbose: Whether to print progress
    """
    if verbose:
        print("\nApplying AWQ quantization with transferred scales...")
    
    quantized_count = 0
    skipped_count = 0
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if name not in transferred_scales:
                # No transferred scales, skip or use default quantization
                skipped_count += 1
                continue
            
            scales = transferred_scales[name]
            
            # Extract parameters
            outlier_indices = scales['outlier_indices'].to(m.weight.device)
            scale_factor = scales['scale_factor']
            w_bit = scales['w_bit']
            q_group_size = scales['q_group_size']
            
            # Preserve original dtype
            orig_dtype = m.weight.data.dtype
            
            # Apply AWQ transformation
            # Scale up protected channels
            m.weight.data[:, outlier_indices] *= scale_factor
            
            # Quantize all weights
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size
            )
            
            # Scale back down protected channels
            m.weight.data[:, outlier_indices] /= scale_factor
            
            # Ensure dtype consistency
            m.weight.data = m.weight.data.to(orig_dtype)
            
            quantized_count += 1
    
    if verbose:
        print(f"  → Quantized {quantized_count} layers")
        print(f"  → Skipped {skipped_count} layers (no transferred scales)")


# ==============================================================================
# EXPERIMENT ORCHESTRATOR
# ==============================================================================

class TransferQuantizationExperiment:
    """Main experiment orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
        # Models
        self.source_model = None
        self.source_tokenizer = None
        self.target_model = None
        self.target_tokenizer = None
        
        # Data
        self.calib_samples = None
        self.test_dataset = None
        
        # Scaling factors
        self.extractor = ScalingFactorExtractor()
        self.transferred_scales = None
    
    def run_experiment(self):
        """Run complete experiment pipeline."""
        print("=" * 80)
        print("TRANSFER QUANTIZATION EXPERIMENT")
        print("=" * 80)
        print(f"Source Model: {self.config['source_model']}")
        print(f"Target Model: {self.config['target_model']}")
        print(f"Transfer Strategy: {self.config['transfer_strategy']}")
        print("=" * 80)
        
        # Phase 1: Setup
        self._load_datasets()
        
        # Phase 2: Source model quantization
        self._evaluate_source_model()
        
        # Phase 3: Target model evaluation
        self._evaluate_target_model()
        
        # Phase 4: Analysis
        self._analyze_results()
        
        # Phase 5: Save results
        self._save_results()
    
    def _load_datasets(self):
        """Load calibration and test datasets."""
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        
        # Load a simple tokenizer for dataset preparation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['source_model'])
        
        print("\nLoading calibration dataset...")
        self.calib_samples = get_calibration_dataset(
            tokenizer,
            self.config['calibration_dataset'],
            self.config.get('calibration_dataset_config', None),
            self.config['calibration_split'],
            n_samples=self.config['n_calibration_samples'],
            block_size=self.config['calibration_block_size'],
        )
        
        print("\nLoading test dataset...")
        self.test_dataset = get_test_dataset(
            tokenizer,
            self.config['test_dataset'],
            self.config.get('test_dataset_config', None),
            self.config['test_split'],
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        print("Datasets loaded successfully!")
    
    def _evaluate_source_model(self):
        """Evaluate source model (raw and quantized)."""
        print("\n" + "=" * 80)
        print("PHASE 1: SOURCE MODEL EVALUATION")
        print("=" * 80)
        
        # Load source model
        print(f"\nLoading source model: {self.config['source_model']}...")
        self.source_model, self.source_tokenizer = load_model_and_tokenizer(
            self.config['source_model'],
            device_map=self.config.get('device_map', 'auto'),
            torch_dtype=self.config.get('torch_dtype', 'float16'),
        )
        
        # Evaluate raw source model
        print("\n--- Raw Source Model ---")
        raw_source_ppl = evaluate_perplexity(
            self.source_model,
            self.source_tokenizer,
            self.test_dataset,
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        raw_source_size = get_model_size(self.source_model, data_width=16, group_size=-1)
        
        self.results['source_raw'] = {
            'perplexity': raw_source_ppl,
            'size_mb': raw_source_size / (8 * MiB),
        }
        
        print(f"\n✓ Source Raw - PPL: {raw_source_ppl:.2f}, Size: {raw_source_size / (8 * MiB):.2f} MB")
        
        # Collect activation statistics
        print("\n--- Collecting Activation Statistics ---")
        input_feat = get_calib_feat(
            self.source_model,
            self.source_tokenizer,
            self.calib_samples,
            verbose=True
        )
        
        # Extract scaling factors BEFORE quantization
        print("\n--- Extracting Scaling Factors ---")
        awq_config = self.config['awq_config']
        self.extractor.extract_from_awq(
            self.source_model,
            input_feat,
            w_bit=awq_config['w_bit'],
            q_group_size=awq_config['q_group_size'],
            protect_ratio=awq_config['protect_ratio'],
            scale_factor=awq_config['scale_factor'],
        )
        
        # Save scaling factors
        scales_path = self.config.get('scales_save_path', 'scaling_factors.json')
        self.extractor.save_to_file(scales_path)
        
        # Apply AWQ quantization
        print("\n--- Quantizing Source Model ---")
        awq_quantize_model_weight(
            self.source_model,
            w_bit=awq_config['w_bit'],
            q_group_size=awq_config['q_group_size'],
            input_feat=input_feat,
            protect_ratio=awq_config['protect_ratio'],
            scale_factor=awq_config['scale_factor'],
        )
        
        # Evaluate quantized source model
        print("\n--- Quantized Source Model ---")
        quant_source_ppl = evaluate_perplexity(
            self.source_model,
            self.source_tokenizer,
            self.test_dataset,
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        quant_source_size = get_model_size(
            self.source_model,
            data_width=awq_config['w_bit'],
            group_size=awq_config['q_group_size'],
        )
        
        self.results['source_quantized'] = {
            'perplexity': quant_source_ppl,
            'size_mb': quant_source_size / (8 * MiB),
            'degradation_%': (quant_source_ppl / raw_source_ppl - 1) * 100,
        }
        
        print(f"\n✓ Source Quantized - PPL: {quant_source_ppl:.2f}, "
              f"Size: {quant_source_size / (8 * MiB):.2f} MB, "
              f"Degradation: {self.results['source_quantized']['degradation_%']:.2f}%")
        
        # Cleanup source model
        print("\nUnloading source model...")
        unload_model(self.source_model)
        self.source_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _evaluate_target_model(self):
        """Evaluate target model with different quantization approaches."""
        print("\n" + "=" * 80)
        print("PHASE 2: TARGET MODEL EVALUATION")
        print("=" * 80)
        
        # Evaluate raw target model
        self._evaluate_raw_target()
        
        # Evaluate target with normal quantization
        self._evaluate_normal_quantized_target()
        
        # Evaluate target with transferred quantization
        self._evaluate_transferred_quantized_target()
    
    def _evaluate_raw_target(self):
        """Evaluate raw target model."""
        print("\n--- Raw Target Model ---")
        
        print(f"\nLoading target model: {self.config['target_model']}...")
        self.target_model, self.target_tokenizer = load_model_and_tokenizer(
            self.config['target_model'],
            device_map=self.config.get('device_map', 'auto'),
            torch_dtype=self.config.get('torch_dtype', 'float16'),
        )
        
        raw_target_ppl = evaluate_perplexity(
            self.target_model,
            self.target_tokenizer,
            self.test_dataset,
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        raw_target_size = get_model_size(self.target_model, data_width=16, group_size=-1)
        
        self.results['target_raw'] = {
            'perplexity': raw_target_ppl,
            'size_mb': raw_target_size / (8 * MiB),
        }
        
        print(f"\n✓ Target Raw - PPL: {raw_target_ppl:.2f}, Size: {raw_target_size / (8 * MiB):.2f} MB")
        
        # Cleanup
        unload_model(self.target_model)
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _evaluate_normal_quantized_target(self):
        """Evaluate target model with normal AWQ quantization."""
        print("\n--- Normal Quantized Target Model ---")
        
        # Reload target model
        print(f"\nLoading target model: {self.config['target_model']}...")
        self.target_model, self.target_tokenizer = load_model_and_tokenizer(
            self.config['target_model'],
            device_map=self.config.get('device_map', 'auto'),
            torch_dtype=self.config.get('torch_dtype', 'float16'),
        )
        
        # Collect activation statistics for target
        print("\nCollecting activation statistics for target model...")
        input_feat_target = get_calib_feat(
            self.target_model,
            self.target_tokenizer,
            self.calib_samples,
            verbose=True
        )
        
        # Apply normal AWQ quantization
        print("\nQuantizing target model with normal AWQ...")
        awq_config = self.config['awq_config']
        awq_quantize_model_weight(
            self.target_model,
            w_bit=awq_config['w_bit'],
            q_group_size=awq_config['q_group_size'],
            input_feat=input_feat_target,
            protect_ratio=awq_config['protect_ratio'],
            scale_factor=awq_config['scale_factor'],
        )
        
        # Evaluate
        normal_quant_ppl = evaluate_perplexity(
            self.target_model,
            self.target_tokenizer,
            self.test_dataset,
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        quant_size = get_model_size(
            self.target_model,
            data_width=awq_config['w_bit'],
            group_size=awq_config['q_group_size'],
        )
        
        self.results['target_normal_quantized'] = {
            'perplexity': normal_quant_ppl,
            'size_mb': quant_size / (8 * MiB),
            'degradation_%': (normal_quant_ppl / self.results['target_raw']['perplexity'] - 1) * 100,
        }
        
        print(f"\n✓ Target Normal Quantized - PPL: {normal_quant_ppl:.2f}, "
              f"Size: {quant_size / (8 * MiB):.2f} MB, "
              f"Degradation: {self.results['target_normal_quantized']['degradation_%']:.2f}%")
        
        # Cleanup
        unload_model(self.target_model)
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _evaluate_transferred_quantized_target(self):
        """Evaluate target model with transferred quantization."""
        print("\n--- Transferred Quantized Target Model ---")
        
        # Reload target model
        print(f"\nLoading target model: {self.config['target_model']}...")
        self.target_model, self.target_tokenizer = load_model_and_tokenizer(
            self.config['target_model'],
            device_map=self.config.get('device_map', 'auto'),
            torch_dtype=self.config.get('torch_dtype', 'float16'),
        )
        
        # Transfer scaling factors
        print("\nTransferring scaling factors from source to target...")
        transfer = ScalingFactorTransfer(self.extractor.layer_scales)
        self.transferred_scales = transfer.transfer_to_model(
            self.target_model,
            strategy=self.config['transfer_strategy']
        )
        
        # Apply transferred quantization
        print("\nQuantizing target model with transferred scales...")
        awq_quantize_with_transferred_scales(
            self.target_model,
            self.transferred_scales,
            verbose=True
        )
        
        # Evaluate
        transfer_quant_ppl = evaluate_perplexity(
            self.target_model,
            self.target_tokenizer,
            self.test_dataset,
            n_samples=self.config['n_test_samples'],
            block_size=self.config['test_block_size'],
        )
        
        awq_config = self.config['awq_config']
        quant_size = get_model_size(
            self.target_model,
            data_width=awq_config['w_bit'],
            group_size=awq_config['q_group_size'],
        )
        
        self.results['target_transferred_quantized'] = {
            'perplexity': transfer_quant_ppl,
            'size_mb': quant_size / (8 * MiB),
            'degradation_%': (transfer_quant_ppl / self.results['target_raw']['perplexity'] - 1) * 100,
        }
        
        print(f"\n✓ Target Transferred Quantized - PPL: {transfer_quant_ppl:.2f}, "
              f"Size: {quant_size / (8 * MiB):.2f} MB, "
              f"Degradation: {self.results['target_transferred_quantized']['degradation_%']:.2f}%")
        
        # Cleanup
        unload_model(self.target_model)
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _analyze_results(self):
        """Analyze and compare all results."""
        print("\n" + "=" * 80)
        print("ANALYSIS & COMPARISON")
        print("=" * 80)
        
        print("\n" + "-" * 80)
        print(f"{'Model Variant':<40} | {'PPL':>10} | {'Size (MB)':>12} | {'Degradation':>12}")
        print("-" * 80)
        
        # Source models
        print(f"{'Source (350M) - Raw':<40} | "
              f"{self.results['source_raw']['perplexity']:>10.2f} | "
              f"{self.results['source_raw']['size_mb']:>12.2f} | "
              f"{'—':>12}")
        
        print(f"{'Source (350M) - Quantized':<40} | "
              f"{self.results['source_quantized']['perplexity']:>10.2f} | "
              f"{self.results['source_quantized']['size_mb']:>12.2f} | "
              f"{self.results['source_quantized']['degradation_%']:>11.2f}%")
        
        print("-" * 80)
        
        # Target models
        print(f"{'Target (1B) - Raw':<40} | "
              f"{self.results['target_raw']['perplexity']:>10.2f} | "
              f"{self.results['target_raw']['size_mb']:>12.2f} | "
              f"{'—':>12}")
        
        print(f"{'Target (1B) - Normal Quantized':<40} | "
              f"{self.results['target_normal_quantized']['perplexity']:>10.2f} | "
              f"{self.results['target_normal_quantized']['size_mb']:>12.2f} | "
              f"{self.results['target_normal_quantized']['degradation_%']:>11.2f}%")
        
        print(f"{'Target (1B) - Transferred Quantized':<40} | "
              f"{self.results['target_transferred_quantized']['perplexity']:>10.2f} | "
              f"{self.results['target_transferred_quantized']['size_mb']:>12.2f} | "
              f"{self.results['target_transferred_quantized']['degradation_%']:>11.2f}%")
        
        print("-" * 80)
        
        # Key insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        
        normal_deg = self.results['target_normal_quantized']['degradation_%']
        transfer_deg = self.results['target_transferred_quantized']['degradation_%']
        transfer_gap = transfer_deg - normal_deg
        
        print(f"\n1. Quantization Impact on Source (350M):")
        print(f"   Degradation: {self.results['source_quantized']['degradation_%']:.2f}%")
        
        print(f"\n2. Quantization Impact on Target (1B):")
        print(f"   Normal Quantization: {normal_deg:.2f}%")
        print(f"   Transferred Quantization: {transfer_deg:.2f}%")
        
        print(f"\n3. Transfer Effectiveness:")
        print(f"   Gap between Normal and Transferred: {transfer_gap:+.2f}%")
        
        if abs(transfer_gap) < 5:
            effectiveness = "EXCELLENT"
            desc = "Transferred quantization performs nearly as well as normal quantization!"
        elif abs(transfer_gap) < 10:
            effectiveness = "GOOD"
            desc = "Transferred quantization shows promising results with acceptable degradation."
        elif abs(transfer_gap) < 20:
            effectiveness = "MODERATE"
            desc = "Transfer shows some benefit but significant gap remains."
        else:
            effectiveness = "LIMITED"
            desc = "Transfer approach shows limited effectiveness for this model pair."
        
        print(f"   Effectiveness: {effectiveness}")
        print(f"   → {desc}")
        
        print(f"\n4. Size Reduction:")
        source_compression = (1 - self.results['source_quantized']['size_mb'] / 
                            self.results['source_raw']['size_mb']) * 100
        target_compression = (1 - self.results['target_normal_quantized']['size_mb'] / 
                             self.results['target_raw']['size_mb']) * 100
        
        print(f"   Source: {source_compression:.1f}% reduction")
        print(f"   Target: {target_compression:.1f}% reduction")
        
        print("=" * 80)
    
    def _save_results(self):
        """Save results to JSON file."""
        output_path = self.config.get('results_save_path', 'transfer_experiment_results.json')
        
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def create_default_config() -> Dict[str, Any]:
    """Create default experiment configuration."""
    return {
        # Models
        'source_model': 'facebook/MobileLLM-350M',
        'target_model': 'facebook/MobileLLM-1B',
        
        # Transfer strategy: 'direct', 'adaptive', or 'ratio'
        'transfer_strategy': 'adaptive',
        
        # AWQ quantization config
        'awq_config': {
            'w_bit': 4,
            'q_group_size': 128,
            'protect_ratio': 0.01,
            'scale_factor': 2.0,
        },
        
        # Calibration dataset
        'calibration_dataset': 'mit-han-lab/pile-val-backup',
        'calibration_dataset_config': None,
        'calibration_split': 'validation',
        'n_calibration_samples': 128,
        'calibration_block_size': 512,
        
        # Test dataset
        'test_dataset': 'mit-han-lab/pile-val-backup',
        'test_dataset_config': None,
        'test_split': 'validation',
        'n_test_samples': 20,
        'test_block_size': 1024,
        
        # Model loading
        'device_map': 'auto',
        'torch_dtype': 'float16',
        
        # Output paths
        'scales_save_path': 'scaling_factors_350m.json',
        'results_save_path': 'transfer_experiment_results.json',
    }


def main():
    """Main entry point for the experiment."""
    import sys
    
    # Load config from file if provided, otherwise use defaults
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Using default configuration")
        config = create_default_config()
        
        # Save default config for reference
        with open('transfer_config_default.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Default config saved to: transfer_config_default.json")
    
    # Run experiment
    experiment = TransferQuantizationExperiment(config)
    experiment.run_experiment()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
