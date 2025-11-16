"""
Quick test script for transfer quantization experiment.

This tests the core components without loading large models,
useful for debugging and validation.
"""

import torch
import torch.nn as nn
from transfer_quantization_experiment import (
    ScalingFactorExtractor,
    ScalingFactorTransfer,
    awq_quantize_with_transferred_scales
)


def create_dummy_model(hidden_dim: int = 512) -> nn.Module:
    """Create a simple dummy model for testing."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 256),
    )


def test_scaling_factor_extraction():
    """Test scaling factor extraction."""
    print("=" * 80)
    print("TEST 1: Scaling Factor Extraction")
    print("=" * 80)
    
    # Create dummy model
    model = create_dummy_model(hidden_dim=512)
    
    # Create dummy activation features
    input_feat = {}
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Simulate activation statistics
            input_dim = module.weight.shape[1]
            input_feat[str(i)] = [torch.randn(input_dim) for _ in range(10)]
    
    # Extract scaling factors
    extractor = ScalingFactorExtractor()
    extractor.extract_from_awq(
        model,
        input_feat,
        w_bit=4,
        q_group_size=128,
        protect_ratio=0.01,
        scale_factor=2.0
    )
    
    # Verify extraction
    assert len(extractor.layer_scales) > 0, "No scales extracted!"
    
    for name, scales in extractor.layer_scales.items():
        assert 'importance_scores' in scales
        assert 'outlier_indices' in scales
        assert 'n_protect' in scales
        print(f"  ✓ Layer {name}: {scales['n_protect']} protected channels")
    
    # Test save/load
    extractor.save_to_file('test_scales.json')
    
    new_extractor = ScalingFactorExtractor()
    new_extractor.load_from_file('test_scales.json')
    
    assert len(new_extractor.layer_scales) == len(extractor.layer_scales)
    print(f"\n✓ Successfully extracted and saved {len(extractor.layer_scales)} layer scales")
    
    return extractor


def test_scaling_factor_transfer():
    """Test scaling factor transfer."""
    print("\n" + "=" * 80)
    print("TEST 2: Scaling Factor Transfer")
    print("=" * 80)
    
    # Create source model (smaller)
    source_model = create_dummy_model(hidden_dim=512)
    
    # Create target model (larger)
    target_model = create_dummy_model(hidden_dim=768)
    
    # Extract from source
    input_feat = {}
    for i, (name, module) in enumerate(source_model.named_modules()):
        if isinstance(module, nn.Linear):
            input_dim = module.weight.shape[1]
            input_feat[str(i)] = [torch.randn(input_dim) for _ in range(10)]
    
    extractor = ScalingFactorExtractor()
    extractor.extract_from_awq(
        source_model,
        input_feat,
        w_bit=4,
        q_group_size=128,
        protect_ratio=0.01,
        scale_factor=2.0
    )
    
    # Test different transfer strategies
    strategies = ['direct', 'adaptive', 'ratio']
    
    for strategy in strategies:
        print(f"\n--- Testing '{strategy}' strategy ---")
        
        transfer = ScalingFactorTransfer(extractor.layer_scales)
        transferred_scales = transfer.transfer_to_model(target_model, strategy=strategy)
        
        print(f"  Transferred scales: {len(transferred_scales)} layers")
        
        if strategy == 'adaptive':
            # Adaptive should transfer to all layers
            assert len(transferred_scales) > 0, f"No scales transferred with {strategy}!"
    
    print(f"\n✓ All transfer strategies tested successfully")


def test_quantization_with_transfer():
    """Test quantization with transferred scales."""
    print("\n" + "=" * 80)
    print("TEST 3: Quantization with Transferred Scales")
    print("=" * 80)
    
    # Create models
    source_model = create_dummy_model(hidden_dim=512)
    target_model = create_dummy_model(hidden_dim=512)  # Same size for direct transfer
    
    # Extract from source
    input_feat = {}
    for i, (name, module) in enumerate(source_model.named_modules()):
        if isinstance(module, nn.Linear):
            input_dim = module.weight.shape[1]
            input_feat[str(i)] = [torch.randn(input_dim) for _ in range(10)]
    
    extractor = ScalingFactorExtractor()
    extractor.extract_from_awq(
        source_model,
        input_feat,
        w_bit=4,
        q_group_size=128,
        protect_ratio=0.01,
        scale_factor=2.0
    )
    
    # Transfer to target
    transfer = ScalingFactorTransfer(extractor.layer_scales)
    transferred_scales = transfer.transfer_to_model(target_model, strategy='adaptive')
    
    # Save original weights
    original_weights = {}
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear):
            original_weights[name] = module.weight.data.clone()
    
    # Apply quantization
    awq_quantize_with_transferred_scales(target_model, transferred_scales, verbose=True)
    
    # Verify weights changed
    changed_count = 0
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear) and name in original_weights:
            if not torch.allclose(module.weight.data, original_weights[name]):
                changed_count += 1
    
    assert changed_count > 0, "No weights were quantized!"
    print(f"\n✓ Quantized {changed_count} layers successfully")
    
    # Verify no NaNs or Infs
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear):
            assert not torch.isnan(module.weight).any(), f"NaN in {name}"
            assert not torch.isinf(module.weight).any(), f"Inf in {name}"
    
    print(f"✓ All weights are valid (no NaN/Inf)")


def test_dimension_mismatch_handling():
    """Test handling of dimension mismatches."""
    print("\n" + "=" * 80)
    print("TEST 4: Dimension Mismatch Handling")
    print("=" * 80)
    
    # Create models with different dimensions
    source_model = create_dummy_model(hidden_dim=256)  # Small
    target_model = create_dummy_model(hidden_dim=1024)  # Large
    
    # Extract from source
    input_feat = {}
    for i, (name, module) in enumerate(source_model.named_modules()):
        if isinstance(module, nn.Linear):
            input_dim = module.weight.shape[1]
            input_feat[str(i)] = [torch.randn(input_dim) for _ in range(10)]
    
    extractor = ScalingFactorExtractor()
    extractor.extract_from_awq(
        source_model,
        input_feat,
        w_bit=4,
        q_group_size=128,
        protect_ratio=0.01,
        scale_factor=2.0
    )
    
    # Transfer with adaptive strategy
    transfer = ScalingFactorTransfer(extractor.layer_scales)
    transferred_scales = transfer.transfer_to_model(target_model, strategy='adaptive')
    
    print(f"\n  Source model size: 256 hidden dim")
    print(f"  Target model size: 1024 hidden dim")
    print(f"  Transferred scales: {len(transferred_scales)} layers")
    
    # Verify dimensions were adjusted
    for name, scales in transferred_scales.items():
        print(f"  Layer {name}: {scales['input_dim']} input dim, {scales['n_protect']} protected")
        # Check that dimensions match target model
        for target_name, module in target_model.named_modules():
            if target_name == name and isinstance(module, nn.Linear):
                assert scales['input_dim'] == module.weight.shape[1], \
                    f"Dimension mismatch for {name}"
    
    print(f"\n✓ Dimension mismatches handled correctly")


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\n" + "=" * 80)
    print("TEST 5: End-to-End Workflow")
    print("=" * 80)
    
    # 1. Create source model and extract
    print("\n1. Extracting from source model...")
    source_model = create_dummy_model(hidden_dim=512)
    
    input_feat = {}
    for i, (name, module) in enumerate(source_model.named_modules()):
        if isinstance(module, nn.Linear):
            input_dim = module.weight.shape[1]
            input_feat[str(i)] = [torch.randn(input_dim) for _ in range(10)]
    
    extractor = ScalingFactorExtractor()
    extractor.extract_from_awq(
        source_model,
        input_feat,
        w_bit=4,
        q_group_size=128,
        protect_ratio=0.01,
        scale_factor=2.0
    )
    
    # 2. Save scales
    print("\n2. Saving scaling factors...")
    extractor.save_to_file('test_end_to_end_scales.json')
    
    # 3. Load scales (simulating new session)
    print("\n3. Loading scaling factors...")
    new_extractor = ScalingFactorExtractor()
    new_extractor.load_from_file('test_end_to_end_scales.json')
    
    # 4. Create target model
    print("\n4. Creating target model...")
    target_model = create_dummy_model(hidden_dim=768)
    
    # 5. Transfer scales
    print("\n5. Transferring scales...")
    transfer = ScalingFactorTransfer(new_extractor.layer_scales)
    transferred_scales = transfer.transfer_to_model(target_model, strategy='adaptive')
    
    # 6. Quantize target
    print("\n6. Quantizing target model...")
    awq_quantize_with_transferred_scales(target_model, transferred_scales, verbose=False)
    
    # 7. Verify
    print("\n7. Verifying quantization...")
    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear):
            assert not torch.isnan(module.weight).any()
            assert not torch.isinf(module.weight).any()
    
    print(f"\n✓ End-to-end workflow completed successfully!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRANSFER QUANTIZATION EXPERIMENT - UNIT TESTS")
    print("=" * 80)
    
    try:
        test_scaling_factor_extraction()
        test_scaling_factor_transfer()
        test_quantization_with_transfer()
        test_dimension_mismatch_handling()
        test_end_to_end()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe transfer quantization experiment is ready to use.")
        print("Run: python transfer_quantization_experiment.py")
        
        return True
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print("TEST FAILED! ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import os
    
    # Set device to CPU for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    success = run_all_tests()
    
    # Cleanup test files
    import glob
    for f in glob.glob('test_*.json'):
        try:
            os.remove(f)
            print(f"Cleaned up: {f}")
        except:
            pass
    
    exit(0 if success else 1)
