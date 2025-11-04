# coding=utf-8
# Unit tests for utils_quant.py
# Test backward compatibility: single-bit, no-noise configuration should behave identically

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the current directory to the path to import utils_quant
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_quant import QuantizeLinear, LsqBinaryTernaryExtension, StretchedElasticQuant


def test_single_bit_no_noise_backward_compatibility():
    """Test that single-bit, no-noise configuration behaves identically to original implementation."""
    print("Testing single-bit, no-noise backward compatibility...")
    
    # Create two identical layers: one with old API (default), one explicitly
    torch.manual_seed(42)
    layer_old = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=1,
        weight_layerwise=False,
        bias=False
    )
    
    torch.manual_seed(42)
    layer_new = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=1,
        weight_layerwise=False,
        bias=False,
        noise_injection=False,  # Explicitly disable noise
        w_bits_list=None  # Use single w_bits
    )
    
    # Copy weights to ensure they're identical
    with torch.no_grad():
        layer_new.weight.copy_(layer_old.weight)
        if hasattr(layer_old, 'weight_clip_val') and layer_old.weight_clip_val is not None:
            if isinstance(layer_old.weight_clip_val, nn.Parameter):
                layer_new.weight_clip_val.copy_(layer_old.weight_clip_val)
            else:
                layer_new.weight_clip_val = layer_old.weight_clip_val.clone()
    
    # Create test input
    torch.manual_seed(123)
    input_tensor = torch.randn(3, 10)
    
    # Forward pass
    output_old = layer_old(input_tensor)
    output_new = layer_new(input_tensor)
    
    # Check outputs are identical
    assert torch.allclose(output_old, output_new, atol=1e-6), \
        "Outputs differ in single-bit, no-noise configuration!"
    print("✓ Single-bit, no-noise backward compatibility test passed")
    
    # Test backward pass
    loss_old = output_old.sum()
    loss_new = output_new.sum()
    
    loss_old.backward()
    loss_new.backward()
    
    # Check gradients are identical
    assert torch.allclose(layer_old.weight.grad, layer_new.weight.grad, atol=1e-6), \
        "Weight gradients differ in single-bit, no-noise configuration!"
    if hasattr(layer_old, 'weight_clip_val') and isinstance(layer_old.weight_clip_val, nn.Parameter):
        assert torch.allclose(layer_old.weight_clip_val.grad, layer_new.weight_clip_val.grad, atol=1e-6), \
            "Clip value gradients differ in single-bit, no-noise configuration!"
    print("✓ Backward pass test passed")


def test_multi_bit_functionality():
    """Test multi-bit training functionality."""
    print("\nTesting multi-bit functionality...")
    
    layer = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits_list=[1, 2, 4],
        weight_layerwise=False,
        bias=False
    )
    
    # Test set_bits
    layer.set_bits(2)
    assert layer.cur_w_bits == 2, "set_bits failed!"
    print("✓ set_bits works correctly")
    
    # Test forward with different bit widths
    input_tensor = torch.randn(3, 10)
    
    layer.set_bits(1)
    output_1bit = layer(input_tensor)
    
    layer.set_bits(2)
    output_2bit = layer(input_tensor)
    
    layer.set_bits(4)
    output_4bit = layer(input_tensor)
    
    # All should produce valid outputs
    assert output_1bit.shape == (3, 5), "1-bit output shape incorrect!"
    assert output_2bit.shape == (3, 5), "2-bit output shape incorrect!"
    assert output_4bit.shape == (3, 5), "4-bit output shape incorrect!"
    print("✓ Multi-bit forward passes work correctly")


def test_noise_injection():
    """Test noise injection functionality."""
    print("\nTesting noise injection...")
    
    # Test pre-quantization noise
    layer_pre = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=2,
        weight_layerwise=False,
        bias=False,
        noise_injection=True,
        pre_quantization_noise=True,
        noise_sigma_weights=0.1,
        noise_sigma_clipvals=0.0  # Disable clip value noise for simpler test
    )
    
    # Test post-quantization noise
    layer_post = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=2,
        weight_layerwise=False,
        bias=False,
        noise_injection=True,
        post_quantization_noise=True,
        noise_sigma_weights=0.1,
        noise_sigma_clipvals=0.0
    )
    
    input_tensor = torch.randn(3, 10)
    
    # Both should produce outputs
    output_pre = layer_pre(input_tensor)
    output_post = layer_post(input_tensor)
    
    assert output_pre.shape == (3, 5), "Pre-quantization noise output shape incorrect!"
    assert output_post.shape == (3, 5), "Post-quantization noise output shape incorrect!"
    print("✓ Noise injection works correctly")


def test_noise_disabled_by_default():
    """Test that noise is disabled by default (backward compatibility)."""
    print("\nTesting noise disabled by default...")
    
    layer = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=2,
        weight_layerwise=False,
        bias=False
    )
    
    assert layer.noise_injection == False, "Noise should be disabled by default!"
    assert layer.pre_quantization_noise == False, "Pre-quantization noise should be disabled by default!"
    assert layer.post_quantization_noise == False, "Post-quantization noise should be disabled by default!"
    print("✓ Noise is disabled by default")


def test_quantization_functions():
    """Test that quantization functions work correctly."""
    print("\nTesting quantization functions...")
    
    # Test LsqBinaryTernaryExtension
    input_tensor = torch.randn(5, 10)
    alpha = nn.Parameter(torch.randn(5, 1))
    num_bits = 4
    layerwise = False
    
    output = LsqBinaryTernaryExtension.apply(input_tensor, alpha, num_bits, layerwise)
    assert output.shape == input_tensor.shape, "LsqBinaryTernaryExtension output shape incorrect!"
    print("✓ LsqBinaryTernaryExtension works correctly")
    
    # Test StretchedElasticQuant
    output = StretchedElasticQuant.apply(input_tensor, alpha, num_bits, layerwise)
    assert output.shape == input_tensor.shape, "StretchedElasticQuant output shape incorrect!"
    print("✓ StretchedElasticQuant works correctly")


def test_different_bit_widths():
    """Test different bit widths work correctly."""
    print("\nTesting different bit widths...")
    
    input_tensor = torch.randn(3, 10)
    
    for w_bits in [1, 2, 4]:
        layer = QuantizeLinear(
            in_features=10,
            out_features=5,
            w_bits=w_bits,
            weight_layerwise=False,
            bias=False
        )
        
        output = layer(input_tensor)
        assert output.shape == (3, 5), f"{w_bits}-bit output shape incorrect!"
        print(f"✓ {w_bits}-bit quantization works correctly")


def test_weight_clip_val_initialization():
    """Test weight clip value initialization."""
    print("\nTesting weight clip value initialization...")
    
    # Test single bit width
    layer = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=2,
        weight_layerwise=False,
        bias=False
    )
    
    assert hasattr(layer, 'weight_clip_val'), "weight_clip_val should exist for w_bits=2!"
    assert layer.weight_clip_val is not None, "weight_clip_val should not be None!"
    print("✓ Single bit width clip value initialization works")
    
    # Test multi-bit with shared clip values
    layer_multi = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits_list=[1, 2],
        multiple_bits_share_clipvals=True,
        weight_layerwise=False,
        bias=False
    )
    
    assert hasattr(layer_multi, 'weight_clip_val'), "weight_clip_val should exist with shared clipvals!"
    print("✓ Multi-bit shared clip values initialization works")
    
    # Test 16-bit (no clip value needed)
    layer_16bit = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits=16,
        weight_layerwise=False,
        bias=False
    )
    
    assert layer_16bit.weight_clip_val is None, "weight_clip_val should be None for 16-bit!"
    print("✓ 16-bit (no clip value) initialization works")


def test_random_bit_assignment():
    """Test random bit assignment functionality."""
    print("\nTesting random bit assignment...")
    
    layer = QuantizeLinear(
        in_features=10,
        out_features=5,
        w_bits_list=[1, 2, 4],
        multiple_bits_random_assign=True,
        multiple_bits_random_assign_prob=1.0,  # Always random
        weight_layerwise=False,
        bias=False
    )
    
    input_tensor = torch.randn(3, 10)
    
    # Run multiple forward passes - should use different bit widths
    outputs = []
    for _ in range(10):
        output = layer(input_tensor)
        outputs.append(output)
        assert output.shape == (3, 5), "Random assignment output shape incorrect!"
    
    print("✓ Random bit assignment works correctly")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running unit tests for utils_quant.py")
    print("=" * 60)
    
    try:
        test_single_bit_no_noise_backward_compatibility()
        test_multi_bit_functionality()
        test_noise_injection()
        test_noise_disabled_by_default()
        test_quantization_functions()
        test_different_bit_widths()
        test_weight_clip_val_initialization()
        test_random_bit_assignment()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

