#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for DeformC3k2 module."""

import sys
import torch

def test_deform_modules():
    """Test all Deform modules."""
    print("=" * 60)
    print("Testing DeformC3k2 Module")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.DeformC3k2 import (
            DeformC3k2, DeformBottleneck, DeformConv2d, DeformC3k, DeformC3k2Block
        )
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Basic DeformC3k2
    print("\n1. Basic DeformC3k2 Test")
    try:
        x = torch.randn(1, 64, 64, 64).to(device)
        model = DeformC3k2(64, 128, n=1, c3k=False, e=0.5, shortcut=True, kernel_size=3).to(device)
        y = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 2: DeformC3k2 with c3k=True
    print("\n2. DeformC3k2 with c3k=True")
    try:
        model = DeformC3k2(64, 128, n=2, c3k=True, e=0.5, shortcut=True, kernel_size=3).to(device)
        y = model(x)
        print(f"   Input: {x.shape} → Output: {y.shape}")
        assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 3: DeformC3k2 with Attention
    print("\n3. DeformC3k2 with Attention")
    try:
        model = DeformC3k2(64, 128, n=1, c3k=False, attn=True, e=0.5, shortcut=True, kernel_size=3).to(device)
        y = model(x)
        print(f"   Input: {x.shape} → Output: {y.shape}")
        assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 4: DeformBottleneck Standalone
    print("\n4. DeformBottleneck Standalone")
    try:
        model = DeformBottleneck(64, 64, shortcut=True, kernel_size=3).to(device)
        y = model(x)
        print(f"   Input: {x.shape} → Output: {y.shape}")
        assert y.shape == (1, 64, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 5: DeformC3k2 with Different Kernel Sizes
    print("\n5. DeformC3k2 with Different Kernel Sizes")
    for ks in [3, 5, 7]:
        try:
            model = DeformC3k2(64, 128, n=1, kernel_size=ks).to(device)
            y = model(x)
            print(f"   kernel_size={ks}: Input {x.shape} → Output {y.shape}")
            assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        except Exception as e:
            print(f"   ❌ kernel_size={ks} failed: {e}")
            return False
    print("   ✅ All kernel sizes OK")
    
    # Test 6: Multi-Block DeformC3k2
    print("\n6. Multi-Block DeformC3k2")
    try:
        model = DeformC3k2(64, 128, n=3, kernel_size=3).to(device)
        y = model(x)
        print(f"   Input: {x.shape} → Output: {y.shape}")
        assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 7: DeformC3k2Block with Coord Attention
    print("\n7. DeformC3k2Block with Coord Attention")
    try:
        model = DeformC3k2Block(64, 128, n=2, use_coord_attn=True).to(device)
        y = model(x)
        print(f"   Input: {x.shape} → Output: {y.shape}")
        assert y.shape == (1, 128, 64, 64), "Output shape mismatch"
        loss = y.mean()
        loss.backward()
        print("   ✅ Backward pass OK")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_deform_modules()
    sys.exit(0 if success else 1)
