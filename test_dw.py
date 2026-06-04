"""Test script to verify DWC3k2 full model builds correctly."""
import sys
sys.path.insert(0, '.')

import torch
import yaml
from ultralytics.nn.tasks import parse_model
from ultralytics.utils.torch_utils import get_num_params, get_num_gradients


def count_flops_manual(model, input_shape=(1, 3, 640, 640)):
    """Manual FLOPs calculation using fvcore or simple method."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, x)
        return flops.total() / 1e9
    except ImportError:
        return None


def test_model(yaml_path, scale='s'):
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    d['scale'] = scale
    model, save = parse_model(d, ch=3, verbose=True)
    model.eval()
    
    total_params = get_num_params(model)
    total_grads = get_num_gradients(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {yaml_path}")
    print(f"Scale: {scale}")
    print(f"Layers: {len(model)}")
    print(f"Parameters: {total_params:,}")
    print(f"Gradients: {total_grads:,}")
    
    # Try FLOPs
    gflops = count_flops_manual(model)
    if gflops:
        print(f"GFLOPs: {gflops:.1f}")
    
    print(f"{'='*60}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        try:
            y = model(x)
            print("Forward pass: OK")
            if isinstance(y, (list, tuple)):
                for i, yi in enumerate(y):
                    print(f"  Det head {i}: {yi.shape}")
            else:
                print(f"  Output: {y.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
    
    return model


if __name__ == '__main__':
    print("=== YOLO11-test-DW (DWC3k2 neck) ===")
    test_model('ultralytics/cfg/models/11/yolo11-test-dw.yaml', 's')
    
    print("\n=== YOLO11-test (original C3k2 neck) ===")
    test_model('ultralytics/cfg/models/11/yolo11-test.yaml', 's')