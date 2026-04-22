from torch.utils.flop_counter import FlopCounterMode
import torch
from ultralytics import YOLO

# model = YOLO('yolo11n-dwwtconv-dywtattn.yaml').model
# model = YOLO('yolo11n-dywtattn.yaml').model
# model = YOLO('yolo11n-dwwtconv.yaml').model
# model = YOLO('yolo11n-dywt-moe.yaml').model
# model = YOLO('yolo11n-moe-v7.yaml').model
# model = YOLO('yolo11n-wtconv-ternary-hipa.yaml').model
# model = YOLO('yolo11n-wavelet-hipa.yaml').model
# model = YOLO('yolo11n-wavelet-hipa-dswtg.yaml').model
model = YOLO('yolo11n-wavelet-hipa-dyc3k2.yaml').model
# model = YOLO('yolo11n-dwwtconv-dyhead.yaml').model
# model = YOLO('yolo11n.yaml').model
model.eval()
dummy = torch.randn(1, 3, 640, 640).to(next(model.parameters()).device)

with FlopCounterMode(model, display=False) as flop_counter:
    model(dummy)

flops = flop_counter.get_total_flops()
print(f"GFLOPs (approx): {flops/1e9:.2f}")