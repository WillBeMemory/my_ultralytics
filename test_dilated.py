"""Verify DilatedBottleneck C2f_Simple works correctly."""
import torch
from ultralytics.nn.modules.FPN_PAN_BiFPN import C2f_Simple, FPN_PAN_BiFPN

configs = [
    (768, 256, 2, "fpn_p3"),
    (512, 128, 2, "fpn_p2"),
    (512, 256, 2, "pan_p3"),
    (1024, 512, 2, "pan_p4"),
]

print("C2f_Simple (DilatedBottleneck) verification:")
total = 0
for c1, c2, n, name in configs:
    m = C2f_Simple(c1, c2, n=n)
    x = torch.randn(1, c1, 32, 32)
    m.eval()
    with torch.no_grad():
        y = m(x)
    p = sum(p.numel() for p in m.parameters())
    total += p
    print(f"  {name}: {x.shape} -> {y.shape}, params={p:,}")
print(f"  TOTAL: {total:,} (baseline: 3,587,968)")

# Full FPN_PAN_BiFPN
print("\nFPN_PAN_BiFPN end-to-end:")
model = FPN_PAN_BiFPN([256, 512, 512], [128, 256, 512], num_bifpn_layers=1, use_refine=False)
p2 = torch.randn(1, 256, 80, 80)
p3 = torch.randn(1, 512, 40, 40)
p4 = torch.randn(1, 512, 20, 20)
model.eval()
with torch.no_grad():
    out = model([p2, p3, p4])
for i, o in enumerate(out):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
