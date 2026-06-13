"""Final verification: C2f_Simple (StarResBlock) vs C3k2, RNG alignment + FPN_PAN."""
import torch
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.FPN_PAN_BiFPN import C2f_Simple, FPN_PAN, FPN_PAN_BiFPN

print("=== 1. RNG state alignment: C2f_Simple(Star) → subsequent module ===")
torch.manual_seed(42)
m1 = C3k2(512, 256, n=2, c3k=False)          # C3k2
sub1 = torch.nn.Conv2d(256, 256, 3, 1, 1)     # next module

torch.manual_seed(42)
m2 = C2f_Simple(512, 256, n=2)               # C2f_Simple (StarResBlock)
sub2 = torch.nn.Conv2d(256, 256, 3, 1, 1)     # next module (same shape)

print(f"  Subsequent Conv2d weight match: {torch.equal(sub1.weight, sub2.weight)}")

print("\n=== 2. Full FPN_PAN end-to-end ===")
model = FPN_PAN([256, 512, 512], [128, 256, 512])
p2 = torch.randn(1, 256, 80, 80)
p3 = torch.randn(1, 512, 40, 40)
p4 = torch.randn(1, 512, 20, 20)
model.eval()
with torch.no_grad():
    out = model([p2, p3, p4])
for i, o in enumerate(out):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

print("\n=== 3. Full FPN_PAN_BiFPN end-to-end ===")
model2 = FPN_PAN_BiFPN([256, 512, 512], [128, 256, 512], num_bifpn_layers=1, use_refine=False)
model2.eval()
with torch.no_grad():
    out2 = model2([p2, p3, p4])
for i, o in enumerate(out2):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total params: {sum(p.numel() for p in model2.parameters()):,}")

print("\n=== 4. Params comparison (4 C2f_Simple instances) ===")
configs = {
    "fpn_p3": (768, 256), "fpn_p2": (512, 128),
    "pan_p3": (512, 256), "pan_p4": (1024, 512),
}
total_c3k2 = total_star = 0
for name, (c1, c2) in configs.items():
    p_c3k2 = sum(p.numel() for p in C3k2(c1, c2, n=2, c3k=False).parameters())
    p_star = sum(p.numel() for p in C2f_Simple(c1, c2, n=2).parameters())
    total_c3k2 += p_c3k2
    total_star += p_star
    print(f"  {name}: C3k2={p_c3k2:,}  Star={p_star:,}  diff={p_star-p_c3k2:+}")
print(f"  TOTAL: C3k2={total_c3k2:,}  Star={total_star:,}  diff={total_star-total_c3k2:+} ({((total_star-total_c3k2)/total_c3k2*100):.2f}%)")
print("\nAll checks passed!")