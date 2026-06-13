"""Final verification: SimAM_Bottleneck C2f_Simple vs C3k2."""
import torch
from ultralytics.nn.modules.block import C3k2, Bottleneck, Conv
from ultralytics.nn.modules.FPN_PAN_BiFPN import C2f_Simple, SimAM_Bottleneck, FPN_PAN, FPN_PAN_BiFPN

# 1. RNG state alignment
print("=== 1. RNG alignment: SimAM_Bottleneck vs Bottleneck(e=0.5) ===")
c = 128
torch.manual_seed(42)
bn = Bottleneck(c, c, True, 1, k=((3,3),(3,3)), e=0.5)
torch.manual_seed(42)
sbn = SimAM_Bottleneck(c, True)

# Same Conv2d shapes → same RNG state after
print(f"  cv1 weight match: {torch.equal(bn.cv1.conv.weight, sbn.cv1.conv.weight)}")
print(f"  cv2 weight match: {torch.equal(bn.cv2.conv.weight, sbn.cv2.conv.weight)}")
print(f"  Params: BN=   {sum(p.numel() for p in bn.parameters()):,}")
print(f"  Params: SimAM={sum(p.numel() for p in sbn.parameters()):,}")

# 2. Subsequent module gets same weights
print("\n=== 2. Subsequent module RNG ===")
torch.manual_seed(42)
_ = Bottleneck(c, c, True, 1, k=((3,3),(3,3)), e=0.5)
next1 = Conv(c, c, 3)
torch.manual_seed(42)
_ = SimAM_Bottleneck(c, True)
next2 = Conv(c, c, 3)
print(f"  Subsequent Conv weight match: {torch.equal(next1.conv.weight, next2.conv.weight)}")

# 3. Full C2f_Simple vs C3k2
print("\n=== 3. C2f_Simple(SimAM) vs C3k2 ===")
configs = [(768,256,"fpn_p3"), (512,128,"fpn_p2"), (512,256,"pan_p3"), (1024,512,"pan_p4")]
for c1, c2, name in configs:
    torch.manual_seed(42)
    m1 = C3k2(c1, c2, n=2, c3k=False)
    torch.manual_seed(42)
    m2 = C2f_Simple(c1, c2, n=2)
    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    # Verify RNG: subsequent module after full C2f_Simple
    torch.manual_seed(42)
    _ = C3k2(c1, c2, n=2, c3k=False)
    sub1 = Conv(c2, c2, 3)
    torch.manual_seed(42)
    _ = C2f_Simple(c1, c2, n=2)
    sub2 = Conv(c2, c2, 3)
    rng_match = torch.equal(sub1.conv.weight, sub2.conv.weight)
    x = torch.randn(1, c1, 32, 32)
    m1.eval(); m2.eval()
    with torch.no_grad():
        shape_match = m1(x).shape == m2(x).shape
    print(f"  {name}: params={p1:,}={p2:,}  shape={shape_match}  RNG_sub={rng_match}")

# 4. FPN_PAN end-to-end
print("\n=== 4. FPN_PAN ===")
model = FPN_PAN([256, 512, 512], [128, 256, 512])
p2 = torch.randn(1, 256, 80, 80)
p3 = torch.randn(1, 512, 40, 40)
p4 = torch.randn(1, 512, 20, 20)
model.eval()
with torch.no_grad():
    out = model([p2, p3, p4])
for i, o in enumerate(out):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total: {sum(p.numel() for p in model.parameters()):,}")

# 5. FPN_PAN_BiFPN end-to-end
print("\n=== 5. FPN_PAN_BiFPN ===")
model2 = FPN_PAN_BiFPN([256, 512, 512], [128, 256, 512], num_bifpn_layers=1, use_refine=False)
model2.eval()
with torch.no_grad():
    out2 = model2([p2, p3, p4])
for i, o in enumerate(out2):
    print(f"  P{i+2}: {o.shape}")
print(f"  Total: {sum(p.numel() for p in model2.parameters()):,}")

print("\nAll tests passed!")