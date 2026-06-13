"""Verify StarResBlock RNG equivalence to Bottleneck(e=0.5)."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, Bottleneck, C3k2

# ====== StarResBlock: StarNet (CVPR 2024) inspired ======
class StarResBlock(nn.Module):
    """StarNet-inspired block. Split channels → parallel 3×3 conv → element-wise multiply → project.

    Star operation (element-wise multiplication) generates implicit high-dimensional
    non-linear feature space without widening the network (CVPR 2024).

    RNG consumption: 3 × Conv2d = (c/2)²×9 + (c/2)²×9 + (c/2)×c×9 = c²×9 ≡ Bottleneck(e=0.5)
    Params: c²×9 + 4c (vs Bottleneck(e=0.5)'s c²×9 + 3c, +c negligible)
    """
    def __init__(self, c, shortcut=True):
        super().__init__()
        c_half = c // 2
        self.cv_a = Conv(c_half, c_half, 3)    # RNG: c²/4 × 9
        self.cv_b = Conv(c_half, c_half, 3)    # RNG: c²/4 × 9
        self.cv_out = Conv(c_half, c, 3)       # RNG: c²/2 × 9
        self.add = shortcut

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        out = self.cv_out(self.cv_a(a) * self.cv_b(b))
        return x + out if self.add else out


# ====== C2f_Simple with StarResBlock ======
class C2f_Simple_Star(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # RNG alignment: C3k2's C2f.__init__ creates n×Bottleneck(e=1.0)
        for _ in range(n):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m = nn.ModuleList(
            StarResBlock(self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ====== Test: RNG state after creating block ======
c = 128
print(f"=== RNG consumption per block (c={c}) ===")

# Bottleneck(e=0.5)
torch.manual_seed(42)
bn = Bottleneck(c, c, True, 1, k=((3, 3), (3, 3)), e=0.5)
rng_after_bn = torch.get_rng_state()
print(f"Bottleneck(e=0.5): params={sum(p.numel() for p in bn.parameters()):,}")

# StarResBlock  
torch.manual_seed(42)
sr = StarResBlock(c, True)
rng_after_sr = torch.get_rng_state()
print(f"StarResBlock:      params={sum(p.numel() for p in sr.parameters()):,}")

# RNG state after both should be identical
print(f"RNG state match: {torch.equal(rng_after_bn, rng_after_sr)}")

# ====== Test: subsequent module gets same random numbers ======
print(f"\n=== Subsequent module RNG verification ===")
torch.manual_seed(42)
bn = Bottleneck(c, c, True, 1, k=((3, 3), (3, 3)), e=0.5)
next_bn = Bottleneck(c, c, True, 1, k=((3, 3), (3, 3)), e=0.5)
bn_sum = sum(v.sum().item() for k, v in next_bn.state_dict().items() if 'conv.weight' in k)

torch.manual_seed(42)
sr = StarResBlock(c, True)
next_bn2 = Bottleneck(c, c, True, 1, k=((3, 3), (3, 3)), e=0.5)
sr_sum = sum(v.sum().item() for k, v in next_bn2.state_dict().items() if 'conv.weight' in k)

print(f"After Bottleneck: next BN weight sum = {bn_sum:.6f}")
print(f"After StarResBlock: next BN weight sum = {sr_sum:.6f}")
print(f"Match: {abs(bn_sum - sr_sum) < 1e-6}")

# ====== Full C3k2 vs C2f_Simple_Star ======
print(f"\n=== Full C2f_Simple_Star vs C3k2 ===")
configs = [
    (768, 256, 2, "fpn_p3"),
    (512, 128, 2, "fpn_p2"),
    (512, 256, 2, "pan_p3"),
    (1024, 512, 2, "pan_p4"),
]
all_ok = True
for c1, c2, n, name in configs:
    torch.manual_seed(42)
    m_c3k2 = C3k2(c1, c2, n=n, c3k=False)
    torch.manual_seed(42)
    m_star = C2f_Simple_Star(c1, c2, n=n)
    p_c3k2 = sum(p.numel() for p in m_c3k2.parameters())
    p_star = sum(p.numel() for p in m_star.parameters())
    x = torch.randn(1, c1, 32, 32)
    m_c3k2.eval(); m_star.eval()
    with torch.no_grad():
        out_match = m_c3k2(x).shape == m_star(x).shape
    print(f"  {name}: params={p_c3k2:,} / {p_star:,} (diff={p_star-p_c3k2:+,})  shape={out_match}")
print(f"All OK: {all_ok and out_match}")

# ====== RNG state after FULL C2f_Simple_Star vs C3k2 ======
print(f"\n=== RNG state after full module creation ===")
torch.manual_seed(42)
m_c3k2 = C3k2(768, 256, n=2, c3k=False)
rng_after_c3k2 = torch.get_rng_state()
# Create a dummy conv to see the RNG state effect
dummy_c3k2 = Conv(256, 256, 3)

torch.manual_seed(42)
m_star = C2f_Simple_Star(768, 256, n=2)
rng_after_star = torch.get_rng_state()
dummy_star = Conv(256, 256, 3)

print(f"RNG state after C3k2 == after Star: {torch.equal(rng_after_c3k2, rng_after_star)}")
print(f"Dummy conv weight match: {torch.equal(dummy_c3k2.conv.weight, dummy_star.conv.weight)}")