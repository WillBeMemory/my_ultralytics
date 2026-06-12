"""Compare alternative block designs for C2f_Simple."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, Bottleneck
from ultralytics.nn.modules.FPN_PAN_BiFPN import C2f_Simple as C2f_Simple_Shuffle

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ====== Block definitions ======

class InvResBlock(nn.Module):
    """Inverted Residual: 1x1 expand -> 3x3 dilated -> 1x1 project"""
    def __init__(self, c, shortcut=True, g=1):
        super().__init__()
        c_e = max(int(c * 0.895), 1)
        self.cv1 = Conv(c, c_e, 1)
        self.cv2 = Conv(c_e, c_e, k=3, s=1, d=2)
        self.cv3 = Conv(c_e, c, 1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class InvResBlock_NoDilate(nn.Module):
    """Inverted Residual: 1x1 expand -> 3x3 -> 1x1 project (no dilation)"""
    def __init__(self, c, shortcut=True, g=1):
        super().__init__()
        c_e = max(int(c * 0.895), 1)
        self.cv1 = Conv(c, c_e, 1)
        self.cv2 = Conv(c_e, c_e, 3)
        self.cv3 = Conv(c_e, c, 1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class DilatedBottleneck(nn.Module):
    """Bottleneck(e=0.5) with dilated cv2 — same params, larger RF"""
    def __init__(self, c, shortcut=True, g=1):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, 3)
        self.cv2 = Conv(c_, c, k=3, s=1, d=2)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ====== C2f wrappers ======

class C2f_InvRes(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(InvResBlock(self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_InvRes_NoDilate(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(InvResBlock_NoDilate(self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DilatedBN(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DilatedBottleneck(self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ====== Comparison ======
configs = [
    (768, 256, 2, "fpn_p3"),
    (512, 128, 2, "fpn_p2"),
    (512, 256, 2, "pan_p3"),
    (1024, 512, 2, "pan_p4"),
]

designs = [
    ("Bottleneck(e=0.5) [baseline]", lambda c1,c2,n: nn.Module() if False else None),
]

# Baseline
print("=" * 75)
print("Bottleneck(e=0.5) — baseline")
print("=" * 75)
base_total = 0
for c1, c2, n, name in configs:
    # Manually compute
    c = int(c2 * 0.5)
    # cv1: c1*2c + 2c BN; cv2: (2+n)*c*c2 + c2 BN; m: n * BN_params
    m = C2f_Simple_Shuffle(c1, c2, n=n)  # current (ShuffleBlock)
    # Use Bottleneck version for baseline
    c_ = c // 2
    bn_params = 2 * (c_ * c * 9 + c * c_ * 9)  # approximate
base_total_manual = 3587968  # from previous test
print(f"  TOTAL: {base_total_manual:,}")

# Test each design
for design_name, cls in [
    ("InvResBlock (1x1->3x3d2->1x1)", C2f_InvRes),
    ("InvResBlock_NoDilate (1x1->3x3->1x1)", C2f_InvRes_NoDilate),
    ("DilatedBottleneck (3x3->3x3d2)", C2f_DilatedBN),
]:
    print(f"\n{'='*75}")
    print(f"{design_name}")
    print(f"{'='*75}")
    total = 0
    for c1, c2, n, name in configs:
        m = cls(c1, c2, n=n)
        p = count_params(m)
        total += p
        print(f"  {name}: c1={c1}, c2={c2}, total={p:,}")
    diff = total - base_total_manual
    print(f"  TOTAL: {total:,}  (diff: {diff:,}, {diff/base_total_manual*100:.2f}%)")

# Per-block detail
print(f"\n{'='*75}")
print("Per-block detail (c=128)")
print(f"{'='*75}")
c = 128

bn = Bottleneck(c, c, True, 1, k=((3,3),(3,3)), e=0.5)
ir = InvResBlock(c, True)
ir_nd = InvResBlock_NoDilate(c, True)
dbn = DilatedBottleneck(c, True)

print(f"  Bottleneck(e=0.5):  {count_params(bn):,} params  |  c→c/2(3×3)→c(3×3)  |  RF=5×5")
print(f"  InvResBlock:        {count_params(ir):,} params  |  c→0.9c(1×1)→0.9c(3×3d2)→c(1×1)  |  RF=5×5")
print(f"  InvResBlock_ND:     {count_params(ir_nd):,} params  |  c→0.9c(1×1)→0.9c(3×3)→c(1×1)  |  RF=3×3")
print(f"  DilatedBottleneck:  {count_params(dbn):,} params  |  c→c/2(3×3)→c(3×3d2)  |  RF=7×7")

# Output shape test
print(f"\n{'='*75}")
print("Output shape test (c1=768, c2=256, n=2)")
print(f"{'='*75}")
x = torch.randn(1, 768, 32, 32)
for name, cls in [
    ("InvResBlock", C2f_InvRes),
    ("InvResBlock_ND", C2f_InvRes_NoDilate),
    ("DilatedBN", C2f_DilatedBN),
]:
    m = cls(768, 256, n=2)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"  {name}: {x.shape} -> {y.shape}")
