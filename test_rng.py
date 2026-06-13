"""Precise RNG trace: C3k2(n=2, c3k=False) vs C2f_Simple"""
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, Bottleneck, C3k2, C2f

# Use get_rng_state to trace exact RNG consumption
def count_rng_consumed(fn, *args, **kwargs):
    """Count how many random values are consumed by fn call."""
    # kaiming_uniform_ uses torch.rand internally
    state_before = torch.get_rng_state()
    result = fn(*args, **kwargs)
    state_after = torch.get_rng_state()
    # Count random values consumed by comparing state
    # Actually, let's just count the number of Conv2d weights
    return result

c1, c2, n = 384, 256, 2
print(f"c1={c1}, c2={c2}, n={n}, self.c={int(c2*0.5)}")

# ====== C3k2 ======
torch.manual_seed(42)
m_c3k2 = C3k2(c1, c2, n=n, c3k=False)
c3k2_sd = m_c3k2.state_dict()
print(f"\n=== C3k2 state_dict (shape + sum of first 10 conv weights) ===")
for i, (k, v) in enumerate(c3k2_sd.items()):
    if 'conv.weight' in k:
        print(f"  {k}: shape={v.shape}, sum={v.sum().item():.6f}")

# ====== C2f_Simple WITH RNG alignment ======
class C2f_Simple_Aligned(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # RNG alignment: C3k2 creates n Bottleneck(e=1.0) in C2f.__init__ then discards them
        for _ in range(n):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

torch.manual_seed(42)
m_aligned = C2f_Simple_Aligned(c1, c2, n=n)
aligned_sd = m_aligned.state_dict()
print(f"\n=== C2f_Simple_Aligned state_dict ===")
for i, (k, v) in enumerate(aligned_sd.items()):
    if 'conv.weight' in k:
        print(f"  {k}: shape={v.shape}, sum={v.sum().item():.6f}")

# ====== C2f_Simple WITHOUT RNG alignment ======
class C2f_Simple_NoAlign(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

torch.manual_seed(42)
m_noalign = C2f_Simple_NoAlign(c1, c2, n=n)
noalign_sd = m_noalign.state_dict()
print(f"\n=== C2f_Simple_NoAlign state_dict ===")
for i, (k, v) in enumerate(noalign_sd.items()):
    if 'conv.weight' in k:
        print(f"  {k}: shape={v.shape}, sum={v.sum().item():.6f}")

# ====== Compare ======
print(f"\n=== Comparison ===")
# Only compare trainable params (conv.weight, bn.weight, bn.bias)
train_keys = [k for k in c3k2_sd if 'running' not in k and 'num_batches' not in k]
print(f"Aligned == C3k2: ", end="")
all_match = True
for k in train_keys:
    m = torch.equal(c3k2_sd[k], aligned_sd[k])
    if not m:
        all_match = False
        print(f"\n  MISMATCH {k}: max_diff={(c3k2_sd[k]-aligned_sd[k]).abs().max().item():.6f}")
print(f"{all_match}")

print(f"NoAlign == C3k2: ", end="")
all_match2 = True
for k in train_keys:
    m = torch.equal(c3k2_sd[k], noalign_sd[k])
    if not m:
        all_match2 = False
        print(f"\n  MISMATCH {k}: max_diff={(c3k2_sd[k]-noalign_sd[k]).abs().max().item():.6f}")
print(f"{all_match2}")

# ====== Test ALL 4 instances in FPN_PAN ======
print(f"\n=== FPN_PAN full test ===")
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
    m_simple = C2f_Simple_Aligned(c1, c2, n=n)
    for k in m_c3k2.state_dict():
        if 'running' in k or 'num_batches' in k:
            continue
        if not torch.equal(m_c3k2.state_dict()[k], m_simple.state_dict()[k]):
            all_ok = False
            print(f"  {name}: MISMATCH {k}")
            break
    else:
        print(f"  {name}: OK")
print(f"\nAll OK: {all_ok}")