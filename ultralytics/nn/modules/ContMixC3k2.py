# Ultralytics 🚀 AGPL-3.0 License

"""ContMixC3k2: C3k2 with OverLoCK-style dynamic convolution bottlenecks.

Reference:
    OverLoCK (CVPR 2025 Oral): An Overview-first-Look-Closely-next ConvNet
    with Context-Mixing Dynamic Kernels.
    https://arxiv.org/abs/2502.20087
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import PSABlock, Conv
from ultralytics.nn.modules.block import Bottleneck as StandardBottleneck
from ultralytics.nn.modules.block import C3k


class ContMixConv(nn.Module):
    """Content-Adaptive Dynamic Convolution (inspired by OverLoCK ContMix).

    Generates per-position convolution kernels via Query-Key interaction,
    then applies them through neighborhood sampling (F.unfold).

    Args:
        channels (int): Input/output channels.
        kernel_size (int): Spatial extent of the dynamic kernel (default 5).
        heads (int): Number of attention heads for kernel generation (default 4).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, heads: int = 4):
        super().__init__()
        if heads > in_channels:
            heads = max(in_channels // 8, 1)
        assert in_channels % heads == 0, f"in_channels ({in_channels}) must be divisible by heads ({heads})"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.heads = heads
        self.head_dim = in_channels // heads
        self.k2 = kernel_size * kernel_size
        self.pad = kernel_size // 2

        # Scale factor to prevent Q/K values from growing too large
        self.scale = self.head_dim ** -0.25

        # Q/K/V projections (1×1 convs, operate on in_channels)
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Project Q·K interaction to per-head kernel weights
        self.weight_proj = nn.Sequential(
            nn.Linear(heads, heads * 2),
            nn.GELU(),
            nn.Linear(heads * 2, heads * self.k2),
        )

        # Relative Position Bias (learnable per-head, per-kernel-position)
        self.rpb = nn.Parameter(torch.zeros(heads, self.k2))

        # SE channel attention (on in_channels)
        se_reduction = max(in_channels // 16, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Output projection: expand/shrink from in_channels to out_channels
        self.out_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.rpb, std=0.02)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        ks = self.kernel_size
        k2 = self.k2

        # 1. QKV projections
        q = self.q_proj(x).reshape(B, self.heads, self.head_dim, H * W)  # (B, h, d, N)
        k = self.k_proj(x).reshape(B, self.heads, self.head_dim, H * W)  # (B, h, d, N)
        v = self.v_proj(x)  # (B, C, H, W)

        # 2. Global context from K via average pooling over spatial dims
        k_global = k.mean(dim=-1)  # (B, h, d)

        # 3. Q·K interaction → dynamic kernel weights per position
        # q: (B, h, d, N), k_global: (B, h, d) → attn: (B, h, N)
        attn = (q * k_global.unsqueeze(-1)).sum(dim=2) * self.scale  # (B, h, N)
        attn = attn.transpose(1, 2)  # (B, N, h)

        # Project to kernel space: (B, N, h) → (B, N, h*k²)
        kernel_w = self.weight_proj(attn)  # (B, N, h*k²)
        kernel_w = kernel_w.reshape(B, H, W, self.heads, k2)  # (B, H, W, h, k²)

        # Add Relative Position Bias
        kernel_w = kernel_w + self.rpb.view(1, 1, 1, self.heads, k2)

        # Softmax over kernel positions
        kernel_w = F.softmax(kernel_w, dim=-1)  # (B, H, W, h, k²)

        # 4. Extract local neighborhoods via F.unfold
        # v: (B, C, H, W) → unfold: (B, C*k², N)
        v_unfold = F.unfold(v, (ks, ks), dilation=1, padding=self.pad)  # (B, C*k², H*W)
        v_unfold = v_unfold.reshape(B, self.heads, self.head_dim, k2, H * W)
        # (B, h, d, k², N) → (B, N, h, k², d)
        v_unfold = v_unfold.permute(0, 4, 1, 3, 2)  # (B, N, h, k², d)

        # 5. Apply dynamic kernels
        # kernel_w: (B, H, W, h, k²) → (B, N, h, k², 1)
        kernel_w = kernel_w.reshape(B, H * W, self.heads, k2, 1)  # (B, N, h, k², 1)

        out = (v_unfold * kernel_w).sum(dim=3)  # (B, N, h, d) → sum over k²
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 6. SE channel attention
        out = out * self.se(out)

        # 7. Output projection + residual (handle channel mismatch)
        out = self.out_proj(out)
        out = self.out_norm(out)
        if self.in_channels == self.out_channels:
            out = self.act(out + identity)
        else:
            out = self.act(out)

        return out


class ContMixBottleneck(nn.Module):
    """Bottleneck with ContMix dynamic convolution replacing the 3×3 conv.

    Standard Bottleneck:  cv1(1×1) → cv2(3×3 Conv)
    ContMixBottleneck:    cv1(1×1) → cv2(ContMix dynamic conv)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 0.5,
        kernel_size: int = 5,
        heads: int = 4,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ContMixConv(c_, c2, kernel_size=kernel_size, heads=heads)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class ContMixC3k2(nn.Module):
    """C3k2 with ContMix dynamic convolution in Bottleneck blocks.

    Drop-in replacement for C3k2. Uses the same CSP structure but replaces
    standard Bottleneck 3×3 convs with content-adaptive dynamic convolutions.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of blocks.
        c3k (bool): Use C3k block (2 Bottlenecks per block) instead of single Bottleneck.
        e (float): Expansion ratio for hidden channels.
        attn (bool): Append PSABlock after each Bottleneck (attention mode).
        g (int): Groups (ignored, ContMix manages groups internally via heads).
        shortcut (bool): Use residual connections.
        kernel_size (int): Dynamic kernel spatial extent (default 5).
        heads (int): Number of heads for kernel generation (default 4).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        kernel_size: int = 5,
        heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.n = n

        # CSP head/tail (same as standard C2f/C3k2)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Build block list
        self.m = nn.ModuleList()
        for _ in range(n):
            if c3k:
                # C3k mode: 2 ContMixBottleneck blocks per module
                block = nn.Sequential(
                    ContMixBottleneck(self.c, self.c, shortcut, e=1.0,
                                      kernel_size=kernel_size, heads=heads),
                    ContMixBottleneck(self.c, self.c, shortcut, e=1.0,
                                      kernel_size=kernel_size, heads=heads),
                )
                self.m.append(block)
            elif attn:
                # Attention mode: ContMixBottleneck + original PSABlock
                block = nn.Sequential(
                    ContMixBottleneck(self.c, self.c, shortcut,
                                      kernel_size=kernel_size, heads=heads),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                self.m.append(block)
            else:
                # Standard mode: single ContMixBottleneck
                self.m.append(
                    ContMixBottleneck(self.c, self.c, shortcut,
                                      kernel_size=kernel_size, heads=heads)
                )

        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]
        a = y[1]
        for m_block in self.m:
            a = m_block(a)
            outputs.append(a)
        return self.act(self.cv2(torch.cat(outputs, dim=1)))


# ================== Test ==================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)

    # Test 1: ContMixConv standalone
    print("\n1. ContMixConv")
    x = torch.randn(1, 64, 32, 32, device=device)
    conv = ContMixConv(64, 64, kernel_size=5, heads=4).to(device)
    y = conv(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [1, 64, 32, 32])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 2: ContMixBottleneck
    print("\n2. ContMixBottleneck")
    x = torch.randn(1, 64, 32, 32, device=device)
    bn = ContMixBottleneck(64, 64, shortcut=True).to(device)
    y = bn(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [1, 64, 32, 32])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 3: ContMixC3k2 basic mode
    print("\n3. ContMixC3k2 (basic)")
    x = torch.randn(1, 64, 64, 64, device=device)
    model = ContMixC3k2(64, 128, n=2, e=0.5, shortcut=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [1, 128, 64, 64])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 4: ContMixC3k2 with c3k=True
    print("\n4. ContMixC3k2 (c3k=True)")
    x = torch.randn(1, 64, 64, 64, device=device)
    model = ContMixC3k2(64, 128, n=2, c3k=True, e=0.5, shortcut=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [1, 128, 64, 64])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 5: ContMixC3k2 with attn=True
    print("\n5. ContMixC3k2 (attn=True)")
    x = torch.randn(1, 64, 64, 64, device=device)
    model = ContMixC3k2(64, 128, n=2, attn=True, e=0.5, shortcut=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [1, 128, 64, 64])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 6: Different input resolutions
    print("\n6. Resolution tests")
    for hw in [(16, 16), (32, 32), (64, 64), (80, 80), (43, 43)]:
        x = torch.randn(1, 64, hw[0], hw[1], device=device)
        model = ContMixC3k2(64, 128, n=1, e=0.5).to(device)
        y = model(x)
        print(f"   {hw[0]}×{hw[1]}: {x.shape} → {y.shape} ✅")

    # Test 7: Parameter count comparison
    print("\n7. Parameter count")
    from ultralytics.nn.modules.block import C3k2
    c1, c2 = 64, 128
    std = C3k2(c1, c2, n=2, e=0.5)
    wt = ContMixC3k2(c1, c2, n=2, e=0.5)
    std_p = sum(p.numel() for p in std.parameters())
    wt_p = sum(p.numel() for p in wt.parameters())
    print(f"   Standard C3k2: {std_p:,} params")
    print(f"   ContMixC3k2:   {wt_p:,} params (+{(wt_p - std_p) / std_p * 100:.0f}%)")
    print(f"   ✅ All tests passed!")
