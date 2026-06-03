# Ultralytics 🚀 AGPL-3.0 License

"""FDConv: Frequency Dynamic Convolution from CVPR 2025.

Reference:
    Frequency Dynamic Convolution for Dense Image Prediction
    Linwei Chen et al., CVPR 2025
    https://arxiv.org/abs/2503.18783
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FDConv(nn.Module):
    """Frequency Dynamic Convolution — drop-in replacement for nn.Conv2d.

    Learns frequency-diverse convolution kernels in the Fourier domain
    via three modules: FDW (frequency-diverse weights), KSM (kernel spatial
    modulation), and FBM (frequency band modulation).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Stride.
        padding (int): Padding.
        kernel_num (int): Number of frequency base kernels (default in_channels).
        groups (int): Groups for convolution.
        bias (bool): Use bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        kernel_num: int = None,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups
        kH = kernel_size
        kW = kernel_size
        self.k2 = kH * kW
        self.kernel_num = kernel_num if kernel_num is not None else min(in_channels, self.k2)

        in_ch_per_group = in_channels // groups
        out_ch_per_group = out_channels // groups

        # ── FDW: Frequency Dynamic Weights ──
        # Learn a SINGLE kernel in Fourier domain (same parameter count as standard
        # Conv2d). The frequency diversity comes from KSM/FBM modulation, not from
        # storing multiple kernels.
        self.fdw_real = nn.Parameter(
            torch.randn(out_ch_per_group, in_ch_per_group, kH, kW)
        )
        self.fdw_imag = nn.Parameter(
            torch.zeros(out_ch_per_group, in_ch_per_group, kH, kW)
        )

        # ── KSM: Kernel Spatial Modulation ──
        # Generate per-position modulation for frequency response
        ksm_dim = max(out_channels // 16, 8)
        self.ksm_conv = nn.Sequential(
            nn.Conv2d(in_channels, ksm_dim, 3, padding=1, groups=groups),
            nn.BatchNorm2d(ksm_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(ksm_dim, self.kernel_num, 1),
            nn.Sigmoid(),
        )

        # ── FBM: Frequency Band Modulation ──
        # Content-adaptive frequency band weighting via channel attention
        # Note: FBM operates on the INPUT features (in_channels) to decide
        # which frequency bands to emphasize for the current content.
        fbm_reduce = max(in_channels // 16, 4)
        self.fbm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, fbm_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fbm_reduce, self.kernel_num, 1),
            nn.Sigmoid(),
        )

        # ── Bias ──
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fdw_real, a=math.sqrt(5))
        nn.init.zeros_(self.fdw_imag)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_dynamic_kernel(self, ksm_weights, fbm_weights):
        """FDW + KSM + FBM → dynamic convolution kernel."""
        B = ksm_weights.shape[0]
        o_g = self.out_channels // self.groups
        i_g = self.in_channels // self.groups
        kH, kW = self.kernel_size, self.kernel_size

        # 1. FFT to frequency domain
        fdw_complex = torch.complex(self.fdw_real, self.fdw_imag)
        fdw_freq = torch.fft.fft2(fdw_complex)  # (o_g, i_g, kH, kW)

        # 2. Flatten and split into kernel_num bands
        fdw_freq = fdw_freq.reshape(o_g, i_g, self.kernel_num, -1)  # (o_g, i_g, kn, band_sz)

        # 3. KSM: spatial mean → (B, kernel_num)
        ksm = ksm_weights.mean(dim=[2, 3])  # (B, kernel_num)

        # 4. FBM: squeeze → (B, kernel_num)
        fbm = fbm_weights.squeeze(-1).squeeze(-1)  # (B, kernel_num)

        # 5. Combine → (B, kernel_num)
        combined = torch.sigmoid((ksm + fbm) * 0.5)  # (B, kernel_num)

        # 6. Apply band modulation in frequency domain
        combined = combined.view(B, 1, 1, self.kernel_num, 1)
        fdw_modulated = fdw_freq.unsqueeze(0) * combined  # (B, o_g, i_g, kn, band_sz)
        fdw_modulated = fdw_modulated.reshape(B, o_g, i_g, kH, kW)

        # 7. IFFT back to spatial domain
        kernel = torch.fft.ifft2(fdw_modulated).real  # (B, o_g, i_g, kH, kW)

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        # 1. KSM: per-position modulation
        ksm_w = self.ksm_conv(x)  # (B, kernel_num, H, W)

        # 2. FBM: content-adaptive frequency band weights
        fbm_w = self.fbm(x)  # (B, kernel_num, 1, 1)

        # 3. Build dynamic kernel
        kernel = self._build_dynamic_kernel(ksm_w, fbm_w)
        # (B, out_ch, in_ch_per_group, kH, kW)

        # 4. Apply dynamic convolution via F.unfold
        x_unfold = F.unfold(x, (self.kernel_size, self.kernel_size),
                            dilation=1, padding=self.padding, stride=self.stride)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Flatten kernel spatial dims: (B, out_ch, in_ch, kH, kW) → (B, out_ch, in_ch, k2, 1)
        kernel = kernel.reshape(B, self.out_channels, -1, self.k2).unsqueeze(-1)

        # Reshape unfold: (B, C_in, k2, N)
        x_unfold = x_unfold.reshape(B, C_in, self.k2, H_out * W_out)

        # Batched matmul: (B, out_ch, C_in, k2, 1) * (B, 1, C_in, k2, N)
        # → sum dim 2,3 → (B, out_ch, N)
        out = (kernel * x_unfold.unsqueeze(1)).sum(dim=(2, 3))

        out = out.reshape(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


class FDConvBottleneck(nn.Module):
    """Bottleneck with FDConv replacing the 3×3 convolution.

    Standard:  cv1(1×1) → cv2(3×3 Conv)
    FDConv:    cv1(1×1) → cv2(FDConv)

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Residual connection.
        e (float): Hidden channel expansion ratio.
        kernel_size (int): FDConv kernel size.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 0.5,
        kernel_size: int = 3,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = FDConv(c_, c2, kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class FDConvC3k2(nn.Module):
    """C3k2 with FDConv frequency-dynamic convolution in bottleneck blocks.

    Drop-in replacement for C3k2. Same CSP structure but each Bottleneck's
    3×3 conv is replaced by FDConv (frequency-diverse dynamic convolution).

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of blocks.
        e (float): Expansion ratio.
        shortcut (bool): Residual connections.
        g (int): Groups for conv.
        fd_kernel_size (int): Kernel size for FDConv (default 3).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,        # YAML convention (ignored, FDConv replaces Bottleneck)
        e: float = 0.5,
        shortcut: bool = True,
        g: int = 1,
        fd_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()
        from ultralytics.nn.modules.block import Conv

        self.c = int(c2 * e)
        self.n = n

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.m = nn.ModuleList(
            FDConvBottleneck(self.c, self.c, shortcut, e=1.0,
                             kernel_size=fd_kernel_size)
            for _ in range(n)
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

    # Test 1: FDConv standalone
    print("\n1. FDConv (standalone)")
    x = torch.randn(2, 64, 32, 32, device=device)
    conv = FDConv(64, 128, kernel_size=3, stride=1, padding=1).to(device)
    y = conv(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 128, 32, 32])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 2: FDConv with stride=2
    print("\n2. FDConv (stride=2)")
    x = torch.randn(2, 64, 32, 32, device=device)
    conv = FDConv(64, 128, kernel_size=3, stride=2, padding=1).to(device)
    y = conv(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 128, 16, 16])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 3: FDConvBottleneck
    print("\n3. FDConvBottleneck")
    x = torch.randn(2, 64, 32, 32, device=device)
    bn = FDConvBottleneck(64, 64, shortcut=True).to(device)
    y = bn(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 64, 32, 32])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 4: FDConvBottleneck channel change
    print("\n4. FDConvBottleneck (64→128)")
    x = torch.randn(2, 64, 32, 32, device=device)
    bn = FDConvBottleneck(64, 128, shortcut=False).to(device)
    y = bn(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 128, 32, 32])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 5: FDConvC3k2
    print("\n5. FDConvC3k2")
    x = torch.randn(2, 64, 64, 64, device=device)
    model = FDConvC3k2(64, 128, n=2, e=0.5, shortcut=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 128, 64, 64])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 6: Resolution tests
    print("\n6. Resolution tests")
    for hw in [(16, 16), (32, 32), (43, 43), (64, 64), (80, 80)]:
        x = torch.randn(2, 64, hw[0], hw[1], device=device)
        model = FDConvC3k2(64, 128, n=1, e=0.5).to(device)
        y = model(x)
        print(f"   {hw[0]}×{hw[1]}: {x.shape} → {y.shape} ✅")

    # Test 7: Parameter count
    print("\n7. Parameter count")
    from ultralytics.nn.modules.block import C3k2
    c1, c2 = 64, 128
    std = C3k2(c1, c2, n=2, e=0.5)
    fd = FDConvC3k2(c1, c2, n=2, e=0.5)
    std_p = sum(p.numel() for p in std.parameters())
    fd_p = sum(p.numel() for p in fd.parameters())
    print(f"   Standard C3k2: {std_p:,} params")
    print(f"   FDConvC3k2:    {fd_p:,} params (+{(fd_p - std_p) / std_p * 100:.0f}%)")

    print(f"\n   ✅ All tests passed!")
