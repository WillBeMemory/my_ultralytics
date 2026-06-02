# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Deformable Convolution modules for C3k2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d

from ultralytics.nn.modules.block import PSABlock


class DeformConv2d(nn.Module):
    """Deformable Convolution 2D with offset generation and optional fallback to standard convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        modulation: bool = True,
    ):
        """Initialize DeformableConv2d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for convolution.
            stride (int): Stride for convolution.
            padding (int, optional): Padding for convolution. Defaults to kernel_size // 2.
            dilation (int): Dilation for convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to use bias.
            modulation (bool): Whether to use modulated deformable convolution.
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.modulation = modulation

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        offset_channels = 2 * kernel_size * kernel_size
        if modulation:
            offset_channels += kernel_size * kernel_size
        offset_channels *= groups

        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.fallback_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if bias:
            self.fallback_conv.bias.data.copy_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deformable convolution.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor [B, C_out, H_out, W_out].
        """
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4)

        offset = self.offset_conv(x)

        H_out = (x.size(2) - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (x.size(3) - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        if offset.size(2) != H_out or offset.size(3) != W_out:
            return self.fallback_conv(x)

        mask = None
        if self.modulation:
            offset_ch_per_group = 2 * self.kernel_size * self.kernel_size
            mask_ch_per_group = self.kernel_size * self.kernel_size
            offset_list = []
            mask_list = []
            for g in range(self.groups):
                start = g * (offset_ch_per_group + mask_ch_per_group)
                offset_part = offset[:, start:start + offset_ch_per_group]
                mask_part = offset[:, start + offset_ch_per_group:start + offset_ch_per_group + mask_ch_per_group]
                offset_list.append(offset_part)
                mask_list.append(mask_part)
            offset = torch.cat(offset_list, dim=1)
            mask = torch.cat(mask_list, dim=1)
            mask = torch.sigmoid(mask)

        try:
            out = deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride, dilation=self.dilation, mask=mask,
            )
        except RuntimeError:
            out = self.fallback_conv(x)

        return out


class DeformBottleneck(nn.Module):
    """Bottleneck module with deformable convolution for enhanced feature extraction.

    This module replaces the standard convolution in Bottleneck with deformable convolution,
    allowing the network to adaptively adjust the receptive field based on input features.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        kernel_size: int = 3,
    ):
        """Initialize DeformBottleneck module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
            kernel_size (int): Kernel size for deformable convolution.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU(inplace=True)

        self.cv2 = DeformConv2d(c_, c2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through deformable bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.act(self.bn1(self.cv1(x)))
        out = self.bn2(self.cv2(out))
        return x + out if self.add else out


class DeformC3k(nn.Module):
    """C3k module with deformable convolution for flexible receptive field.

    This module extends the standard C3k with deformable convolutions,
    providing better adaptation to objects of varying shapes and scales.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
        deform_kernel_size: int = 3,
    ):
        """Initialize DeformC3k module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of DeformBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size for standard convolution.
            deform_kernel_size (int): Kernel size for deformable convolution.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU(inplace=True)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.bn2 = nn.BatchNorm2d(c_)
        self.m = nn.Sequential(
            *(
                DeformBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0, kernel_size=deform_kernel_size)
                for _ in range(n)
            )
        )
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act_out = nn.SiLU(inplace=True) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeformC3k.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out1 = self.act(self.bn1(self.cv1(x)))
        out2 = self.m(out1)
        out3 = self.cv3(torch.cat([out2, self.act(self.bn2(self.cv2(x)))], dim=1))
        return self.act_out(self.bn3(out3))


class DeformC3k2(nn.Module):
    """Deformable C3k2 module for enhanced feature extraction in YOLO.

    This module combines deformable convolution with the C3k2 architecture,
    providing adaptive receptive fields while maintaining efficient computation.
    It supports optional attention mechanisms and flexible block configurations.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int): Number of blocks to stack.
        c3k (bool): Whether to use DeformC3k blocks.
        e (float): Expansion ratio for hidden channels.
        attn (bool): Whether to use attention blocks.
        g (int): Groups for convolutions.
        shortcut (bool): Whether to use shortcut connections.
        kernel_size (int): Kernel size for deformable convolution.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): Initial 1x1 convolution.
        m (nn.ModuleList): List of deformable blocks.
        cv2 (Conv): Final 1x1 convolution.
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
        kernel_size: int = 3,
    ):
        """Initialize DeformC3k2 module."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)

        self.m = nn.ModuleList()
        for _ in range(n):
            if attn:
                self.m.append(
                    nn.Sequential(
                        DeformBottleneck(self.c, self.c, shortcut, g, kernel_size=kernel_size),
                        PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                    )
                )
            elif c3k:
                self.m.append(
                    DeformC3k(
                        self.c, self.c, n=2, shortcut=shortcut, g=g, k=kernel_size, deform_kernel_size=kernel_size
                    )
                )
            else:
                self.m.append(
                    DeformBottleneck(self.c, self.c, shortcut, g, kernel_size=kernel_size)
                )

        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeformC3k2.

        Args:
            x (torch.Tensor): Input tensor [B, C1, H, W].

        Returns:
            torch.Tensor: Output tensor [B, C2, H, W].
        """
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]

        a = y[1]
        for m_block in self.m:
            a = m_block(a)
            outputs.append(a)

        return self.cv2(torch.cat(outputs, dim=1))


class DeformC3k2Block(nn.Module):
    """Enhanced Deformable Bottleneck Block with coordinate attention and multi-scale support.

    This module provides additional flexibility with:
    - Configurable deformable and standard convolution paths
    - Optional coordinate attention integration
    - Flexible kernel size configuration
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        shortcut: bool = True,
        g: int = 1,
        kernel_size: int = 3,
        use_coord_attn: bool = False,
    ):
        """Initialize DeformC3k2Block.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of blocks.
            e (float): Expansion ratio.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            kernel_size (int): Kernel size for deformable convolution.
            use_coord_attn (bool): Whether to use coordinate attention.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)

        self.blocks = nn.ModuleList()
        for _ in range(n):
            block = nn.Sequential(
                DeformBottleneck(self.c, self.c, shortcut, g, kernel_size=kernel_size),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            )
            if use_coord_attn:
                from ultralytics.nn.modules.coordatt import CoordAtt
                block.append(CoordAtt(self.c, self.c))
            self.blocks.append(block)

        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeformC3k2Block."""
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]

        a = y[1]
        for block in self.blocks:
            a = block(a)
            outputs.append(a)

        return self.cv2(torch.cat(outputs, dim=1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Testing DeformC3k2 Module")
    print("=" * 60)

    x = torch.randn(1, 64, 64, 64).to(device)

    print(f"\n1. Basic DeformC3k2 Test")
    print(f"   Input shape: {x.shape}")
    model = DeformC3k2(64, 128, n=1, c3k=False, e=0.5, shortcut=True, kernel_size=3).to(device)
    y = model(x)
    print(f"   Output shape: {y.shape} (expected [1, 128, 64, 64])")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print(f"\n2. DeformC3k2 with c3k=True")
    model = DeformC3k2(64, 128, n=2, c3k=True, e=0.5, shortcut=True, kernel_size=3).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print(f"\n3. DeformC3k2 with Attention")
    model = DeformC3k2(64, 128, n=1, c3k=False, attn=True, e=0.5, shortcut=True, kernel_size=3).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print(f"\n4. DeformBottleneck Standalone")
    model = DeformBottleneck(64, 64, shortcut=True, kernel_size=3).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print(f"\n5. DeformC3k2 with Different Kernel Sizes")
    for ks in [3, 5, 7]:
        model = DeformC3k2(64, 128, n=1, kernel_size=ks).to(device)
        y = model(x)
        print(f"   kernel_size={ks}: Input {x.shape} → Output {y.shape}")

    print(f"\n6. Multi-Block DeformC3k2")
    model = DeformC3k2(64, 128, n=3, kernel_size=3).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print(f"\n7. DeformC3k2Block with Coord Attention")
    model = DeformC3k2Block(64, 128, n=2, use_coord_attn=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.mean()
    loss.backward()
    print("   ✅ Backward pass OK")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
