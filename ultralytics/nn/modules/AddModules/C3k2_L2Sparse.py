# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""C3k2 with L2-Norm Top-K mask and true sparse convolution (spconv) enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C3k, PSABlock
from ultralytics.nn.modules.conv import Conv


class L2MaskSparseConv(nn.Module):
    """L2-Norm Top-K true sparse convolution enhancement block.

    1. Selects Top-K spatial positions by L2 norm.
    2. Dilates the mask to 3x3 neighborhood around each selected position,
       so the sparse convolution can capture spatial context.
    3. Applies real sparse convolution (spconv.SubMConv2d) that physically
       skips computation at non-selected positions.

    Args:
        channels (int): Number of input/output channels.
        keep_ratio (float): Ratio of spatial positions to keep before dilation.
        min_keeps (int): Minimum number of positions to keep before dilation.
        kernel_size (int): Kernel size for sparse convolution (3 or 5).
        alpha_init (float): Initial value for the learnable fusion weight.
    """

    def __init__(
        self,
        channels: int,
        keep_ratio: float = 0.25,
        min_keeps: int = 16,
        kernel_size: int = 3,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.kernel_size = kernel_size

        # Learnable fusion weight (starts small for stable training)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # True sparse convolution via spconv.SubMConv2d
        #   - Only computes at active sites (Top-K + dilated neighbors)
        #   - Physically skips zero regions (real sparsity, not mask gating)
        import spconv.pytorch as spconv
        self.sparse_conv = spconv.SubMConv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)  # spconv output is [N, C]
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Enhanced tensor [B, C, H, W].
        """
        import spconv.pytorch as spconv

        B, C, H, W = x.shape
        N = H * W

        # 1. Compute L2 norm per spatial position → importance map
        l2 = torch.norm(x, p=2, dim=1)  # [B, H, W]

        # 2. Top-K selection → seed positions
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_idx = torch.topk(l2.flatten(1), k=k, dim=-1)  # [B, K]

        # 3. Create spatial mask from Top-K seeds → dilate to 3x3 neighborhood
        mask_flat = torch.zeros(B, N, device=x.device, dtype=torch.float32)
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask_2d = mask_flat.view(B, 1, H, W)  # [B, 1, H, W]

        # Dilation: expand each seed to its 3x3 neighborhood
        dilated = F.max_pool2d(mask_2d, kernel_size=3, stride=1, padding=1)  # [B, 1, H, W]

        # 4. Get all active positions from dilated mask
        #    torch.nonzero returns [num_active, 3] = [batch_idx, y, x]
        positions = torch.nonzero(dilated.squeeze(1), as_tuple=False)  # [total_active, 3]
        total_active = positions.shape[0]

        # 5. Build spconv indices [total_active, 3] and extract features [total_active, C]
        batch_indices = positions[:, 0].long()
        y_indices = positions[:, 1].long()
        x_indices = positions[:, 2].long()
        indices = torch.stack([batch_indices, y_indices, x_indices], dim=1).int()
        features = x[batch_indices, :, y_indices, x_indices]  # [total_active, C]

        # 6. Build SparseConvTensor and apply true sparse convolution
        sparse_in = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[H, W],
            batch_size=B,
        )
        sparse_out = self.sparse_conv(sparse_in)
        sparse_out = sparse_out.replace_feature(self.act(self.bn(sparse_out.features)))

        # 7. Convert to dense: non-selected positions are automatically zero
        enhanced = sparse_out.dense()  # [B, C, H, W]

        # 8. Residual fusion with learnable weight
        return x + self.alpha * enhanced


class C3k2_L2Sparse(nn.Module):
    """C3k2 with L2-Norm Top-K true sparse convolution (spconv) enhancement.

    Architecture (inherits C3k2 structure):
        x ──→ cv1 ──→ chunk(2) ──┬── y[0] ──────────────────→ outputs
                                  └── y[1] ──→ Bottleneck ──→ L2MaskSparseConv ──→ outputs
                                              (× n blocks)

    Each Bottleneck output is enhanced by L2MaskSparseConv, which:
      1. Selects top-K spatial positions by L2 norm
      2. Applies real spconv.SubMConv2d (physically skips non-selected sites)
      3. Fuses back via learnable alpha * sparse_output + identity

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of Bottleneck blocks.
        c3k (bool): Use C3k blocks instead of Bottleneck.
        e (float): Expansion ratio for hidden channels.
        attn (bool): Use attention blocks (Bottleneck + PSABlock).
        g (int): Groups for convolutions.
        shortcut (bool): Use shortcut connections in Bottleneck.
        keep_ratio (float): Top-K keep ratio for sparse conv (default 0.25).
        min_keeps (int): Minimum spatial positions to keep (default 16).
        sparse_kernel_size (int): Kernel size for sparse conv (default 3).
        alpha_init (float): Initial fusion weight (default 0.1).
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
        keep_ratio: float = 0.25,
        min_keeps: int = 16,
        sparse_kernel_size: int = 3,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # --- C3k2 standard structure ---
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.m = nn.ModuleList()
        for _ in range(n):
            if attn:
                self.m.append(
                    nn.Sequential(
                        Bottleneck(self.c, self.c, shortcut, g),
                        PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                    )
                )
            elif c3k:
                self.m.append(C3k(self.c, self.c, 2, shortcut, g))
            else:
                self.m.append(Bottleneck(self.c, self.c, shortcut, g))

        # --- True sparse conv enhancement (after each bottleneck) ---
        self.sparse_convs = nn.ModuleList(
            L2MaskSparseConv(
                self.c,
                keep_ratio=keep_ratio,
                min_keeps=min_keeps,
                kernel_size=sparse_kernel_size,
                alpha_init=alpha_init,
            )
            for _ in range(n)
        )

        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C1, H, W].

        Returns:
            Output tensor [B, C2, H, W].
        """
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]

        a = y[1]
        for m_block, sparse_conv in zip(self.m, self.sparse_convs):
            a = m_block(a)            # standard C3k2 bottleneck
            a = sparse_conv(a)        # L2-TopK true sparse conv enhancement
            outputs.append(a)

        return self.cv2(torch.cat(outputs, dim=1))


# ==================== 测试 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("WARNING: spconv requires CUDA. Tests will be skipped on CPU.")
        exit(0)

    x = torch.randn(2, 128, 40, 40).to(device)

    # 1. C3k2_L2Sparse (基础模式)
    print("\n1. C3k2_L2Sparse (Bottleneck mode)")
    model = C3k2_L2Sparse(128, 256, n=2, e=0.5, keep_ratio=0.25).to(device)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Input:  {list(x.shape)}")
    print(f"   Output: {list(y.shape)} (expected [2, 256, 40, 40])")
    print(f"   Params: {params:,}")
    loss = y.mean()
    loss.backward()
    print("   Backward pass OK")

    # 2. C3k2_L2Sparse (c3k 模式)
    print("\n2. C3k2_L2Sparse (C3k mode)")
    model = C3k2_L2Sparse(128, 256, n=2, c3k=True, e=0.5, keep_ratio=0.3).to(device)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Input:  {list(x.shape)}")
    print(f"   Output: {list(y.shape)}")
    print(f"   Params: {params:,}")
    loss = y.mean()
    loss.backward()
    print("   Backward pass OK")

    # 3. C3k2_L2Sparse (attn 模式)
    print("\n3. C3k2_L2Sparse (Attention mode)")
    model = C3k2_L2Sparse(128, 256, n=2, attn=True, e=0.5, keep_ratio=0.25).to(device)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Input:  {list(x.shape)}")
    print(f"   Output: {list(y.shape)}")
    print(f"   Params: {params:,}")
    loss = y.mean()
    loss.backward()
    print("   Backward pass OK")

    # 4. 不同 keep_ratio
    print("\n4. Different keep_ratio")
    for r in [0.1, 0.25, 0.5]:
        model = C3k2_L2Sparse(128, 256, n=1, keep_ratio=r).to(device)
        y = model(x)
        print(f"   keep_ratio={r}: output shape {list(y.shape)}")

    # 5. 不同分辨率
    print("\n5. Different input sizes")
    for size in [(20, 20), (40, 40), (80, 80)]:
        H, W = size
        xi = torch.randn(2, 128, H, W).to(device)
        model = C3k2_L2Sparse(128, 256, n=2, keep_ratio=0.25).to(device)
        yi = model(xi)
        print(f"   {H}x{W}: {list(xi.shape)} -> {list(yi.shape)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)