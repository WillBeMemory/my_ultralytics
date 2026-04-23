# ============================================
# File: ultralytics/nn/modules/CARAFE.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """
    CARAFE: 内容感知特征重组 (Content-Aware ReAssembly of FEatures)
    无需输入通道参数，像 nn.Upsample 一样即插即用。

    参数:
        scale_factor (int): 上采样倍率，默认 2
        kernel_size (int): 特征重组时使用的邻域大小 (K)，默认 3
        compress_ratio (int): 核预测时的通道压缩比，默认 4
    """

    def __init__(self, scale_factor=2, kernel_size=3, compress_ratio=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.compress_ratio = compress_ratio

        # 内部模块将在第一次 forward 时根据输入通道数动态创建
        self._initialized = False
        self.channel_compressor = None
        self.encoder = None
        self.kernel_normalizer = nn.Softmax(dim=1)

    def _init_layers(self, in_channels: int, device: torch.device, dtype: torch.dtype):
        """根据输入通道数创建内部卷积层"""
        compressed_channels = max(1, in_channels // self.compress_ratio)

        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, 1),
            nn.BatchNorm2d(compressed_channels),
            nn.ReLU(inplace=True)
        ).to(device=device, dtype=dtype)

        self.encoder = nn.Conv2d(
            compressed_channels,
            (self.scale_factor ** 2) * (self.kernel_size ** 2),
            kernel_size=3, padding=1
        ).to(device=device, dtype=dtype)

        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._init_layers(x.shape[1], x.device, x.dtype)

        B, C, H, W = x.shape

        # Step 1: 预测上采样核
        compressed = self.channel_compressor(x)  # (B, C_comp, H, W)
        kernels = self.encoder(compressed)  # (B, σ²K², H, W)

        # Step 2: 重排上采样核
        kernels = kernels.permute(0, 2, 3, 1)  # (B, H, W, σ²K²)
        kernels = kernels.reshape(B, H, W, self.scale_factor ** 2, self.kernel_size ** 2)
        kernels = self.kernel_normalizer(kernels)  # 归一化，使权重和为1

        # Step 3: 提取邻域特征
        K = self.kernel_size
        pad = K // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='replicate')
        x_unfolded = F.unfold(x_padded, K, stride=1)  # (B, C*K², H*W)
        x_unfolded = x_unfolded.view(B, C, K ** 2, H * W).permute(0, 3, 2, 1)  # (B, H*W, K², C)

        # Step 4: 特征重组（向量化实现）
        # 将邻域特征与上采样核相乘并重组
        kernels = kernels.reshape(B, H, W, self.scale_factor ** 2, K ** 2)  # (B, H, W, σ², K²)
        kernels = kernels.permute(0, 1, 2, 4, 3)  # (B, H, W, K², σ²)
        kernels = kernels.reshape(B, H * W, K ** 2, self.scale_factor ** 2)  # (B, H*W, K², σ²)

        # 邻域特征: (B, H*W, K², C)
        # 爱因斯坦求和: 对每个位置，用 K² 维核权重对 C 维特征加权
        out = torch.einsum('bnkc,bnks->bnsc', x_unfolded, kernels)  # (B, H*W, C, σ²)
        out = out.reshape(B, H, W, C, self.scale_factor ** 2)  # (B, H, W, C, σ²)
        out = out.permute(0, 3, 1, 4, 2)  # (B, C, H, σ², W)
        out = out.reshape(B, C, H * self.scale_factor, W * self.scale_factor)

        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试不同通道数的输入
    for ch in [64, 128, 256]:
        x = torch.randn(2, ch, 20, 20).to(device)
        carafe = CARAFE(scale_factor=2, kernel_size=3, compress_ratio=4).to(device)
        out = carafe(x)
        print(f"Input channels: {ch:3d}, Input shape: {x.shape}, Output shape: {out.shape}")
        print(f"  Params: {sum(p.numel() for p in carafe.parameters()):,}")
        loss = out.mean()
        loss.backward()
        print("  Backward OK")