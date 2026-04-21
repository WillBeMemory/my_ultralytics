# ============================================
# File: ultralytics/nn/modules/DSWT_GhostConv.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_filters(in_channels: int):
    """生成 Haar 小波的分解/重构滤波器，形状 (in_channels*4, 1, 2, 2)"""
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    return filters.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """小波分解：输入 (B,C,H,W)，输出 (B,C*4,H/2,W/2)"""
    B, C, H, W = x.shape
    x = F.conv2d(x, filters, stride=2, groups=C, padding=0)
    return x


def wavelet_reconstruct(coeffs: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """小波重构：输入 (B,C,4,H,W)，输出 (B,C,H*2,W*2)"""
    B, C, _, H, W = coeffs.shape
    coeffs = coeffs.reshape(B, C * 4, H, W)
    x = F.conv_transpose2d(coeffs, filters, stride=2, groups=C, padding=0)
    return x


class DepthwiseSeparableWaveletConv(nn.Module):
    """
    深度可分离小波卷积 (DSWT)，支持步长 stride。
    流程：深度卷积 (DWConv) → 小波分解 (WT) → 小波重构 (IWT) → 逐点卷积 (PWConv)
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.channels = channels
        self.stride = stride

        # 1. 深度卷积（空间滤波，支持步长）
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=channels, bias=False
        )

        # 2. 小波滤波器（不可训练）
        self.register_buffer('dec_filters', haar_filters(channels))
        self.register_buffer('rec_filters', haar_filters(channels))

        # 3. 逐点卷积（通道混合）
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 深度卷积（可能包含 stride 下采样）
        x = self.dwconv(x)

        # 处理奇数尺寸
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # 小波分解
        coeffs = wavelet_decompose(x, self.dec_filters)  # (B, C*4, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)

        # 小波重构（恢复原尺寸）
        x = wavelet_reconstruct(coeffs, self.rec_filters)  # (B, C, H, W)

        # 裁剪回原始尺寸（若因填充变大）
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        # 逐点卷积
        x = self.pwconv(x)
        return x


class DSWT_GhostConv(nn.Module):
    """
    深度可分离小波幽灵卷积 (DSWT-GhostConv)。
    将 GhostConv 内部的廉价操作替换为 DepthwiseSeparableWaveletConv。

    参数顺序（兼容 YOLO 配置文件）：
        in_channels (int)   : 输入通道数
        out_channels (int)  : 输出通道数
        n (int)             : 重复次数（此处忽略，保留接口兼容）
        kernel_size (int)   : 卷积核尺寸，默认 3
        stride (int)        : 步长，默认 1
        ratio (int)         : 压缩比，内在特征图数量 = out_channels // ratio，默认 2
    """

    def __init__(self, in_channels: int, out_channels: int, n: int = 1,
                 kernel_size: int = 3, stride: int = 1, ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.ratio = ratio

        self.intrinsic_channels = out_channels // ratio
        self.ghost_channels = out_channels - self.intrinsic_channels

        # 1. 生成内在特征图的标准卷积（1x1 点卷积，无下采样）
        self.primary_conv = nn.Conv2d(in_channels, self.intrinsic_channels, kernel_size=1, bias=False)

        # 2. 当 stride > 1 时，用于对内在特征图下采样的深度卷积（与廉价操作中的 dwconv 一致）
        if stride > 1:
            self.intrinsic_downsample = nn.Conv2d(
                self.intrinsic_channels, self.intrinsic_channels, kernel_size,
                stride=stride, padding=kernel_size // 2, groups=self.intrinsic_channels, bias=False
            )
        else:
            self.intrinsic_downsample = nn.Identity()

        # 3. 廉价操作：深度可分离小波卷积（用于繁衍幽灵特征图，并执行 stride 下采样）
        self.cheap_operation = DepthwiseSeparableWaveletConv(
            channels=self.intrinsic_channels, kernel_size=kernel_size, stride=stride
        )

        # 4. 最终融合层（可选，用于平滑拼接后的特征）
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成内在特征图（保持原始空间尺寸）
        intrinsic = self.primary_conv(x)  # (B, intrinsic, H, W)

        # 对内在特征图下采样（若 stride > 1）
        intrinsic_ds = self.intrinsic_downsample(intrinsic)  # (B, intrinsic, H', W')

        # 通过廉价操作繁衍幽灵特征图（可能包含 stride 下采样）
        ghost = self.cheap_operation(intrinsic)  # (B, intrinsic, H', W')

        # 如果 ghost_channels < intrinsic_channels，则只取部分通道
        if self.ghost_channels < self.intrinsic_channels:
            ghost = ghost[:, :self.ghost_channels, :, :]
            # 内在特征图也可能需要截取以匹配总通道数
            if intrinsic_ds.size(1) > self.intrinsic_channels - self.ghost_channels:
                intrinsic_ds = intrinsic_ds[:, :self.intrinsic_channels - self.ghost_channels, :, :]

        # 拼接内在特征图和幽灵特征图
        out = torch.cat([intrinsic_ds, ghost], dim=1)  # (B, out_channels, H', W')

        # 最终融合
        out = self.final_conv(out)
        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 stride=1
    model1 = DSWT_GhostConv(in_channels=64, out_channels=128, n=1, kernel_size=3, stride=1).to(device)
    x1 = torch.randn(2, 64, 160, 160).to(device)
    out1 = model1(x1)
    print(f"stride=1: Input {x1.shape} -> Output {out1.shape}")

    # 测试 stride=2
    model2 = DSWT_GhostConv(in_channels=64, out_channels=128, n=1, kernel_size=3, stride=2).to(device)
    x2 = torch.randn(2, 64, 160, 160).to(device)
    out2 = model2(x2)
    print(f"stride=2: Input {x2.shape} -> Output {out2.shape}")

    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params:,}")

    loss = out2.mean()
    loss.backward()
    print("Backward pass completed.")