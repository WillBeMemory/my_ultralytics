import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

__all__ = ['DepthwiseSeparableConvWithWTConv2d']


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dims) * init_scale)

    def forward(self, x):
        return x * self.scale


def haar_decompose_filters(in_channels: int):
    """Haar 分解滤波器 (DWT)"""
    dec_lo = torch.tensor([1/2, 1/2])
    dec_hi = torch.tensor([1/2, -1/2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),   # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),   # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),   # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)    # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    filters = filters.reshape(in_channels * 4, 1, 2, 2)
    return filters


def haar_reconstruct_filters(in_channels: int):
    """Haar 重建滤波器 (IDWT)"""
    rec_lo = torch.tensor([1., 1.])
    rec_hi = torch.tensor([1., -1.])
    base = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),   # LL
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),   # LH
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),   # HL
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)    # HH
    ], dim=0)  # (4, 2, 2)
    # 归一化保持能量一致（与分解配合）
    base = base / 2.0
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    filters = filters.reshape(in_channels * 4, 1, 2, 2)
    return filters


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    x = F.conv2d(x, filters, stride=2, groups=C, padding=0)
    return x


def wavelet_reconstruct(coeffs: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    B, C4, H, W = coeffs.shape
    C = C4 // 4
    x = F.conv_transpose2d(coeffs, filters, stride=2, groups=C, padding=0)
    return x


class WTConv2d(nn.Module):
    """
    递归小波卷积模块（修正版）。
    使用 Haar 小波分解与逆变换，在多个尺度上对子带进行卷积增强。
    要求输入输出通道数相同，空间尺寸保持不变。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 depth: int = 2, bias: bool = True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.depth = depth

        # 注册分解与重建滤波器
        dec_filters = haar_decompose_filters(in_channels)
        rec_filters = haar_reconstruct_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)

        # 基卷积（处理原始尺度特征）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size//2, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 每个小波层级的子带卷积
        self.wavelet_convs = nn.ModuleList()
        for level in range(depth):
            level_convs = nn.ModuleDict({
                'LL': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2,
                                groups=in_channels, bias=False),
                'LH': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2,
                                groups=in_channels, bias=False),
                'HL': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2,
                                groups=in_channels, bias=False),
                'HH': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2,
                                groups=in_channels, bias=False),
            })
            self.wavelet_convs.append(level_convs)
        # 整体的 wavelet 分支缩放，初始较小以优先基卷积
        self.wavelet_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)

    def _decompose_recursive(self, x: torch.Tensor, level: int):
        """递归分解，返回最深 LL 和每一层高频子带列表"""
        if level == self.depth:
            return x, []   # 最深层的 LL
        coeffs = wavelet_decompose(x, self.dec_filters)  # (B, C*4, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]
        final_ll, high_list = self._decompose_recursive(ll, level + 1)
        current_high = {'LH': lh, 'HL': hl, 'HH': hh}
        high_list.insert(0, current_high)
        return final_ll, high_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 填充到 2^depth 的倍数，确保整除
        divisor = 2 ** self.depth
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        # 递归分解获得最深 LL 和高频列表
        final_ll, high_list = self._decompose_recursive(x_pad, 0)

        # 对每一层的高频子带进行卷积增强
        processed_high_list = []
        for level in range(self.depth):
            convs = self.wavelet_convs[level]
            high = high_list[level]
            processed_high = {
                'LH': convs['LH'](high['LH']),
                'HL': convs['HL'](high['HL']),
                'HH': convs['HH'](high['HH']),
            }
            processed_high_list.append(processed_high)

        # 最深层的 LL 也进行卷积
        processed_final_ll = self.wavelet_convs[-1]['LL'](final_ll)

        # 逆小波重建，自底向上
        x_wt = processed_final_ll
        for level in reversed(range(self.depth)):
            high = processed_high_list[level]
            lh = high['LH']
            hl = high['HL']
            hh = high['HH']
            combined = torch.stack([x_wt, lh, hl, hh], dim=2)  # (B, C, 4, Hc, Wc)
            B_, C_, _, Hc, Wc = combined.shape
            combined = combined.reshape(B_, C_ * 4, Hc, Wc)
            x_wt = wavelet_reconstruct(combined, self.rec_filters)  # 尺寸翻倍

        # 裁剪掉之前填充的部分，恢复到原始尺寸
        x_wt = x_wt[:, :, :H, :W]
        # 可学习的缩放
        x_wt = self.wavelet_scale(x_wt)

        # 基卷积处理原始特征
        x_base = self.base_conv(x)
        x_base = self.base_scale(x_base)

        return x_base + x_wt


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """
    深度可分离卷积 + 小波增强模块。
    使用 WTConv2d 作为深度卷积，再接点卷积；支持步长 >1 时通过平均池化实现下采样。
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, wt_depth=2):
        super().__init__()
        # 深度小波卷积（c1 -> c1，尺寸不变，除非 stride 用池化处理）
        self.depthwise = WTConv2d(c1, c1, kernel_size=k, depth=wt_depth, bias=False)
        # 点卷积
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 下采样用平均池化
        self.downsample = nn.AvgPool2d(s, stride=s) if s > 1 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)       # 保持尺寸不变
        x = self.downsample(x)      # 如果需要下采样
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试带下采样的模块
    model_down = DepthwiseSeparableConvWithWTConv2d(64, 128, k=3, s=2, wt_depth=2).to(device)
    x = torch.randn(2, 64, 80, 80).to(device)
    out = model_down(x)
    print(f"下采样测试 - 输入尺寸: {x.shape}, 输出尺寸: {out.shape}")

    # 测试不同深度下保持尺寸的模块
    for d in [1, 2, 3]:
        model_d = DepthwiseSeparableConvWithWTConv2d(64, 64, k=3, s=1, wt_depth=d).to(device)
        out_d = model_d(x)
        print(f"depth={d}, stride=1, 输出尺寸: {out_d.shape}, 参数量: {sum(p.numel() for p in model_d.parameters()):,}")