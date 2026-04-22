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


def haar_filters(in_channels: int):
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
    return filters, filters  # dec_filters, rec_filters


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
    递归小波卷积模块（尺寸修复版）。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, depth: int = 2, bias: bool = True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.depth = depth

        dec, rec = haar_filters(in_channels)
        self.register_buffer('dec_filters', dec)
        self.register_buffer('rec_filters', rec)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size//2, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

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
        self.wavelet_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)

    def _decompose_recursive(self, x: torch.Tensor, level: int):
        if level == self.depth:
            return x, []
        coeffs = wavelet_decompose(x, self.dec_filters)
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
        pad_h = (2**self.depth - H % (2**self.depth)) % (2**self.depth)
        pad_w = (2**self.depth - W % (2**self.depth)) % (2**self.depth)
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            x_pad = x

        final_ll, high_list = self._decompose_recursive(x_pad, 0)

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
        processed_final_ll = self.wavelet_convs[-1]['LL'](final_ll)

        x_wt = processed_final_ll
        for level in reversed(range(self.depth)):
            high = processed_high_list[level]
            lh = high['LH']
            hl = high['HL']
            hh = high['HH']
            combined = torch.stack([x_wt, lh, hl, hh], dim=2)
            B, C, _, H_curr, W_curr = combined.shape
            combined = combined.reshape(B, C * 4, H_curr, W_curr)
            x_wt = wavelet_reconstruct(combined, self.rec_filters)

        # 关键修复：强制对齐尺寸
        if x_wt.shape[2] != H or x_wt.shape[3] != W:
            x_wt = F.interpolate(x_wt, size=(H, W), mode='bilinear', align_corners=False)

        x_wt = self.wavelet_scale(x_wt)

        x_base = self.base_conv(x)
        x_base = self.base_scale(x_base)

        return x_base + x_wt


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, wt_depth=2):
        super().__init__()
        self.depthwise = WTConv2d(c1, c1, kernel_size=k, depth=wt_depth, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.downsample = None
        if s > 1:
            self.downsample = nn.AvgPool2d(s, stride=s)

    def forward(self, x):
        x = self.depthwise(x)
        if self.downsample:
            x = self.downsample(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthwiseSeparableConvWithWTConv2d(64, 128, k=3, s=2, wt_depth=2).to(device)
    x = torch.randn(2, 64, 80, 80).to(device)
    out = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {out.shape}")

    for d in [1, 2, 3]:
        model_d = DepthwiseSeparableConvWithWTConv2d(64, 128, k=3, s=1, wt_depth=d).to(device)
        out_d = model_d(x)
        print(f"depth={d}, 输出尺寸: {out_d.shape}, 参数量: {sum(p.numel() for p in model_d.parameters()):,}")