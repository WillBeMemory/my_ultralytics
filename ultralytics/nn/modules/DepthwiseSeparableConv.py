import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def create_2d_wavelet_filter(wave, channels, dtype=torch.float):
    """生成任意小波基的二维分解/重构滤波器 (4*C, 1, len, len)"""
    w = pywt.Wavelet(wave)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype)

    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0).unsqueeze(1)  # (4, 1, len, len)

    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0).unsqueeze(1)

    dec_filters = dec_filters.repeat(channels, 1, 1, 1)
    rec_filters = rec_filters.repeat(channels, 1, 1, 1)
    return dec_filters, rec_filters


class WTConv2d(nn.Module):
    """
    修正版小波卷积（深度可分离，支持多级分解）
    - 所有卷积均为分组卷积（groups = channels），实现深度可分离。
    - 可选 BatchNorm 和 SiLU 激活，增强非线性表达能力。
    - 修复了官方源码中多级重构时的尺寸不匹配问题。
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 wt_levels=2, wavelet='bior4.4', bias=True, use_activation=True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.use_activation = use_activation

        # 固定小波滤波器
        dec_filters, rec_filters = create_2d_wavelet_filter(wavelet, in_channels)
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)
        self.pad = dec_filters.shape[2] // 2 - 1

        # 基础深度卷积分支
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=autopad(kernel_size), groups=in_channels, bias=bias)
        self.base_bn = nn.BatchNorm2d(in_channels) if use_activation else nn.Identity()
        self.base_act = nn.SiLU(inplace=True) if use_activation else nn.Identity()
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 各级子带深度卷积
        self.wavelet_convs = nn.ModuleList()
        self.wavelet_bns = nn.ModuleList()
        self.wavelet_acts = nn.ModuleList()
        self.wavelet_scales = nn.ParameterList()
        for _ in range(wt_levels):
            conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                             padding=autopad(kernel_size), groups=in_channels * 4, bias=False)
            self.wavelet_convs.append(conv)
            self.wavelet_bns.append(nn.BatchNorm2d(in_channels * 4) if use_activation else nn.Identity())
            self.wavelet_acts.append(nn.SiLU(inplace=True) if use_activation else nn.Identity())
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

        # 下采样方式：stride > 1 时使用平均池化（kernel_size=2）
        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=2, stride=stride)
        else:
            self.do_stride = None

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        coeffs = F.conv2d(x, self.dec_filters, stride=2, groups=C)
        return coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])

    def _idwt(self, coeffs, target_h, target_w):
        B, C, _, h, w = coeffs.shape
        flat = coeffs.reshape(B, C * 4, h, w)
        out = F.conv_transpose2d(flat, self.rec_filters, stride=2, groups=C, padding=self.pad)
        return out[:, :, :target_h, :target_w]

    def forward(self, x):
        B, C, H, W = x.shape
        orig_h, orig_w = H, W

        # 多级分解与子带卷积
        levels = []
        current = x
        for lvl in range(self.wt_levels):
            coeffs = self._dwt(current)
            flat = coeffs.reshape(B, C * 4, coeffs.shape[-2], coeffs.shape[-1])
            conv_out = self.wavelet_convs[lvl](flat)
            conv_out = self.wavelet_bns[lvl](conv_out)
            conv_out = self.wavelet_acts[lvl](conv_out)
            conv_out = conv_out * self.wavelet_scales[lvl]
            processed = conv_out.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
            levels.append(processed)
            current = processed[:, :, 0]  # LL 进入下一级

        # 逐级重构（修正尺寸对齐）
        current = None
        for lvl in reversed(range(self.wt_levels)):
            if lvl == self.wt_levels - 1:
                ll = levels[lvl][:, :, 0]
            else:
                ll = current
            lh = levels[lvl][:, :, 1]
            hl = levels[lvl][:, :, 2]
            hh = levels[lvl][:, :, 3]
            coeffs = torch.stack([ll, lh, hl, hh], dim=2)
            target_h = levels[lvl].shape[-2] * 2
            target_w = levels[lvl].shape[-1] * 2
            if lvl == 0:
                target_h, target_w = orig_h, orig_w
            current = self._idwt(coeffs, target_h, target_w)

        # 基础卷积分支（带 BN 和激活）
        base = self.base_conv(x)
        base = self.base_bn(base)
        base = self.base_act(base)
        base = base * self.base_scale

        out = base + current

        if self.do_stride is not None:
            out = self.do_stride(out)
        return out


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """
    深度可分离小波卷积 = 小波深度卷积 (WTConv2d) + 1×1 逐点卷积
    支持 BN 与 SiLU 激活，可替换标准深度可分离卷积。
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 wt_levels=2, wavelet='bior4.4', use_activation=False):
        super().__init__()
        # 小波增强的深度卷积（内部已包含 BN 和激活）
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=1, wt_levels=wt_levels, wavelet=wavelet,
                                  use_activation=use_activation)
        # 逐点卷积 + BN + 激活
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels) if use_activation else nn.Identity()
        self.pw_act = nn.SiLU(inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


# ---------- 简单测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试深度可分离小波卷积（带 BN 和激活）
    model = DepthwiseSeparableConvWithWTConv2d(64, 128, kernel_size=5, stride=1,
                                               wt_levels=2, wavelet='bior4.4').to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"DepthwiseSeparableConvWithWTConv2d: input {x.shape} → output {y.shape}")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")

    # 测试 stride=2 下采样
    model2 = DepthwiseSeparableConvWithWTConv2d(64, 128, kernel_size=5, stride=2,
                                                wt_levels=2, wavelet='bior4.4').to(device)
    model2.train()
    x2 = torch.randn(2, 64, 32, 32).to(device)
    y2 = model2(x2)
    print(f"stride=2: input {x2.shape} → output {y2.shape} (expected [2,128,16,16])")
    loss2 = y2.mean()
    loss2.backward()
    print("Gradients OK")