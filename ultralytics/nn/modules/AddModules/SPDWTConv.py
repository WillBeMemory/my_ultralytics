import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def create_haar_filters(channels, dtype=torch.float):
    """生成 Haar 小波分解和重构滤波器 (4*C, 1, 2, 2)"""
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    LL = lo.unsqueeze(0) * lo.unsqueeze(1)
    LH = lo.unsqueeze(0) * hi.unsqueeze(1)
    HL = hi.unsqueeze(0) * lo.unsqueeze(1)
    HH = hi.unsqueeze(0) * hi.unsqueeze(1)
    dec_filters = torch.stack([LL, LH, HL, HH])      # (4, 2, 2)
    dec_filters = dec_filters.unsqueeze(1)            # (4, 1, 2, 2)
    dec_filters = dec_filters.repeat(channels, 1, 1, 1)  # (4*C, 1, 2, 2)
    rec_filters = dec_filters.clone()
    return dec_filters, rec_filters


class HaarWTConv2d(nn.Module):
    """纯 PyTorch 实现的多级 Haar 小波卷积（修正版）"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, wt_levels=2, bias=True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d 要求 in_channels == out_channels"
        self.channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        dec_filters, rec_filters = create_haar_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same',
                                   groups=in_channels, bias=bias)
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.wavelet_convs = nn.ModuleList()
        self.wavelet_scales = nn.ParameterList()
        for _ in range(wt_levels):
            conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                             groups=in_channels * 4, bias=False)
            self.wavelet_convs.append(conv)
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        coeffs = F.conv2d(x, self.dec_filters, stride=2, groups=C)
        return coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])

    def _idwt(self, coeffs, target_h, target_w):
        B, C, _, h, w = coeffs.shape
        flat = coeffs.reshape(B, C * 4, h, w)
        out = F.conv_transpose2d(flat, self.rec_filters, stride=2, groups=C)
        return out[:, :, :target_h, :target_w]

    def forward(self, x):
        B, C, H, W = x.shape
        orig_h, orig_w = H, W

        # -------- 多级分解与子带卷积 --------
        levels = []
        current = x
        for lvl in range(self.wt_levels):
            coeffs = self._dwt(current)
            flat = coeffs.reshape(B, C * 4, coeffs.shape[-2], coeffs.shape[-1])
            conv_out = self.wavelet_convs[lvl](flat) * self.wavelet_scales[lvl]
            processed = conv_out.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
            levels.append(processed)
            current = processed[:, :, 0]

        # -------- 逐级重构（修复尺寸对齐）--------
        current = None
        for lvl in reversed(range(self.wt_levels)):
            if lvl == self.wt_levels - 1:          # 最深级
                ll = levels[lvl][:, :, 0]          # 直接使用该级低频
            else:                                   # 其他级
                # 将上一级重构图像分解，取低频分量作为本级的低频
                ll = self._dwt(current)[:, :, 0]

            lh = levels[lvl][:, :, 1]
            hl = levels[lvl][:, :, 2]
            hh = levels[lvl][:, :, 3]
            coeffs = torch.stack([ll, lh, hl, hh], dim=2)

            target_h = levels[lvl].shape[-2] * 2
            target_w = levels[lvl].shape[-1] * 2
            if lvl == 0:
                target_h, target_w = orig_h, orig_w

            current = self._idwt(coeffs, target_h, target_w)

        # -------- 基础卷积支路 --------
        base = self.base_conv(x) * self.base_scale
        if self.base_conv.bias is not None:
            base = base + self.base_conv.bias.view(1, -1, 1, 1)

        out = base + current
        if self.do_stride is not None:
            out = self.do_stride(out)
        return out


class SPDWTConv(nn.Module):
    """
    SPD + WTConv 下采样模块
    - pixel_unshuffle 无损空间到深度
    - 修正版多级 Haar 小波卷积（所有子带处理）
    - 1×1 卷积压缩到输出通道
    """
    def __init__(self, c1, c2, k=1, s=2, kernel_size=5, wt_levels=2):
        super().__init__()
        self.s = s
        if s == 2:
            c_mid = c1 * 4
            self.wtconv = HaarWTConv2d(in_channels=c_mid, out_channels=c_mid,
                                       kernel_size=kernel_size, stride=1,
                                       wt_levels=wt_levels, bias=True)
            self.compress = nn.Conv2d(c_mid, c2, kernel_size=k, stride=1,
                                     padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU(inplace=True)
        else:
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                                 padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            x = F.pixel_unshuffle(x, 2)      # (B, 4C, H/2, W/2)
            x = self.wtconv(x)
            x = self.compress(x)
            return self.act(self.bn(x))
        else:
            return self.act(self.bn(self.conv(x)))


# ---------- 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SPDWTConv(c1=64, c2=128, k=3, s=2, kernel_size=5, wt_levels=2).to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [2,128,16,16])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")