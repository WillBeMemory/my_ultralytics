import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.block import Conv


# ================== Haar 小波滤波器 ==================
def create_haar_filters(channels, dtype=torch.float):
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    LL = lo.unsqueeze(0) * lo.unsqueeze(1)
    LH = lo.unsqueeze(0) * hi.unsqueeze(1)
    HL = hi.unsqueeze(0) * lo.unsqueeze(1)
    HH = hi.unsqueeze(0) * hi.unsqueeze(1)
    filters = torch.stack([LL, LH, HL, HH]).unsqueeze(1).repeat(channels, 1, 1, 1)
    return filters, filters.clone()


class WTConv2d(nn.Module):
    """
    多级 Haar 小波深度卷积（彻底修复尺寸对齐）
    严格按照原始 WTConv 逻辑：下采样得到的低频与上一级重构后再次下采样得到的低频相加
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, wt_levels=2, bias=True, use_activation=True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.channels = in_channels
        self.wt_levels = wt_levels
        self.use_activation = use_activation

        dec_filters, rec_filters = create_haar_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)
        self.pad = 0  # Haar 2x2 无需额外 padding

        # 基础深度可分离卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels, bias=bias)
        self.base_bn = nn.BatchNorm2d(in_channels) if use_activation else nn.Identity()
        self.base_act = nn.SiLU(inplace=True) if use_activation else nn.Identity()
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 各级子带深度可分离卷积
        self.wavelet_convs = nn.ModuleList()
        self.wavelet_bns = nn.ModuleList()
        self.wavelet_acts = nn.ModuleList()
        self.wavelet_scales = nn.ParameterList()
        for _ in range(wt_levels):
            conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                             padding=kernel_size // 2, groups=in_channels, bias=False)
            self.wavelet_convs.append(conv)
            self.wavelet_bns.append(nn.BatchNorm2d(in_channels * 4) if use_activation else nn.Identity())
            self.wavelet_acts.append(nn.SiLU(inplace=True) if use_activation else nn.Identity())
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.5))

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

        # 保存各级处理后的 LL 和 H，以及原始输入尺寸
        ll_list = []
        h_list = []
        shapes = []
        current = x
        for lvl in range(self.wt_levels):
            shapes.append(current.shape[2:])
            coeffs = self._dwt(current)
            flat = coeffs.reshape(B, C * 4, coeffs.shape[-2], coeffs.shape[-1])
            conv_out = self.wavelet_convs[lvl](flat)
            conv_out = self.wavelet_bns[lvl](conv_out)
            conv_out = self.wavelet_acts[lvl](conv_out)
            conv_out = conv_out * self.wavelet_scales[lvl]
            processed = conv_out.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
            ll_list.append(processed[:, :, 0])          # LL
            h_list.append(processed[:, :, 1:4])          # (LH, HL, HH)
            current = processed[:, :, 0]

        # 重构
        next_ll = 0
        for lvl in reversed(range(self.wt_levels)):
            curr_ll = ll_list[lvl]          # 当前级 LL
            curr_h = h_list[lvl]            # 当前级高频 (B,C,3,H_l,W_l)
            curr_shape = shapes[lvl]        # 当前级输入尺寸 (H_in, W_in)

            # 非最深级：将 next_ll 下采样到当前级 LL 尺寸，并与 curr_ll 相加
            if lvl < self.wt_levels - 1:
                # 先将 next_ll 调整到偶数尺寸（避免 DWT 错位）
                nh, nw = next_ll.shape[2], next_ll.shape[3]
                pad_h = (2 - nh % 2) % 2
                pad_w = (2 - nw % 2) % 2
                if pad_h or pad_w:
                    next_ll_padded = F.pad(next_ll, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    next_ll_padded = next_ll
                next_ll_dwt = self._dwt(next_ll_padded)[:, :, 0]  # 低频
                # 如果尺寸仍有 1px 偏差，插值到 curr_ll 尺寸
                if next_ll_dwt.shape[-2:] != curr_ll.shape[-2:]:
                    next_ll_dwt = F.interpolate(next_ll_dwt, size=curr_ll.shape[-2:], mode='nearest')
                curr_ll = curr_ll + next_ll_dwt
            # 最深级：next_ll 为 0，直接使用 curr_ll

            lh, hl, hh = curr_h[:, :, 0], curr_h[:, :, 1], curr_h[:, :, 2]
            # 确保高频与 LL 尺寸一致（应该已经一致，但仍做保险）
            if curr_ll.shape[-2:] != lh.shape[-2:]:
                lh = F.interpolate(lh, size=curr_ll.shape[-2:], mode='nearest')
                hl = F.interpolate(hl, size=curr_ll.shape[-2:], mode='nearest')
                hh = F.interpolate(hh, size=curr_ll.shape[-2:], mode='nearest')

            coeffs = torch.stack([curr_ll, lh, hl, hh], dim=2)
            rec = self._idwt(coeffs, curr_shape[0], curr_shape[1])
            next_ll = rec

        # 小波支路最终输出，裁剪到原始尺寸
        wave_out = next_ll[:, :, :orig_h, :orig_w]

        # 基础支路
        base = self.base_conv(x)
        base = self.base_bn(base)
        base = self.base_act(base)
        base = base * self.base_scale

        return base + wave_out


class StarBlock(nn.Module):
    def __init__(self, channels, e=0.5):
        super().__init__()
        c_ = int(channels * e)
        self.conv1 = nn.Conv2d(channels, c_, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, c_, 1, bias=False)
        self.fusion = nn.Conv2d(c_, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        star = x1 * x2
        out = self.fusion(star)
        out = self.bn(out)
        out = self.act(out + identity)
        return out


class WTBlock(nn.Module):
    def __init__(self, c, kernel_size=5, wt_levels=2, use_star=True, star_e=0.5, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.wtconv = WTConv2d(c, c, kernel_size=kernel_size, wt_levels=wt_levels, use_activation=True)
        self.star = StarBlock(c, e=star_e) if use_star else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.wtconv(x)
        out = self.star(out)
        if self.shortcut:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


class C3k2WT(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 kernel_size=5, wt_levels=2, use_star=True, star_e=0.5, **kwargs):
        super().__init__()
        self.c = int(c2 * e)
        self.n = n
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList([
            WTBlock(self.c, kernel_size=kernel_size, wt_levels=wt_levels,
                    use_star=use_star, star_e=star_e, shortcut=shortcut)
            for _ in range(n)
        ])
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]
        a = y[1]
        for m in self.m:
            a = m(a)
            outputs.append(a)
        return self.act(self.cv2(torch.cat(outputs, dim=1)))


# 测试
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    x = torch.randn(1, 64, 43, 43).to(device)
    model = C3k2WT(64, 128, n=1, c3k=False, e=0.5, wt_levels=2, shortcut=True).to(device)
    model.train()
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [1,128,43,43])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")