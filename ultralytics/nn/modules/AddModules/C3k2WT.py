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
    """轻量 Haar 小波深度卷积（修正多级重构尺寸对齐）"""
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

        # 基础深度可分离卷积分支
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
                             padding=kernel_size // 2, groups=in_channels * 4, bias=False)
            self.wavelet_convs.append(conv)
            self.wavelet_bns.append(nn.BatchNorm2d(in_channels * 4) if use_activation else nn.Identity())
            self.wavelet_acts.append(nn.SiLU(inplace=True) if use_activation else nn.Identity())
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

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

        # 多级分解并保存处理后的全部子带
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
            current = processed[:, :, 0]          # LL 进入下一级

        # 逐级重构（修正尺寸对齐）
        current = None
        for lvl in reversed(range(self.wt_levels)):
            if lvl == self.wt_levels - 1:          # 最深级：直接使用处理后的 LL
                ll = levels[lvl][:, :, 0]
            else:
                # 关键修正：将上一级重构图像 current 进行小波分解，取其 LL 分量
                # 这样得到的低频尺寸与当前级的高频子带完全匹配
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

        # 基础深度卷积分支
        base = self.base_conv(x)
        base = self.base_bn(base)
        base = self.base_act(base)
        base = base * self.base_scale

        return base + current


class StarBlock(nn.Module):
    """星操作块：两个 1×1 投影 → 逐元素相乘 → 融合，带残差连接"""
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
    """WTConv2d + 可选 StarBlock 的特征精炼块"""
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
    """
    融合小波卷积的 C3k2 模块，接口完全兼容 C3k2。

    YAML 示例（与原版 C3k2 参数顺序一致）：
    [-1, 2, C3k2WT, [128, 1, False, 0.25]]
    [-1, 2, C3k2WT, [128, 1, False, 0.25, True, 1, 5, 2, True, 0.5]]
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 kernel_size=5, wt_levels=2, use_star=True, star_e=0.5, **kwargs):
        super().__init__()
        self.c = int(c2 * e)          # 分支通道数
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
        y = list(self.cv1(x).chunk(2, dim=1))   # [b, a]
        a = y[1]   # 增强分支

        outputs = [y[0], a]
        for m in self.m:
            a = m(a)
            outputs.append(a)

        out = torch.cat(outputs, dim=1)
        out = self.act(self.cv2(out))
        return out