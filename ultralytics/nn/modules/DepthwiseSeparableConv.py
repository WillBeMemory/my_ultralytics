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
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    LL = lo.unsqueeze(0) * lo.unsqueeze(1)
    LH = lo.unsqueeze(0) * hi.unsqueeze(1)
    HL = hi.unsqueeze(0) * lo.unsqueeze(1)
    HH = hi.unsqueeze(0) * hi.unsqueeze(1)
    filters = torch.stack([LL, LH, HL, HH]).unsqueeze(1).repeat(channels, 1, 1, 1)
    return filters, filters.clone()


class WTConv2d(nn.Module):
    """轻量 Haar 小波深度卷积，内部所有卷积均为深度可分离"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 wt_levels=2, bias=True, use_activation=True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d requires in_channels == out_channels"
        self.channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.use_activation = use_activation

        dec_filters, rec_filters = create_haar_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)
        self.pad = 0

        # 基础深度可分离卷积分支
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=autopad(kernel_size), groups=in_channels, bias=bias)
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
                             padding=autopad(kernel_size), groups=in_channels * 4, bias=False)
            self.wavelet_convs.append(conv)
            self.wavelet_bns.append(nn.BatchNorm2d(in_channels * 4) if use_activation else nn.Identity())
            self.wavelet_acts.append(nn.SiLU(inplace=True) if use_activation else nn.Identity())
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

        # 下采样方式
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
            current = processed[:, :, 0]

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

        base = self.base_conv(x)
        base = self.base_bn(base)
        base = self.base_act(base)
        base = base * self.base_scale

        out = base + current
        if self.do_stride is not None:
            out = self.do_stride(out)
        return out


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


class StarPointwise(nn.Module):
    """带星操作的逐点卷积：先 1×1 映射到目标通道，再用星操作增强通道交互"""
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        self.mapping = nn.Conv2d(c1, c2, 1, bias=False)  # stride 固定为 1，下采样由 depthwise 完成
        self.star = StarBlock(c2, e=e)

    def forward(self, x):
        x = self.mapping(x)
        x = self.star(x)
        return x


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    """
    深度可分离 Haar 小波卷积（星操作增强版）
    - 深度卷积：WTConv2d (Haar 小波多尺度分解，负责空间下采样)
    - 逐点卷积：StarPointwise (1×1 映射 + 星操作通道交互)
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 wt_levels=2, star_e=0.5, use_activation=True):
        super().__init__()
        # 小波深度卷积（stride 控制下采样，通过平均池化实现）
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, wt_levels=wt_levels, use_activation=use_activation)
        # 星操作增强的逐点卷积（不改变空间尺寸）
        self.pointwise = StarPointwise(in_channels, out_channels, e=star_e)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 简单测试
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试 stride=2 下采样
    model = DepthwiseSeparableConvWithWTConv2d(64, 128, kernel_size=3, stride=2, wt_levels=2).to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"stride=2: input {x.shape} → output {y.shape} (expected [2,128,16,16])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")