import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 标准 Bottleneck ==================
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.cv1(x)))
        out = self.bn2(self.cv2(out))
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 背景填充模块（局部统计量 + 两级强度） ==================
class AdaptiveBackgroundFill(nn.Module):
    """
    分段背景填充：
    - 使用局部均值（3×3 平均池化）作为高低分界。
    - 低于局部均值的背景区域 → 强压至全局最小值 (fill_strength_high)。
    - 高于局部均值的背景区域 → 弱压至局部均值 (fill_strength_low)。
    - 非背景区域保持原值。
    """
    def __init__(self, ch, pool_size=3,
                 fill_strength_high=0.9, fill_strength_low=0.3,
                 bg_thresh_ratio=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.fill_strength_high = fill_strength_high
        self.fill_strength_low = fill_strength_low
        self.bg_thresh_ratio = bg_thresh_ratio

    def forward(self, x):
        B, C, H, W = x.shape

        # 填充目标：全局最小值
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)

        # 局部均值 (3x3 平均池化)
        pad = self.pool_size // 2
        local_mean = F.avg_pool2d(x, self.pool_size, stride=1, padding=pad)

        # 局部对比度判定背景
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + eps
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)

        # 划分高低区域（仅在背景内）
        high_mask = (x > local_mean).to(dtype=x.dtype) * bg_mask    # 高值背景
        low_mask  = (x <= local_mean).to(dtype=x.dtype) * bg_mask   # 低值背景

        # 分段填充
        out = x.clone()
        # 低值背景：强压至 baseline
        out = out * (1 - self.fill_strength_high * low_mask) + baseline * self.fill_strength_high * low_mask
        # 高值背景：弱压至 local_mean
        out = out * (1 - self.fill_strength_low * high_mask) + local_mean * self.fill_strength_low * high_mask

        return out


# ================== 通道感知边缘增强（注意力） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        dtype = x.dtype
        ch_sharp = self.ch_sharp.to(dtype)
        ch_thresh = self.ch_thresh.to(dtype)
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out


# ================== 完整模块：SoftFillEdgeEnhance ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3,
                 fill_strength_high=0.9, fill_strength_low=0.3,
                 bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(
            c1, pool_size,
            fill_strength_high=fill_strength_high,
            fill_strength_low=fill_strength_low,
            bg_thresh_ratio=bg_thresh_ratio
        )
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh,
                                                 edge_sharp, edge_thresh)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        if len(self.bottlenecks) > 0:
            target_dtype = self.bottlenecks[0].cv1.weight.dtype
        else:
            target_dtype = x.dtype
        x = x.to(target_dtype)

        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)
        return out