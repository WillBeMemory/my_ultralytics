import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ['SoftFillEdgeEnhance']


# -------------------------- YOLO官方标准Bottleneck(无修改) --------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# -------------------------- 修复维度报错：Same Padding 保证尺寸不变 --------------------------
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, channels, pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.padding = pool_size // 2  # 强制Same卷积，输入=输出尺寸
        self.avg_pool = nn.AvgPool2d(pool_size, 1, self.padding)
        self.fill_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_abs = x.abs()
        max_s = F.max_pool2d(x_abs, self.pool_size, 1, self.padding)
        min_s = -F.max_pool2d(-x_abs, self.pool_size, 1, self.padding)

        local_contrast = (max_s - min_s).clamp(min=1e-6)
        bg_mask = (local_contrast < torch.mean(local_contrast, dim=1, keepdim=True)).float()
        bg = self.avg_pool(x)

        return x * (1 - bg_mask * self.fill_strength) + bg * bg_mask * self.fill_strength


# -------------------------- 通道注意力(维度安全) --------------------------
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.padding = pool_size // 2
        self.ch_sharp = nn.Parameter(torch.tensor(5.0))
        self.edge_sharp = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        x_abs = x.abs()

        # 通道权重
        max_ch = x_abs.flatten(2).max(-1, keepdim=True)[0].view(x.shape[0], x.shape[1], 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = (max_ch - avg_ch).clamp(min=1e-6)
        ch_weight = torch.sigmoid(diff_ch * self.ch_sharp)

        # 空间权重
        avg_s = F.avg_pool2d(x_abs, self.pool_size, 1, self.padding)
        max_s = F.max_pool2d(x_abs, self.pool_size, 1, self.padding)
        diff_s = (max_s - avg_s).clamp(min=1e-6)
        edge_weight = torch.sigmoid(diff_s * self.edge_sharp)

        return x * ch_weight * edge_weight


# -------------------------- 核心模块【严格遵守YOLO接口：c1, c2 固定参数】 --------------------------
class SoftFillEdgeEnhance(nn.Module):
    """
    YOLO官方接口规范：__init__(c1, c2)
    c1: 输入通道
    c2: 输出通道 (本模块残差结构，c1必须=c2)
    你的YAML写 [-1, 1, SoftFillEdgeEnhance, [256]] → 自动解析 c1=256, c2=256
    """

    def __init__(self, c1, c2, pool_size=3, shortcut=True):
        super().__init__()
        # 强制通道一致（残差模块要求）
        assert c1 == c2, "SoftFillEdgeEnhance 输入输出通道必须一致 (c1=c2)"
        self.c1 = c1
        self.shortcut = shortcut

        # 核心功能
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size)
        self.bottlenecks = nn.Sequential(Bottleneck(c1, c1), Bottleneck(c1, c1))
        self.proj = Conv(c1, c2, 1)

    def forward(self, x):
        identity = x
        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)
        return out + identity if self.shortcut else out