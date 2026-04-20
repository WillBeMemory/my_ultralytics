import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # 确保 Conv 模块已导入

_all_ = ["SPPF_LSKA"]
class LSKA(nn.Module):
    """大可分离核注意力模块 (Large Separable Kernel Attention)"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        # 深度可分离卷积的分解：先水平后垂直
        self.conv0_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), groups=dim)
        self.conv0_v = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), groups=dim)
        self.conv1_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), groups=dim)
        self.conv1_v = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), groups=dim)

        # 用于生成最终注意力权重的 1x1 卷积
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        # 分解卷积路径
        attn = self.conv0_h(x)
        attn = self.conv0_v(attn)
        attn = self.conv1_h(attn)
        attn = self.conv1_v(attn)
        # 生成注意力图
        attn = self.conv(attn)
        # 与原始输入相乘，实现注意力加权
        return u * attn

class SPPF_LSKA(nn.Module):
    """结合 LSKA 的快速空间金字塔池化模块"""
    def __init__(self, c1, c2, k=5, lska_kernel_size=7):
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数，与原始 SPPF 一致
        self.cv1 = Conv(c1, c_, 1, 1)          # 降维
        self.cv2 = Conv(c_ * 4, c2, 1, 1)      # 融合后升维

        # LSKA 注意力模块，放在池化之后、融合之前
        self.lska = LSKA(c_ * 4, kernel_size=lska_kernel_size)

        # 最大池化层，与 SPPF 保持一致
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)  # (B, c_, H, W)

        # 串行池化，生成多尺度特征
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # 拼接四个尺度（包括原始特征）
        concat_features = torch.cat((x, y1, y2, y3), 1)  # (B, c_*4, H, W)

        # 应用 LSKA 注意力
        attended_features = self.lska(concat_features)

        # 最终卷积输出
        return self.cv2(attended_features)
