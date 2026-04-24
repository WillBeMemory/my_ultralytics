# ============================================
# File: ultralytics/nn/modules/GSConv.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class ChannelShuffle(nn.Module):
    """通道混洗：将输入通道分组重新排列，实现组间信息交互。"""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(B, C, H, W)


class GSConv(nn.Module):
    """
    Group Shuffle Convolution (GSConv)

    原理：混合标准卷积（SC）和深度可分离卷积（DSC），通过通道混洗补偿 DSC 的信息损失，
    实现轻量化且高精度的特征提取。建议主要用于 Neck（颈部网络）的轻量化。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核尺寸，默认 3
        s (int): 步长，默认 1
        p (int, optional): 填充，默认 None 则自动计算
        g (int): 分组数，默认 1
        act (bool): 是否使用激活函数，默认 True
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = p or k // 2

        # 标准卷积分支（SC）：捕获通道信息
        self.conv_sc = Conv(c1, c2 // 2, k, s, p, g=g, act=act)

        # 深度可分离卷积分支（DSC）：高效空间特征提取
        self.conv_dw = nn.Conv2d(
            c1, c1, k, s, p, groups=c1, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(c1)
        self.act_dw = nn.SiLU() if act else nn.Identity()
        self.pw_conv = nn.Conv2d(c1, c2 // 2, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2 // 2)
        self.act_pw = nn.SiLU() if act else nn.Identity()

        # 通道混洗，用于融合两组特征
        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x):
        # 标准卷积分支
        out_sc = self.conv_sc(x)  # (B, C//2, H, W)

        # 深度可分离卷积分支
        out_dw = self.conv_dw(x)  # (B, C, H, W)
        out_dw = self.bn_dw(out_dw)
        out_dw = self.act_dw(out_dw)
        out_dw = self.pw_conv(out_dw)  # (B, C//2, H, W)
        out_dw = self.bn_pw(out_dw)
        out_dw = self.act_pw(out_dw)

        # 拼接两组特征
        out = torch.cat([out_sc, out_dw], dim=1)  # (B, C, H, W)

        # 通道混洗，使信息均匀混合
        out = self.shuffle(out)
        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GSConv(64, 128, k=3, s=2).to(device)  # 模拟 stride=2 的下采样
    x = torch.randn(2, 64, 160, 160).to(device)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")  # 期望 [2, 128, 80, 80]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")