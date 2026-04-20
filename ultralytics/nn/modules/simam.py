import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C3k, PSABlock, C3k2

__all__ = ['SimAM', 'C3k2_SimAM']


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module (ICML 2021)
    """
    def __init__(self, lambda_=1e-4):
        super().__init__()
        self.lambda_ = lambda_
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算均值和方差
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        # 能量函数 (公式 3)
        e = 4 * (var + self.lambda_) / ((x - mu).pow(2) + 2 * var + self.lambda_ + 1e-6)
        # 注意力权重 = sigmoid(1/e)
        weight = self.sigmoid(1 / e)
        return x * weight


class BottleneckWithSimAM(nn.Module):
    """
    通用包装器：将任意瓶颈模块的输出通过 SimAM 后再返回
    """
    def __init__(self, bottleneck):
        super().__init__()
        self.bottleneck = bottleneck
        self.simam = SimAM()

    def forward(self, x):
        out = self.bottleneck(x)
        out = self.simam(out)
        return out


class C3k2_SimAM(C3k2):
    """
    C3k2 模块的 SimAM 增强版本
    支持原 C3k2 的所有参数，并在每个内部模块（Bottleneck/C3k/PSABlock）后插入 SimAM
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        # 调用父类 C3k2 的初始化，以构建 cv1, cv2, self.c 等基础结构
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)

        # 重新构建 self.m：将原有的每个内部模块用 BottleneckWithSimAM 包装
        self.m = nn.ModuleList()
        for _ in range(n):
            if attn:
                # attention 模式：Bottleneck + PSABlock
                orig = nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, g),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
                )
            elif c3k:
                # c3k 模式：使用 C3k 模块
                orig = C3k(self.c, self.c, 2, shortcut, g)
            else:
                # 默认模式：普通 Bottleneck
                orig = Bottleneck(self.c, self.c, shortcut, g)

            self.m.append(BottleneckWithSimAM(orig))

    def forward(self, x):
        """与 C3k2 完全相同，直接复用父类的 forward 即可（因为 self.m 已被替换）"""
        return super().forward(x)