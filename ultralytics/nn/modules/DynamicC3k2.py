# ============================================
# File: ultralytics/nn/modules/DynamicC3k2.py
# ============================================

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d


class DSWTBottleneck(nn.Module):
    """
    深度可分离小波瓶颈模块。
    结构：1x1 压缩 → 深度可分离小波卷积 → 残差连接。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, wt_depth: int = 2):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DepthwiseSeparableConvWithWTConv2d(c_, c2, k=3, wt_depth=wt_depth)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DynamicC3k2(nn.Module):
    """
    动态 C3k2 模块（C3 风格通道分割 + 深度可分离小波卷积）。

    通道分割采用 C3 方式：两个并行的 1x1 卷积分别生成主路径和捷径特征。
    内部块使用 DSWTBottleneck，兼具轻量化与小波变换的多频带分离优势。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 占位参数（兼容 YAML）
        num_bottlenecks (int): 实际堆叠的 DSWTBottleneck 数量，默认 2
        e (float): 隐藏通道压缩比，c_ = int(c2 * e)
        shortcut (bool): 瓶颈模块是否使用残差连接
        g (int): 分组卷积数
        wt_depth (int): 小波递归分解深度，默认 2
        **kwargs: 吸收 YAML 可能传入的其他参数
    """

    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            num_bottlenecks: int = 2,
            e: float = 0.5,
            shortcut: bool = True,
            g: int = 1,
            wt_depth: int = 2,
            **kwargs
    ):
        super().__init__()
        self.c_ = int(c2 * e)  # 隐藏通道数（主路径与捷径的通道数）
        self.num_bottlenecks = num_bottlenecks

        # C3 风格的双路 1x1 压缩
        self.cv1 = Conv(c1, self.c_, 1, 1)  # 主路径压缩
        self.cv2 = Conv(c1, self.c_, 1, 1)  # 捷径压缩

        # 主路径：堆叠 num_bottlenecks 个 DSWTBottleneck
        self.m = nn.ModuleList([
            DSWTBottleneck(self.c_, self.c_, shortcut, g, e=0.5, wt_depth=wt_depth)
            for _ in range(num_bottlenecks)
        ])

        # 输出融合层：捷径(c_) + 主路径输出(c_) = 2*c_ → c2
        self.cv3 = Conv(2 * self.c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 双路压缩
        x_main = self.cv1(x)  # 主路径特征
        x_shortcut = self.cv2(x)  # 捷径特征

        # 主路径串行通过 DSWTBottleneck
        for m in self.m:
            x_main = m(x_main)

        # 拼接并融合
        out = torch.cat([x_main, x_shortcut], dim=1)
        return self.cv3(out)


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DynamicC3k2(c1=256, c2=256, n=1, num_bottlenecks=2, e=0.25, wt_depth=2).to(device)
    x = torch.randn(2, 256, 20, 20).to(device)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")