# ============================================
# File: ultralytics/nn/modules/DynamicC3k2.py
# ============================================

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d


class DSWTBottleneck(nn.Module):
    """
    深度可分离小波瓶颈模块（支持递归深度）。
    结构：1x1 压缩 → 深度可分离小波卷积（递归深度可调） → 残差连接。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, wt_depth: int = 2):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        # 传入 wt_depth 控制小波递归深度
        self.cv2 = DepthwiseSeparableConvWithWTConv2d(c_, c2, k=3, wt_depth=wt_depth)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DynamicC3k2(nn.Module):
    """
    动态 C3k2 模块（深度可分离小波卷积增强版，支持递归深度）。
    内部块使用 DSWTBottleneck，兼具轻量化与小波变换的多频带分离优势。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 占位参数（兼容 YAML）
        num_bottlenecks (int): 实际堆叠的 DSWTBottleneck 数量，默认 2
        e (float): 隐藏通道压缩比，c = int(c2 * e)
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
            wt_depth: int = 2,
            shortcut: bool = True,
            g: int = 1,

            **kwargs
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.num_bottlenecks = num_bottlenecks

        # 外层 CSP 压缩：输入 -> 2*c 通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # 内部块：num_bottlenecks 个 DSWTBottleneck
        self.m = nn.ModuleList([
            DSWTBottleneck(self.c, self.c, shortcut, g, e=0.5, wt_depth=wt_depth)
            for _ in range(num_bottlenecks)
        ])

        # 输出融合层：捷径(c) + 主路径原始输入(c) + 内部块输出(num_bottlenecks * c) = (2+num_bottlenecks)*c
        cv2_in_ch = (2 + num_bottlenecks) * self.c
        self.cv2 = Conv(cv2_in_ch, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))  # y[0]: 捷径(c), y[1]: 主路径入口(c)
        cur = y[-1]
        for m in self.m:
            cur = m(cur)
            y.append(cur)
        return self.cv2(torch.cat(y, dim=1))


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试不同递归深度
    for depth in [1, 2, 3]:
        model = DynamicC3k2(c1=256, c2=256, n=1, num_bottlenecks=2, e=0.25, wt_depth=depth).to(device)
        x = torch.randn(2, 256, 20, 20).to(device)
        out = model(x)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"wt_depth={depth}, Output: {out.shape}, Params: {total_params:,}")