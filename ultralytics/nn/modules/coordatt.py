import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from .block import Bottleneck,PSABlock,C2f,C3k

_all_ = ["CoordAtt","C3k2_CA"]



class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 定义两个方向的池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 输出形状 (b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 输出形状 (b,c,1,w)

        mip = max(8, inp // reduction)  # 中间通道数

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 坐标信息嵌入
        x_h = self.pool_h(x)  # (n,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n,c,1,w) -> (n,c,w,1)

        # 拼接并卷积
        y = torch.cat([x_h, x_w], dim=2)  # (n,c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离回两个方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (n,c,1,w)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # (n,oup,h,1)
        a_w = self.conv_w(x_w).sigmoid()  # (n,oup,1,w)

        # 应用注意力（通常与输入相乘）
        out = identity * a_h * a_w
        return out

class C3k2_CA(C2f):
    """
    C3k2 module with Coordinate Attention.
    Inherits from C2f and replaces the original attention (PSABlock) with CoordAtt.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,        # 保留 attn 参数，用于控制是否使用注意力
        g: int = 1,
        shortcut: bool = True,
        reduction: int = 32,        # 新增：CoordAtt 的缩减率
    ):
        """
        Initialize C3k2_CA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention (CoordAtt) after each block.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
            reduction (int): Reduction ratio for CoordAtt.
        """
        super().__init__(c1, c2, n, shortcut, g, e)

        # 构建 n 个块，每个块可以是 Bottleneck、C3k 或 Bottleneck+CoordAtt
        self.m = nn.ModuleList(
            self._build_block(attn, c3k,shortcut, reduction,g) for _ in range(n)
        )

    def _build_block(self, attn: bool, c3k: bool,shortcut, reduction: int,g):
        """Build a single block with optional CoordAtt."""
        # 基础部分：Bottleneck 或 C3k
        if c3k:
            # 当 c3k=True 时，使用 C3k 模块（假设 C3k 已定义，且其参数与 Bottleneck 兼容）
            block = C3k(self.c, self.c, 2, shortcut, g)  # 注意参数可能需要根据实际 C3k 调整
        else:
            block = Bottleneck(self.c, self.c,shortcut, g)

        # 如果启用注意力，则在 block 后接 CoordAtt
        if attn:
            # CoordAtt 要求输入输出通道数相同，这里用 self.c
            return nn.Sequential(block, CoordAtt(self.c, self.c, reduction))
        else:
            return block


# 示例使用
if __name__ == '__main__':
    block = CoordAtt(64, 64)  # 实例化Coordinate Attention模块
    input = torch.rand(1, 64, 64, 64)  # 创建一个随机输入
    output = block(input)  # 通过模块处理输入
    print(output.shape())  # 打印输入和输出的尺寸