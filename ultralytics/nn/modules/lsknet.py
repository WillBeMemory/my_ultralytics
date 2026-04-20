import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck
from .conv import Conv, autopad
from .block import Bottleneck,PSABlock,C2f,C3k

_all_ = ["LSK","C3k2_LSK"]

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class LSK(nn.Module):
    """Large Selective Kernel module for spatial attention."""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv0(x)           # 5×5 深度卷积
        attn = self.conv_spatial(attn) # 7×7 空洞卷积，扩大感受野
        attn = self.conv1(attn)
        attn = self.sigmoid(attn)
        return x * attn

class LSKBottleneck(Bottleneck):
    """Bottleneck with LSK module inserted after the second conv."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.lsk = LSK(c2)  # LSK applied after cv2

    def forward(self, x):
        identity = x
        out = self.cv2(self.cv1(x))
        out = self.lsk(out)
        return identity + out if self.add else out

class C3k2_LSK(C2f):
    """
    C3k2 module with optional LSK attention in bottlenecks.
    Inherits from C2f and adds a flag `lsk` to enable LSK-enhanced bottlenecks.
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
        lsk: bool = False,  # 新增参数：是否使用 LSK 瓶颈
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        # 重新构建 self.m，支持 lsk 选项
        self.m = nn.ModuleList()
        for _ in range(n):
            if attn:
                # attention 模式：Bottleneck + PSABlock
                self.m.append(nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, g),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                ))
            elif c3k:
                # c3k 模式：使用 C3k 模块
                self.m.append(C3k(self.c, self.c, 2, shortcut, g))
            elif lsk:
                # lsk 模式：使用 LSKBottleneck
                self.m.append(LSKBottleneck(self.c, self.c, shortcut, g, e=1.0))
            else:
                # 默认：普通 Bottleneck
                self.m.append(Bottleneck(self.c, self.c, shortcut, g))

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    block = LSKblock(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(input.size(), output.size())