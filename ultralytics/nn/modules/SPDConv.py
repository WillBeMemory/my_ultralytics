import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SPDConv(nn.Module):
    """
    SPD + 两个深度可分离卷积，实现轻量无损下采样与特征提取。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 深度卷积核大小（默认3）
        s (int): 步长（2 启用 SPD 下采样 + 双深度可分离卷积；1 退化为单个深度可分离卷积）
        act (bool): 是否使用 SiLU 激活，默认 True
    """
    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.s = s
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        if s == 2:
            c_mid = c1 * 4   # SPD 膨胀后的通道数

            # 第一个深度可分离卷积：4*c1 → c2
            self.dsconv1 = nn.Sequential(
                # Depthwise 3x3
                nn.Conv2d(c_mid, c_mid, k, stride=1, padding=autopad(k),
                          groups=c_mid, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.SiLU(inplace=True),

                # Pointwise 1x1
                nn.Conv2d(c_mid, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )

            # 第二个深度可分离卷积：c2 → c2（精炼）
            self.dsconv2 = nn.Sequential(
                # Depthwise 3x3
                nn.Conv2d(c2, c2, k, stride=1, padding=autopad(k),
                          groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),

                # Pointwise 1x1
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )

        else:
            # 退化模式：一个深度可分离卷积（保持与普通卷积类似的接口）
            self.conv = nn.Sequential(
                nn.Conv2d(c1, c1, k, stride=1, padding=autopad(k),
                          groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape

            # 确保空间尺寸为偶数（SPD 需要）
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

            # 1. 无损空间到深度（尺寸减半，通道数 ×4）
            x = F.pixel_unshuffle(x, downscale_factor=2)   # (B, 4C, H/2, W/2)

            # 2. 第一个深度可分离卷积（4C → c2）
            x = self.dsconv1(x)

            # 3. 第二个深度可分离卷积（c2 → c2，精炼特征）
            x = self.dsconv2(x)

            return x
        else:
            # 退化：单深度可分离卷积（不改变尺寸）
            return self.conv(x)