import torch
import torch.nn as nn
import torch.nn.functional as F


class SPDConv(nn.Module):
    """
    可配置的 SPD 下采样模块。

    参数:
        c1           : 输入通道数 (框架自动传入)
        c2           : 输出通道数 (仅在 compress=True 时生效)
        factor       : 下采样倍率，默认 2
        compress     : 是否用 1×1 卷积压缩通道。若为 False，则只做 pixel_unshuffle，输出通道 = c1 * factor²
        kernel_size  : 压缩卷积的核大小，默认 1
        pass_through : 是否在卷积后加入残差连接 (仅当 c1*factor² == c2 且 compress=True 时有效)
    """

    def __init__(self, c1, c2, factor=2, compress=True, kernel_size=1, pass_through=False):
        super().__init__()
        self.factor = factor
        self.compress = compress

        # 无损空间到深度
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=factor)
        expanded_ch = c1 * factor ** 2

        if compress:
            self.conv = nn.Sequential(
                nn.Conv2d(expanded_ch, c2, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
            self.pass_through = pass_through and expanded_ch == c2
        else:
            # 不压缩，直接输出膨胀后的通道
            self.conv = nn.Identity()
            self.pass_through = False

    def forward(self, x):
        out = self.space_to_depth(x)  # (B, C*F², H/F, W/F)
        out = self.conv(out)
        return out