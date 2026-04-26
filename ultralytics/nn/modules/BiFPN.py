import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import HWD
from ultralytics.nn.modules.block import C3k2


class BiFPN(nn.Module):
    """
    YOLO11 默认 Neck 的复刻版，自底向上路径使用 HWD 小波下采样。

    参数:
        channels : [c3, c4, c5] 输入通道（经 width 缩放后的实际值）
        out_channels : [o3, o4, o5] 输出通道，默认 o3=c3//2, o4=c4, o5=c5
        c3k2_kwargs : 传给 C3k2 的额外参数，如 n=1, shortcut=False
    """

    def __init__(self, channels, out_channels=None, c3k2_kwargs=None):
        super().__init__()
        c3, c4, c5 = channels
        self.o3 = o3 = out_channels[0] if out_channels else c3 // 2
        self.o4 = o4 = out_channels[1] if out_channels else c4
        self.o5 = o5 = out_channels[2] if out_channels else c5

        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)

        # ---- 自顶向下（保持不变） ----
        self.p4_td_c3k2 = C3k2(c4 + c5, o4, **kwa)
        self.p3_td_c3k2 = C3k2(c3 + o4, o3, **kwa)

        # ---- 自底向上（使用 HWD 小波下采样） ----
        self.down1 = HWD(o3, o4)   # 输入 o3，输出 o4，空间尺寸减半
        self.down2 = HWD(o4, o5)   # 输入 o4，输出 o5，空间尺寸减半

        self.p4_out_c3k2 = C3k2(o4 + o4, o4, **kwa)   # 拼接后通道 2*o4 -> o4
        self.p5_out_c3k2 = C3k2(c5 + o5, o5, **kwa)   # 拼接后通道 c5+o5 -> o5

    def forward(self, features):
        p3, p4, p5 = features

        # ---- 自顶向下 ----
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.p4_td_c3k2(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = self.p3_td_c3k2(torch.cat([p3, p4_up], dim=1))

        # ---- 自底向上 ----
        p3_down = self.down1(p3_td)   # HWD: o3 -> o4，尺寸减半
        p4_out = self.p4_out_c3k2(torch.cat([p4_td, p3_down], dim=1))

        p4_down = self.down2(p4_out)  # HWD: o4 -> o5，尺寸减半
        p5_out = self.p5_out_c3k2(torch.cat([p5, p4_down], dim=1))

        return [p3_td, p4_out, p5_out]