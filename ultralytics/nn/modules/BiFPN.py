# ============================================
# File: ultralytics/nn/modules/BiFPN.py
# 完全复刻 YOLO11 默认 Neck（无投影，纯 Concat + C3k2）
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2


class BiFPN(nn.Module):
    """
    YOLO11 默认 Neck 的精确复刻，仅用 Concat + C3k2，无任何额外投影。

    参数:
        channels : [c3, c4, c5] 输入通道（已经过 width 缩放的实际值）
        out_channels : [o3, o4, o5] 期望输出通道，默认 o3=c3//2, o4=c4, o5=c5
        c3k2_kwargs : 传给 C3k2 的额外参数，如 n=1, shortcut=False
    """
    def __init__(self, channels, out_channels=None, c3k2_kwargs=None):
        super().__init__()
        c3, c4, c5 = channels
        self.o3 = o3 = out_channels[0] if out_channels else c3 // 2
        self.o4 = o4 = out_channels[1] if out_channels else c4
        self.o5 = o5 = out_channels[2] if out_channels else c5

        # C3k2 默认参数
        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)

        # ---- 自顶向下 ----
        # P4_td: Concat(P4, Up(P5)) -> C3k2，输入 c4+c5，输出 o4（=c4）
        self.p4_td_c3k2 = C3k2(c4 + c5, o4, **kwa)

        # P3_td: Concat(P3, Up(P4_td)) -> C3k2，输入 c3+o4（因为P4_td输出o4），输出 o3
        self.p3_td_c3k2 = C3k2(c3 + o4, o3, **kwa)

        # ---- 自底向上 (PAN) ----
        # P4_out: Concat(P4_td(out=o4), Down(P3_out(out=o3))) -> C3k2，输入 o4+o3，输出 o4
        self.p4_out_c3k2 = C3k2(o4 + o3, o4, **kwa)

        # P5_out: Concat(P5, Down(P4_out(out=o4))) -> C3k2，输入 c5+o4，输出 o5
        self.p5_out_c3k2 = C3k2(c5 + o4, o5, **kwa)

    def forward(self, features):
        p3, p4, p5 = features   # 原始通道 c3, c4, c5

        # ---- 自顶向下 ----
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.p4_td_c3k2(torch.cat([p4, p5_up], dim=1))   # 输出 o4

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = self.p3_td_c3k2(torch.cat([p3, p4_up], dim=1))   # 输出 o3

        # ---- 自底向上 ----
        p3_down = F.avg_pool2d(p3_td, 2, 2)                       # 下采样，通道 o3
        p4_out = self.p4_out_c3k2(torch.cat([p4_td, p3_down], dim=1))  # 输入 o4+o3，输出 o4

        p4_down = F.avg_pool2d(p4_out, 2, 2)                      # 下采样，通道 o4
        p5_out = self.p5_out_c3k2(torch.cat([p5, p4_down], dim=1))      # 输入 c5+o4，输出 o5

        return [p3_td, p4_out, p5_out]


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # YOLO11n 缩放后实际通道 (width=0.25)
    c3, c4, c5 = 128, 128, 256       # 对应原始 512,512,1024
    out_channels = [64, 128, 256]     # 可选，默认会自动设为 c3//2, c4, c5

    model = BiFPN([c3, c4, c5], out_channels).to(device)
    print(model)

    # 构造假输入
    p3 = torch.randn(2, 128, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    out = model([p3, p4, p5])

    print("\n=== 输出形状 ===")
    names = ["P3_out", "P4_out", "P5_out"]
    for name, feat in zip(names, out):
        print(f"{name}: {feat.shape}")

    expected = [(2, 64, 80, 80), (2, 128, 40, 40), (2, 256, 20, 20)]
    for o, e in zip(out, expected):
        assert o.shape == e, f"形状不匹配: 期望 {e}, 实际 {o.shape}"
    print("所有形状验证通过！")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")