import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import HWD
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.WaveletFusionUp import WaveletFusionUp


class BiFPN(nn.Module):
    """
    方式一：小波融合 (替换上采样) + Concat + C3k2 细化
    """
    def __init__(self, channels, out_channels=None, c3k2_kwargs=None):
        super().__init__()
        c3, c4, c5 = channels
        self.o3 = o3 = out_channels[0] if out_channels else c3 // 2
        self.o4 = o4 = out_channels[1] if out_channels else c4
        self.o5 = o5 = out_channels[2] if out_channels else c5

        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)

        # ---- 自顶向下 ----
        # P5 → P4：小波融合输出 c5 通道，与 P4 拼接后 C3k2
        self.fuse_p5_p4 = WaveletFusionUp(low_ch=c4, high_ch=c5, k=3)
        self.p4_td_c3k2 = C3k2(c4 + c5, o4, **kwa)

        # P4_td → P3：小波融合输出 o4 通道，与 P3 拼接后 C3k2
        self.fuse_p4_p3 = WaveletFusionUp(low_ch=c3, high_ch=o4, k=3)
        self.p3_td_c3k2 = C3k2(c3 + o4, o3, **kwa)

        # ---- 自底向上 (HWD 下采样) ----
        self.down1 = HWD(o3, o4)
        self.down2 = HWD(o4, o5)
        self.p4_out_c3k2 = C3k2(o4 + o4, o4, **kwa)
        self.p5_out_c3k2 = C3k2(c5 + o5, o5, **kwa)

    def forward(self, features):
        p3, p4, p5 = features

        # 自顶向下
        p5_fused = self.fuse_p5_p4(p4, p5)                # (B, c5, H/16, W/16)
        p4_td = self.p4_td_c3k2(torch.cat([p4, p5_fused], dim=1))  # (B, o4, H/16, W/16)

        p4_fused = self.fuse_p4_p3(p3, p4_td)             # (B, o4, H/8, W/8)
        p3_td = self.p3_td_c3k2(torch.cat([p3, p4_fused], dim=1))  # (B, o3, H/8, W/8)

        # 自底向上
        p3_down = self.down1(p3_td)                       # (B, o4, H/16, W/16)
        p4_out = self.p4_out_c3k2(torch.cat([p4_td, p3_down], dim=1))

        p4_down = self.down2(p4_out)                      # (B, o5, H/32, W/32)
        p5_out = self.p5_out_c3k2(torch.cat([p5, p4_down], dim=1))

        return [p3_td, p4_out, p5_out]


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    c3, c4, c5 = 128, 128, 256
    model = BiFPN([c3, c4, c5]).to(device)

    p3 = torch.randn(2, 128, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    out = model([p3, p4, p5])
    print("=== 输出形状 ===")
    for name, feat in zip(["P3_out", "P4_out", "P5_out"], out):
        print(f"{name}: {feat.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")