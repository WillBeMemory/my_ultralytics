import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.WaveletFusionUp import WaveletFusionUp


class BiFPN(nn.Module):
    """
    四尺度自顶向下 BiFPN（接收 P2, P3, P4, P5）。
    仅自顶向下：P5 → P4 → P3 → P2，每步使用双输入小波融合 + C3k2 精炼。
    输出：融合后的 P2, P3, P4, P5（通道与输入一致）。
    """
    def __init__(self, channels, out_channels=None, c3k2_kwargs=None, fusion_compression=1):
        super().__init__()
        c2, c3, c4, c5 = channels

        # 输出通道：默认与输入通道保持一致
        self.o2 = out_channels[0] if out_channels else c2
        self.o3 = out_channels[1] if out_channels else c3
        self.o4 = out_channels[2] if out_channels else c4
        self.o5 = out_channels[3] if out_channels else c5

        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)

        # ---- 自顶向下融合 ----
        # P5 → P4
        self.fuse_p5_p4 = WaveletFusionUp(low_ch=c4, high_ch=c5, k=3, compression=fusion_compression)
        self.p4_td_c3k2 = C3k2(c4 + c5, self.o4, **kwa)

        # P4_td → P3
        self.fuse_p4_p3 = WaveletFusionUp(low_ch=c3, high_ch=self.o4, k=3, compression=fusion_compression)
        self.p3_td_c3k2 = C3k2(c3 + self.o4, self.o3, **kwa)

        # P3_td → P2
        self.fuse_p3_p2 = WaveletFusionUp(low_ch=c2, high_ch=self.o3, k=3, compression=fusion_compression)
        self.p2_td_c3k2 = C3k2(c2 + self.o3, self.o2, **kwa)

    def forward(self, features):
        p2, p3, p4, p5 = features  # p2: (B, c2, H/4, W/4), p3: (B, c3, H/8, W/8), ...

        # P5 → P4
        p5_fused = self.fuse_p5_p4(p4, p5)                # (B, c5, H/16, W/16)
        p4_td = self.p4_td_c3k2(torch.cat([p4, p5_fused], dim=1))  # (B, o4, H/16, W/16)

        # P4_td → P3
        p4_fused = self.fuse_p4_p3(p3, p4_td)             # (B, o4, H/8, W/8)
        p3_td = self.p3_td_c3k2(torch.cat([p3, p4_fused], dim=1))  # (B, o3, H/8, W/8)

        # P3_td → P2
        p3_fused = self.fuse_p3_p2(p2, p3_td)             # (B, o3, H/4, W/4)
        p2_td = self.p2_td_c3k2(torch.cat([p2, p3_fused], dim=1))  # (B, o2, H/4, W/4)

        # 返回四个尺度的融合结果（P5 保持原始不变）
        return [p2_td, p3_td, p4_td, p5]


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟缩放后的实际通道（例如 YOLO11n width=0.25 的某种变体）
    c2, c3, c4, c5 = 32, 64, 128, 256
    model = BiFPN([c2, c3, c4, c5], fusion_compression=2).to(device)

    p2 = torch.randn(2, c2, 160, 160).to(device)
    p3 = torch.randn(2, c3, 80, 80).to(device)
    p4 = torch.randn(2, c4, 40, 40).to(device)
    p5 = torch.randn(2, c5, 20, 20).to(device)

    out = model([p2, p3, p4, p5])
    names = ["P2_td", "P3_td", "P4_td", "P5"]
    for name, feat in zip(names, out):
        print(f"{name}: {feat.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")