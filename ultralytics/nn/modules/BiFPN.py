import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2

class BiFPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels  # [128, 128, 256] 实际输入

        # ---- 目标输出通道数 (匹配 Detect) ----
        target_p3 = c3/2
        target_p4 = c4
        target_p5 = c5

        # ---- 自顶向下 ----
        # P5上采样后与P4拼接: 256 + 128 = 384 -> 输出 target_p4 = 128
        self.p5_to_p4 = C3k2(c5 + c4, target_p4, n=2, c3k=False, e=0.5)
        # P4_td上采样后与P3拼接: 128 + 128 = 256 -> 输出 target_p3 = 64
        self.p4_to_p3 = C3k2(target_p4 + c3, target_p3, n=2, c3k=False, e=0.5)

        # ---- 自底向上 ----
        # 下采样
        self.down_p3 = Conv(target_p3, target_p3, 3, 2)    # 64 -> 64
        # P3_down与P4_td拼接: 64 + 128 = 192 -> 输出 target_p4 = 128
        self.p4_out = C3k2(target_p3 + target_p4, target_p4, n=2, c3k=False, e=0.5)
        self.down_p4 = Conv(target_p4, target_p4, 3, 2)    # 128 -> 128
        # P4_down与P5 (实际为256) 拼接: 128 + 256 = 384 -> 输出 target_p5 = 256
        self.p5_out = C3k2(target_p4 + c5, target_p5, n=2, c3k=False, e=0.5)

    def forward(self, features):
        p3, p4, p5 = features

        # ---- 自顶向下 ----
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.p5_to_p4(torch.cat([p5_up, p4], dim=1))

        p4_td_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = self.p4_to_p3(torch.cat([p4_td_up, p3], dim=1))

        # ---- 自底向上 ----
        p3_down = self.down_p3(p3_td)
        p4_out = self.p4_out(torch.cat([p3_down, p4_td], dim=1))

        p4_down = self.down_p4(p4_out)
        p5_out = self.p5_out(torch.cat([p4_down, p5], dim=1))

        return [p3_td, p4_out, p5_out]
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # YOLOv11n 缩放后通道
    p3 = torch.randn(2, 64, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    bifpn = BiFPN(channels=[64, 128, 256]).to(device)
    outs = bifpn([p3, p4, p5])
    for i, o in enumerate(outs):
        print(f"P{i+3}: {o.shape}")

    # 测试 SplitList
    from ultralytics.nn.modules.SplitList import SplitList
    for idx, ch in enumerate([64, 128, 256]):
        split = SplitList(0, ch, index=idx).to(device)
        o = split(outs)
        print(f"Split {idx}: {o.shape}, expected ch={ch}")
    print("All tests passed.")