# ============================================
# File: ultralytics/nn/modules/BiFPN.py
# 新版 BiFPN：支持原始双向（6 输入）与纯自顶向下（3 输入）
# 当输入特征数量为 3 时，自动切换为 Top‑Down 模式，仅由 P5 引导融合
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.GSConv import GSConv   # 可选


class BiFPN_Add(nn.Module):
    """快速归一化加权融合，支持任意数量的输入"""
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class BiFPN(nn.Module):
    """
    PAN‑like BiFPN（仅用 P5 HIPA 引导）
    输入: [P3_orig, P4_orig, P5_hipa]
    输出: [P3_out, P4_out, P5_out]
    """
    def __init__(self, channels, out_channels=None, use_depthwise=False):
        super().__init__()
        c3, c4, c5 = channels
        o3 = out_channels[0] if out_channels else c3 // 2
        o4 = out_channels[1] if out_channels else c4
        o5 = out_channels[2] if out_channels else c5

        # 原始特征投影到各尺度目标通道
        self.p3_proj = nn.Conv2d(c3, o3, 1, bias=False)
        self.p4_proj = nn.Conv2d(c4, o4, 1, bias=False)
        self.p5_proj = nn.Conv2d(c5, o5, 1, bias=False)  # 虽然 P5 不变，但保留以统一

        # 通道对齐：P5/o5 → P4/o4, P5/o5 → P3/o3, P3/o3 → P4/o4, P4/o4 → P5/o5
        self.p5_to_p4 = nn.Conv2d(o5, o4, 1, bias=False)
        self.p5_to_p3 = nn.Conv2d(o5, o3, 1, bias=False)
        self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)
        self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)

        # 融合加权（每处两个输入）
        self.fuse_p4_td = BiFPN_Add(2)   # P4_orig + P5_up
        self.fuse_p3_td = BiFPN_Add(2)   # P3_orig + P5_up

        self.fuse_p4_out = BiFPN_Add(2)  # P4_td + P3_down
        self.fuse_p5_out = BiFPN_Add(2)  # P5 + P4_down

        # 平滑卷积
        Conv = lambda c: nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False) if use_depthwise \
                         else nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.smooth_p4_td = Conv(o4)
        self.smooth_p3_td = Conv(o3)
        self.smooth_p4_out = Conv(o4)
        self.smooth_p5_out = Conv(o5)

    def forward(self, features):
        p3, p4, p5 = features   # p5 即 HIPA 输出

        p3 = self.p3_proj(p3)   # (B, o3, H/8, W/8)
        p4 = self.p4_proj(p4)   # (B, o4, H/16, W/16)
        p5 = self.p5_proj(p5)   # (B, o5, H/32, W/32)

        # ---- 自顶向下（P5 直达 P4、P3） ----
        p5_up_to_p4 = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p5_up_to_p4 = self.p5_to_p4(p5_up_to_p4)   # o5 -> o4
        p4_td = self.smooth_p4_td(self.fuse_p4_td([p4, p5_up_to_p4]))

        p5_up_to_p3 = F.interpolate(p5, size=p3.shape[-2:], mode='nearest')
        p5_up_to_p3 = self.p5_to_p3(p5_up_to_p3)   # o5 -> o3
        p3_td = self.smooth_p3_td(self.fuse_p3_td([p3, p5_up_to_p3]))

        # ---- 自底向上 ----
        p3_down = F.avg_pool2d(p3_td, 2, 2)        # -> o3 at P4 size
        p3_down = self.p3_to_p4(p3_down)           # o3 -> o4
        p4_out = self.smooth_p4_out(self.fuse_p4_out([p4_td, p3_down]))

        p4_down = F.avg_pool2d(p4_out, 2, 2)       # -> o4 at P5 size
        p4_down = self.p4_to_p5(p4_down)           # o4 -> o5
        p5_out = self.smooth_p5_out(self.fuse_p5_out([p5, p4_down]))  # 注意这里 p5 还是 HIPA 输出，不是 p5_td

        return [p3_td, p4_out, p5_out]

# ============================================
# 测试代码（自顶向下模式）
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 YOLO11n 缩放后通道 (width=0.25)
    # 原始配置：[512, 512, 1024] → 缩放后 [128, 128, 256]
    channels = [128, 128, 256]   # 对应 P3, P4, P5 实际输入通道

    model = BiFPN(channels, use_depthwise=True,).to(device)

    # 创建假输入
    p3 = torch.randn(2, 128, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    out = model([p3, p4, p5])
    print("=== Top‑Down BiFPN Output Shapes ===")
    for i, name in enumerate(["P3", "P4", "P5"]):
        print(f"{name}: {out[i].shape}")
    # 期望：P3 (2,64,80,80), P4 (2,128,40,40), P5 (2,256,20,20)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")