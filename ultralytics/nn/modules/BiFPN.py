import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.SPDConv import SPDConv
from ultralytics.nn.modules.block import Conv, C3k2


class BiFPN_Add(nn.Module):
    """快速归一化加权融合"""
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
    P2 通过一次 SPDConv(factor=4) 降至 P4 尺寸，再分别上下采样至 P3/P5 尺寸。
    大幅减少重复 SPD 计算量，同时保留空间信息无损转化。
    """
    def __init__(self, channels, out_channels=None, use_c3k2=False):
        super().__init__()
        c2, c3, c4, c5 = channels
        if out_channels is None:
            out_channels = [c3, c4, c5]
        o3, o4, o5 = out_channels
        self.expand_ch = 256

        # 各层投影到统一通道
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)
        self.p5_proj = Conv(c5, self.expand_ch, 1, act=False)

        # ★ 核心改动：P2 仅通过一次 SPDConv 降至 P4 尺寸 (160→40)
        self.p2_to_p4 = SPDConv(self.expand_ch, self.expand_ch, factor=4, compress=True)

        # 上采样模块（P4 尺寸 → P3 尺寸）
        self.up_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')

        # 下采样模块（P4 尺寸 → P5 尺寸），用两次轻量 2× 下采样
        self.down_to_p5 = nn.Sequential(
            nn.AvgPool2d(2, 2),   # 40→20
        )

        # 以下融合结构完全不变
        self.star_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.star_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.star_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)

        self.td_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.td_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.bu_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.bu_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)

        self.fuse_p4_td = BiFPN_Add(2)
        self.fuse_p3_td = BiFPN_Add(2)
        self.fuse_p4_bu = BiFPN_Add(2)
        self.fuse_p5_bu = BiFPN_Add(2)

        self.down3 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1,
                      groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1,
                      groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )

        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)
        self.compress_p5 = Conv(self.expand_ch, o5, 1, act=False)

        if use_c3k2:
            self.refine_p4_td = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_p3_td = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_p4_bu = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_p5_bu = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
        else:
            self.refine_p4_td = nn.Identity()
            self.refine_p3_td = nn.Identity()
            self.refine_p4_bu = nn.Identity()
            self.refine_p5_bu = nn.Identity()

    def forward(self, features):
        p2, p3, p4, p5 = features
        p2_low = self.p2_proj(p2)   # (B,256,160,160)
        p3_low = self.p3_proj(p3)   # (B,256,80,80)
        p4_low = self.p4_proj(p4)   # (B,256,40,40)
        p5_low = self.p5_proj(p5)   # (B,256,20,20)

        # P2 经过一次 SPD(f4) → 40×40
        p2_to_p4 = self.p2_to_p4(p2_low)   # (B,256,40,40)

        # 上采样到 P3 尺寸，下采样到 P5 尺寸
        p2_to_p3 = self.up_to_p3(p2_to_p4)  # (B,256,80,80)
        p2_to_p5 = self.down_to_p5(p2_to_p4) # (B,256,20,20)

        # 星操作增强
        p3_enh = self.star_p3(p2_to_p3 * p3_low)
        p4_enh = self.star_p4(p2_to_p4 * p4_low)
        p5_enh = self.star_p5(p2_to_p5 * p5_low)

        # 双向融合（与之前完全一致）
        p5_up = F.interpolate(self.td_p5(p5_enh), size=p4_enh.shape[-2:], mode='nearest')
        p4_td = self.refine_p4_td(self.fuse_p4_td([p4_enh, p5_up]))
        p4_up = F.interpolate(self.td_p4(p4_td), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_p3_td(self.fuse_p3_td([p3_enh, p4_up]))

        p3_down = self.down3(p3_td)
        p4_bu = self.refine_p4_bu(self.fuse_p4_bu([p4_td, p3_down]))
        p4_down = self.down4(p4_bu)
        p5_bu = self.refine_p5_bu(self.fuse_p5_bu([p5_enh, p4_down]))

        p3_out = self.compress_p3(p3_td)
        p4_out = self.compress_p4(p4_bu)
        p5_out = self.compress_p5(p5_bu)
        return [p3_out, p4_out, p5_out]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 backbone 输出的多尺度特征（根据之前设计的 YAML）
    # P2: 160x160, 128ch; P3: 80x80, 256ch; P4: 40x40, 512ch; P5: 20x20, 1024ch
    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)
    p5 = torch.randn(bs, 1024, 20, 20).to(device)

    # BiFPN 配置：输入通道列表 [c2, c3, c4, c5] 和输出通道 [o3, o4, o5]
    channels = [128, 256, 512, 1024]
    out_channels = [256, 512, 1024]
    model = BiFPN(channels, out_channels, use_c3k2=False).to(device)

    # 前向传播
    with torch.no_grad():
        outs = model([p2, p3, p4, p5])

    expected_shapes = [
        (bs, 256, 80, 80),
        (bs, 512, 40, 40),
        (bs, 1024, 20, 20)
    ]

    print("\n=== BiFPN Output Shape Check ===")
    for i, (out, exp) in enumerate(zip(outs, expected_shapes), start=3):
        print(f"P{i}_out: {out.shape} (expected {exp})", end="")
        if out.shape == exp:
            print(" ✅")
        else:
            print(" ❌ MISMATCH")
            raise RuntimeError(f"Shape mismatch at P{i}_out")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # 可尝试反向传播检验梯度流（无哑错误）
    try:
        loss = outs[-1].sum()
        loss.backward()
        print("Backward passed successfully.")
    except Exception as e:
        print(f"Backward failed: {e}")