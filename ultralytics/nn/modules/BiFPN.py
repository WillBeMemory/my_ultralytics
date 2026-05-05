import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class BiFPN_Add(nn.Module):
    """快速归一化加权融合（用于双向路径）"""
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
    轻量版：P2 与 P3/P4/P5 使用星操作增强，双向加权融合，融合通道降低到 256。
    输入: P2(1024,16), P3(256,32), P4(256,16), P5(512,8)
    输出: P3_out(128), P4_out(256), P5_out(512)
    """
    def __init__(self, channels, out_channels=None, use_c3k2=False):
        super().__init__()
        c2, c3, c4, c5 = channels          # [1024, 256, 256, 512]
        if out_channels is None:
            out_channels = [c3, c4, c5]
        o3, o4, o5 = out_channels          # 目标输出通道

        # ★ 统一的低维融合通道，大幅降低计算量
        self.expand_ch = 256

        # ---------- P2 降维投影 ----------
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)   # 1024 → 256

        # ---------- P5 扩张到 256（P3/P4 本身就是 256，不额外扩张） ----------
        self.expand_p5 = Conv(c5, self.expand_ch, 1, act=False)   # 512 → 256
        # P3、P4 无需扩张，直接使用 256 通道

        # ---------- P2 尺寸适配 ----------
        self.p2_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')   # 16 → 32
        self.p2_to_p4 = nn.Identity()                                  # 16 不变
        self.p2_to_p5 = nn.MaxPool2d(2, 2)                             # 16 → 8

        # ---------- 星操作后投影（融合 P2 与各层） ----------
        self.star_p2_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.star_p2_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.star_p2_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)

        # ---------- 双向加权融合投影 ----------
        self.td_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.td_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.bu_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.bu_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)

        # ---------- 加权融合节点 ----------
        self.fuse_p4_td = BiFPN_Add(2)   # P4 + up(P5)
        self.fuse_p3_td = BiFPN_Add(2)   # P3 + up(P4_td)
        self.fuse_p4_bu = BiFPN_Add(2)   # P4_td + down(P3_bu)
        self.fuse_p5_bu = BiFPN_Add(2)   # P5 + down(P4_bu)

        # ---------- 下采样（深度可分离）----------
        self.down3 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1, groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1, groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )

        # ---------- 压缩到目标输出通道 ----------
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)
        self.compress_p5 = Conv(self.expand_ch, o5, 1, act=False)

        # ---------- 可选精炼（默认关闭，降低计算量） ----------
        if use_c3k2:
            kwa = dict(n=1, shortcut=False, e=0.5)
            self.refine_p4_td = C3k2(self.expand_ch, self.expand_ch, **kwa)
            self.refine_p3_td = C3k2(self.expand_ch, self.expand_ch, **kwa)
            self.refine_p4_bu = C3k2(self.expand_ch, self.expand_ch, **kwa)
            self.refine_p5_bu = C3k2(self.expand_ch, self.expand_ch, **kwa)
        else:
            self.refine_p4_td = nn.Identity()
            self.refine_p3_td = nn.Identity()
            self.refine_p4_bu = nn.Identity()
            self.refine_p5_bu = nn.Identity()

    def forward(self, features):
        p2, p3, p4, p5 = features

        # 1. P2 降维
        p2_low = self.p2_proj(p2)           # (B, 256, 16, 16)

        # 2. P3/P4 直接使用，P5 扩张到 256
        p3_low = p3                          # (B, 256, 32, 32)
        p4_low = p4                          # (B, 256, 16, 16)
        p5_low = self.expand_p5(p5)          # (B, 256, 8, 8)

        # 3. P2 尺寸适配
        p2_p3 = self.p2_to_p3(p2_low)        # (B,256,32,32)
        p2_p4 = self.p2_to_p4(p2_low)        # (B,256,16,16)
        p2_p5 = self.p2_to_p5(p2_low)        # (B,256,8,8)

        # 4. 星操作：P2 ⊙ 各层 后接投影平滑
        p3_enh = self.star_p2_p3(p2_p3 * p3_low)
        p4_enh = self.star_p2_p4(p2_p4 * p4_low)
        p5_enh = self.star_p2_p5(p2_p5 * p5_low)

        # 5. 双向加权融合
        # 自顶向下
        p5_up = F.interpolate(self.td_p5(p5_enh), size=p4_enh.shape[-2:], mode='nearest')
        p4_td = self.refine_p4_td(self.fuse_p4_td([p4_enh, p5_up]))

        p4_up = F.interpolate(self.td_p4(p4_td), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_p3_td(self.fuse_p3_td([p3_enh, p4_up]))

        # 自底向上
        p3_down = self.down3(p3_td)
        p4_bu = self.refine_p4_bu(self.fuse_p4_bu([p4_td, p3_down]))

        p4_down = self.down4(p4_bu)
        p5_bu = self.refine_p5_bu(self.fuse_p5_bu([p5_enh, p4_down]))

        # 6. 压缩输出
        p3_out = self.compress_p3(p3_td)
        p4_out = self.compress_p4(p4_bu)
        p5_out = self.compress_p5(p5_bu)

        return [p3_out, p4_out, p5_out]


# ==================== 简单 main 测试 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")

    in_channels  = [1024, 256, 256, 512]
    out_channels = [128, 256, 512]

    bifpn = BiFPN(in_channels, out_channels, use_c3k2=False).to(device)

    bs = 2
    p2 = torch.randn(bs, 1024, 16, 16).to(device)
    p3 = torch.randn(bs,  256, 32, 32).to(device)
    p4 = torch.randn(bs,  256, 16, 16).to(device)
    p5 = torch.randn(bs,  512,  8,  8).to(device)

    inputs = [p2, p3, p4, p5]
    outputs = bifpn(inputs)

    expected = [
        (bs, 128, 32, 32),
        (bs, 256, 16, 16),
        (bs, 512,  8,  8),
    ]

    print("=== BiFPN 输出形状 ===")
    for name, out, exp in zip(["P3_out", "P4_out", "P5_out"], outputs, expected):
        print(f"{name}: {out.shape} (期望 {exp})")
        assert out.shape == exp, f"形状不匹配！期望 {exp}，实际 {out.shape}"

    print("\n✅ 所有输出形状验证通过！")
    total_params = sum(p.numel() for p in bifpn.parameters())
    print(f"总参数量: {total_params:,}")