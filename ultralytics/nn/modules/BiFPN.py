import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class BiFPN_Add(nn.Module):
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class BiFPN(nn.Module):
    def __init__(self, channels, out_channels=None, use_c3k2=False):
        super().__init__()
        c2, c3, c4, c5 = channels          # [1024, 256, 256, 512]
        if out_channels is None:
            out_channels = [c3, c4, c5]
        o3, o4, o5 = out_channels           # 目标输出通道，例如 [128, 256, 512]

        self.expand_ch = c2                 # 统一扩展通道数 = P2 通道数

        # ---------- 扩张 ----------
        self.expand_p3 = Conv(c3, self.expand_ch, 1, act=False)   # 256 → 1024
        self.expand_p4 = Conv(c4, self.expand_ch, 1, act=False)
        self.expand_p5 = Conv(c5, self.expand_ch, 1, act=False)   # 512 → 1024

        # ---------- P2 注入路径 ----------
        self.p2_to_p3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),          # 16 → 32
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )
        self.p2_to_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)   # 尺寸不变
        self.p2_to_p5 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1, groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )                                                         # 16 → 8

        # ---------- 自顶向下投影 ----------
        self.td_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)  # P5 → P4
        self.td_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)  # P4 → P3

        # ---------- 自底向上投影 ----------
        self.bu_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)  # P3 → P4
        self.bu_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)  # P4 → P5

        # ---------- 加权融合节点 ----------
        self.fuse_p4_td = BiFPN_Add(2)   # P4_exp + up(P5_exp)
        self.fuse_p3_td = BiFPN_Add(2)   # P3_exp + up(P4_td)
        self.fuse_p4_bu = BiFPN_Add(2)   # P4_td + down(P3_bu)
        self.fuse_p5_bu = BiFPN_Add(2)   # P5_exp + down(P4_bu)

        # ---------- 下采样（深度可分离） ----------
        self.down3 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1, groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.expand_ch, self.expand_ch, 3, stride=2, padding=1, groups=self.expand_ch, bias=False),
            Conv(self.expand_ch, self.expand_ch, 1, act=False)
        )

        # ---------- 压缩 ----------
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)   # 1024 → o3
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)   # 1024 → o4
        self.compress_p5 = Conv(self.expand_ch, o5, 1, act=False)   # 1024 → o5

        # ---------- 可选精炼 ----------
        kwa = dict(n=1, shortcut=False, e=0.5)
        self.refine_p4_td = C3k2(self.expand_ch, self.expand_ch, **kwa) if use_c3k2 else nn.Identity()
        self.refine_p3_td = C3k2(self.expand_ch, self.expand_ch, **kwa) if use_c3k2 else nn.Identity()
        self.refine_p4_bu = C3k2(self.expand_ch, self.expand_ch, **kwa) if use_c3k2 else nn.Identity()
        self.refine_p5_bu = C3k2(self.expand_ch, self.expand_ch, **kwa) if use_c3k2 else nn.Identity()

    def forward(self, features):
        p2, p3, p4, p5 = features

        # 1. 扩张
        p3_exp = self.expand_p3(p3)   # (B, 1024, 32, 32)
        p4_exp = self.expand_p4(p4)   # (B, 1024, 16, 16)
        p5_exp = self.expand_p5(p5)   # (B, 1024, 8, 8)

        # 2. P2 注入
        p3_exp = p3_exp + self.p2_to_p3(p2)
        p4_exp = p4_exp + self.p2_to_p4(p2)
        p5_exp = p5_exp + self.p2_to_p5(p2)

        # 3. 自顶向下
        p5_up = F.interpolate(self.td_p5(p5_exp), size=p4_exp.shape[-2:], mode='nearest')
        p4_td = self.refine_p4_td(self.fuse_p4_td([p4_exp, p5_up]))

        p4_up = F.interpolate(self.td_p4(p4_td), size=p3_exp.shape[-2:], mode='nearest')
        p3_td = self.refine_p3_td(self.fuse_p3_td([p3_exp, p4_up]))

        # 4. 自底向上
        p3_down = self.down3(p3_td)
        p4_bu = self.refine_p4_bu(self.fuse_p4_bu([p4_td, p3_down]))

        p4_down = self.down4(p4_bu)
        p5_bu = self.refine_p5_bu(self.fuse_p5_bu([p5_exp, p4_down]))

        # 5. 压缩输出
        p3_out = self.compress_p3(p3_td)
        p4_out = self.compress_p4(p4_bu)
        p5_out = self.compress_p5(p5_bu)

        return [p3_out, p4_out, p5_out]


# ==================== 简单 main 测试 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")

    # 输入通道（你的 backbone 实际值）
    in_channels  = [1024, 256, 256, 512]   # P2, P3, P4, P5
    # 输出通道（P3 压缩到 128，其他不变）
    out_channels = [128, 256, 512]         # P3_out, P4_out, P5_out

    bifpn = BiFPN(in_channels, out_channels, use_c3k2=False).to(device)

    # 构造虚拟输入（尺寸与 backbone 输出一致）
    bs = 2
    p2 = torch.randn(bs, 1024, 16, 16).to(device)
    p3 = torch.randn(bs,  256, 32, 32).to(device)
    p4 = torch.randn(bs,  256, 16, 16).to(device)
    p5 = torch.randn(bs,  512,  8,  8).to(device)

    inputs = [p2, p3, p4, p5]
    outputs = bifpn(inputs)

    # 期望输出形状
    expected = [
        (bs, 128, 32, 32),   # P3_out
        (bs, 256, 16, 16),   # P4_out
        (bs, 512,  8,  8),   # P5_out
    ]

    print("=== BiFPN 输出形状 ===")
    for name, out, exp in zip(["P3_out", "P4_out", "P5_out"], outputs, expected):
        print(f"{name}: {out.shape} (期望 {exp})")
        assert out.shape == exp, f"形状不匹配！期望 {exp}，实际 {out.shape}"

    print("\n✅ 所有输出形状验证通过！")
    print(f"总参数量: {sum(p.numel() for p in bifpn.parameters()):,}")