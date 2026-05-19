import torch
import torch.nn as nn
import torch.nn.functional as F

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
    双向特征金字塔 (BiFPN)，无 SPDConv，无星操作。
    利用深层语义生成空间掩膜，对 P2 特征进行门控抑制背景，
    再通过标准卷积级联下采样到 P3/P4/P5 尺寸，与各层特征加权融合。
    """
    def __init__(self, channels, out_channels=None, use_c3k2=False):
        super().__init__()
        c2, c3, c4, c5 = channels
        if out_channels is None:
            out_channels = [c3, c4, c5]
        o3, o4, o5 = out_channels
        self.expand_ch = 256

        # ---------- 各层投影到统一通道 ----------
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)
        self.p5_proj = Conv(c5, self.expand_ch, 1, act=False)

        # ---------- 掩膜生成器 ----------
        # 为 P3/P4/P5 各生成一个单通道注意力图，上采样到 P2 尺寸后融合
        self.mask_conv3 = Conv(self.expand_ch, 1, 1, act=False)
        self.mask_conv4 = Conv(self.expand_ch, 1, 1, act=False)
        self.mask_conv5 = Conv(self.expand_ch, 1, 1, act=False)

        # ---------- P2 级联下采样 (标准卷积) ----------
        # 通过三次 stride=2 卷积将 P2 特征逐步降到 P3/P4/P5 尺寸
        self.down_p2_to_p3 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 160→80
        self.down_p3_to_p4 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 80→40
        self.down_p4_to_p5 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 40→20

        # ---------- 双向融合前的初始融合 (P2 下采样 + 原始投影) ----------
        self.fuse_init_p3 = BiFPN_Add(2)
        self.fuse_init_p4 = BiFPN_Add(2)
        self.fuse_init_p5 = BiFPN_Add(2)

        # ---------- TD 路径 ----------
        self.td_p5 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.td_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.fuse_td_p4 = BiFPN_Add(2)
        self.fuse_td_p3 = BiFPN_Add(2)

        # ---------- BU 路径 ----------
        self.bu_p3 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.bu_p4 = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.fuse_bu_p4 = BiFPN_Add(2)
        self.fuse_bu_p5 = BiFPN_Add(2)

        # ---------- BU 下采样 (标准卷积) ----------
        self.down_bu3 = Conv(self.expand_ch, self.expand_ch, 3, 2)
        self.down_bu4 = Conv(self.expand_ch, self.expand_ch, 3, 2)

        # ---------- 可选精炼 ----------
        if use_c3k2:
            self.refine_td_p4 = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_td_p3 = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_bu_p4 = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
            self.refine_bu_p5 = C3k2(self.expand_ch, self.expand_ch, n=1, shortcut=False, e=0.5)
        else:
            self.refine_td_p4 = nn.Identity()
            self.refine_td_p3 = nn.Identity()
            self.refine_bu_p4 = nn.Identity()
            self.refine_bu_p5 = nn.Identity()

        # ---------- 输出压缩 ----------
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)
        self.compress_p5 = Conv(self.expand_ch, o5, 1, act=False)

    def forward(self, features):
        p2, p3, p4, p5 = features

        # 1. 投影
        p2_low = self.p2_proj(p2)   # (B,256,160,160)
        p3_low = self.p3_proj(p3)   # (B,256,80,80)
        p4_low = self.p4_proj(p4)   # (B,256,40,40)
        p5_low = self.p5_proj(p5)   # (B,256,20,20)

        # 2. 掩膜生成 (P3/P4/P5 各自生成注意力图，上采样后融合)
        mask3 = self.mask_conv3(p3_low)   # (B,1,80,80)
        mask4 = self.mask_conv4(p4_low)   # (B,1,40,40)
        mask5 = self.mask_conv5(p5_low)   # (B,1,20,20)

        mask4_up = F.interpolate(mask4, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)
        mask3_up = F.interpolate(mask3, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)
        mask5_up = F.interpolate(mask5, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)

        # 简单求和融合 + Sigmoid
        mask = torch.sigmoid(mask3_up + mask4_up + mask5_up)   # (B,1,160,160)

        # 3. P2 门控
        p2_gated = p2_low * mask

        # 4. 标准卷积级联下采样
        p2_p3 = self.down_p2_to_p3(p2_gated)   # (B,256,80,80)
        p2_p4 = self.down_p3_to_p4(p2_p3)      # (B,256,40,40)
        p2_p5 = self.down_p4_to_p5(p2_p4)      # (B,256,20,20)

        # 5. 初始融合：P2 下采样特征 + 原始投影特征
        p3_enh = self.fuse_init_p3([p3_low, p2_p3])
        p4_enh = self.fuse_init_p4([p4_low, p2_p4])
        p5_enh = self.fuse_init_p5([p5_low, p2_p5])

        # 6. 双向融合 (TD + BU)
        # ---- Top-Down ----
        p5_up = F.interpolate(self.td_p5(p5_enh), size=p4_enh.shape[-2:], mode='nearest')
        p4_td = self.refine_td_p4(self.fuse_td_p4([p4_enh, p5_up]))

        p4_up = F.interpolate(self.td_p4(p4_td), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_td_p3(self.fuse_td_p3([p3_enh, p4_up]))

        # ---- Bottom-Up ----
        p3_down = self.down_bu3(p3_td)
        p4_bu = self.refine_bu_p4(self.fuse_bu_p4([p4_td, p3_down]))

        p4_down = self.down_bu4(p4_bu)
        p5_bu = self.refine_bu_p5(self.fuse_bu_p5([p5_enh, p4_down]))

        # 7. 输出压缩
        p3_out = self.compress_p3(p3_td)
        p4_out = self.compress_p4(p4_bu)
        p5_out = self.compress_p5(p5_bu)

        return [p3_out, p4_out, p5_out]


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)
    p5 = torch.randn(bs, 1024, 20, 20).to(device)

    channels = [128, 256, 512, 1024]
    out_channels = [256, 512, 1024]
    model = BiFPN(channels, out_channels, use_c3k2=False).to(device)

    outs = model([p2, p3, p4, p5])

    expected = [
        (bs, 256, 80, 80),
        (bs, 512, 40, 40),
        (bs, 1024, 20, 20)
    ]

    for i, (out, exp) in enumerate(zip(outs, expected), start=3):
        status = "✅" if out.shape == exp else "❌"
        print(f"P{i}_out: {out.shape} expected {exp} {status}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    loss = outs[-1].sum()
    loss.backward()
    print("Backward passed successfully.")