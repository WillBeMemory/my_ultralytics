import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class BiFPN_Add(nn.Module):
    """快速归一化加权融合（官方 BiFPN 核心，ReLU 归一化）"""
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（官方 BiFPN 精炼模块）"""
    def __init__(self, ch, kernel_size=3, act=True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size, stride=1,
                            padding=kernel_size // 2, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class BiFPNLayer(nn.Module):
    """单层 BiFPN 双向融合（Top‑Down + Bottom‑Up），可重复使用"""
    def __init__(self, ch, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        # Top‑Down 路径
        self.td_p5 = Conv(ch, ch, 1, act=False)
        self.td_p4 = Conv(ch, ch, 1, act=False)
        self.fuse_td_p4 = BiFPN_Add(2)
        self.fuse_td_p3 = BiFPN_Add(2)

        # Bottom‑Up 路径
        self.bu_p3 = Conv(ch, ch, 1, act=False)
        self.bu_p4 = Conv(ch, ch, 1, act=False)
        self.fuse_bu_p4 = BiFPN_Add(2)
        self.fuse_bu_p5 = BiFPN_Add(2)

        # 下采样（标准 stride=2 卷积）
        self.down_bu3 = Conv(ch, ch, 3, 2)
        self.down_bu4 = Conv(ch, ch, 3, 2)

        # 精炼模块（深度可分离卷积）
        if use_refine:
            self.refine_td_p4 = DepthwiseSeparableConv(ch)
            self.refine_td_p3 = DepthwiseSeparableConv(ch)
            self.refine_bu_p4 = DepthwiseSeparableConv(ch)
            self.refine_bu_p5 = DepthwiseSeparableConv(ch)
        else:
            self.refine_td_p4 = nn.Identity()
            self.refine_td_p3 = nn.Identity()
            self.refine_bu_p4 = nn.Identity()
            self.refine_bu_p5 = nn.Identity()

    def forward(self, p3_enh, p4_enh, p5_enh):
        # Top‑Down
        p5_up = F.interpolate(self.td_p5(p5_enh), size=p4_enh.shape[-2:], mode='nearest')
        p4_td = self.refine_td_p4(self.fuse_td_p4([p4_enh, p5_up]))

        p4_up = F.interpolate(self.td_p4(p4_td), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_td_p3(self.fuse_td_p3([p3_enh, p4_up]))

        # Bottom‑Up
        p3_down = self.down_bu3(p3_td)
        p4_bu = self.refine_bu_p4(self.fuse_bu_p4([p4_td, p3_down]))

        p4_down = self.down_bu4(p4_bu)
        p5_bu = self.refine_bu_p5(self.fuse_bu_p5([p5_enh, p4_down]))

        return p3_td, p4_bu, p5_bu


class BiFPN(nn.Module):
    """
    官方精简版 BiFPN（通道自适应 + P2 门控级联下采样）

    - 统一通道 expand_ch 自动取 c4（P4 通道数），保证在不同模型缩放比下计算量可控。
    - 利用 P3/P4/P5 生成语义掩膜，抑制 P2 背景区域。
    - 门控后的 P2 经标准卷积级联下采样至 P3/P4/P5，并加权融合。
    - 支持 num_layers 个 BiFPNLayer 进行重复双向融合。
    - 精炼模块使用深度可分离卷积（use_refine 可关闭）。
    """
    def __init__(self, channels, out_channels=None, num_layers=1, use_refine=True, expand_ch=None):
        super().__init__()
        c2, c3, c4, c5 = channels
        if out_channels is None:
            out_channels = [c3, c4, c5]
        o3, o4, o5 = out_channels

        # 自动确定内部融合通道数（默认取 P4 通道数，与 Neck 的自然宽度对齐）
        self.expand_ch = expand_ch if expand_ch is not None else c4
        self.num_layers = num_layers

        # ---------- 各层投影到统一通道 ----------
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)
        self.p5_proj = Conv(c5, self.expand_ch, 1, act=False)

        # ---------- 掩膜生成器 ----------
        self.mask_conv3 = Conv(self.expand_ch, 1, 1, act=False)
        self.mask_conv4 = Conv(self.expand_ch, 1, 1, act=False)
        self.mask_conv5 = Conv(self.expand_ch, 1, 1, act=False)

        # ---------- P2 级联下采样 (标准 stride=2) ----------
        self.down_p2_to_p3 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 160→80
        self.down_p3_to_p4 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 80→40
        self.down_p4_to_p5 = Conv(self.expand_ch, self.expand_ch, 3, 2)   # 40→20

        # ---------- 初始加权融合 ----------
        self.init_fuse_p3 = BiFPN_Add(2)
        self.init_fuse_p4 = BiFPN_Add(2)
        self.init_fuse_p5 = BiFPN_Add(2)

        # ---------- 重复 BiFPN 层 ----------
        self.layers = nn.ModuleList([
            BiFPNLayer(self.expand_ch, use_refine=use_refine)
            for _ in range(num_layers)
        ])

        # ---------- 输出压缩 ----------
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)
        self.compress_p5 = Conv(self.expand_ch, o5, 1, act=False)

    def forward(self, features):
        p2, p3, p4, p5 = features

        # 1. 投影到统一通道
        p2_low = self.p2_proj(p2)   # (B, C, 160, 160)
        p3_low = self.p3_proj(p3)   # (B, C, 80, 80)
        p4_low = self.p4_proj(p4)   # (B, C, 40, 40)
        p5_low = self.p5_proj(p5)   # (B, C, 20, 20)

        # 2. 生成语义掩膜并门控 P2
        m3 = self.mask_conv3(p3_low)
        m4 = self.mask_conv4(p4_low)
        m5 = self.mask_conv5(p5_low)
        m3_up = F.interpolate(m3, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)
        m4_up = F.interpolate(m4, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)
        m5_up = F.interpolate(m5, size=p2_low.shape[-2:], mode='bilinear', align_corners=False)
        mask = torch.sigmoid(m3_up + m4_up + m5_up)
        p2_gated = p2_low * mask

        # 3. P2 级联下采样 → P3/P4/P5
        p2_p3 = self.down_p2_to_p3(p2_gated)   # (B, C, 80, 80)
        p2_p4 = self.down_p3_to_p4(p2_p3)      # (B, C, 40, 40)
        p2_p5 = self.down_p4_to_p5(p2_p4)      # (B, C, 20, 20)

        # 4. 初始加权融合
        p3_enh = self.init_fuse_p3([p3_low, p2_p3])
        p4_enh = self.init_fuse_p4([p4_low, p2_p4])
        p5_enh = self.init_fuse_p5([p5_low, p2_p5])

        # 5. 通过多个 BiFPN 层
        for layer in self.layers:
            p3_enh, p4_enh, p5_enh = layer(p3_enh, p4_enh, p5_enh)

        # 6. 输出压缩
        p3_out = self.compress_p3(p3_enh)
        p4_out = self.compress_p4(p4_enh)
        p5_out = self.compress_p5(p5_enh)

        return [p3_out, p4_out, p5_out]


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    # 模拟 YOLO11n 的通道配置（width_multiple=0.25 缩放后）
    # P2: 128, P3: 256, P4: 512, P5: 1024  → 缩放后：32, 64, 128, 256？这里直接用典型值
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)
    p5 = torch.randn(bs, 1024, 20, 20).to(device)

    channels = [128, 256, 512, 1024]
    out_channels = [256, 512, 1024]
    # expand_ch 会自动取 c4=512，你可手动指定 expand_ch=256 来覆盖
    model = BiFPN(channels, out_channels, num_layers=1, use_refine=False).to(device)

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