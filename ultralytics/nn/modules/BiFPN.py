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
    """单层 BiFPN 双向融合（Top‑Down + Bottom‑Up），用于 P2/P3/P4 三层"""
    def __init__(self, ch, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        # Top‑Down 路径：P4 → P3 → P2
        self.td_p4 = Conv(ch, ch, 1, act=False)
        self.td_p3 = Conv(ch, ch, 1, act=False)
        self.fuse_td_p3 = BiFPN_Add(2)
        self.fuse_td_p2 = BiFPN_Add(2)

        # Bottom‑Up 路径：P2 → P3 → P4
        self.bu_p2 = Conv(ch, ch, 1, act=False)
        self.bu_p3 = Conv(ch, ch, 1, act=False)
        self.fuse_bu_p3 = BiFPN_Add(2)
        self.fuse_bu_p4 = BiFPN_Add(2)

        # 下采样（标准 stride=2 卷积）
        self.down_bu2 = Conv(ch, ch, 3, 2)
        self.down_bu3 = Conv(ch, ch, 3, 2)

        # 精炼模块（深度可分离卷积）
        if use_refine:
            self.refine_td_p3 = DepthwiseSeparableConv(ch)
            self.refine_td_p2 = DepthwiseSeparableConv(ch)
            self.refine_bu_p3 = DepthwiseSeparableConv(ch)
            self.refine_bu_p4 = DepthwiseSeparableConv(ch)
        else:
            self.refine_td_p3 = nn.Identity()
            self.refine_td_p2 = nn.Identity()
            self.refine_bu_p3 = nn.Identity()
            self.refine_bu_p4 = nn.Identity()

    def forward(self, p2_enh, p3_enh, p4_enh):
        # Top‑Down: P4 → P3 → P2
        p4_up = F.interpolate(self.td_p4(p4_enh), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_td_p3(self.fuse_td_p3([p3_enh, p4_up]))

        p3_up = F.interpolate(self.td_p3(p3_td), size=p2_enh.shape[-2:], mode='nearest')
        p2_td = self.refine_td_p2(self.fuse_td_p2([p2_enh, p3_up]))

        # Bottom‑Up: P2 → P3 → P4
        p2_down = self.down_bu2(p2_td)
        p3_bu = self.refine_bu_p3(self.fuse_bu_p3([p3_td, p2_down]))

        p3_down = self.down_bu3(p3_bu)
        p4_bu = self.refine_bu_p4(self.fuse_bu_p4([p4_enh, p3_down]))

        return p2_td, p3_bu, p4_bu


class BiFPN(nn.Module):
    """
    带 P2 语义门控的 BiFPN（无级联下采样，轻量）
    - 统一通道 expand_ch 默认取 o3（P3 的输出通道数）
    - 使用 P3/P4 生成空间掩膜，对 P2 进行门控，抑制背景
    - 支持 num_layers 个 BiFPNLayer 进行重复双向融合
    - 精炼模块使用深度可分离卷积（use_refine 可关闭）
    """
    def __init__(self, channels, out_channels=None, num_layers=1, use_refine=True, expand_ch=None):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.expand_ch = expand_ch if expand_ch is not None else o3
        self.num_layers = num_layers

        # ---------- 各层投影到统一通道 ----------
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)

        # ---------- P2 门控掩膜生成（极轻量） ----------
        self.mask_conv3 = Conv(self.expand_ch, 1, 1, act=False)   # P3 → 1 通道
        self.mask_conv4 = Conv(self.expand_ch, 1, 1, act=False)   # P4 → 1 通道

        # ---------- 多层双向融合 ----------
        self.layers = nn.ModuleList([
            BiFPNLayer(self.expand_ch, use_refine=use_refine)
            for _ in range(num_layers)
        ])

        # ---------- 输出压缩 ----------
        self.compress_p2 = Conv(self.expand_ch, o2, 1, act=False)
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)

    def forward(self, features):
        p2, p3, p4 = features

        # 1. 投影到统一通道
        p2 = self.p2_proj(p2)   # (B, C, 160, 160)
        p3 = self.p3_proj(p3)   # (B, C, 80, 80)
        p4 = self.p4_proj(p4)   # (B, C, 40, 40)

        # 2. 生成 P2 门控掩膜（P3/P4 → P2 尺寸）
        m3 = self.mask_conv3(p3)                            # (B, 1, 80, 80)
        m4 = self.mask_conv4(p4)                            # (B, 1, 40, 40)
        m3_up = F.interpolate(m3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        m4_up = F.interpolate(m4, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        mask = torch.sigmoid(m3_up + m4_up)                 # (B, 1, 160, 160)
        p2 = p2 * mask                                       # 背景抑制

        # 3. 双向融合
        for layer in self.layers:
            p2, p3, p4 = layer(p2, p3, p4)

        # 4. 输出压缩
        p2_out = self.compress_p2(p2)
        p3_out = self.compress_p3(p3)
        p4_out = self.compress_p4(p4)

        return [p2_out, p3_out, p4_out]


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]
    model = BiFPN(channels, out_channels, num_layers=1, use_refine=False).to(device)

    outs = model([p2, p3, p4])

    expected = [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]
    for i, (out, exp) in enumerate(zip(outs, expected), start=2):
        status = "✅" if out.shape == exp else "❌"
        print(f"P{i}_out: {out.shape} expected {exp} {status}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    loss = outs[-1].sum()
    loss.backward()
    print("Backward passed successfully.")