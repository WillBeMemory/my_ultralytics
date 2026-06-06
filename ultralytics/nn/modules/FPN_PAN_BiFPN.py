import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class BiFPN_Add(nn.Module):
    """快速归一化加权融合（BiFPN 核心，ReLU 归一化）"""

    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 — 仅 DW 3×3，通道混合由前置 mid_conv(1×1) 完成"""

    def __init__(self, ch, kernel_size=3, act=True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size, stride=1,
                            padding=kernel_size // 2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class FPN_PAN(nn.Module):
    """
    标准 FPN-PAN，与 YOLO11 默认 Neck 结构完全一致：
    - FPN (Top-Down):   Upsample → Concat → C3k2
    - PAN (Bottom-Up):  Conv(s=2) → Concat → C3k2
    - C3k2 内部 n=1（YOLO11 默认，n 控制 Bottleneck 数量）
    """

    def __init__(self, channels, out_channels=None, c3k2_n=1):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self._out_channels = out_channels
        self._c3k2_n = c3k2_n
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        o2, o3, o4 = self._out_channels

        # ========== FPN (Top-Down) ==========
        self.fpn_p3 = C3k2(p3.shape[1] + p4.shape[1], o3, n=self._c3k2_n, c3k=False)
        self.fpn_p2 = C3k2(p2.shape[1] + o3, o2, n=self._c3k2_n, c3k=False)

        # ========== PAN (Bottom-Up) ==========
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C3k2(o3 + o3, o3, n=self._c3k2_n, c3k=False)
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C3k2(o4 + p4.shape[1], o4, n=self._c3k2_n, c3k=False)

        self.to(p2.device)
        self._initialized = True

    def forward(self, features):
        p2, p3, p4 = features

        if not self._initialized:
            self._init_layers(p2, p3, p4)

        # ========== FPN (Top-Down) ==========
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3_fpn = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        p3_up = F.interpolate(p3_fpn, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2_fpn = self.fpn_p2(torch.cat([p2, p3_up], dim=1))

        # ========== PAN (Bottom-Up) ==========
        p2_down = self.pan_p2_down(p2_fpn)
        p3_pan = self.pan_p3(torch.cat([p3_fpn, p2_down], dim=1))

        p3_down = self.pan_p3_down(p3_pan)
        p4_pan = self.pan_p4(torch.cat([p4, p3_down], dim=1))

        return [p2_fpn, p3_pan, p4_pan]


class BiFPNLayer(nn.Module):
    """
    BiFPN 单层，P2/P3/P4 两两加权融合（Top-Down + Bottom-Up）
    - 使用 BiFPN_Add 做加权融合（而非 Concat）
    - 1x1 Conv 做通道投影保证维度一致
    - DWConv 做融合后精炼
    """

    def __init__(self, channels=None, use_refine=True, mode='bilinear'):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self.mode = mode
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ========== Top-Down：P4→P3, P3→P2 ==========
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        self.td_p3_fuse = BiFPN_Add(2)
        self.td_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        self.td_p2_fuse = BiFPN_Add(2)
        self.td_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        self.bu_p3_fuse = BiFPN_Add(2)
        self.bu_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        self.bu_p4_fuse = BiFPN_Add(2)
        self.bu_p4_refine = DepthwiseSeparableConv(c4) if self.use_refine else nn.Identity()

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)

        # ========== Top-Down ==========
        p4_up = F.interpolate(self.td_p4_to_p3(p4_in), size=p3_in.shape[-2:], mode=self.mode, align_corners=False)
        p3_td = self.td_p3_fuse([p3_in, p4_up])
        p3_td = self.td_p3_refine(p3_td)

        p3_up = F.interpolate(self.td_p3_to_p2(p3_td), size=p2_in.shape[-2:], mode=self.mode, align_corners=False)
        p2_td = self.td_p2_fuse([p2_in, p3_up])
        p2_td = self.td_p2_refine(p2_td)

        # ========== Bottom-Up ==========
        p2_down = self.bu_p2_to_p3(p2_td)
        p3_out = self.bu_p3_fuse([p3_td, p2_down])
        p3_out = self.bu_p3_refine(p3_out)

        p3_down = self.bu_p3_to_p4(p3_out)
        p4_out = self.bu_p4_fuse([p4_in, p3_down])
        p4_out = self.bu_p4_refine(p4_out)

        return p2_td, p3_out, p4_out


class FPN_PAN_BiFPN(nn.Module):
    """
    优化版 FPN-PAN-BiFPN 组合模块

    优化点：
    1. 残差连接：BiFPN 学习残差，避免覆盖 FPN-PAN 的优质特征
    2. P2 语义门控：P3/P4 生成空间掩膜抑制 P2 背景噪声（SAR 关键）
    3. 双线性上采样：替换 nearest 插值，小目标边界更平滑
    4. c3k2_n 可配置：FPN-PAN 中 C3k2 的深度可调
    """

    def __init__(self, channels, out_channels=None, num_bifpn_layers=1, use_refine=False,
                 c3k2_n=1, p2_gate=True, residual=True, upsample_mode='bilinear'):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_bifpn_layers = num_bifpn_layers
        self.p2_gate = p2_gate
        self.residual = residual

        # FPN-PAN（标准 YOLO11 结构，c3k2_n 可调）
        self.fpn_pan = FPN_PAN(channels, out_channels, c3k2_n=c3k2_n)

        # P2 语义门控：利用 P3/P4 的高层语义生成空间掩膜，抑制 P2 背景噪声
        if p2_gate:
            self.p2_gate_conv3 = Conv(o3, 1, 1, act=False)  # P3 → 单通道掩膜
            self.p2_gate_conv4 = Conv(o4, 1, 1, act=False)  # P4 → 单通道掩膜

        # BiFPN 层
        if num_bifpn_layers > 0:
            self.bifpn_layers = nn.ModuleList([
                BiFPNLayer(out_channels, use_refine, mode=upsample_mode)
                for _ in range(num_bifpn_layers)
            ])
        else:
            self.bifpn_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        # ========== FPN-PAN ==========
        p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])

        # ========== P2 语义门控 ==========
        if self.p2_gate:
            # P3/P4 高层语义 → 空间注意力掩膜 → 抑制 P2 背景
            m3 = self.p2_gate_conv3(p3_out)                           # (B, 1, 80, 80)
            m4 = self.p2_gate_conv4(p4_out)                           # (B, 1, 40, 40)
            m3_up = F.interpolate(m3, size=p2_out.shape[-2:], mode='bilinear', align_corners=False)
            m4_up = F.interpolate(m4, size=p2_out.shape[-2:], mode='bilinear', align_corners=False)
            gate = torch.sigmoid(m3_up + m4_up)                       # (B, 1, H, W)
            p2_out = p2_out * gate

        # ========== BiFPN 精炼（残差连接） ==========
        if self.bifpn_layers:
            p2_bifpn, p3_bifpn, p4_bifpn = p2_out, p3_out, p4_out
            for layer in self.bifpn_layers:
                p2_bifpn, p3_bifpn, p4_bifpn = layer(p2_bifpn, p3_bifpn, p4_bifpn)

            if self.residual:
                # BiFPN 学习残差，保留 FPN-PAN 的原始特征
                p2_out = p2_out + p2_bifpn
                p3_out = p3_out + p3_bifpn
                p4_out = p4_out + p4_bifpn
            else:
                p2_out, p3_out, p4_out = p2_bifpn, p3_bifpn, p4_bifpn

        return [p2_out, p3_out, p4_out]


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]

    print("\n=== Testing FPN_PAN (标准 YOLO11 结构) ===")
    fpn_pan = FPN_PAN(channels, out_channels).to(device)
    outs = fpn_pan([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan.parameters())
    print(f"  FPN_PAN Params: {total_params:,}")

    print("\n=== Testing BiFPNLayer (两两加权融合) ===")
    bifpn_layer = BiFPNLayer(out_channels, use_refine=True).to(device)
    p2_out, p3_out, p4_out = bifpn_layer(p2, p3, p4)
    for name, out, exp in [
        ("P2", p2_out, (bs, 128, 160, 160)),
        ("P3", p3_out, (bs, 256, 80, 80)),
        ("P4", p4_out, (bs, 512, 40, 40))
    ]:
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{status}]")
    print(f"  BiFPNLayer Params: {sum(p.numel() for p in bifpn_layer.parameters()):,}")

    print("\n=== Testing FPN_PAN_BiFPN (优化版: residual + p2_gate + refine) ===")
    fpn_pan_bifpn = FPN_PAN_BiFPN(
        channels, out_channels,
        num_bifpn_layers=1,
        use_refine=True,
        p2_gate=True,
        residual=True,
    ).to(device)
    outs = fpn_pan_bifpn([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_bifpn.parameters())
    print(f"  FPN_PAN_BiFPN Params: {total_params:,}")

    print("\n=== Testing FPN_PAN_BiFPN (3 BiFPN layers) ===")
    fpn_pan_bifpn3 = FPN_PAN_BiFPN(
        channels, out_channels,
        num_bifpn_layers=3,
        use_refine=True,
        p2_gate=True,
        residual=True,
    ).to(device)
    outs = fpn_pan_bifpn3([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_bifpn3.parameters())
    print(f"  FPN_PAN_BiFPN (3 layers) Params: {total_params:,}")