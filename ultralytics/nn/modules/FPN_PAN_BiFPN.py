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
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self._out_channels = out_channels
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        o2, o3, o4 = self._out_channels

        # ========== FPN (Top-Down) ==========
        # P4 → Upsample → Concat(P3) → C3k2 → P3_fpn
        self.fpn_p3 = C3k2(p3.shape[1] + p4.shape[1], o3, n=1, c3k=False)
        # P3_fpn → Upsample → Concat(P2) → C3k2 → P2_fpn
        self.fpn_p2 = C3k2(p2.shape[1] + o3, o2, n=1, c3k=False)

        # ========== PAN (Bottom-Up) ==========
        # P2_fpn → Conv(s=2) → Concat(P3_fpn) → C3k2 → P3_pan
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C3k2(o3 + o3, o3, n=1, c3k=False)
        # P3_pan → Conv(s=2) → Concat(P4) → C3k2 → P4_pan
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C3k2(o4 + p4.shape[1], o4, n=1, c3k=False)

        self.to(p2.device)
        self._initialized = True

    def forward(self, features):
        p2, p3, p4 = features

        if not self._initialized:
            self._init_layers(p2, p3, p4)

        # ========== FPN (Top-Down) ==========
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_fpn = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        p3_up = F.interpolate(p3_fpn, size=p2.shape[-2:], mode='nearest')
        p2_fpn = self.fpn_p2(torch.cat([p2, p3_up], dim=1))

        # ========== PAN (Bottom-Up) ==========
        p2_down = self.pan_p2_down(p2_fpn)
        p3_pan = self.pan_p3(torch.cat([p3_fpn, p2_down], dim=1))

        p3_down = self.pan_p3_down(p3_pan)
        p4_pan = self.pan_p4(torch.cat([p4, p3_down], dim=1))

        return [p2_fpn, p3_pan, p4_pan]


class FullyConnectedBiFPNLayer(nn.Module):
    """
    全连接 BiFPN 层：P2/P3/P4 之间全部互相采样融合

    融合方式：
    - P2 ↔ P3：P2 和 P3 双向融合
    - P3 ↔ P4：P3 和 P4 双向融合
    - P2 ↔ P4：P2 和 P4 直接双向融合（跨尺度）

    每条路径包含：
    1. 1x1 Conv 通道投影
    2. 上下采样（对齐空间尺寸）
    3. BiFPN_Add 加权融合
    """
    def __init__(self, channels=None, use_refine=True):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ===== P2 ↔ P3 双向融合 =====
        # P2 → P3: P2 降采样后与 P3 融合
        self.p2_to_p3_conv = Conv(c2, c3, 3, 2)  # 降采样
        self.p2_p3_fuse = BiFPN_Add(2)
        self.p2_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # P3 → P2: P3 上采样后与 P2 融合
        self.p3_to_p2_conv = Conv(c3, c2, 1, act=False)  # 通道投影
        self.p3_p2_fuse = BiFPN_Add(2)
        self.p3_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        # ===== P3 ↔ P4 双向融合 =====
        # P3 → P4: P3 降采样后与 P4 融合
        self.p3_to_p4_conv = Conv(c3, c4, 3, 2)
        self.p3_p4_fuse = BiFPN_Add(2)
        self.p3_p4_refine = DepthwiseSeparableConv(c4) if self.use_refine else nn.Identity()

        # P4 → P3: P4 上采样后与 P3 融合
        self.p4_to_p3_conv = Conv(c4, c3, 1, act=False)
        self.p4_p3_fuse = BiFPN_Add(2)
        self.p4_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # ===== P2 ↔ P4 跨尺度直接融合 =====
        # P2 → P4: P2 两次降采样后与 P4 融合
        self.p2_to_p4_conv1 = Conv(c2, c2, 3, 2)  # H/2, W/2
        self.p2_to_p4_conv2 = Conv(c2, c4, 3, 2)  # H/4, W/4
        self.p2_p4_fuse = BiFPN_Add(2)
        self.p2_p4_refine = DepthwiseSeparableConv(c4) if self.use_refine else nn.Identity()

        # P4 → P2: P4 上采样后与 P2 融合
        self.p4_to_p2_conv = Conv(c4, c2, 1, act=False)
        self.p4_p2_fuse = BiFPN_Add(2)
        self.p4_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        self._c2, self._c3, self._c4 = c2, c3, c4
        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)
        c2, c3, c4 = self._c2, self._c3, self._c4

        # ===== P2 ↔ P3 融合 =====
        # P2 → P3
        p2_down = self.p2_to_p3_conv(p2_in)  # c2→c3, H/2
        p2_p3_fused = self.p2_p3_fuse([p3_in, p2_down])
        p2_p3_fused = self.p2_p3_refine(p2_p3_fused)

        # P3 → P2
        p3_up = F.interpolate(self.p3_to_p2_conv(p3_in), size=p2_in.shape[-2:], mode='nearest')
        p3_p2_fused = self.p3_p2_fuse([p2_in, p3_up])
        p3_p2_fused = self.p3_p2_refine(p3_p2_fused)

        # ===== P3 ↔ P4 融合 =====
        # P3 → P4
        p3_down = self.p3_to_p4_conv(p3_in)  # c3→c4, H/2
        p3_p4_fused = self.p3_p4_fuse([p4_in, p3_down])
        p3_p4_fused = self.p3_p4_refine(p3_p4_fused)

        # P4 → P3
        p4_up = F.interpolate(self.p4_to_p3_conv(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p4_p3_fused = self.p4_p3_fuse([p3_in, p4_up])
        p4_p3_fused = self.p4_p3_refine(p4_p3_fused)

        # ===== P2 ↔ P4 跨尺度融合 =====
        # P2 → P4 (两次降采样)
        p2_down_half = self.p2_to_p4_conv1(p2_in)  # H/2
        p2_down_quarter = self.p2_to_p4_conv2(p2_down_half)  # H/4
        p2_p4_fused = self.p2_p4_fuse([p4_in, p2_down_quarter])
        p2_p4_fused = self.p2_p4_refine(p2_p4_fused)

        # P4 → P2 (两次上采样)
        p4_up = F.interpolate(self.p4_to_p2_conv(p4_in), size=p2_in.shape[-2:], mode='nearest')
        p4_p2_fused = self.p4_p2_fuse([p2_in, p4_up])
        p4_p2_fused = self.p4_p2_refine(p4_p2_fused)

        # ===== 最终输出 =====
        # 所有融合结果加权求和
        # P2_out = f(P2, P3→P2, P4→P2)
        p2_out = self._fuse_outputs([p2_in, p3_p2_fused, p4_p2_fused], c2)
        # P3_out = f(P3, P2→P3, P4→P3)
        p3_out = self._fuse_outputs([p3_in, p2_p3_fused, p4_p3_fused], c3)
        # P4_out = f(P4, P3→P4, P2→P4)
        p4_out = self._fuse_outputs([p4_in, p3_p4_fused, p2_p4_fused], c4)

        return p2_out, p3_out, p4_out

    def _fuse_outputs(self, features, out_channels):
        """对多个融合特征进行加权求和"""
        # 使用平均融合（也可改用可学习权重）
        out = sum(features) / len(features)
        return out


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN-BiFPN 组合模块
    - FPN-PAN：与 YOLO11 默认结构完全一致（Conv + C3k2 + Upsample + Concat）
    - BiFPN：全连接双向融合（P2↔P3, P3↔P4, P2↔P4）
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1, use_refine=True, num_cycles=1):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_bifpn_layers = num_bifpn_layers
        self.num_cycles = num_cycles

        # FPN-PAN（标准 YOLO11 结构）
        self.fpn_pan = FPN_PAN(channels, out_channels)

        # 全连接 BiFPN 层
        if num_bifpn_layers > 0:
            self.bifpn_layers = nn.ModuleList([
                FullyConnectedBiFPNLayer(out_channels, use_refine) for _ in range(num_bifpn_layers)
            ])
        else:
            self.bifpn_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        # FPN-PAN
        p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])

        # BiFPN 精炼（可选多层）
        if self.bifpn_layers:
            for layer in self.bifpn_layers:
                p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)

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

    print("\n=== Testing FullyConnectedBiFPNLayer ===")
    bifpn = FullyConnectedBiFPNLayer(out_channels, use_refine=True).to(device)
    p2_out, p3_out, p4_out = bifpn(p2, p3, p4)
    for name, out, exp in [
        ("P2", p2_out, (bs, 128, 160, 160)),
        ("P3", p3_out, (bs, 256, 80, 80)),
        ("P4", p4_out, (bs, 512, 40, 40))
    ]:
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in bifpn.parameters())
    print(f"  FullyConnectedBiFPNLayer Params: {total_params:,}")

    print("\n=== Testing FPN_PAN_BiFPN (1 layer) ===")
    fpn_pan_bifpn = FPN_PAN_BiFPN(
        channels, out_channels,
        num_bifpn_layers=1,
        use_refine=True
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
