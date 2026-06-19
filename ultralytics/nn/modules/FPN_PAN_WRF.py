import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class WeightedAdd(nn.Module):
    """快速归一化加权融合（fast normalized fusion，ReLU 归一化）。"""
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


class WRF(nn.Module):
    """
    加权再融合层（Weighted Re-Fusion），P2/P3/P4 两两加权融合（Top-Down + Bottom-Up）。
    - 使用 WeightedAdd 做加权融合（而非 Concat）
    - 1x1 Conv 做通道投影保证维度一致
    - DWConv 做融合后精炼（use_refine 开关控制，默认在 FPN_PAN_WRF 中关闭）
    注：本层设计为寄生在已全融合的 FPN-PAN 之后，故默认不启用精炼（use_refine=False）。
    """
    def __init__(self, channels=None, use_refine=True):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ========== Top-Down：P4→P3, P3→P2 ==========
        # P4 → 1x1 Conv(512→256) → Upsample → WeightedAdd(P3, P4_up) → DWConv refine
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        self.td_p3_fuse = WeightedAdd(2)
        self.td_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # P3_td → 1x1 Conv(256→128) → Upsample → WeightedAdd(P2, P3_up) → DWConv refine
        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        self.td_p2_fuse = WeightedAdd(2)
        self.td_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        # P2_td → Conv(s=2, 128→256) → WeightedAdd(P3_td, P2_down) → DWConv refine
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        self.bu_p3_fuse = WeightedAdd(2)
        self.bu_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # P3_out → Conv(s=2, 256→512) → WeightedAdd(P4, P3_down) → DWConv refine
        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        self.bu_p4_fuse = WeightedAdd(2)
        self.bu_p4_refine = DepthwiseSeparableConv(c4) if self.use_refine else nn.Identity()

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)

        # ========== Top-Down ==========
        # P4 → P3 融合
        p4_up = F.interpolate(self.td_p4_to_p3(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.td_p3_fuse([p3_in, p4_up])
        p3_td = self.td_p3_refine(p3_td)

        # P3 → P2 融合
        p3_up = F.interpolate(self.td_p3_to_p2(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_td = self.td_p2_fuse([p2_in, p3_up])
        p2_td = self.td_p2_refine(p2_td)

        # ========== Bottom-Up ==========
        # P2 → P3 融合
        p2_down = self.bu_p2_to_p3(p2_td)
        p3_out = self.bu_p3_fuse([p3_td, p2_down])
        p3_out = self.bu_p3_refine(p3_out)

        # P3 → P4 融合
        p3_down = self.bu_p3_to_p4(p3_out)
        p4_out = self.bu_p4_fuse([p4_in, p3_down])
        p4_out = self.bu_p4_refine(p4_out)

        return p2_td, p3_out, p4_out


class FPN_PAN_WRF(nn.Module):
    """
    FPN-PAN + WRF 组合颈部。
    - FPN-PAN：与 YOLO11 默认结构完全一致（Conv + C3k2 + Upsample + Concat），完成全尺度融合
    - WRF：P2/P3/P4 两两加权再融合（WeightedAdd），可堆叠多层；默认 1 层、无精炼
    """
    def __init__(self, channels, out_channels=None, num_wrf_layers=1, use_refine=False):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_wrf_layers = num_wrf_layers

        # FPN-PAN（标准 YOLO11 结构）
        self.fpn_pan = FPN_PAN(channels, out_channels)

        # WRF 再融合层
        if num_wrf_layers > 0:
            self.wrf_layers = nn.ModuleList([
                WRF(out_channels, use_refine) for _ in range(num_wrf_layers)
            ])
        else:
            self.wrf_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        # FPN-PAN
        p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])

        # WRF 再融合
        if self.wrf_layers:
            for layer in self.wrf_layers:
                p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)

        return [p2_out, p3_out, p4_out]


# Backward-compat (deprecated; canonical name is FPN_PAN_WRF).
# 保留旧名别名，使内嵌旧 yaml（模块名 FPN_PAN_BiFPN）的老 .pt 权重仍可经 tasks.parse_model 的 globals() 解析加载。
FPN_PAN_BiFPN = FPN_PAN_WRF


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

    print("\n=== Testing WRF (加权再融合) ===")
    wrf_layer = WRF(out_channels, use_refine=True).to(device)
    p2_out, p3_out, p4_out = wrf_layer(p2, p3, p4)
    for name, out, exp in [
        ("P2", p2_out, (bs, 128, 160, 160)),
        ("P3", p3_out, (bs, 256, 80, 80)),
        ("P4", p4_out, (bs, 512, 40, 40))
    ]:
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{status}]")
    print(f"  WRF Params: {sum(p.numel() for p in wrf_layer.parameters()):,}")

    print("\n=== Testing FPN_PAN_WRF (1 WRF layer) ===")
    fpn_pan_wrf = FPN_PAN_WRF(
        channels, out_channels,
        num_wrf_layers=1,
        use_refine=True
    ).to(device)
    outs = fpn_pan_wrf([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_wrf.parameters())
    print(f"  FPN_PAN_WRF Params: {total_params:,}")

    print("\n=== Testing FPN_PAN_WRF (3 WRF layers) ===")
    fpn_pan_wrf3 = FPN_PAN_WRF(
        channels, out_channels,
        num_wrf_layers=3,
        use_refine=True
    ).to(device)
    outs = fpn_pan_wrf3([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_wrf3.parameters())
    print(f"  FPN_PAN_WRF (3 layers) Params: {total_params:,}")
