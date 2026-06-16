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
        self.fpn_p3 = C3k2(p3.shape[1] + p4.shape[1], o3, n=2, c3k=False)
        # P3_fpn → Upsample → Concat(P2) → C3k2 → P2_fpn
        self.fpn_p2 = C3k2(p2.shape[1] + o3, o2, n=2, c3k=False)

        # ========== PAN (Bottom-Up) ==========
        # P2_fpn → Conv(s=2) → Concat(P3_fpn) → C3k2 → P3_pan
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C3k2(o3 + o3, o3, n=2, c3k=False)
        # P3_pan → Conv(s=2) → Concat(P4) → C3k2 → P4_pan
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C3k2(o4 + p4.shape[1], o4, n=2, c3k=False)

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



class BiFPNLayer(nn.Module):
    """
    BiFPN 单层，P2/P3/P4 两两加权融合（Top-Down + Bottom-Up）
    - 使用 BiFPN_Add 做加权融合（而非 Concat）
    - 1x1 Conv 做通道投影保证维度一致
    - refine_type: 'dwconv' 使用深度可分离卷积精炼，'c3k2' 使用 C3k2 精炼
    """
    def __init__(self, channels=None, use_refine=True, refine_type='c3k2'):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self.refine_type = refine_type
        self._initialized = False

    def _make_refine(self, ch):
        if not self.use_refine:
            return nn.Identity()
        if self.refine_type == 'c3k2':
            return C3k2(ch, ch, n=2, c3k=False)
        return DepthwiseSeparableConv(ch)

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]
        o2, o3, o4 = self._channels  # 目标输出通道

        # ========== 内部使用 backbone 原始通道做融合 ==========
        # ========== Top-Down：P4→P3, P3→P2 ==========
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        self.td_p3_fuse = BiFPN_Add(2)
        self.td_p3_refine = self._make_refine(c3)

        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        self.td_p2_fuse = BiFPN_Add(2)
        self.td_p2_refine = self._make_refine(c2)

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        self.bu_p3_fuse = BiFPN_Add(2)
        self.bu_p3_refine = self._make_refine(c3)

        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        self.bu_p4_fuse = BiFPN_Add(2)
        self.bu_p4_refine = self._make_refine(c4)

        # ========== 输出投影：backbone通道 → out_channels ==========
        self.out_proj_p2 = Conv(c2, o2, 1, act=False) if c2 != o2 else nn.Identity()
        self.out_proj_p3 = Conv(c3, o3, 1, act=False) if c3 != o3 else nn.Identity()
        self.out_proj_p4 = Conv(c4, o4, 1, act=False) if c4 != o4 else nn.Identity()

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)

        # ========== Top-Down（backbone原始通道空间） ==========
        p4_up = F.interpolate(self.td_p4_to_p3(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.td_p3_fuse([p3_in, p4_up])
        p3_td = self.td_p3_refine(p3_td)

        p3_up = F.interpolate(self.td_p3_to_p2(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_td = self.td_p2_fuse([p2_in, p3_up])
        p2_td = self.td_p2_refine(p2_td)

        # ========== Bottom-Up ==========
        p2_down = self.bu_p2_to_p3(p2_td)
        p3_out = self.bu_p3_fuse([p3_td, p2_down])
        p3_out = self.bu_p3_refine(p3_out)

        p3_down = self.bu_p3_to_p4(p3_out)
        p4_out = self.bu_p4_fuse([p4_in, p3_down])
        p4_out = self.bu_p4_refine(p4_out)

        # ========== 输出投影 ==========
        p2_out = self.out_proj_p2(p2_td)
        p3_out = self.out_proj_p3(p3_out)
        p4_out = self.out_proj_p4(p4_out)

        return p2_out, p3_out, p4_out


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN-BiFPN 组合模块
    - use_fpn_pan=True:  FPN-PAN → BiFPN 精炼（原始模式）
    - use_fpn_pan=False: 仅 BiFPN（替代 FPN-PAN，论文核心方案）
    - refine_type:       'dwconv' 或 'c3k2'
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1,
                 use_refine=True, use_fpn_pan=False, refine_type='c3k2'):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]

        self.num_bifpn_layers = num_bifpn_layers
        self.use_fpn_pan = use_fpn_pan

        # FPN-PAN（可选）
        if use_fpn_pan:
            self.fpn_pan = FPN_PAN(channels, out_channels)
        else:
            self.fpn_pan = None

        # BiFPN 层
        if num_bifpn_layers > 0:
            self.bifpn_layers = nn.ModuleList([
                BiFPNLayer(out_channels, use_refine, refine_type)
                for _ in range(num_bifpn_layers)
            ])
        else:
            self.bifpn_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        if self.fpn_pan is not None:
            # FPN-PAN + BiFPN 精炼模式
            p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])
            if self.bifpn_layers is not None:
                for layer in self.bifpn_layers:
                    p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)
        else:
            # 纯 BiFPN 模式（替代 FPN-PAN）
            p2_out, p3_out, p4_out = p2, p3, p4
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

    def test_model(name, model, inputs):
        outs = model(inputs)
        expected = [(bs, 128, 160, 160), (bs, 256, 80, 80), (bs, 512, 40, 40)]
        for i, (out, exp) in enumerate(zip(outs, expected), start=2):
            status = "OK" if out.shape == exp else "FAIL"
            print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total_params:,}")

    print("\n=== 1. FPN_PAN (标准 YOLO11 结构) ===")
    test_model("FPN_PAN", FPN_PAN(channels, out_channels).to(device), [p2, p3, p4])

    print("\n=== 2. BiFPNLayer (DWConv精炼) ===")
    bifpn_dw = BiFPNLayer(out_channels, use_refine=True, refine_type='dwconv').to(device)
    p2o, p3o, p4o = bifpn_dw(p2, p3, p4)
    for name, out, exp in [("P2", p2o, (bs, 128, 160, 160)), ("P3", p3o, (bs, 256, 80, 80)), ("P4", p4o, (bs, 512, 40, 40))]:
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{'OK' if out.shape == exp else 'FAIL'}]")
    print(f"  Params: {sum(p.numel() for p in bifpn_dw.parameters()):,}")

    print("\n=== 3. BiFPNLayer (C3k2精炼) ===")
    bifpn_c3 = BiFPNLayer(out_channels, use_refine=True, refine_type='c3k2').to(device)
    p2o, p3o, p4o = bifpn_c3(p2, p3, p4)
    for name, out, exp in [("P2", p2o, (bs, 128, 160, 160)), ("P3", p3o, (bs, 256, 80, 80)), ("P4", p4o, (bs, 512, 40, 40))]:
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{'OK' if out.shape == exp else 'FAIL'}]")
    print(f"  Params: {sum(p.numel() for p in bifpn_c3.parameters()):,}")

    print("\n=== 4. FPN_PAN + BiFPN (DWConv精炼) ===")
    test_model("FPN_PAN+DW", FPN_PAN_BiFPN(
        channels, out_channels, num_bifpn_layers=1, use_refine=True,
        use_fpn_pan=True, refine_type='dwconv'
    ).to(device), [p2, p3, p4])

    print("\n=== 5. FPN_PAN + BiFPN (C3k2精炼) ===")
    test_model("FPN_PAN+C3k2", FPN_PAN_BiFPN(
        channels, out_channels, num_bifpn_layers=1, use_refine=True,
        use_fpn_pan=True, refine_type='c3k2'
    ).to(device), [p2, p3, p4])

    print("\n=== 6. 纯 BiFPN (C3k2精炼, 替代FPN-PAN) ===")
    test_model("Pure_C3k2", FPN_PAN_BiFPN(
        channels, out_channels, num_bifpn_layers=1, use_refine=True,
        use_fpn_pan=False, refine_type='c3k2'
    ).to(device), [p2, p3, p4])

    print("\n=== 7. 纯 BiFPN (2层, C3k2精炼) ===")
    test_model("Pure_C3k2x2", FPN_PAN_BiFPN(
        channels, out_channels, num_bifpn_layers=2, use_refine=True,
        use_fpn_pan=False, refine_type='c3k2'
    ).to(device), [p2, p3, p4])

    # 模拟实际 YOLO11s 场景：backbone [128,256,256] → out_channels [64,128,256]
    print("\n=== 8. 纯 BiFPN (通道不匹配场景, C3k2精炼) ===")
    p2_real = torch.randn(bs, 128, 160, 160).to(device)
    p3_real = torch.randn(bs, 256, 80, 80).to(device)
    p4_real = torch.randn(bs, 256, 40, 40).to(device)
    real_channels = [128, 256, 256]
    real_out = [64, 128, 256]
    model_real = FPN_PAN_BiFPN(
        real_channels, real_out, num_bifpn_layers=1, use_refine=True,
        use_fpn_pan=False, refine_type='c3k2'
    ).to(device)
    outs = model_real([p2_real, p3_real, p4_real])
    for i, (out, exp_ch) in enumerate(zip(outs, real_out), start=2):
        status = "OK" if out.shape[1] == exp_ch else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected channels={exp_ch} [{status}]")
    print(f"  Params: {sum(p.numel() for p in model_real.parameters()):,}")