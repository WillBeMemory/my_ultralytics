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


class CW_BiFPN_Add(nn.Module):
    """逐通道加权融合（Channel-Wise BiFPN Add）
    每个通道独立学习融合权重，比标量 BiFPN_Add 表达力更强
    """
    def __init__(self, num_inputs=2, channels=64):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, channels, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)  # [num_inputs, C]
        w_norm = w / (w.sum(dim=0, keepdim=True) + self.eps)  # [num_inputs, C]
        return sum(w_norm[i].view(1, -1, 1, 1) * xs[i] for i in range(len(xs)))


class Pixel_BiFPN_Add(nn.Module):
    """逐像素加权融合（ASFF 思想，Liu et al. 2019）。
    每个空间位置 (x,y) 独立学一组对 num_inputs 个输入的权重——
    权重由「拼接后的特征」经轻量卷积预测并 softmax 归一化得到，内容自适应。

    对比：
      - BiFPN_Add : 全局标量权重            [num_inputs]
      - CW_BiFPN_Add: 逐通道权重            [num_inputs, C]
      - Pixel_BiFPN_Add: 逐像素权重         [B, num_inputs, H, W]   ← 表达力最强
    """
    def __init__(self, num_inputs=2, channels=64, compress=4):
        super().__init__()
        mid = max(channels // compress, 8)   # 压缩通道，降参防过拟合
        self.weight_predictor = nn.Sequential(
            Conv(channels * num_inputs, mid, 1, act=False),
            Conv(mid, mid, 3, act=True),
            nn.Conv2d(mid, num_inputs, 1),   # 输出 [B, num_inputs, H, W]
        )
        self.num_inputs = num_inputs

    def forward(self, xs):
        stacked = torch.cat(xs, dim=1)                 # [B, num_inputs*C, H, W]
        w = self.weight_predictor(stacked)             # [B, num_inputs, H, W]
        w = F.softmax(w, dim=1).unsqueeze(2)           # softmax → [B, num_inputs, 1, H, W]
        out = sum(w[:, i] * xs[i] for i in range(self.num_inputs))  # 广播加权求和
        return out


def _make_fuse(fuse_type, num_inputs=2, channels=64):
    """融合算子工厂：fuse_type ∈ {'scalar','channel','pixel'}。

    scalar : BiFPN_Add        — 全局标量权重（EfficientDet）
    channel: CW_BiFPN_Add     — 逐通道权重（默认，与历史版本一致）
    pixel  : Pixel_BiFPN_Add  — 逐像素权重（ASFF）
    """
    fuse_type = (fuse_type or 'channel').lower()
    if fuse_type == 'scalar':
        return BiFPN_Add(num_inputs)
    elif fuse_type == 'channel':
        return CW_BiFPN_Add(num_inputs, channels)
    elif fuse_type == 'pixel':
        return Pixel_BiFPN_Add(num_inputs, channels)
    else:
        raise ValueError(f"fuse_type must be 'scalar'|'channel'|'pixel', got '{fuse_type}'")


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
    标准 YOLO11 FPN-PAN，使用 Concat 融合（与 YOLO11 默认 neck 一致）：
    - FPN (Top-Down):   Upsample → Concat → C3k2
    - PAN (Bottom-Up):  Conv(s=2) → Concat → C3k2

    本类固定使用 Concat，不参与 fuse_type 切换；逐像素/逐通道加权融合
    只用于后续串联的 BiFPNLayer（见 FPN_PAN_BiFPN）。
    """
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self._out_channels = out_channels
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]
        o2, o3, o4 = self._out_channels

        # ========== FPN (Top-Down): Concat 融合 ==========
        # P4 → Upsample → Concat(P3) → C3k2     输入通道 = c3 + c4
        self.fpn_p3 = C3k2(c3 + c4, o3, n=2, c3k=False)
        # P3_fpn → Upsample → Concat(P2) → C3k2  输入通道 = c2 + o3
        self.fpn_p2 = C3k2(c2 + o3, o2, n=2, c3k=False)

        # ========== PAN (Bottom-Up): Concat 融合 ==========
        # P2_fpn → Conv(s=2) → Concat(P3_fpn) → C3k2  输入通道 = o3 + o3
        self.bu_p2_to_p3 = Conv(o2, o3, 3, 2)
        self.pan_p3 = C3k2(o3 * 2, o3, n=2, c3k=False)
        # P3_pan → Conv(s=2) → Concat(P4) → C3k2       输入通道 = c4 + o4
        self.bu_p3_to_p4 = Conv(o3, o4, 3, 2)
        self.pan_p4 = C3k2(c4 + o4, o4, n=2, c3k=False)

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
        p2_down = self.bu_p2_to_p3(p2_fpn)
        p3_pan = self.pan_p3(torch.cat([p3_fpn, p2_down], dim=1))

        p3_down = self.bu_p3_to_p4(p3_pan)
        p4_pan = self.pan_p4(torch.cat([p4, p3_down], dim=1))

        return [p2_fpn, p3_pan, p4_pan]


class BiFPNLayer(nn.Module):
    """
    BiFPN 单层，P2/P3/P4 两两加权融合（Top-Down + Bottom-Up）
    - 使用加权融合（CW_Add/像素加权）做融合（而非 Concat）
    - 1x1 Conv 做通道投影保证维度一致
    - DWConv 做融合后精炼

    fuse_type ∈ {'scalar','channel','pixel'} 控制融合权重的粒度。
    """
    def __init__(self, channels=None, use_refine=False, fuse_type='channel'):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self.fuse_type = fuse_type
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ========== Top-Down：P4→P3, P3→P2 ==========
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        self.td_p3_fuse = _make_fuse(self.fuse_type, 2, c3)
        self.td_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        self.td_p2_fuse = _make_fuse(self.fuse_type, 2, c2)
        self.td_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        self.bu_p3_fuse = _make_fuse(self.fuse_type, 2, c3)
        self.bu_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        self.bu_p4_fuse = _make_fuse(self.fuse_type, 2, c4)
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


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN-BiFPN 组合模块
    - FPN-PAN：与 YOLO11 默认结构完全一致（Conv + C3k2 + Upsample + Concat）。
    - BiFPN：在 FPN-PAN 之后串联若干层 BiFPNLayer 做精炼，可堆叠多层。

    fuse_type 只作用于 BiFPNLayer 部分（FPN-PAN 固定为 Concat）：
      - 'channel'（默认，逐通道权重）。
      - 'pixel' 启用 ASFF 风格的逐像素加权融合（Liu et al. 2019）。
      - 'scalar' 回退到 BiFPN 全局标量权重，用于消融下界对照。
    num_bifpn_layers=0 时退化为纯 FPN-PAN（不含加权融合）。
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1,
                 use_refine=False, fuse_type='channel'):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_bifpn_layers = num_bifpn_layers
        self.fuse_type = fuse_type

        # FPN-PAN（标准 YOLO11 结构，固定 Concat）
        self.fpn_pan = FPN_PAN(channels, out_channels)

        # BiFPN 层（fuse_type 仅在此生效）
        if num_bifpn_layers > 0:
            self.bifpn_layers = nn.ModuleList([
                BiFPNLayer(out_channels, use_refine, fuse_type=fuse_type)
                for _ in range(num_bifpn_layers)
            ])
        else:
            self.bifpn_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        # FPN-PAN
        p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])

        # BiFPN 精炼
        if self.bifpn_layers:
            for layer in self.bifpn_layers:
                p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)

        return [p2_out, p3_out, p4_out]


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160, device=device)
    p3 = torch.randn(bs, 256, 80, 80, device=device)
    p4 = torch.randn(bs, 512, 40, 40, device=device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]
    expected = [(bs, 128, 160, 160), (bs, 256, 80, 80), (bs, 512, 40, 40)]

    def check(name, outs, expected):
        ok = True
        for out, exp in zip(outs, expected):
            status = "OK" if out.shape == exp else "FAIL"
            ok &= out.shape == exp
            print(f"    {name}: {list(out.shape)} expected {list(exp)} [{status}]")
        return ok

    FUSE_CLASSES = (BiFPN_Add, CW_BiFPN_Add, Pixel_BiFPN_Add)

    def count(mod, cls):
        return sum(1 for m in mod.modules() if isinstance(m, cls))

    # ---- 1) FPN_PAN 单独：应为纯 Concat，不含任何加权融合算子 ----
    print("\n=== FPN_PAN 单独 (应为纯 Concat) ===")
    fpn_pan = FPN_PAN(channels, out_channels).to(device)
    outs = fpn_pan([p2, p3, p4])
    assert check("FPN_PAN", outs, expected)
    n_fuse = sum(count(fpn_pan, c) for c in FUSE_CLASSES)
    print(f"    加权融合算子总数 (应为 0): {n_fuse} | params: {sum(p.numel() for p in fpn_pan.parameters()):,}")
    assert n_fuse == 0, "FPN_PAN 不应包含任何加权融合算子"
    loss = sum(o.float().sum() for o in outs); loss.backward()

    # ---- 2) FPN_PAN_BiFPN: 加权融合只出现在 BiFPNLayer ----
    for n_layers, fuse, exp_cls in [(0, 'pixel', None),
                                    (1, 'scalar', BiFPN_Add),
                                    (1, 'channel', CW_BiFPN_Add),
                                    (1, 'pixel', Pixel_BiFPN_Add)]:
        model = FPN_PAN_BiFPN(channels, out_channels,
                              num_bifpn_layers=n_layers, use_refine=True,
                              fuse_type=fuse).to(device)
        outs = model([p2, p3, p4])
        ok = check(f"layers={n_layers},{fuse}", outs, expected)
        # FPN_PAN 部分始终 0 个加权算子
        n_in_fpnp = sum(count(model.fpn_pan, c) for c in FUSE_CLASSES)
        # BiFPNLayer 部分: layers=1 → 4 个对应算子; layers=0 → 0
        n_active = count(model, exp_cls) if exp_cls else 0
        loss = sum(o.float().sum() for o in outs); loss.backward()
        print(f"    layers={n_layers} {fuse:7s} | FPN_PAN 内加权算子={n_in_fpnp} "
              f"| {exp_cls.__name__ if exp_cls else '-':16s}={n_active} "
              f"| params={sum(p.numel() for p in model.parameters()):,}")
        assert ok and n_in_fpnp == 0, "FPN_PAN 部分必须为纯 Concat"
        assert n_active == (4 if n_layers == 1 else 0), f"BiFPN 融合点计数错误: {n_active}"

    # ---- 3) Pixel_BiFPN_Add 单算子权重检查 (B=3 防 batch/inputs 维混淆) ----
    print("\n=== Pixel_BiFPN_Add 单算子权重检查 (B=3) ===")
    add = Pixel_BiFPN_Add(2, 64).to(device)
    a = torch.randn(3, 64, 16, 16, device=device)
    b = torch.randn(3, 64, 16, 16, device=device)
    y = add([a, b])
    w = F.softmax(add.weight_predictor(torch.cat([a, b], dim=1)), dim=1)
    s = w.sum(dim=1)
    assert w.shape == (3, 2, 16, 16), f"weight shape {w.shape}"
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), "像素权重未在 inputs 维归一化为 1"
    assert w.std() > 1e-3, f"像素权重退化为常量, std={w.std().item():.2e}"
    print(f"    out {tuple(y.shape)} | per-pixel weights {tuple(w.shape)} | "
          f"sum≈1: {s.mean().item():.4f} | std={w.std().item():.4f}")
    print("\nAll checks passed.")

