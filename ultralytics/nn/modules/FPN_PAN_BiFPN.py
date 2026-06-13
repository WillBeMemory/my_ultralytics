import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


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


class GhostModule(nn.Module):
    """GhostNet Ghost Module (CVPR 2020): primary conv + cheap DW conv to generate features.

    Instead of one Conv(c_in, c_out, k), use:
      - Primary conv: Conv(c_in, c_out//ratio, k) — full cross-channel mixing
      - Cheap DW conv: DW(c_out//ratio, c_out//ratio, dw_k) — spatial refinement
      - Concat primary + cheap → c_out channels

    Same output channels, fewer parameters, no channel compression.
    """

    def __init__(self, c_in, c_out, k=1, ratio=2, dw_k=3, stride=1, act=True):
        super().__init__()
        init_c = c_out // ratio
        new_c = init_c * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(c_in, init_c, k, stride, k // 2, bias=False),
            nn.BatchNorm2d(init_c),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_c, new_c, dw_k, 1, dw_k // 2, groups=init_c, bias=False),
            nn.BatchNorm2d(new_c),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBottleneck(nn.Module):
    """GhostNet-style Bottleneck — same params as Bottleneck(e=0.5) but no channel compression.

    - cv1: GhostModule(c, c, k=3, ratio=2) — 3×3 primary + DW, full c channels output
    - cv2: GhostModule(c, c, k=3, ratio=2, act=False) — same, no activation
    - No channel compression: c→c throughout (vs c→c/2→c in standard Bottleneck)
    - Primary conv provides full cross-channel mixing (no grouping)
    - Cheap DW conv adds spatial refinement for free
    - Params: ~c²×9 + c×9 ≈ Bottleneck(e=0.5)'s c²×9 (+0.32%)
    """

    def __init__(self, c, shortcut=True, g=1):
        super().__init__()
        self.cv1 = GhostModule(c, c, k=3, ratio=2, dw_k=3, act=True)
        self.cv2 = GhostModule(c, c, k=3, ratio=2, dw_k=3, act=False)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_Simple(nn.Module):
    """
    C2f with GhostBottleneck — same params as C3k2(n=2, c3k=False), no channel compression.
    Replaces Bottleneck(e=0.5) with GhostBottleneck:
    - Nearly identical parameter count (+0.32%)
    - No channel compression (c→c throughout vs c→c/2→c)
    - Full cross-channel mixing in primary conv (standard conv, no grouping)
    - DW conv adds implicit spatial refinement
    """

    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FPN_PAN(nn.Module):
    """
    FPN-PAN with C2f_Simple (n=2 GhostBottleneck, no PSA/C3k).
    - FPN (Top-Down):   Upsample → Concat → C2f_Simple(n=2)
    - PAN (Bottom-Up):  Conv(s=2) → Concat → C2f_Simple(n=2)
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
        self.fpn_p3 = C2f_Simple(p3.shape[1] + p4.shape[1], o3, n=2)
        self.fpn_p2 = C2f_Simple(p2.shape[1] + o3, o2, n=2)

        # ========== PAN (Bottom-Up) ==========
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C2f_Simple(o3 + o3, o3, n=2)
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C2f_Simple(o4 + p4.shape[1], o4, n=2)

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
    - DWConv 做融合后精炼
    """
    def __init__(self, channels=None, use_refine=True):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
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

        return p2_td, p3_out, p4_out


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN-BiFPN 组合模块
    - FPN-PAN：C2f_Simple(n=2) 替代 C3k2，去掉 PSA/C3k，2-branch CSP
    - BiFPN：P2/P3/P4 两两加权融合（BiFPN_Add），可堆叠多层
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1, use_refine=False):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_bifpn_layers = num_bifpn_layers

        # FPN-PAN
        self.fpn_pan = FPN_PAN(channels, out_channels)

        # BiFPN 层
        if num_bifpn_layers > 0:
            self.bifpn_layers = nn.ModuleList([
                BiFPNLayer(out_channels, use_refine) for _ in range(num_bifpn_layers)
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
