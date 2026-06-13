import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


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


class StarResBlock(nn.Module):
    """StarNet-inspired block (CVPR 2024). Element-wise multiplication for implicit high-dim features.

    Architecture:
      x → chunk(c/2, c/2) → cv_a(3×3) ↘
                            cv_b(3×3) → multiply → cv_out(3×3) → + x

    Star operation (element-wise multiplication of two feature branches) generates an
    implicit high-dimensional non-linear feature space without widening the network.
    Each branch transforms half the channels independently, then their product creates
    complex inter-channel interactions.

    RNG equivalence:
      Branch A: Conv(c/2, c/2, 3) → weight = (c/2)²×9 = c²/4 × 9  RNG values
      Branch B: Conv(c/2, c/2, 3) → weight = c²/4 × 9  RNG values
      Project:  Conv(c/2, c, 3)   → weight = c²/2 × 9  RNG values
      Total: c² × 9  ≡  Bottleneck(e=0.5)'s c×(c/2)×9 + (c/2)×c×9

    Params: c²×9 + 4c  (vs Bottleneck(e=0.5)'s c²×9 + 3c, diff = +c, <0.1%)
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_half = c // 2
        self.cv_a = Conv(c_half, c_half, 3)    # transform branch A
        self.cv_b = Conv(c_half, c_half, 3)    # transform branch B
        self.cv_out = Conv(c_half, c, 3)       # project back to c
        self.add = shortcut

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        out = self.cv_out(self.cv_a(a) * self.cv_b(b))
        return x + out if self.add else out


class C2f_Simple(nn.Module):
    """
    C2f with StarResBlock — StarNet (CVPR 2024) style feature refinement.
    RNG consumption equivalent to C3k2(n=2, c3k=False), ensuring identical
    initialization for all subsequent modules in the model.

    C3k2 execution trace (n=2, c3k=False):
      1. C2f.__init__ creates: cv1, cv2, n × Bottleneck(e=1.0)  ─── RNG
      2. C3k2.__init__ overwrites: n × Bottleneck(e=0.5)          ─── RNG

    C2f_Simple execution:
      1. cv1, cv2 (same)
      2. n × dummy Bottleneck(e=1.0) — discarded, RNG alignment
      3. n × StarResBlock — actual computation, same total RNG as step 2 above
    """

    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # RNG alignment: C2f.__init__ creates n × Bottleneck(e=1.0) that C3k2 discards
        for _ in range(n):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m = nn.ModuleList(
            StarResBlock(self.c, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FPN_PAN(nn.Module):
    """
    FPN-PAN with C2f_Simple (n=2 StarResBlock, StarNet CVPR 2024).
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
