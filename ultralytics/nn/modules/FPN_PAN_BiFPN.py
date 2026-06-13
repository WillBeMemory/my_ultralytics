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


class LargeKernelBottleneck(nn.Module):
    """Bottleneck with 7×7 DW conv for the second stage (m[1]).

    Architecture:
      x → cv1: Conv(c, c/2, 3) → SiLU → BN     (channel compression, same as standard)
        → dw:  DW_Conv(c/2, c/2, 7) → BN → SiLU (large kernel spatial mixing, NEW)
        → cv2: Conv(c/2, c, 3) → BN              (channel expansion, no SiLU)
        → + x

    The 7×7 DW conv expands receptive field without changing channel count.
    Only adds c/2 × 49 params per bottleneck (vs c/2 × c × 49 for standard 7×7).

    RNG: cv1/cv2 manually initialized from dummy Bottleneck(e=0.5).
         DW conv created in RNG bubble, keeps its own random init.
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, 3)
        self.dw = nn.Sequential(
            nn.Conv2d(c_, c_, 7, 1, 3, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = Conv(c_, c, 3, act=False)
        self.add = shortcut

    def _init_from_bottleneck(self, dummy):
        """Copy cv1/cv2 weights from a standard Bottleneck(e=0.5).

        The DW conv keeps its own random initialization.
        """
        # cv1: same shape, direct copy
        self.cv1.conv.weight.data.copy_(dummy.cv1.conv.weight.data)
        for name in ('weight', 'bias', 'running_mean', 'running_var'):
            getattr(self.cv1.bn, name).data.copy_(getattr(dummy.cv1.bn, name).data)
        # cv2: same shape, direct copy
        self.cv2.conv.weight.data.copy_(dummy.cv2.conv.weight.data)
        for name in ('weight', 'bias', 'running_mean', 'running_var'):
            getattr(self.cv2.bn, name).data.copy_(getattr(dummy.cv2.bn, name).data)

    def forward(self, x):
        out = self.dw(self.cv1(x))
        out = self.cv2(out)
        return x + out if self.add else out


class C2f_Simple(nn.Module):
    """
    C2f with mixed bottlenecks: m[0] = standard Bottleneck(e=0.5),
    m[1] = LargeKernelBottleneck (7×7 DW conv + cv2 no SiLU).

    RNG alignment: dummy Bottleneck(e=1.0) + Bottleneck + dummy Bottleneck(e=0.5)
    consume the same RNG as C3k2. LargeKernelBottleneck is created in an RNG bubble
    (save/restore) so subsequent modules' initialization is unaffected.
    """

    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # RNG alignment: C2f.__init__ creates n × Bottleneck(e=1.0) that C3k2 discards
        for _ in range(n):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        # m[0] = standard Bottleneck(e=0.5) — same RNG as C3k2's m[0]
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g),
        ])
        # m[1]: consume same RNG as Bottleneck(e=0.5) for alignment, then create large kernel
        dummy = Bottleneck(self.c, self.c, shortcut, g)  # consume RNG ≡ C3k2's m[1]
        # Save RNG state (now aligned with post-C3k2)
        rng_state = torch.random.get_rng_state()
        # Create large kernel module in RNG bubble
        large_kernel_m = LargeKernelBottleneck(self.c, shortcut)
        large_kernel_m._init_from_bottleneck(dummy)
        # Restore RNG state so subsequent modules see same state as C3k2
        torch.random.set_rng_state(rng_state)
        self.m.append(large_kernel_m)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FPN_PAN(nn.Module):
    """
    FPN-PAN with C2f_Simple (n=2: Bottleneck + LargeKernelBottleneck with 7×7 DW).
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
    - FPN-PAN：C2f_Simple(n=2, Bottleneck + LargeKernelBottleneck) 替代 C3k2
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
