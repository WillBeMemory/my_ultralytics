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


class AsymBottleneck(nn.Module):
    """非对称卷积 Bottleneck：3×1 → 1×3 顺序分解 (ACNet 思路)。

    将标准 3×3 分解为 3×1 + 1×3 两个连续卷积，参数量仅 2/3 但感受野等价。
    3×1 捕获垂直方向特征，1×3 捕获水平方向特征，顺序组合覆盖十字方向。

    Architecture:
      x → cv1: Conv(c, c/2, (3,1)) → SiLU → BN   (垂直方向)
        → cv_mid: Conv(c/2, c/2, (1,3)) → SiLU → BN  (水平方向)
        → cv2: Conv(c/2, c, 3) → BN                  (通道扩展，无 SiLU)
        → + x
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, (3, 1))
        self.cv_mid = Conv(c_, c_, (1, 3))
        self.cv2 = Conv(c_, c, 3, act=False)
        self.add = shortcut

    def forward(self, x):
        out = self.cv_mid(self.cv1(x))
        out = self.cv2(out)
        return x + out if self.add else out


class DilatedBottleneck(nn.Module):
    """空洞卷积 Bottleneck：3×3 dilation=2 扩大感受野。

    在 c/2 窄通道中插入 DW 空洞卷积，捕获更广域的空间上下文，
    同时保持标准卷积的跨通道混合能力。

    Architecture:
      x → cv1: Conv(c, c/2, 3) → SiLU → BN           (通道压缩)
        → dw:  DW_Conv(c/2, c/2, 3, dilation=2) → BN → SiLU  (空洞空间混合)
        → cv2: Conv(c/2, c, 3) → BN                    (通道扩展，无 SiLU)
        → + x
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, 3)
        self.dw = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = Conv(c_, c, 3, act=False)
        self.add = shortcut

    def forward(self, x):
        out = self.dw(self.cv1(x))
        out = self.cv2(out)
        return x + out if self.add else out


class C2f_Simple(nn.Module):
    """
    C2f with multi-oriented bottlenecks (n=4):
      m[0]: 标准 3×3 Bottleneck       — 基线，全方向特征提取
      m[1]: AsymBottleneck (3×1→1×3)  — 垂直+水平十字方向
      m[2]: AsymBottleneck (1×3→3×1)  — 水平+垂直十字方向（反向）
      m[3]: DilatedBottleneck (d=2)   — 空洞卷积，扩大感受野

    RNG alignment: 先消耗与 C3k2(n=2) 相同的 RNG（dummy + 2×Bottleneck + dummy cv2），
    再在 RNG 气泡中创建所有非标准模块，保证后续模块初始化不受影响。
    """

    def __init__(self, c1, c2, n=4, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # === RNG alignment: 消耗与 C3k2(n=2, c3k=False) 完全相同的随机数 ===
        # C2f.__init__ 创建 2×Bottleneck(e=1.0) 被 C3k2 丢弃
        for _ in range(2):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        # C3k2 重新创建 2×Bottleneck(e=0.5)
        for _ in range(2):
            Bottleneck(self.c, self.c, shortcut, g)
        # C3k2 的 cv2: Conv((2+2)*c, c2, 1)
        Conv((2 + 2) * self.c, c2, 1)

        # 此时 RNG 状态与 C3k2(n=2) 构造完成后完全一致
        rng_state = torch.random.get_rng_state()

        # === RNG 气泡：在此创建所有实际模块，不影响后续 RNG ===
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g),       # m[0]: 标准 3×3
            AsymBottleneck(self.c, shortcut),               # m[1]: 3×1 → 1×3
            AsymBottleneck(self.c, shortcut),               # m[2]: 1×3 → 3×1 (同结构，训练中分化)
            DilatedBottleneck(self.c, shortcut),            # m[3]: 空洞卷积
        ])

        # 恢复 RNG 状态，后续模块看到的随机数与 C3k2 完全相同
        torch.random.set_rng_state(rng_state)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FPN_PAN(nn.Module):
    """
    FPN-PAN with C2f_Simple (n=4: 标准3×3 + 3×1→1×3 + 1×3→3×1 + 空洞卷积).
    - FPN (Top-Down):   Upsample → Concat → C2f_Simple(n=4)
    - PAN (Bottom-Up):  Conv(s=2) → Concat → C2f_Simple(n=4)
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
        self.fpn_p3 = C2f_Simple(p3.shape[1] + p4.shape[1], o3, n=4)
        self.fpn_p2 = C2f_Simple(p2.shape[1] + o3, o2, n=4)

        # ========== PAN (Bottom-Up) ==========
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C2f_Simple(o3 + o3, o3, n=4)
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C2f_Simple(o4 + p4.shape[1], o4, n=4)

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
    - FPN-PAN：C2f_Simple(n=4, 多方向 Bottleneck) 替代 C3k2
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
