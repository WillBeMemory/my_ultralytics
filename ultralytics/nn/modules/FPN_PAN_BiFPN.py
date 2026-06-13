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


class CrossDilatedBottleneck(nn.Module):
    """十字 + 空洞 DW Bottleneck：cv1 压缩后用 DW 十字卷积和 DW 空洞卷积做空间混合。

    十字卷积 = DW 3×1 (竖) + DW 1×3 (横) 并行求和，覆盖十字方向。
    空洞卷积 = DW 3×3 dilation=2，扩大感受野覆盖 5×5 区域。
    三路 DW 并行求和后 SiLU 激活，再 cv2 扩展通道。

    DW 参数极少：c/2 × (3+3+9) = 7.5c，远小于标准卷积 c/2 × c × 9。

    Architecture:
      x → cv1: Conv(c, c/2, 3) → SiLU → BN          (通道压缩，标准卷积跨通道混合)
        → [DW 3×1 + BN] + [DW 1×3 + BN] + [DW 3×3 d=2 + BN]  (十字 + 空洞空间混合)
        → SiLU
        → cv2: Conv(c/2, c, 3) → BN                   (通道扩展，无 SiLU)
        → + x
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c, c_, 3)
        # 十字方向：竖 + 横
        self.dw_v = nn.Conv2d(c_, c_, (3, 1), 1, (1, 0), groups=c_, bias=False)
        self.dw_v_bn = nn.BatchNorm2d(c_)
        self.dw_h = nn.Conv2d(c_, c_, (1, 3), 1, (0, 1), groups=c_, bias=False)
        self.dw_h_bn = nn.BatchNorm2d(c_)
        # 空洞方向：扩大感受野
        self.dw_d = nn.Conv2d(c_, c_, 3, 1, 2, dilation=2, groups=c_, bias=False)
        self.dw_d_bn = nn.BatchNorm2d(c_)
        # 通道扩展
        self.cv2 = Conv(c_, c, 3, act=False)
        self.add = shortcut

    def forward(self, x):
        out = self.cv1(x)
        # 十字 + 空洞并行求和
        out = self.dw_v_bn(self.dw_v(out)) + self.dw_h_bn(self.dw_h(out)) + self.dw_d_bn(self.dw_d(out))
        out = F.silu(out)
        out = self.cv2(out)
        return x + out if self.add else out


class RepConv(nn.Module):
    """重参数化卷积 (ACNet 风格)：训练时 3×3 + 1×3 + 3×1 三分支并行，推理时融合为单个 3×3。

    训练时：
      x → 3×3 Conv + BN ──┐
        → 1×3 Conv + BN ──┼──→ 求和 → SiLU → out
        → 3×1 Conv + BN ──┘

    推理时（调用 merge() 后）：
      x → 3×3 Conv + BN → SiLU → out   ← 与标准 Conv 完全等价

    融合原理：
      1. 每个分支的 BN 吸收进 Conv2d（得到带 bias 的等价卷积）
      2. 1×3 和 3×1 核 zero-pad 到 3×3
      3. 三个 3×3 核相加 = 融合后的单核
    """

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.c1, self.c2, self.k, self.s = c1, c2, k, s
        # 主分支：3×3
        self.conv3x3 = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn3x3 = nn.BatchNorm2d(c2)
        # 水平分支：1×k
        self.conv1xk = nn.Conv2d(c1, c2, (1, k), s, (0, k // 2), bias=False)
        self.bn1xk = nn.BatchNorm2d(c2)
        # 垂直分支：k×1
        self.convkx1 = nn.Conv2d(c1, c2, (k, 1), s, (k // 2, 0), bias=False)
        self.bnkx1 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        self._merged = False

    def forward(self, x):
        if self._merged:
            return self.act(self.bn3x3(self.conv3x3(x)))
        return self.act(
            self.bn3x3(self.conv3x3(x)) +
            self.bn1xk(self.conv1xk(x)) +
            self.bnkx1(self.convkx1(x))
        )

    def merge(self):
        """将三分支融合为单个 3×3 卷积，推理时零额外开销。"""
        if self._merged:
            return
        # 1. 每个分支：Conv2d + BN → 等价的 Conv2d(with bias)
        w3, b3 = self._fuse_bn(self.conv3x3, self.bn3x3)
        w1k, b1k = self._fuse_bn(self.conv1xk, self.bn1xk)
        wk1, bk1 = self._fuse_bn(self.convkx1, self.bnkx1)

        # 2. Pad 1×k 和 k×1 到 k×k
        pad = self.k // 2
        w1k = F.pad(w1k, (0, 0, pad, pad))   # 上下补零
        wk1 = F.pad(wk1, (pad, pad, 0, 0))   # 左右补零

        # 3. 三核相加
        w = w3 + w1k + wk1
        b = b3 + b1k + bk1

        # 4. 写回主分支，BN 设为直通（output = input + bias）
        self.conv3x3.weight.data.copy_(w)
        self.bn3x3.weight.data.fill_(1.0)
        self.bn3x3.bias.data.copy_(b)
        self.bn3x3.running_mean.zero_()
        self.bn3x3.running_var.fill_(1.0)
        self.bn3x3.num_batches_tracked.zero_()

        # 5. 删除额外分支
        del self.conv1xk, self.bn1xk, self.convkx1, self.bnkx1
        self._merged = True

    @staticmethod
    def _fuse_bn(conv, bn):
        """将 Conv2d + BatchNorm2d 融合为等价的 Conv2d(with bias)。

        BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
              = (gamma / sqrt(var + eps)) * x + (beta - gamma * mean / sqrt(var + eps))
        """
        w = conv.weight.data.clone()
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = (var + eps).sqrt()
        scale = gamma / std
        fused_w = w * scale.reshape(-1, 1, 1, 1)
        fused_b = beta - mean * scale
        return fused_w, fused_b


class RepBottleneck(nn.Module):
    """重参数化 Bottleneck：cv1/cv2 使用 RepConv (ACNet 风格)。

    训练时：cv1/cv2 各有 3×3 + 1×3 + 3×1 三分支，增强水平/垂直方向特征提取。
    调用 merge() 后：等价于标准 Bottleneck(e=0.5)，推理零额外开销。

    Architecture:
      x → cv1: RepConv(c, c/2, 3) → SiLU     (三分支：3×3 + 1×3 + 3×1)
        → cv2: RepConv(c/2, c, 3, act=False)   (三分支：3×3 + 1×3 + 3×1)
        → + x
    """

    def __init__(self, c, shortcut=True):
        super().__init__()
        c_ = c // 2
        self.cv1 = RepConv(c, c_, 3)
        self.cv2 = RepConv(c_, c, 3, act=False)
        self.add = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def merge(self):
        """融合所有 RepConv 为标准 3×3 卷积。"""
        self.cv1.merge()
        self.cv2.merge()


class C2f_Simple(nn.Module):
    """
    C2f with n=2: m[0] = 标准 Bottleneck, m[1] = RepBottleneck (ACNet 重参数化)。

    训练时 m[1] 使用三分支卷积增强特征提取，调用 merge() 后等价于标准 Bottleneck。
    RNG alignment: 先消耗与 C3k2(n=2) 相同的 RNG，再在气泡中创建 m[1]。
    """

    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # RNG alignment: C2f.__init__ 创建 n × Bottleneck(e=1.0) 被 C3k2 丢弃
        for _ in range(n):
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        # m[0] = 标准 Bottleneck(e=0.5)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g),
        ])
        # m[1]: 消耗与 C3k2 m[1] 相同的 RNG，然后在气泡中创建 RepBottleneck
        dummy = Bottleneck(self.c, self.c, shortcut, g)
        rng_state = torch.random.get_rng_state()
        self.m.append(RepBottleneck(self.c, shortcut))
        torch.random.set_rng_state(rng_state)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def merge(self):
        """融合所有 RepBottleneck，训练后调用一次即可。"""
        for m in self.m:
            if hasattr(m, 'merge'):
                m.merge()


class FPN_PAN(nn.Module):
    """
    FPN-PAN with C2f_Simple (n=2: Bottleneck + RepBottleneck).
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
    - FPN-PAN：C2f_Simple(n=2, Bottleneck + RepBottleneck) 替代 C3k2
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
