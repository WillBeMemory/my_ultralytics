import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class BiFPN_Add(nn.Module):
    """快速归一化加权融合（官方 BiFPN 核心，ReLU 归一化）"""
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 — 仅 DW 3×3，通道混合由前置 mid_conv(1×1) 完成，去掉冗余 PW"""
    def __init__(self, ch, kernel_size=3, act=True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size, stride=1,
                            padding=kernel_size // 2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class DWConvDownsample(nn.Module):
    """深度可分离下采样：DW 3×3 stride=2 + PW 1×1，替代标准 Conv(s=2) 节省 ~88% FLOPs"""
    def __init__(self, c1, c2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, 2, 1, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn_pw(self.pw(self.act(self.bn_dw(self.dw(x))))))


class LightweightFPN_PAN(nn.Module):
    """
    轻量级 FPN-PAN 模块 - 严格按照 YOLO11 风格
    - Upsample (先上采样，通道不变)
    - Concat (和 backbone 拼接)
    - Conv (融合 + 降通道，用深度可分离卷积代替 C3k2)
    """
    def __init__(self, channels, out_channels=None, use_refine=True):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        o2, o3, o4 = out_channels
        
        self._out_channels = out_channels
        self._use_refine = use_refine
        
        # 动态初始化
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        o2, o3, o4 = self._out_channels
        
        # ========== FPN (Top-Down) ==========
        # P4 → P3: Upsample -> Concat (p3 + p4_up) -> Conv
        self.fpn_p3_mid_conv = Conv(p3.shape[1] + p4.shape[1], o3, 1, act=False)
        self.fpn_p3_refine = DepthwiseSeparableConv(o3) if self._use_refine else nn.Identity()
        
        # P3 → P2: Upsample -> Concat (p2 + p3_up) -> Conv
        self.fpn_p2_mid_conv = Conv(p2.shape[1] + o3, o2, 1, act=False)
        self.fpn_p2_refine = DepthwiseSeparableConv(o2) if self._use_refine else nn.Identity()
        
        # ========== PAN (Bottom-Up) ==========
        # P2 → P3: DWConvDownsample(stride=2) -> Concat (p3_fpn + p2_down) -> Conv
        self.pan_p2_down_conv = DWConvDownsample(o2, o3)
        self.pan_p3_mid_conv = Conv(o3 + o3, o3, 1, act=False)
        self.pan_p3_refine = DepthwiseSeparableConv(o3) if self._use_refine else nn.Identity()
        
        # P3 → P4: DWConvDownsample(stride=2) -> Concat (p4 + p3_down) -> Conv
        self.pan_p3_down_conv = DWConvDownsample(o3, o4)
        self.pan_p4_mid_conv = Conv(o4 + p4.shape[1], o4, 1, act=False)
        self.pan_p4_refine = DepthwiseSeparableConv(o4) if self._use_refine else nn.Identity()
        
        self.to(p2.device)
        self._initialized = True

    def forward(self, features):
        p2, p3, p4 = features
        
        if not self._initialized:
            self._init_layers(p2, p3, p4)
        
        # ========== FPN (Top-Down) ==========
        # P4 → P3
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_concat = torch.cat([p3, p4_up], dim=1)
        p3_mid = self.fpn_p3_mid_conv(p3_concat)
        p3_fpn = self.fpn_p3_refine(p3_mid)
        
        # P3 → P2
        p3_up = F.interpolate(p3_fpn, size=p2.shape[-2:], mode='nearest')
        p2_concat = torch.cat([p2, p3_up], dim=1)
        p2_mid = self.fpn_p2_mid_conv(p2_concat)
        p2_fpn = self.fpn_p2_refine(p2_mid)
        
        # ========== PAN (Bottom-Up) ==========
        # P2 → P3
        p2_down = self.pan_p2_down_conv(p2_fpn)
        p3_concat_pan = torch.cat([p3_fpn, p2_down], dim=1)
        p3_mid_pan = self.pan_p3_mid_conv(p3_concat_pan)
        p3_pan = self.pan_p3_refine(p3_mid_pan)
        
        # P3 → P4
        p3_down = self.pan_p3_down_conv(p3_pan)
        p4_concat_pan = torch.cat([p4, p3_down], dim=1)
        p4_mid_pan = self.pan_p4_mid_conv(p4_concat_pan)
        p4_pan = self.pan_p4_refine(p4_mid_pan)
        
        return [p2_fpn, p3_pan, p4_pan]


class BiFPNLayer(nn.Module):
    """单层 BiFPN 双向融合（Top-Down + Bottom-Up），用于 P2/P3/P4 三层"""
    def __init__(self, channels=None, use_refine=True):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # Top-Down 路径：P4 → P3 → P2
        self.td_p4 = Conv(c4, c3, 1, act=False)  # 1x1 降通道
        # 不用 BiFPN_Add 了，直接用 Concat + Conv，这样不需要通道完全一致
        self.td_p3_mid = Conv(c3 + c3, c3, 1, act=False)
        self.td_p3 = Conv(c3, c2, 1, act=False)  # 1x1 降通道
        self.td_p2_mid = Conv(c2 + c2, c2, 1, act=False)

        # Bottom-Up 路径：P2 → P3 → P4
        self.down_bu2 = DWConvDownsample(c2, c3)
        self.bu_p3_mid = Conv(c3 + c3, c3, 1, act=False)
        self.down_bu3 = DWConvDownsample(c3, c4)
        self.bu_p4_mid = Conv(c4 + c4, c4, 1, act=False)

        # 精炼模块（深度可分离卷积）
        if self.use_refine:
            self.refine_td_p3 = DepthwiseSeparableConv(c3)
            self.refine_td_p2 = DepthwiseSeparableConv(c2)
            self.refine_bu_p3 = DepthwiseSeparableConv(c3)
            self.refine_bu_p4 = DepthwiseSeparableConv(c4)
        else:
            self.refine_td_p3 = nn.Identity()
            self.refine_td_p2 = nn.Identity()
            self.refine_bu_p3 = nn.Identity()
            self.refine_bu_p4 = nn.Identity()

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)
        
        # Top-Down: P4 → P3 → P2
        p4_up = F.interpolate(self.td_p4(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_concat = torch.cat([p3_in, p4_up], dim=1)
        p3_mid = self.td_p3_mid(p3_concat)
        p3_td = self.refine_td_p3(p3_mid)
        
        p3_up = F.interpolate(self.td_p3(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_concat = torch.cat([p2_in, p3_up], dim=1)
        p2_mid = self.td_p2_mid(p2_concat)
        p2_td = self.refine_td_p2(p2_mid)

        # Bottom-Up: P2 → P3 → P4
        p2_down = self.down_bu2(p2_td)
        p3_concat_bu = torch.cat([p3_td, p2_down], dim=1)
        p3_mid_bu = self.bu_p3_mid(p3_concat_bu)
        p3_bu = self.refine_bu_p3(p3_mid_bu)
        
        p3_down = self.down_bu3(p3_bu)
        p4_concat_bu = torch.cat([p4_in, p3_down], dim=1)
        p4_mid_bu = self.bu_p4_mid(p4_concat_bu)
        p4_bu = self.refine_bu_p4(p4_mid_bu)

        return p2_td, p3_bu, p4_bu


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN-BiFPN 组合模块 - 保持 YOLO 默认风格
    - 各阶段保持自己的通道数
    - 先做 FPN-PAN，再做可选的 BiFPN 精炼
    - 所有内部精炼模块都使用深度可分离卷积
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1, use_refine=True):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels
        
        self.num_bifpn_layers = num_bifpn_layers
        
        # FPN-PAN
        self.fpn_pan = LightweightFPN_PAN(channels, out_channels, use_refine)
        
        # BiFPN 层（可选，仅当 num_bifpn_layers > 0 时使用）
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
        
        # BiFPN 精炼（可选）
        if self.bifpn_layers:
            for layer in self.bifpn_layers:
                p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)
        
        return [p2_out, p3_out, p4_out]


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]
    
    print("\nTesting LightweightFPN_PAN:")
    fpn_pan = LightweightFPN_PAN(channels, out_channels, use_refine=True).to(device)
    outs = fpn_pan([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        print(f"P{i}: {out.shape} expected {exp}")
    total_params = sum(p.numel() for p in fpn_pan.parameters())
    print(f"Params: {total_params:,}")

    print("\nTesting FPN_PAN_BiFPN (with 1 BiFPN layer):")
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
        print(f"P{i}: {out.shape} expected {exp}")
    total_params = sum(p.numel() for p in fpn_pan_bifpn.parameters())
    print(f"Params: {total_params:,}")
