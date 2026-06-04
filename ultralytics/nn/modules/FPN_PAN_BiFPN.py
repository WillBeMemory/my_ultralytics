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
    """深度可分离卷积（官方 BiFPN 精炼模块）"""
    def __init__(self, ch, kernel_size=3, act=True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size, stride=1,
                            padding=kernel_size // 2, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class LightweightFPNLayer(nn.Module):
    """轻量级 FPN（Top‑Down）层，内部用深度可分离卷积"""
    def __init__(self, ch, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        self.p4_proj = Conv(ch, ch, 1, act=False)
        self.p3_proj = Conv(ch, ch, 1, act=False)

        if use_refine:
            self.refine_p3 = DepthwiseSeparableConv(ch)
            self.refine_p2 = DepthwiseSeparableConv(ch)
        else:
            self.refine_p3 = nn.Identity()
            self.refine_p2 = nn.Identity()

    def forward(self, p2_in, p3_in, p4_in):
        # Top‑Down: P4 → P3 → P2
        p4_up = F.interpolate(self.p4_proj(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.refine_p3(p3_in + p4_up)

        p3_up = F.interpolate(self.p3_proj(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_td = self.refine_p2(p2_in + p3_up)

        return p2_td, p3_td, p4_in


class LightweightPANLayer(nn.Module):
    """轻量级 PAN（Bottom‑Up）层，内部用深度可分离卷积"""
    def __init__(self, ch, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        self.down_p2 = Conv(ch, ch, 3, 2)
        self.down_p3 = Conv(ch, ch, 3, 2)

        if use_refine:
            self.refine_p3 = DepthwiseSeparableConv(ch)
            self.refine_p4 = DepthwiseSeparableConv(ch)
        else:
            self.refine_p3 = nn.Identity()
            self.refine_p4 = nn.Identity()

    def forward(self, p2_td, p3_td, p4_td):
        # Bottom‑Up: P2 → P3 → P4
        p2_down = self.down_p2(p2_td)
        p3_bu = self.refine_p3(p3_td + p2_down)

        p3_down = self.down_p3(p3_bu)
        p4_bu = self.refine_p4(p4_td + p3_down)

        return p2_td, p3_bu, p4_bu


class LightweightFPN_PAN(nn.Module):
    """
    轻量级 FPN + PAN 模块
    - FPN: Top‑Down 路径（P4 → P3 → P2）
    - PAN: Bottom‑Up 路径（P2 → P3 → P4）
    - 内部精炼模块使用深度可分离卷积
    """
    def __init__(self, channels, out_channels=None, use_refine=True, expand_ch=None):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.expand_ch = expand_ch if expand_ch is not None else o3

        # 投影到统一通道
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)

        # FPN + PAN
        self.fpn = LightweightFPNLayer(self.expand_ch, use_refine=use_refine)
        self.pan = LightweightPANLayer(self.expand_ch, use_refine=use_refine)

        # 输出压缩
        self.compress_p2 = Conv(self.expand_ch, o2, 1, act=False)
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)

    def forward(self, features):
        p2, p3, p4 = features

        # 1. 投影到统一通道
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)

        # 2. FPN (Top‑Down)
        p2_td, p3_td, p4_td = self.fpn(p2, p3, p4)

        # 3. PAN (Bottom‑Up)
        p2_out, p3_out, p4_out = self.pan(p2_td, p3_td, p4_td)

        # 4. 输出压缩
        p2_out = self.compress_p2(p2_out)
        p3_out = self.compress_p3(p3_out)
        p4_out = self.compress_p4(p4_out)

        return [p2_out, p3_out, p4_out]


class BiFPNLayer(nn.Module):
    """单层 BiFPN 双向融合（Top‑Down + Bottom‑Up），用于 P2/P3/P4 三层"""
    def __init__(self, ch, use_refine=True):
        super().__init__()
        self.use_refine = use_refine

        # Top‑Down 路径：P4 → P3 → P2
        self.td_p4 = Conv(ch, ch, 1, act=False)
        self.td_p3 = Conv(ch, ch, 1, act=False)
        self.fuse_td_p3 = BiFPN_Add(2)
        self.fuse_td_p2 = BiFPN_Add(2)

        # Bottom‑Up 路径：P2 → P3 → P4
        self.bu_p2 = Conv(ch, ch, 1, act=False)
        self.bu_p3 = Conv(ch, ch, 1, act=False)
        self.fuse_bu_p3 = BiFPN_Add(2)
        self.fuse_bu_p4 = BiFPN_Add(2)

        # 下采样（标准 stride=2 卷积）
        self.down_bu2 = Conv(ch, ch, 3, 2)
        self.down_bu3 = Conv(ch, ch, 3, 2)

        # 精炼模块（深度可分离卷积）
        if use_refine:
            self.refine_td_p3 = DepthwiseSeparableConv(ch)
            self.refine_td_p2 = DepthwiseSeparableConv(ch)
            self.refine_bu_p3 = DepthwiseSeparableConv(ch)
            self.refine_bu_p4 = DepthwiseSeparableConv(ch)
        else:
            self.refine_td_p3 = nn.Identity()
            self.refine_td_p2 = nn.Identity()
            self.refine_bu_p3 = nn.Identity()
            self.refine_bu_p4 = nn.Identity()

    def forward(self, p2_enh, p3_enh, p4_enh):
        # Top‑Down: P4 → P3 → P2
        p4_up = F.interpolate(self.td_p4(p4_enh), size=p3_enh.shape[-2:], mode='nearest')
        p3_td = self.refine_td_p3(self.fuse_td_p3([p3_enh, p4_up]))

        p3_up = F.interpolate(self.td_p3(p3_td), size=p2_enh.shape[-2:], mode='nearest')
        p2_td = self.refine_td_p2(self.fuse_td_p2([p2_enh, p3_up]))

        # Bottom‑Up: P2 → P3 → P4
        p2_down = self.down_bu2(p2_td)
        p3_bu = self.refine_bu_p3(self.fuse_bu_p3([p3_td, p2_down]))

        p3_down = self.down_bu3(p3_bu)
        p4_bu = self.refine_bu_p4(self.fuse_bu_p4([p4_enh, p3_down]))

        return p2_td, p3_bu, p4_bu


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN‑PAN‑BiFPN 组合模块：
    - FPN (Top‑Down) + PAN (Bottom‑Up) → 输出 P2/P3/P4 特征
    - BiFPN（多层双向加权融合）→ 再精炼一次
    - 所有内部精炼模块都使用深度可分离卷积
    """
    def __init__(self, channels, out_channels=None, 
                 num_bifpn_layers=1, use_refine=True, expand_ch=None):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.expand_ch = expand_ch if expand_ch is not None else o3
        self.num_bifpn_layers = num_bifpn_layers

        # 各层投影到统一通道
        self.p2_proj = Conv(c2, self.expand_ch, 1, act=False)
        self.p3_proj = Conv(c3, self.expand_ch, 1, act=False)
        self.p4_proj = Conv(c4, self.expand_ch, 1, act=False)

        # FPN (Top‑Down)
        self.fpn_p4_proj = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.fpn_p3_proj = Conv(self.expand_ch, self.expand_ch, 1, act=False)
        self.fpn_refine_p3 = DepthwiseSeparableConv(self.expand_ch) if use_refine else nn.Identity()
        self.fpn_refine_p2 = DepthwiseSeparableConv(self.expand_ch) if use_refine else nn.Identity()

        # PAN (Bottom‑Up)
        self.pan_down_p2 = Conv(self.expand_ch, self.expand_ch, 3, 2)
        self.pan_down_p3 = Conv(self.expand_ch, self.expand_ch, 3, 2)
        self.pan_refine_p3 = DepthwiseSeparableConv(self.expand_ch) if use_refine else nn.Identity()
        self.pan_refine_p4 = DepthwiseSeparableConv(self.expand_ch) if use_refine else nn.Identity()

        # BiFPN 层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(self.expand_ch, use_refine=use_refine)
            for _ in range(num_bifpn_layers)
        ])

        # 输出压缩
        self.compress_p2 = Conv(self.expand_ch, o2, 1, act=False)
        self.compress_p3 = Conv(self.expand_ch, o3, 1, act=False)
        self.compress_p4 = Conv(self.expand_ch, o4, 1, act=False)

    def forward(self, features):
        p2, p3, p4 = features

        # 1. 投影到统一通道
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)

        # 2. FPN (Top‑Down)
        p4_up_fpn = F.interpolate(self.fpn_p4_proj(p4), size=p3.shape[-2:], mode='nearest')
        p3_td = self.fpn_refine_p3(p3 + p4_up_fpn)
        p3_up_fpn = F.interpolate(self.fpn_p3_proj(p3_td), size=p2.shape[-2:], mode='nearest')
        p2_td = self.fpn_refine_p2(p2 + p3_up_fpn)

        # 3. PAN (Bottom‑Up)
        p2_down_pan = self.pan_down_p2(p2_td)
        p3_bu = self.pan_refine_p3(p3_td + p2_down_pan)
        p3_down_pan = self.pan_down_p3(p3_bu)
        p4_bu = self.pan_refine_p4(p4 + p3_down_pan)

        # 4. BiFPN 多层融合
        p2_out, p3_out, p4_out = p2_td, p3_bu, p4_bu
        for layer in self.bifpn_layers:
            p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)

        # 5. 输出压缩
        p2_out = self.compress_p2(p2_out)
        p3_out = self.compress_p3(p3_out)
        p4_out = self.compress_p4(p4_out)

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

    print("\nTesting FPN_PAN_BiFPN:")
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
