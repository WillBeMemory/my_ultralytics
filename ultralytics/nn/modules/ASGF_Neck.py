import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class SpatialAttention(nn.Module):
    """
    空间注意力：生成空间自适应权重
    借鉴 ASFF 思想：每个位置学习独立的融合权重
    """
    def __init__(self, num_inputs, channels):
        super().__init__()
        self.num_inputs = num_inputs
        # 1x1 卷积生成空间权重图
        self.weight_conv = nn.Conv2d(channels * num_inputs, num_inputs, 1, bias=False)

    def forward(self, features):
        """
        features: [feat1, feat2, ...] 已对齐到相同尺寸
        返回：加权融合后的特征
        """
        # 拼接所有特征
        concat = torch.cat(features, dim=1)  # (B, C*num_inputs, H, W)
        # 生成空间权重图
        weights = self.weight_conv(concat)   # (B, num_inputs, H, W)
        weights = F.softmax(weights, dim=1)  # 每个位置权重和为1

        # 加权融合
        fused = sum(w.unsqueeze(1) * f for w, f in zip(
            torch.unbind(weights, dim=1), features))
        return fused


class GlobalContext(nn.Module):
    """
    全局上下文分支
    借鉴 Gold-YOLO：AdaptiveAvgPool2d(1) 获取全局信息
    不使用 Norm（因为输出 spatial size=1，Norm 会报错）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, out_channels, 1, 1) 全局特征
        """
        global_feat = self.pool(x)  # (B, C, 1, 1)
        global_feat = self.conv(global_feat)  # (B, out_channels, 1, 1)
        global_feat = self.act(global_feat)
        return global_feat


class InjectionAdd(nn.Module):
    """
    信息注入模块（来自 Gold-YOLO）
    将全局特征注入到局部特征
    支持不同通道数的局部和全局特征
    """
    def __init__(self, local_channels, global_channels):
        super().__init__()
        self.local_conv = Conv(local_channels, local_channels, 1, act=False)
        # 全局特征通道数可能不同，需要投影到局部特征通道数
        self.global_proj = Conv(global_channels, local_channels, 1, act=False)

    def forward(self, local_feat, global_feat):
        """
        local_feat: (B, C_local, H, W)
        global_feat: (B, C_global, 1, 1) 或 (B, C_global, H, W)
        """
        if global_feat.shape[2:] != local_feat.shape[2:]:
            global_feat = F.interpolate(global_feat, size=local_feat.shape[2:],
                                        mode='nearest')
        return self.local_conv(local_feat) + self.global_proj(global_feat)


class ASGFLayer(nn.Module):
    """
    Adaptive Spatial-Global Fusion Layer
    结合 ASFF 空间自适应权重 + Gold-YOLO 全局上下文

    核心设计：
    1. 局部融合：空间自适应权重（每个位置独立决定融合比例）
    2. 全局分支：AdaptiveAvgPool2d(1) 获取全局上下文
    3. 信息注入：全局特征通过 Add 注入局部特征
    """
    def __init__(self, channels, use_global=True):
        super().__init__()
        c2, c3, c4 = channels
        self.use_global = use_global
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ========== Top-Down：P4→P3, P3→P2 ==========
        # P4 → 1x1 Conv → Upsample
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        # P3 空间融合（P3_original + P4_up）
        self.td_p3_fuse = SpatialAttention(2, c3)

        # P3 → 1x1 Conv → Upsample
        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        # P2 空间融合（P2_original + P3_up）
        self.td_p2_fuse = SpatialAttention(2, c2)

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        # P2 → Conv(s=2)
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        # P3 空间融合（P3_td + P2_down）
        self.bu_p3_fuse = SpatialAttention(2, c3)

        # P3 → Conv(s=2)
        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        # P4 空间融合（P4_original + P3_down）
        self.bu_p4_fuse = SpatialAttention(2, c4)

        # ========== 全局上下文分支 ==========
        if self.use_global:
            # 收集所有尺度的全局信息
            self.global_p2 = GlobalContext(c2, c2 // 4)
            self.global_p3 = GlobalContext(c3, c3 // 4)
            self.global_p4 = GlobalContext(c4, c4 // 4)

            # 全局融合（不使用 BatchNorm，因为 spatial size=1）
            total_global_ch = (c2 + c3 + c4) // 4
            self.global_fuse = nn.Sequential(
                nn.Conv2d(total_global_ch, total_global_ch // 2, 1, bias=False),
                nn.SiLU(inplace=True)
            )

            # 注入到各层（局部通道数，全局通道数）
            global_ch = total_global_ch // 2
            self.inject_p2 = InjectionAdd(c2, global_ch)
            self.inject_p3 = InjectionAdd(c3, global_ch)
            self.inject_p4 = InjectionAdd(c4, global_ch)

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)

        # ========== 全局上下文（极低计算量）==========
        if self.use_global:
            g2 = self.global_p2(p2_in)  # (B, c2//4, 1, 1)
            g3 = self.global_p3(p3_in)  # (B, c3//4, 1, 1)
            g4 = self.global_p4(p4_in)  # (B, c4//4, 1, 1)

            # 统一尺寸后拼接
            g2_up = g2
            g3_up = F.interpolate(g3, size=(1, 1), mode='nearest')
            g4_up = F.interpolate(g4, size=(1, 1), mode='nearest')
            global_cat = torch.cat([g2_up, g3_up, g4_up], dim=1)
            global_fused = self.global_fuse(global_cat)  # (B, total//2, 1, 1)

        # ========== Top-Down ==========
        # P4 → P3
        p4_up = F.interpolate(self.td_p4_to_p3(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.td_p3_fuse([p3_in, p4_up])

        # P3 → P2
        p3_up = F.interpolate(self.td_p3_to_p2(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_td = self.td_p2_fuse([p2_in, p3_up])

        # ========== Bottom-Up ==========
        # P2 → P3
        p2_down = self.bu_p2_to_p3(p2_td)
        p3_out = self.bu_p3_fuse([p3_td, p2_down])

        # P3 → P4
        p3_down = self.bu_p3_to_p4(p3_out)
        p4_out = self.bu_p4_fuse([p4_in, p3_down])

        # ========== 全局信息注入 ==========
        if self.use_global:
            p2_out = self.inject_p2(p2_td, global_fused)
            p3_out = self.inject_p3(p3_out, global_fused)
            p4_out = self.inject_p4(p4_out, global_fused)
        else:
            p2_out = p2_td

        return p2_out, p3_out, p4_out


class ASGFNeck(nn.Module):
    """
    Adaptive Spatial-Global Fusion Neck
    完整 Neck 模块，包含 FPN-PAN + ASGF 层

    参数：
        channels: 输入通道 [c2, c3, c4]
        out_channels: 输出通道 [o2, o3, o4]
        num_asgf_layers: ASGF 层数
        use_global: 是否使用全局上下文分支
    """
    def __init__(self, channels, out_channels=None, num_asgf_layers=1, use_global=True):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = channels
        o2, o3, o4 = out_channels

        # FPN-PAN 基础结构（使用 C3k2）
        self.fpn_p3 = nn.Sequential(
            Conv(c3 + c4, o3, 1),
            Conv(o3, o3, 3)
        )
        self.fpn_p2 = nn.Sequential(
            Conv(c2 + o3, o2, 1),
            Conv(o2, o2, 3)
        )

        self.pan_p3 = nn.Sequential(
            Conv(o3 + o3, o3, 1),
            Conv(o3, o3, 3)
        )
        self.pan_p4 = nn.Sequential(
            Conv(c4 + o4, o4, 1),
            Conv(o4, o4, 3)
        )

        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3_down = Conv(o3, o4, 3, 2)

        # ASGF 精炼层
        self.asgf_layers = nn.ModuleList([
            ASGFLayer(out_channels, use_global) for _ in range(num_asgf_layers)
        ])

    def forward(self, features):
        p2, p3, p4 = features

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

        # ========== ASGF 精炼 ==========
        p2_out, p3_out, p4_out = p2_fpn, p3_pan, p4_pan
        for layer in self.asgf_layers:
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

    print("\n=== Testing SpatialAttention ===")
    sa = SpatialAttention(2, 256).to(device)
    feat1 = torch.randn(bs, 256, 80, 80).to(device)
    feat2 = torch.randn(bs, 256, 80, 80).to(device)
    out = sa([feat1, feat2])
    print(f"  SpatialAttention: {[f.shape for f in [feat1, feat2]]} → {out.shape}")

    print("\n=== Testing GlobalContext ===")
    gc = GlobalContext(256, 64).to(device)
    out = gc(p3)
    print(f"  GlobalContext: {p3.shape} → {out.shape}")

    print("\n=== Testing ASGFLayer ===")
    asgf = ASGFLayer(out_channels, use_global=True).to(device)
    p2_out, p3_out, p4_out = asgf(p2, p3, p4)
    for name, out, exp in [
        ("P2", p2_out, (bs, 128, 160, 160)),
        ("P3", p3_out, (bs, 256, 80, 80)),
        ("P4", p4_out, (bs, 512, 40, 40))
    ]:
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{status}]")
    print(f"  ASGFLayer Params: {sum(p.numel() for p in asgf.parameters()):,}")

    print("\n=== Testing ASGFNeck ===")
    neck = ASGFNeck(channels, out_channels, num_asgf_layers=1, use_global=True).to(device)
    outs = neck([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in neck.parameters())
    print(f"  ASGFNeck Params: {total_params:,}")

    # 对比：不使用全局分支
    print("\n=== Testing ASGFNeck (no global) ===")
    neck_no_global = ASGFNeck(channels, out_channels, num_asgf_layers=1, use_global=False).to(device)
    outs = neck_no_global([p2, p3, p4])
    total_params_no_global = sum(p.numel() for p in neck_no_global.parameters())
    print(f"  ASGFNeck (no global) Params: {total_params_no_global:,}")
    print(f"  Global branch adds: {total_params - total_params_no_global:,} params")
