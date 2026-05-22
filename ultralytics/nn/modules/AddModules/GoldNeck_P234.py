import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AlignCat_3in_at_P4(nn.Module):
    """将 P2/P3/P4 对齐到 P4 尺寸并拼接（不融合）"""
    def forward(self, inputs):
        p2, p3, p4 = inputs
        target_size = p4.shape[2:]   # 以 P4 尺寸为基准
        aligned = []
        for feat in [p2, p3, p4]:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(feat)
        return torch.cat(aligned, dim=1)


class AlignCat_2in_at_P3(nn.Module):
    """将 P2/P3 对齐到 P3 尺寸并拼接"""
    def forward(self, inputs):
        p2, p3 = inputs
        target_size = p3.shape[2:]
        if p2.shape[2:] != target_size:
            p2 = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        return torch.cat([p2, p3], dim=1)


class SimFusion_2in(nn.Module):
    """两路特征对齐拼接后 1×1 融合"""
    def __init__(self, out_channels):
        super().__init__()
        self.fuse = ConvBNSiLU(out_channels * 2, out_channels, 1)

    def forward(self, inputs):
        feat1, feat2 = inputs
        target_size = feat1.shape[2:]
        if feat2.shape[2:] != target_size:
            feat2 = F.interpolate(feat2, size=target_size, mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([feat1, feat2], dim=1))


class InjectionAdd(nn.Module):
    """注入模块：全局信息 + 局部特征（Add 方式）"""
    def __init__(self, channels):
        super().__init__()
        self.local_conv = ConvBNSiLU(channels, channels, 1)
        self.global_conv = ConvBNSiLU(channels, channels, 1)

    def forward(self, local_feat, global_feat):
        if global_feat.shape[2:] != local_feat.shape[2:]:
            global_feat = F.interpolate(global_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.local_conv(local_feat) + self.global_conv(global_feat)


class RepBlock(nn.Module):
    """轻量精炼块：1×1 + 3×3 深度卷积 + 1×1（残差）"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        layers = []
        for _ in range(n):
            layers.extend([
                ConvBNSiLU(c1, c2, 1),
                ConvBNSiLU(c2, c2, 3, 1, 1),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x) + x


class GoldNeck_P234(nn.Module):
    """
    Gold‑YOLO Gather‑and‑Distribute Neck（优化版：低分辨率全局融合）

    参数接口：
        channels: 输入通道列表 [c2, c3, c4] (对应 P2, P3, P4)
        out_channels: 输出通道列表 [o2, o3, o4] 或 None (默认与输入相同)
        expand_ch: 全局融合通道数，默认 128（大幅降低计算量）

    YAML 调用格式：
        - [[2, 4, 6], 1, GoldNeck_P234, [[128, 256, 512], [128, 256, 512], 1, True, 128]]
    """
    def __init__(self, channels, out_channels=None, num_layers=1, use_refine=True, expand_ch=128, **kwargs):
        super().__init__()
        c2, c3, c4 = channels

        if out_channels is None:
            out_channels = channels
        o2, o3, o4 = out_channels

        # 全局融合通道数（缩小至 128）
        self.fusion_ch = expand_ch

        # ========== 各层投影到统一通道（压缩） ==========
        self.lat_conv2 = ConvBNSiLU(c2, self.fusion_ch)
        self.lat_conv3 = ConvBNSiLU(c3, self.fusion_ch)
        self.lat_conv4 = ConvBNSiLU(c4, self.fusion_ch)

        # ========== Low‑GD 分支（在 P4 尺寸上进行） ==========
        # Low‑FAM：收集 P2/P3/P4 → 对齐到 P4 尺寸后拼接
        self.low_fam = AlignCat_3in_at_P4()
        # Low‑IFM：融合并输出两路全局信息
        self.low_ifm = nn.Sequential(
            ConvBNSiLU(self.fusion_ch * 3, self.fusion_ch, 1),
            RepBlock(self.fusion_ch, self.fusion_ch, n=1),
            ConvBNSiLU(self.fusion_ch, self.fusion_ch * 2, 1),
        )

        # P4 注入
        self.laf_p4 = SimFusion_2in(self.fusion_ch)
        self.inject_p4 = InjectionAdd(self.fusion_ch)
        self.rep_p4 = RepBlock(self.fusion_ch, self.fusion_ch, n=1)

        # P3 注入（局部使用 P2/P3 融合）
        self.laf_p3 = SimFusion_2in(self.fusion_ch)
        self.inject_p3 = InjectionAdd(self.fusion_ch)
        self.rep_p3 = RepBlock(self.fusion_ch, self.fusion_ch, n=1)

        # ========== High‑GD 分支（全局池化，计算量极低） ==========
        self.high_fam = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.fusion_ch, self.fusion_ch, 1))
            for _ in range(2)
        ])
        self.high_ifm = nn.Sequential(
            ConvBNSiLU(self.fusion_ch * 2, self.fusion_ch, 1),
            RepBlock(self.fusion_ch, self.fusion_ch, n=1),
            ConvBNSiLU(self.fusion_ch, self.fusion_ch * 2, 1),
        )

        # Bottom‑Up：P3 + P4 → N4
        self.laf_n4 = AlignCat_2in_at_P3()
        self.fuse_n4 = ConvBNSiLU(self.fusion_ch * 2, self.fusion_ch, 1)
        self.inject_n4 = InjectionAdd(self.fusion_ch)
        self.rep_n4 = RepBlock(self.fusion_ch, self.fusion_ch, n=1)
        self.down_sample_n4 = ConvBNSiLU(self.fusion_ch, self.fusion_ch, 3, 2, 1)  # 80→40

        # ========== 输出压缩到目标通道 ==========
        self.compress_p2 = ConvBNSiLU(self.fusion_ch, o2, 1)
        self.compress_p3 = ConvBNSiLU(self.fusion_ch, o3, 1)
        self.compress_p4 = ConvBNSiLU(self.fusion_ch, o4, 1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x
        else:
            return x

        p2, p3, p4 = features

        # 1. 投影到统一通道
        p2 = self.lat_conv2(p2)    # (B, C, 160, 160)
        p3 = self.lat_conv3(p3)    # (B, C, 80, 80)
        p4 = self.lat_conv4(p4)    # (B, C, 40, 40)

        # ==================== Low‑GD 分支（在 P4 尺寸上） ====================
        # 2a. 收集全局信息（对齐到 P4 尺寸）
        low_align = self.low_fam([p2, p3, p4])          # (B, 3C, 40, 40)
        low_fuse = self.low_ifm(low_align)              # (B, 2C, 40, 40)
        low_g4, low_g3 = low_fuse.chunk(2, dim=1)       # 各 (B, C, 40, 40)

        # 2b. P4 注入（在 40×40 上）
        p4_adj = self.laf_p4([p4, p3])                  # (B, C, 40, 40)
        p4_out = self.inject_p4(p4_adj, low_g4)         # (B, C, 40, 40)
        p4_out = self.rep_p4(p4_out)                    # (B, C, 40, 40)

        # 2c. P3 注入（在 80×80 上，low_g3 上采样）
        p3_adj = self.laf_p3([p3, p2])                  # (B, C, 80, 80)
        p3_out = self.inject_p3(p3_adj, low_g3)         # (B, C, 80, 80)
        p3_out = self.rep_p3(p3_out)                    # (B, C, 80, 80)

        # ==================== High‑GD 分支（极低计算量） ====================
        # 3a. 收集高级全局信息（P3_out, P4_out）
        high_feats = []
        for i, feat in enumerate([p3_out, p4_out]):
            g = self.high_fam[i](feat)                  # (B, C, 1, 1)
            high_feats.append(g)
        high_cat = torch.cat(high_feats, dim=1)         # (B, 2C, 1, 1)
        high_fuse = self.high_ifm(high_cat)             # (B, 2C, 1, 1)
        high_g3, high_g4 = high_fuse.chunk(2, dim=1)    # 各 (B, C, 1, 1)

        # 3b. 注入到 N4（对应输出 P4）
        n4_cat = self.laf_n4([p2, p3_out])              # (B, 2C, 80, 80)
        n4_adj = self.fuse_n4(n4_cat)                   # (B, C, 80, 80)
        n4_adj = self.inject_n4(n4_adj, high_g4)        # (B, C, 80, 80)
        n4_out = self.rep_n4(n4_adj)                    # (B, C, 80, 80)
        n4_out = self.down_sample_n4(n4_out)            # (B, C, 40, 40)

        # 3c. P3 注入
        n3_out = self.inject_p3(p3_out, high_g3)        # (B, C, 80, 80)
        n3_out = self.rep_p3(n3_out)                    # (B, C, 80, 80)

        # 4. 输出压缩
        o2 = self.compress_p2(p2)       # P2 原始细节
        o3 = self.compress_p3(n3_out)
        o4 = self.compress_p4(n4_out)

        return [o2, o3, o4]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 Backbone 输出 P2, P3, P4
    p2 = torch.randn(2, 128, 160, 160).to(device)  # 高分辨率
    p3 = torch.randn(2, 256, 80, 80).to(device)
    p4 = torch.randn(2, 512, 40, 40).to(device)
    features = [p2, p3, p4]

    # 创建 GoldNeck_P234 实例，参数与 BiFPN 风格一致
    neck = GoldNeck_P234(
        channels=[128, 256, 512],  # 输入通道列表 [c2, c3, c4]
        out_channels=[128, 256, 512],  # 输出通道列表 [o2, o3, o4]
    ).to(device)
    print(neck)

    # 前向传播
    outputs = neck(features)
    for i, out in enumerate(outputs, start=2):
        print(f"P{i}_out shape: {out.shape}")

    # 验证输出形状
    expected_shapes = [
        (2, 128, 160, 160),  # P2_out
        (2, 256, 80, 80),  # P3_out
        (2, 512, 40, 40)  # P4_out
    ]
    for i, (out, expected) in enumerate(zip(outputs, expected_shapes)):
        assert out.shape == expected, f"Shape mismatch at P{i + 2}_out: {out.shape} vs {expected}"
    print("✅ All output shapes verified.")

    # 梯度测试
    loss = sum(o.mean() for o in outputs)
    loss.backward()
    print("✅ Backward pass succeeded.")

    total_params = sum(p.numel() for p in neck.parameters())
    print(f"Total parameters: {total_params:,}")