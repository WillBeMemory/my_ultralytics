import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class PRN(nn.Module):
    """
    Progressive Refinement Neck (PRN) - 修正版，已修复 use_refine=False 时的错误

    参数接口与 BiFPN/EFC_FPN 完全一致:
        channels: 输入通道列表 [c2, c3, c4] (对应 P2, P3, P4)
        out_channels: 输出通道列表 [o2, o3, o4] (若不指定则与输入相同)
        num_layers: PRN 的迭代精炼轮数 (官方默认 2)
        use_refine: 是否在每轮融合后使用 C3k2 精炼 (默认 True)

    YAML 调用格式:
        - [[2, 4, 6], 1, PRN, [[128, 256, 512], [128, 256, 512], 2, True]]
    """
    def __init__(self, channels, out_channels=None, num_layers=2, use_refine=False, expand_ch=None, **kwargs):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = channels
        o2, o3, o4 = out_channels

        self.num_layers = num_layers
        self.use_refine = use_refine

        # ========== Lateral Convs: 将各层输入投影到统一通道 ==========
        self.lat_conv2 = Conv(c2, o2, 1, act=False)
        self.lat_conv3 = Conv(c3, o3, 1, act=False)
        self.lat_conv4 = Conv(c4, o4, 1, act=False)

        # ========== 上/下采样算子 ==========
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down2 = Conv(o2, o3, 3, 2)  # P2 -> P3
        self.down3 = Conv(o3, o4, 3, 2)  # P3 -> P4

        # ========== 每轮精炼的融合模块 ==========
        self.td_fuse_layers = nn.ModuleList()
        self.bu_fuse_layers = nn.ModuleList()
        self.td_refine = nn.ModuleList()
        self.bu_refine = nn.ModuleList()

        for _ in range(num_layers):
            # Top-Down 融合卷积
            td_fuses = nn.ModuleDict({
                # P4_td: cur_p4 (o4) + backbone_p4 (o4) = 2*o4
                'fuse_p4': Conv(o4 * 2, o4, 1, act=False),
                # P3_td: cur_p3 (o3) + backbone_p3 (o3) + p4_up (o4) = 2*o3 + o4
                'fuse_p3': Conv(o3 * 2 + o4, o3, 1, act=False),
                # P2_td: cur_p2 (o2) + backbone_p2 (o2) + p3_up (o3) = 2*o2 + o3
                'fuse_p2': Conv(o2 * 2 + o3, o2, 1, act=False),
            })
            self.td_fuse_layers.append(td_fuses)

            # Bottom-Up 融合卷积
            bu_fuses = nn.ModuleDict({
                # P3_bu: p3_td (o3) + backbone_p3 (o3) + p2_down (o3) = 3*o3
                'fuse_p3': Conv(o3 * 3, o3, 1, act=False),
                # P4_bu: p4_td (o4) + backbone_p4 (o4) + p3_down (o4) = 3*o4
                'fuse_p4': Conv(o4 * 3, o4, 1, act=False),
            })
            self.bu_fuse_layers.append(bu_fuses)

            # 可选精炼模块
            if use_refine:
                td_refines = nn.ModuleDict({
                    'refine_p4': C3k2(o4, o4, n=1, shortcut=False, e=0.5),
                    'refine_p3': C3k2(o3, o3, n=1, shortcut=False, e=0.5),
                    'refine_p2': C3k2(o2, o2, n=1, shortcut=False, e=0.5),
                })
                bu_refines = nn.ModuleDict({
                    'refine_p3': C3k2(o3, o3, n=1, shortcut=False, e=0.5),
                    'refine_p4': C3k2(o4, o4, n=1, shortcut=False, e=0.5),
                })
            else:
                td_refines = nn.ModuleDict({
                    'refine_p4': nn.Identity(),
                    'refine_p3': nn.Identity(),
                    'refine_p2': nn.Identity(),
                })
                bu_refines = nn.ModuleDict({
                    'refine_p3': nn.Identity(),
                    'refine_p4': nn.Identity(),
                })
            self.td_refine.append(td_refines)
            self.bu_refine.append(bu_refines)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x
        else:
            return x

        if len(features) < 3:
            return features

        p2, p3, p4 = features

        # ====== 1. Lateral Convs: 统一通道 ======
        p2_lat = self.lat_conv2(p2)
        p3_lat = self.lat_conv3(p3)
        p4_lat = self.lat_conv4(p4)

        # 骨干原始特征引用 (PRN 核心思想: 多轮复用)
        backbone_p2 = p2_lat
        backbone_p3 = p3_lat
        backbone_p4 = p4_lat

        # 初始化当前特征
        cur_p2, cur_p3, cur_p4 = p2_lat, p3_lat, p4_lat

        for layer_idx in range(self.num_layers):
            # ====== 2. Top-Down 路径 ======
            # P4 与原始 P4 融合
            p4_td = torch.cat([cur_p4, backbone_p4], dim=1)
            p4_td = self.td_fuse_layers[layer_idx]['fuse_p4'](p4_td)
            p4_td = self.td_refine[layer_idx]['refine_p4'](p4_td)

            # P4 上采样并与 P3、原始 P3 融合
            p4_up = self.up(p4_td)
            p3_td = torch.cat([cur_p3, backbone_p3, p4_up], dim=1)
            p3_td = self.td_fuse_layers[layer_idx]['fuse_p3'](p3_td)
            p3_td = self.td_refine[layer_idx]['refine_p3'](p3_td)

            # P3 上采样并与 P2、原始 P2 融合
            p3_up = self.up(p3_td)
            p2_td = torch.cat([cur_p2, backbone_p2, p3_up], dim=1)
            p2_td = self.td_fuse_layers[layer_idx]['fuse_p2'](p2_td)
            p2_td = self.td_refine[layer_idx]['refine_p2'](p2_td)

            # ====== 3. Bottom-Up 路径 ======
            # P2 下采样并与 P3_td、原始 P3 融合
            p2_down = self.down2(p2_td)
            p3_bu = torch.cat([p3_td, backbone_p3, p2_down], dim=1)
            p3_bu = self.bu_fuse_layers[layer_idx]['fuse_p3'](p3_bu)
            p3_bu = self.bu_refine[layer_idx]['refine_p3'](p3_bu)

            # P3 下采样并与 P4_td、原始 P4 融合
            p3_down = self.down3(p3_bu)
            p4_bu = torch.cat([p4_td, backbone_p4, p3_down], dim=1)
            p4_bu = self.bu_fuse_layers[layer_idx]['fuse_p4'](p4_bu)
            p4_bu = self.bu_refine[layer_idx]['refine_p4'](p4_bu)

            # 更新当前特征，供下一轮精炼使用
            cur_p2, cur_p3, cur_p4 = p2_td, p3_bu, p4_bu

        return [cur_p2, cur_p3, cur_p4]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 Backbone 输出 P2, P3, P4
    p2 = torch.randn(2, 128, 160, 160).to(device)  # 高分辨率
    p3 = torch.randn(2, 256, 80, 80).to(device)
    p4 = torch.randn(2, 512, 40, 40).to(device)
    features = [p2, p3, p4]

    # 创建 PRN 实例，与 BiFPN 参数完全一致
    prn = PRN(
        channels=[128, 256, 512],  # 输入通道列表
        out_channels=[128, 256, 512],  # 输出通道列表
        num_layers=2,  # 迭代精炼次数
        use_refine=True  # 是否使用 C3k2 精炼
    ).to(device)
    print(prn)

    # 前向传播
    outputs = prn(features)
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

    # 参数量统计
    total_params = sum(p.numel() for p in prn.parameters())
    print(f"Total parameters: {total_params:,}")